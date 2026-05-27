"""LightMEM adapter for the coding_synthetic evaluation pipeline.

Wraps the LightMEM 3-stage memory system (zjunlp/LightMem, ICLR 2026)
behind the ``CodingMemoryMethod`` protocol.

LightMEM was designed for chat histories with timestamps. We adapt it
for document-based ingestion by:
  - Disabling sensory-memory (LLMLingua-2 compression) and topic
    segmentation — our documents are already structured, not raw chat.
  - Feeding each document as a single forced-extract "user" message.
  - Using embedding-based retrieval (Qdrant, local on-disk).
  - **Overriding the upstream chat-only extraction prompt** with a
    domain-neutral fact-extraction prompt (modelled on A-MEM's
    ``METADATA_PROMPT``). The upstream
    ``METADATA_GENERATE_PROMPT`` is hard-coded as a "Personal Information
    Extractor" for User dialogues (Alice/Paris-style examples) which
    silently destroys signal on code/document inputs and was the root
    cause of LightMem scoring below ``truncated``/``prompt`` on
    coding-synth Phase 5. We pass our neutral prompt via the official
    per-call ``METADATA_GENERATE_PROMPT`` argument of
    ``LightMemory.add_memory`` (``lightmem.py:204-211``); the output
    schema (``{source_id, fact}``) is preserved unchanged so the
    downstream Qdrant indexer is not affected.

This tests LightMEM's core: LLM-based knowledge extraction + metadata
generation + embedding index + vector retrieval.

Requirements:
  pip install lightmem  (Python 3.10/3.11 — upstream pins python<3.12)
  A running or local Qdrant instance, or use Qdrant local file mode.
  OPENAI_API_KEY / OPENAI_API_BASE for the LLM backend, or configure
  litellm proxy for Bedrock.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


# Domain-neutral replacement for upstream ``METADATA_GENERATE_PROMPT``.
# Lives in a shared module so the IR-side adapter (which talks to
# LightMem through a JSON-over-stdio subprocess in an isolated venv)
# can import the same canonical text. See
# ``src/memgym/memory/_lightmem_prompts.py`` for the full rationale.
from memgym.memory.strategies._lightmem_prompts import NEUTRAL_METADATA_PROMPT as _NEUTRAL_METADATA_PROMPT


class LightMemMethod:
    """LightMEM (3-stage memory) for coding QA evaluation.

    Each document is ingested via ``add_memory()`` with forced extraction,
    which runs LightMEM's knowledge extraction pipeline (LLM-based fact
    extraction + optional metadata generation) and stores results in a
    Qdrant vector index.  At retrieval time, embedding search returns the
    most relevant extracted memories.

    Parameters
    ----------
    llm_model : str
        Model name for the OpenAI-compatible backend (e.g. "gpt-4o-mini").
    embedding_model : str
        Path or HuggingFace name for sentence-transformers embedder.
    embedding_dims : int
        Embedding dimension (384 for all-MiniLM-L6-v2).
    retrieve_k : int
        Number of memories to retrieve per query.
    api_base : str, optional
        OpenAI API base URL (e.g. litellm proxy for Bedrock).
    api_key : str, optional
        OpenAI API key.
    enable_metadata : bool
        If True, generate metadata for each memory entry.
    enable_summary : bool
        If True, store summarized memory text (otherwise raw).
    qdrant_dir : str, optional
        Directory for Qdrant on-disk storage. Uses temp dir if None.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dims: int = 384,
        retrieve_k: int = 10,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_metadata: bool = True,
        enable_summary: bool = True,
        qdrant_dir: Optional[str] = None,
        max_ingest_chars: int = 15000,
    ) -> None:
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._embedding_dims = embedding_dims
        self._retrieve_k = retrieve_k
        self._api_base = api_base
        self._api_key = api_key
        self._enable_metadata = enable_metadata
        self._enable_summary = enable_summary
        self._qdrant_dir = qdrant_dir or tempfile.mkdtemp(prefix="lightmem_qdrant_")
        self._doc_counter = 0
        self._base_time = datetime(2024, 1, 1, 9, 0)
        self._max_ingest_chars = max_ingest_chars
        self._system = self._create_system()

    def _create_system(self) -> Any:
        from lightmem.memory.lightmem import LightMemory

        collection = f"memgym_{uuid.uuid4().hex[:8]}"
        qdrant_path = os.path.join(self._qdrant_dir, collection)

        config = {
            # Disable heavy preprocessing — our docs are already structured
            "pre_compress": False,
            "topic_segment": False,
            # Use all message content (documents go in as "user" messages)
            "messages_use": "user_only",
            # LLM-based extraction and metadata
            "metadata_generate": self._enable_metadata,
            "text_summary": self._enable_summary,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self._llm_model,
                    **({"api_key": self._api_key} if self._api_key else {}),
                    **({"openai_base_url": self._api_base} if self._api_base else {}),
                },
            },
            # Force extraction on every add_memory call
            "extract_threshold": 0.0,
            # Embedding-based indexing and retrieval
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": self._embedding_model,
                    "embedding_dims": self._embedding_dims,
                    # LightMem's HuggingFace embedder unpacks model_kwargs
                    # via **config.model_kwargs; None blows up with
                    # "argument after ** must be a mapping, not NoneType".
                    "model_kwargs": {},
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection,
                    "embedding_model_dims": self._embedding_dims,
                    "path": qdrant_path,
                },
            },
            "update": "offline",
        }

        # Set env vars for the OpenAI client if provided
        if self._api_key:
            os.environ.setdefault("OPENAI_API_KEY", self._api_key)
        if self._api_base:
            os.environ.setdefault("OPENAI_API_BASE", self._api_base)

        self._collection = collection
        return LightMemory.from_config(config)

    def _make_timestamp(self) -> str:
        """Generate a synthetic timestamp for the next document."""
        ts = self._base_time + timedelta(hours=self._doc_counter)
        weekday = ts.strftime("%a")
        return ts.strftime(f"%Y/%m/%d ({weekday}) %H:%M")

    # -- CodingMemoryMethod interface ----------------------------------------

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        """Ingest one document through LightMEM's extraction pipeline.

        Each document becomes a single "user" message with forced
        segmentation and extraction, so the full LLM extraction pipeline
        runs immediately (no buffering across documents).

        We override the upstream chat-only extraction prompt by passing
        ``METADATA_GENERATE_PROMPT=_NEUTRAL_METADATA_PROMPT`` to
        ``add_memory``. ``task_prompt`` is intentionally not threaded into
        the storage prompt — A-MEM's design (which is the empirical
        comparison anchor here) keeps storage prompts domain-neutral and
        only consumes ``task_description`` at retrieval time for query
        rewrite. Mirroring that here keeps the across-method comparison
        apples-to-apples at the storage layer.
        """
        timestamp = self._make_timestamp()
        self._doc_counter += 1

        message = {
            "role": "user",
            "content": f"[Document: {doc_name}]\n{doc_content[: self._max_ingest_chars]}",
            "time_stamp": timestamp,
        }
        self._system.add_memory(
            messages=[message],
            METADATA_GENERATE_PROMPT=_NEUTRAL_METADATA_PROMPT,
            force_segment=True,
            force_extract=True,
        )

    def retrieve(self, question: str, task_prompt: str) -> str:
        """Retrieve relevant memories for a question via embedding search."""
        result = self._system.retrieve(query=question, limit=self._retrieve_k)
        if isinstance(result, list):
            result = "\n".join(str(r) for r in result)
        return result if result else "(no relevant memories found)"

    def reset(self) -> None:
        """Clear all state by creating a fresh LightMemory instance."""
        self._doc_counter = 0
        self._system = self._create_system()
