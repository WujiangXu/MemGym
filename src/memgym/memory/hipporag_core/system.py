"""Shared HippoRAG core (`pip install hipporag`, v2 main, OSU-NLP-Group).

Both the coding-synthetic and memgym-ir pipelines wrap this class behind a
thin pipeline-specific adapter. The split mirrors how `AgenticMemorySystem`
is shared by `AMemMethod` and `IRAMemMemory`.

Design points:
- Lazy-commit: `add_doc` buffers; the expensive OpenIE + KG + PPR build
  fires once on first `retrieve()` (or explicitly via `commit()`). This
  matches HippoRAG's batch-style `index(docs=...)` API and keeps cost in
  line with the the training phase hang fix's per-instance ingest budget.
- Per-instance storage dir under a session-level temp prefix (mirrors
  `LightMemMethod._qdrant_dir`).
- OpenAI-compatible env vars set in __init__ so litellm proxies / Bedrock
  gateways work transparently.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import List, Optional


class HippoRAGSystem:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 5,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        # HippoRAG instantiates the official `openai.OpenAI` client directly
        # (see hipporag/llm/openai_gpt.py); it does NOT route through litellm.
        # Callers commonly pass the litellm-style "openai/<model>" prefix
        # because that's what the coding-synthetic harness uses for its main
        # answerer. Strip it so the OpenAI SDK gets a real model name.
        if isinstance(llm_model, str) and llm_model.startswith("openai/"):
            llm_model = llm_model[len("openai/"):]
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._api_base = api_base
        self._api_key = api_key
        self._embedding_base_url = embedding_base_url
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="hipporag_")
        self._buffered_docs: List[str] = []
        self._committed = False
        self._system = None
        # Override (not setdefault) so a stale env var can't shadow an
        # explicit caller-provided key.
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
            os.environ["OPENAI_BASE_URL"] = api_base

    def _create_system(self):
        from hipporag import HippoRAG

        save_dir = os.path.join(
            self._storage_dir, f"hipporag_{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(save_dir, exist_ok=True)
        return HippoRAG(
            save_dir=save_dir,
            llm_model_name=self._llm_model,
            llm_base_url=self._api_base,
            embedding_model_name=self._embedding_model,
            embedding_base_url=self._embedding_base_url,
        )

    def add_doc(self, doc_id: str, text: str) -> None:
        """Buffer one document for indexing on first retrieve."""
        if not text or not text.strip():
            return
        # Prefix with doc_id so OpenIE has a stable label per chunk.
        self._buffered_docs.append(f"[{doc_id}]\n{text}")
        # Adding new docs after a commit means we need to re-index.
        self._committed = False

    def commit(self) -> None:
        """Run HippoRAG.index over buffered docs (idempotent)."""
        if self._committed or not self._buffered_docs:
            return
        if self._system is None:
            self._system = self._create_system()
        self._system.index(docs=list(self._buffered_docs))
        self._committed = True

    def retrieve(self, question: str, k: Optional[int] = None) -> List[str]:
        """Retrieve top-k passages. Lazy-commits if needed.

        HippoRAG's `graph_search_with_fact_entities` asserts when no query
        NER phrase overlaps the indexed graph (HippoRAG.py:1412). On a
        sparse graph (small index) or out-of-domain queries this can fire
        even on production-quality data. Catch it and return empty so the
        eval loop keeps going — the answerer will then see
        "(no relevant memories found)" instead of the whole run crashing.
        """
        if not self._buffered_docs:
            return []
        self.commit()
        top_k = k or self._retrieve_k
        try:
            solutions = self._system.retrieve(
                queries=[question], num_to_retrieve=top_k
            )
        except (AssertionError, RuntimeError, ValueError, ZeroDivisionError):
            return []
        if not solutions:
            return []
        first = solutions[0] if isinstance(solutions, list) else solutions
        docs = getattr(first, "docs", None)
        if docs is None:
            try:
                first = solutions[0][0]
                docs = getattr(first, "docs", [])
            except Exception:
                docs = []
        return list(docs or [])

    def reset(self) -> None:
        self._buffered_docs.clear()
        self._committed = False
        self._system = None
