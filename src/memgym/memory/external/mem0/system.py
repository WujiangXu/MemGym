"""Shared Mem0 core (`pip install mem0ai`).

Mem0's contribution is its add-time fact extraction + conflict resolution:
each `add()` call invokes the LLM to extract atomic facts, embed them,
and decide whether the new facts ADD/UPDATE/DELETE prior memories.
Retrieval is a vector similarity search over the resulting fact store.

We wrap `mem0.Memory` behind the standard `add_doc` / `retrieve` / `reset`
interface so it slots into the existing per-pipeline adapters without
each adapter needing to know Mem0's user_id-keyed API.

Cost note: every `add_doc` call is one LLM round-trip (extract+update).
On 500k/1m context buckets this is the dominant wall-clock cost — the
coding-synth adapter respects `--ingest-max-calls` and clamps
`--ingest-chars-per-call` to keep cell wall-clocks under ~12 h.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import List, Optional


class Mem0System:
    """Wrapper around `mem0.Memory` — extract+update fact memory."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 5,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        if isinstance(llm_model, str) and llm_model.startswith("openai/"):
            llm_model = llm_model[len("openai/") :]
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._api_base = api_base
        self._api_key = api_key
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="mem0_")
        self._user_id = f"coding_qa_{uuid.uuid4().hex[:8]}"
        self._memory = None  # lazy

    def _create_system(self):
        from mem0 import Memory

        # LiteLLM provider supports both OpenAI- and Bedrock-flavored model
        # IDs uniformly, matching the rest of the MemGym pipeline's LLM
        # routing. HuggingFace embedder uses the unified MiniLM-v2 default.
        config = {
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": self._llm_model,
                    "api_base": self._api_base,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": self._embedding_model},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": os.path.join(self._storage_dir, "qdrant"),
                    "collection_name": f"mem0_{uuid.uuid4().hex[:8]}",
                },
            },
        }
        return Memory.from_config(config)

    def add_doc(self, doc_id: str, text: str) -> None:
        if not text or not text.strip():
            return
        if self._memory is None:
            self._memory = self._create_system()
        self._memory.add(
            messages=f"[{doc_id}]\n{text}",
            user_id=self._user_id,
            metadata={"doc_id": doc_id},
        )

    def retrieve(self, question: str, k: Optional[int] = None) -> List[str]:
        if self._memory is None:
            return []
        result = self._memory.search(
            query=question,
            user_id=self._user_id,
            limit=k or self._retrieve_k,
        )
        if isinstance(result, dict):
            items = result.get("results", []) or []
        else:
            items = result or []
        passages: List[str] = []
        for item in items:
            if isinstance(item, dict):
                passages.append(
                    item.get("memory") or item.get("text") or str(item)
                )
            else:
                passages.append(str(item))
        return passages

    def reset(self) -> None:
        # Rotate the user_id rather than tearing down the embedder + Qdrant
        # collection. This gives us a clean memory slate at zero load cost,
        # which matters when Phase-5 reuses the same backend across many
        # eval instances.
        self._user_id = f"coding_qa_{uuid.uuid4().hex[:8]}"
