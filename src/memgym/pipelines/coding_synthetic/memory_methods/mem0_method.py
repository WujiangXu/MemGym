"""Mem0 adapter for the coding_synthetic evaluation pipeline.

Wraps `memgym.memory.external.mem0.Mem0System` (shared with the IR
pipeline's `ir_mem0` backend) behind the `CodingMemoryMethod`
protocol.
"""
from __future__ import annotations

from typing import Optional


class Mem0Method:
    """Mem0 (LLM extract+update fact memory) for coding QA."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 5,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_ingest_chars: int = 10000,
    ) -> None:
        self._max_ingest_chars = max_ingest_chars
        self._retrieve_k = retrieve_k
        from memgym.memory.external.mem0 import Mem0System

        self._system = Mem0System(
            llm_model=llm_model,
            embedding_model=embedding_model,
            retrieve_k=retrieve_k,
            api_base=api_base,
            api_key=api_key,
        )

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        self._system.add_doc(doc_name, doc_content[: self._max_ingest_chars])

    def retrieve(self, question: str, task_prompt: str) -> str:
        passages = self._system.retrieve(question, k=self._retrieve_k)
        if not passages:
            return "(no relevant memories found)"
        return "\n---\n".join(passages)

    def reset(self) -> None:
        self._system.reset()
