"""SimpleMem adapter for the coding_synthetic evaluation pipeline.

Wraps `memgym.memory.simplemem.SimpleMemBackend` (shared with the IR
pipeline's `ir_simplemem` backend) behind the `CodingMemoryMethod`
protocol.
"""
from __future__ import annotations

from typing import Any, Optional


class SimpleMemMethod:
    """SimpleMem (semantic structured compression) for coding QA."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        retrieve_k: int = 5,
        max_ingest_chars: int = 15000,
        enable_planning: bool = True,
        enable_reflection: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ) -> None:
        self._max_ingest_chars = max_ingest_chars
        self._retrieve_k = retrieve_k
        from memgym.memory.simplemem_core import SimpleMemBackend

        self._system = SimpleMemBackend(
            llm_model=llm_model,
            api_base=api_base,
            api_key=api_key,
            retrieve_k=retrieve_k,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

    # -- CodingMemoryMethod interface ----------------------------------------

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        self._system.add_doc(doc_name, doc_content[: self._max_ingest_chars])

    def retrieve(self, question: str, task_prompt: str) -> str:
        passages = self._system.retrieve(question, k=self._retrieve_k)
        if not passages:
            return "(no relevant memories found)"
        return "\n---\n".join(passages)

    def reset(self) -> None:
        self._system.reset()
