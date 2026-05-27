"""MemoryBank adapter for the coding_synthetic evaluation pipeline.

Wraps `memgym.memory.external.memorybank.MemoryBankSystem` (shared with the
IR pipeline's `ir_memorybank` backend) behind the `CodingMemoryMethod`
protocol.
"""
from __future__ import annotations


class MemoryBankMethod:
    """MemoryBank (Ebbinghaus decay + retrieval reinforcement) for coding QA."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        chunk_chars: int = 1500,
        decay_hours_scale: float = 24.0,
        reinforcement_factor: float = 1.5,
        strength_cap: float = 100.0,
        max_ingest_chars: int = 15000,
    ) -> None:
        self._max_ingest_chars = max_ingest_chars
        self._retrieve_k = retrieve_k
        from memgym.memory.external.memorybank import MemoryBankSystem

        self._system = MemoryBankSystem(
            embedding_model=embedding_model,
            retrieve_k=retrieve_k,
            chunk_chars=chunk_chars,
            decay_hours_scale=decay_hours_scale,
            reinforcement_factor=reinforcement_factor,
            strength_cap=strength_cap,
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
