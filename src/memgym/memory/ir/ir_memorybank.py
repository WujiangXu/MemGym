"""MemoryBank adapter for IR evaluation.

Wraps the shared `MemoryBankSystem` core (vendored Ebbinghaus decay +
reinforcement algorithm) behind the IR `BaseMemoryManager` interface.

Note: the synthetic IR eval has no real wall-clock spacing between turns,
so the decay term is approximately constant within an instance and the
discriminating signal comes from the reinforcement effect on units that
get retrieved across multiple sub-questions of one multi-hop query.
"""
from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..external.memorybank.system import MemoryBankSystem


class IRMemoryBankMemory(BaseMemoryManager):
    """MemoryBank (Ebbinghaus decay + reinforcement) for IR retrieval."""

    def __init__(
        self,
        max_tokens: int = 100000,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        chunk_chars: int = 1500,
        decay_hours_scale: float = 24.0,
        reinforcement_factor: float = 1.5,
        strength_cap: float = 100.0,
        question: str = "",
        max_ingest_chars: int = 15000,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._chunk_chars = chunk_chars
        self._decay_hours_scale = decay_hours_scale
        self._reinforcement_factor = reinforcement_factor
        self._strength_cap = strength_cap
        self._max_ingest_chars = max_ingest_chars
        self._all_observations: List[str] = []
        self._turn_id = 0
        self._system = self._create_system()

    def _create_system(self) -> MemoryBankSystem:
        return MemoryBankSystem(
            embedding_model=self._embedding_model,
            retrieve_k=self._retrieve_k,
            chunk_chars=self._chunk_chars,
            decay_hours_scale=self._decay_hours_scale,
            reinforcement_factor=self._reinforcement_factor,
            strength_cap=self._strength_cap,
        )

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            self._system.add_doc(
                f"turn_{self._turn_id}",
                text[: self._max_ingest_chars],
            )
            self._turn_id += 1

        tokens = self.count_tokens(self._all_observations)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_memorybank",
                "num_docs": self._turn_id,
            },
        )

    def retrieve_for_question(
        self, question: str, top_k: Optional[int] = None
    ) -> str:
        k = top_k or self._retrieve_k
        passages = self._system.retrieve(question, k=k)
        if not passages:
            return "(no relevant memories found)"
        return "\n---\n".join(passages)

    def get_notes_text(self) -> str:
        return ""

    def reset(self) -> None:
        self._system.reset()
        self._all_observations.clear()
        self._turn_id = 0
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "num_docs": self._turn_id,
            "retrieve_k": self._retrieve_k,
        })
        return stats


register_memory_model("ir_memorybank", IRMemoryBankMemory)
