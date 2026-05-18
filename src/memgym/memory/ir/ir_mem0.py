"""Mem0 adapter for IR evaluation.

Wraps the shared `Mem0System` core (also used by the coding-synthetic
pipeline's `mem0` method) behind the IR `BaseMemoryManager` interface.
Each `manage_context` call hands one observation to Mem0 (which runs
the LLM extract+update step). `retrieve_for_question` runs vector
similarity search over the resulting fact store.
"""
from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..mem0_core.system import Mem0System


class IRMem0Memory(BaseMemoryManager):
    """Mem0 (LLM extract+update fact memory) for IR retrieval."""

    def __init__(
        self,
        max_tokens: int = 100000,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        question: str = "",
        max_ingest_chars: int = 10000,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._api_base = api_base
        self._api_key = api_key
        self._max_ingest_chars = max_ingest_chars
        self._all_observations: List[str] = []
        self._turn_id = 0
        self._system = self._create_system()

    def _create_system(self) -> Mem0System:
        return Mem0System(
            llm_model=self._llm_model,
            embedding_model=self._embedding_model,
            retrieve_k=self._retrieve_k,
            api_base=self._api_base,
            api_key=self._api_key,
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
                "strategy": "ir_mem0",
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


register_memory_model("ir_mem0", IRMem0Memory)
