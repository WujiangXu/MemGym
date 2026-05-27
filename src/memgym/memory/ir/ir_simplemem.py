"""SimpleMem adapter for IR evaluation.

Wraps the shared `SimpleMemBackend` core (also used by the coding-synthetic
pipeline's `simplemem` method) behind the IR `BaseMemoryManager` interface.

SimpleMem expects a dialogue stream `(speaker, text, timestamp)`; we map
each turn observation onto a single `add_dialogue("system", text, ts)`
with a synthetic monotonic timestamp. Retrieval bypasses SimpleMem's
own answerer LLM (`ask()`) and reads the underlying hybrid retriever
directly so this pipeline owns the answerer.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..external.simplemem.system import SimpleMemBackend


class IRSimpleMemMemory(BaseMemoryManager):
    """SimpleMem (semantic structured compression) for IR retrieval."""

    def __init__(
        self,
        max_tokens: int = 100000,
        llm_model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        retrieve_k: int = 10,
        question: str = "",
        max_ingest_chars: int = 15000,
        enable_planning: bool = True,
        enable_reflection: bool = False,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self._llm_model = llm_model
        self._api_base = api_base
        self._api_key = api_key
        self._retrieve_k = retrieve_k
        self._max_ingest_chars = max_ingest_chars
        self._enable_planning = enable_planning
        self._enable_reflection = enable_reflection
        self._all_observations: List[str] = []
        self._turn_id = 0
        self._system = self._create_system()

    def _create_system(self) -> SimpleMemBackend:
        return SimpleMemBackend(
            llm_model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
            retrieve_k=self._retrieve_k,
            enable_planning=self._enable_planning,
            enable_reflection=self._enable_reflection,
        )

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Buffer the observation as a SimpleMem dialogue (lazy-finalized)."""
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
                "strategy": "ir_simplemem",
                "num_docs": self._turn_id,
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Run hybrid retrieval. Triggers `finalize()` on first call."""
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


register_memory_model("ir_simplemem", IRSimpleMemMemory)
