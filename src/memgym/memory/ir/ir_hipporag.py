"""HippoRAG adapter for IR evaluation.

Wraps the shared `HippoRAGSystem` core (also used by the coding-synthetic
pipeline's `hipporag` method) behind the IR `BaseMemoryManager` interface.

Each `manage_context` call buffers the observation as a HippoRAG document.
`retrieve_for_question` triggers the (idempotent) OpenIE + PPR index build
on first call, then runs PPR-ranked retrieval over the resulting graph.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..external.hipporag.system import HippoRAGSystem


class IRHippoRAGMemory(BaseMemoryManager):
    """HippoRAG (knowledge-graph + Personalized PageRank) for IR retrieval."""

    def __init__(
        self,
        max_tokens: int = 100000,
        llm_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        question: str = "",
        max_ingest_chars: int = 15000,
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

    def _create_system(self) -> HippoRAGSystem:
        return HippoRAGSystem(
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
        """Buffer the observation as a HippoRAG document (lazy-indexed)."""
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
                "strategy": "ir_hipporag",
                "num_docs": self._turn_id,
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Run PPR-ranked retrieval. Triggers index build on first call."""
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


register_memory_model("ir_hipporag", IRHippoRAGMemory)
