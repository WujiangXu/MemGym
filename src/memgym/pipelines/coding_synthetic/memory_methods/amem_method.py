"""A-MEM adapter for the coding_synthetic evaluation pipeline.

Wraps the existing ``AgenticMemorySystem`` (src/memgym/memory/amem/system.py)
behind the ``CodingMemoryMethod`` protocol.  Key difference from prompt-based
strategies: ``retrieve()`` does per-question vector+graph search via
``find_related_memories_raw`` instead of dumping a flat notes string.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AMemMethod:
    """A-MEM (Zettelkasten agentic memory) for coding QA evaluation.

    Each document is ingested as a memory note with LLM-generated metadata
    (keywords, context, tags) and Zettelkasten-style links to related notes.
    At retrieval time, vector search + graph neighbor expansion returns the
    most relevant memories for the specific question.
    """

    def __init__(
        self,
        llm_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 5,
        enable_evolution: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_ingest_chars: int = 15000,
    ) -> None:
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._enable_evolution = enable_evolution
        self._api_base = api_base
        self._api_key = api_key
        self._max_ingest_chars = max_ingest_chars
        self._system = self._create_system()

    def _create_system(self) -> Any:
        from memgym.memory.amem.system import AgenticMemorySystem

        return AgenticMemorySystem(
            model_name=self._embedding_model,
            llm_model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
            retrieve_k=self._retrieve_k,
            enable_evolution=self._enable_evolution,
            evo_threshold=50,
            verbose=False,
        )

    # -- CodingMemoryMethod interface ----------------------------------------

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        self._system.set_task_description(task_prompt)
        self._system.add_note(
            content=doc_content[: self._max_ingest_chars],
            tags=[doc_name],
        )

    def retrieve(self, question: str, task_prompt: str) -> str:
        result = self._system.find_related_memories_raw(
            query=question, k=self._retrieve_k
        )
        return result if result else "(no relevant memories found)"

    def reset(self) -> None:
        self._system = self._create_system()
