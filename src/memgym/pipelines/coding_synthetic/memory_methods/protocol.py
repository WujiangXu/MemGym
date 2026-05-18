"""CodingMemoryMethod protocol and prompt-strategy wrapper.

The protocol defines the interface that any memory method must implement
to be used in the coding_synthetic evaluation pipeline. The key design
choice: the training phase calls ``retrieve(question)`` per QA pair, so methods
with real retrieval (A-MEM, LightMEM) return *query-specific* context
while the prompt wrapper always returns the full accumulated notes.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..llm.client import LLMClient
from ..llm.prompts import NOTE_TAKING_QA_PROMPT
from ..types.schemas import NOTE_TAKING_QA_SCHEMA


@runtime_checkable
class CodingMemoryMethod(Protocol):
    """Interface for pluggable memory methods in coding QA evaluation."""

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        """Process one document during the training phase (sequential reading).

        Called once per document in the order they appear in
        ``instance.memory_files``.  The method may store, index, or
        summarize the document however it likes.
        """
        ...

    def retrieve(self, question: str, task_prompt: str) -> str:
        """Return context for answering *question* during the training phase.

        The returned string is injected into the ``{notes}`` slot of
        ``EVICTED_ANSWER_QA_PROMPT_MEMORY_ONLY``.  Methods with real
        retrieval should return only the most relevant memories;
        prompt-based wrappers return the full accumulated notes.
        """
        ...

    def reset(self) -> None:
        """Clear all state.  Called once per instance before ingestion."""
        ...


class PromptStrategyMethod:
    """Wrap an existing prompt-only strategy as a ``CodingMemoryMethod``.

    This preserves backward compatibility: the same note-taking prompt
    and LLM call as before, but packaged behind the protocol so the eval
    loop can treat it identically to A-MEM or LightMEM.
    """

    def __init__(
        self,
        strategy: str,
        client: LLMClient,
        budget: int = 500,
    ) -> None:
        from ..stages.qa.eval import STRATEGY_INSTRUCTIONS

        self.strategy = strategy
        self.client = client
        self.budget = budget
        self.strategy_instruction = STRATEGY_INSTRUCTIONS.get(
            strategy, STRATEGY_INSTRUCTIONS["basic"]
        )
        self._accumulated_notes: str = ""
        self._doc_idx: int = 0
        self._total_docs: int = 0

    # -- CodingMemoryMethod interface ----------------------------------------

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        self._doc_idx += 1
        if self._accumulated_notes:
            prev_section = (
                f"Your notes from previous documents:\n{self._accumulated_notes}"
            )
        else:
            prev_section = "This is the first document \u2014 no previous notes."

        prompt = NOTE_TAKING_QA_PROMPT.format(
            task_prompt=task_prompt,
            filename=doc_name,
            doc_index=self._doc_idx,
            total_docs=self._total_docs,
            document_content=doc_content[:15000],
            previous_notes_section=prev_section,
            strategy_instruction=self.strategy_instruction,
            budget=self.budget,
        )
        result = self.client.get_json_completion(
            prompt=prompt,
            response_format=NOTE_TAKING_QA_SCHEMA,
            temperature=0.0,
        )
        self._accumulated_notes = result.get("notes", "")

    def retrieve(self, question: str, task_prompt: str) -> str:
        return self._accumulated_notes

    def reset(self) -> None:
        self._accumulated_notes = ""
        self._doc_idx = 0
        self._total_docs = 0

    def set_total_docs(self, n: int) -> None:
        """Set total document count (for prompt formatting)."""
        self._total_docs = n
