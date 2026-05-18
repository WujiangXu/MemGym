"""IR-specific LLM Summarizing Memory for research trajectories.

Subclasses LLMSummarizingMemory but uses an IR-specific summarization prompt
that tracks research-relevant fields: entities discovered, bridge facts,
key evidence, candidate answers, and unresolved gaps.
"""

from typing import Any, Dict, List, Optional

import litellm

from ..base import BaseMemoryManager, FilteredContext, register_memory_model

# IR-specific summarization prompt (replaces the generic SWE-bench one)
IR_SUMMARIZATION_PROMPT = """\
You are summarizing a research agent's investigation so far. The agent is \
searching the web to answer a complex multi-hop question. Some of the earlier \
search results are being evicted from context — your summary must preserve \
the critical information.

RESEARCH QUESTION: {question}

PREVIOUS SUMMARY (if any):
{previous_summary}

SEARCH RESULTS BEING EVICTED:
{forgotten_text}

Create a concise summary that preserves:
1. ENTITIES FOUND — specific names, dates, numbers discovered
2. BRIDGE FACTS — facts that connect one entity to another (critical for multi-hop)
3. KEY EVIDENCE — facts that directly help answer the question
4. CANDIDATE ANSWER — best current answer based on evidence so far
5. CONTRADICTIONS — conflicting information found across sources
6. UNRESOLVED — what still needs to be searched

Be specific and factual. Do NOT include vague statements like "several sources were found".
Include exact names, dates, and relationships.
"""


class IRSummarizingMemory(BaseMemoryManager):
    """LLM-based summarization adapted for IR research trajectories.

    When the accumulated context exceeds max_size messages, uses an LLM
    to summarize evicted turns into an IR-focused summary that tracks
    entities, bridge facts, evidence, and candidate answers.
    """

    def __init__(
        self,
        max_size: int = 10,
        keep_first: int = 1,
        summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        condensation_ratio: float = 0.75,
        max_tokens: int = 4000,
        question: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.max_size = max_size
        self.keep_first = keep_first
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.question = question

        self._history: List[Any] = []
        self._summary = ""
        self._condensation_count = 0

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        # Append current observation
        if current_observation:
            self._history.append(current_observation)

        # Check if condensation needed
        if len(self._history) > self.max_size:
            self._condense()

        # Build view: summary (if any) + current history
        view = []
        if self._summary:
            view.append(f"[RESEARCH SUMMARY]\n{self._summary}")
        view.extend(self._history)

        tokens = self.count_tokens(view)
        return FilteredContext(
            content=view,
            metadata={
                "tokens": tokens,
                "was_compacted": self._condensation_count > 0,
                "condensation_count": self._condensation_count,
                "strategy": "ir_summarizing",
                "history_size": len(self._history),
            },
        )

    def _condense(self):
        """Summarize older entries, keep recent ones."""
        target = int(self.max_size * self.condensation_ratio)
        keep_count = max(self.keep_first, target)

        # Split into forgotten and kept
        forgotten = self._history[:len(self._history) - keep_count]
        kept = self._history[len(self._history) - keep_count:]

        if not forgotten:
            return

        forgotten_text = "\n\n".join(str(item) for item in forgotten)

        # Call LLM to summarize
        prompt = IR_SUMMARIZATION_PROMPT.format(
            question=self.question,
            previous_summary=self._summary or "(none)",
            forgotten_text=forgotten_text,
        )

        try:
            response = litellm.completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": "You are a research summarizer. Be concise and factual."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            self._summary = response.choices[0].message.content
        except Exception as e:
            # Fallback: just concatenate the forgotten text
            self._summary = (self._summary + "\n" + forgotten_text)[:2000]

        self._history = kept
        self._condensation_count += 1

    def reset(self) -> None:
        self._history = []
        self._summary = ""
        self._condensation_count = 0

    def get_notes_text(self) -> str:
        """Return current memory state as text."""
        parts = []
        if self._summary:
            parts.append(f"[RESEARCH SUMMARY]\n{self._summary}")
        parts.extend(str(item) for item in self._history)
        return "\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "condensation_count": self._condensation_count,
            "history_size": len(self._history),
            "has_summary": bool(self._summary),
        }


register_memory_model("ir_summarizing", IRSummarizingMemory)
