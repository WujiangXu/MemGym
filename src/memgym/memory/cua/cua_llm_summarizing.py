"""CUA-specific LLM Summarizing Memory for GUI agent trajectories.

Subclasses BaseMemoryManager with a CUA-specific summarization prompt
that tracks GUI state: screen observations, user actions, visual outcomes,
blockers, and progress toward the desktop task goal.
"""

from typing import Any, Dict, List, Optional

import litellm

from ..base import BaseMemoryManager, FilteredContext, register_memory_model

# CUA-specific summarization prompt (replaces the SWE-bench code-centric one)
CUA_SUMMARIZATION_PROMPT = """\
You are summarizing the interaction history of a GUI agent performing a \
desktop computer task. The agent uses screenshots and takes actions like \
clicking, typing, and navigating. Some earlier steps are being evicted \
from context -- your summary must preserve the critical information.

TASK GOAL: {goal}

PREVIOUS SUMMARY (if any):
{previous_summary}

STEPS BEING EVICTED:
{forgotten_text}

Create a concise summary that preserves:
1. GOAL -- the user's original task and success criteria
2. SCREEN STATE -- current application(s) open, visible UI elements, window layout
3. ACTIONS TAKEN -- sequence of clicks, typing, scrolling, keyboard shortcuts performed
4. OUTCOMES -- how the screen changed after each significant action
5. BLOCKERS -- error dialogs, unexpected states, dead ends encountered
6. PROGRESS -- what has been accomplished so far, what remains to be done

Be specific about UI elements (button names, menu paths, field labels). \
Do NOT include vague statements like "several actions were performed". \
Include exact coordinates only when relevant to repeated interactions \
with the same element."""


class CUALLMSummarizingMemory(BaseMemoryManager):
    """LLM-based summarization adapted for CUA GUI agent trajectories.

    When the accumulated context exceeds max_size messages, uses an LLM
    to summarize evicted steps into a GUI-focused summary that tracks
    screen state, actions, outcomes, and progress.
    """

    def __init__(
        self,
        max_size: int = 20,
        keep_first: int = 1,
        summarization_model: str = "gpt-4o-mini",
        condensation_ratio: float = 0.75,
        max_tokens: int = 16000,
        goal: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.max_size = max_size
        self.keep_first = keep_first
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.goal = goal

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
            view.append(f"[GUI AGENT SUMMARY]\n{self._summary}")
        view.extend(self._history)

        tokens = self.count_tokens(view)
        return FilteredContext(
            content=view,
            metadata={
                "tokens": tokens,
                "was_compacted": self._condensation_count > 0,
                "condensation_count": self._condensation_count,
                "strategy": "cua_llm_summarizing",
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
        prompt = CUA_SUMMARIZATION_PROMPT.format(
            goal=self.goal or "(not specified)",
            previous_summary=self._summary or "(none)",
            forgotten_text=forgotten_text,
        )

        try:
            response = litellm.completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": "You are a GUI interaction summarizer. Be concise and specific about UI elements."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            self._summary = response.choices[0].message.content
        except Exception:
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
            parts.append(f"[GUI AGENT SUMMARY]\n{self._summary}")
        parts.extend(str(item) for item in self._history)
        return "\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "condensation_count": self._condensation_count,
            "history_size": len(self._history),
            "has_summary": bool(self._summary),
            "compaction_count": self._condensation_count,
        }


register_memory_model("cua_llm_summarizing", CUALLMSummarizingMemory)
