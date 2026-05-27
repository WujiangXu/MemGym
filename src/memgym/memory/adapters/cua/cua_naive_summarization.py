"""CUA-specific Naive Summarization Memory for GUI agent trajectories.

Simple token-threshold summarization with a CUA-specific prompt that
focuses on GUI interactions: screen state, actions, visual outcomes,
and progress toward the desktop task.
"""

from typing import Any, Dict, List, Optional

from litellm import completion

from ...base import BaseMemoryManager, FilteredContext, register_memory_model

# CUA-specific summarization prompt
CUA_SUMMARIZATION_PROMPT = """You are summarizing context for a GUI agent performing a desktop computer task.

PRESERVE (high priority):
- User's original task goal and success criteria
- Current screen state (application, window, visible UI elements)
- Actions performed (clicks, typing, scrolling, keyboard shortcuts)
- Outcomes of actions (how the screen changed)
- Error dialogs, unexpected states, or dead ends
- Progress toward completing the task
- Menu paths, button names, and field labels referenced

DISCARD:
- Redundant screenshot descriptions already processed
- Repeated failed action attempts (keep only the pattern and resolution)
- Verbose coordinate details unless targeting the same element repeatedly
- Wait/sleep steps with no state change

Output a concise summary maintaining all critical GUI interaction context."""


class CUANaiveSummarizationMemory(BaseMemoryManager):
    """Naive context management for CUA GUI agents.

    Simple strategy:
    - Accumulates observations from GUI interaction steps
    - When total tokens > max_tokens, summarizes everything with CUA prompt
    - Returns summary + recent steps

    Args:
        max_tokens: Threshold for triggering summarization (default: 16000)
        keep_recent: Number of recent messages to keep after summarization
        keep_first: Number of initial messages to preserve (task instruction)
        summarization_model: LLM model for summarization
    """

    def __init__(
        self,
        max_tokens: int = 16000,
        keep_recent: int = 5,
        keep_first: int = 1,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
        summarization_model: str = "gpt-4o-mini",
        **kwargs,
    ):
        config = config or {}
        config["env_type"] = "cua"
        super().__init__(max_tokens, tokenizer_name, config)

        self.summarization_model = summarization_model
        self.keep_recent = keep_recent
        self.keep_first = keep_first

        self._summary: Optional[str] = None

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        # Build full message list
        messages = list(original_context) + [current_observation]

        # Count tokens
        total_tokens = self.count_tokens(messages)

        # If over threshold, summarize
        was_compacted = False
        if total_tokens > self.max_tokens:
            self._summarize_all(messages)
            was_compacted = True

        # Build output
        if self._summary:
            head = messages[:self.keep_first]
            summary_msg = {"role": "user", "content": f"[GUI Summary] {self._summary}"}
            tail = messages[-self.keep_recent:] if len(messages) > self.keep_recent else messages
            content = head + [summary_msg] + tail
        else:
            content = messages

        filtered_tokens = self.count_tokens(content)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=content,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": total_tokens,
                "was_compacted": was_compacted,
                "compression_ratio": total_tokens / max(1, filtered_tokens) if was_compacted else 1.0,
                "direct_messages": True,
                "strategy": "cua_naive",
                "has_summary": self._summary is not None,
                "summarization_count": self._compaction_count,
            },
        )

    def _summarize_all(self, messages: List[Any]) -> None:
        """Summarize messages using CUA-specific prompt."""
        content = "\n".join(str(msg) for msg in messages)
        if self._summary:
            content = f"Previous summary: {self._summary}\n\nNew content:\n{content}"

        try:
            response = completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": CUA_SUMMARIZATION_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_completion_tokens=200,
            )
            self._summary = response.choices[0].message.content.strip()
            self._compaction_count += 1
        except Exception as e:
            print(f"Warning: CUA summarization failed: {e}")

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._summary = None
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        base_stats = super().get_stats()
        base_stats.update({
            "has_summary": self._summary is not None,
            "summarization_model": self.summarization_model,
            "env_type": "cua",
            "keep_recent": self.keep_recent,
            "keep_first": self.keep_first,
            "summary_preview": (
                self._summary[:100] + "..."
                if self._summary and len(self._summary) > 100
                else self._summary
            ),
        })
        return base_stats


register_memory_model("cua_naive", CUANaiveSummarizationMemory)
