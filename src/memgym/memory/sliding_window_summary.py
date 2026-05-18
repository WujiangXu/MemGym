"""
Sliding window + incremental summary memory manager for MemGym.

Keeps a fixed window of the last N messages verbatim, plus a continuously-
maintained rolling summary of everything older. The summary is updated
incrementally as messages fall out of the window — only newly-fallen-off
messages are merged into the existing summary, not the full history.

Key differences from existing strategies:
- vs LLMSummarizing: Fixed window size (not variable), continuous updates
  (not batch-triggered when view exceeds threshold)
- vs Naive: Incremental merge (not full re-summarization), preserves
  more recent context in the window
"""

import time
from typing import Any, Dict, List, Optional

from litellm import completion

from .base import BaseMemoryManager, FilteredContext, register_memory_model


# =============================================================================
# Helper functions (same patterns as llm_summarizing.py)
# =============================================================================

def _truncate_content(content: str, max_chars: int) -> str:
    """Truncate content to fit within max_chars."""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]


def _get_message_content(msg: Any) -> str:
    """Extract string content from a message dict or raw value."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
        return str(content) if content else ""
    return str(msg)


# =============================================================================
# Incremental merge prompts (environment-specific)
# =============================================================================

_BASE_MERGE_PROMPT = """You are maintaining a rolling context summary for an interactive agent.

You will be given:
1. The CURRENT SUMMARY of everything the agent has done so far
2. NEW MESSAGES that just fell out of the agent's recent context window

Your task: Merge the new information into the existing summary to produce an UPDATED SUMMARY.

Rules:
- PRESERVE all important information from the current summary
- ADD key facts from the new messages
- REMOVE redundant or superseded information (e.g., if a bug was fixed, drop the error details but note it was fixed)
- Keep the summary CONCISE but COMPLETE enough for the agent to continue working
- Output ONLY the updated summary text, nothing else."""

MERGE_PROMPTS = {
    "swe": _BASE_MERGE_PROMPT + """

For this SWE-bench coding task, pay special attention to:
- Bug description and reproduction steps
- Files explored and their relevant content
- Patches attempted and their outcomes
- Test commands run and results
- Git operations (branch, commit, diff state)
- File paths, function names, class names""",

    "tau2": _BASE_MERGE_PROMPT + """

For this customer service task, pay special attention to:
- Customer's original request and constraints
- Actions taken and their results
- Commitments or promises made
- Pending items to address""",

    "generic": _BASE_MERGE_PROMPT,
}

SUMMARY_MARKER = "[Summary]"


# =============================================================================
# SlidingWindowSummaryMemory
# =============================================================================

class SlidingWindowSummaryMemory(BaseMemoryManager):
    """
    Fixed sliding window + incrementally-maintained rolling summary.

    Always keeps the last `window_size` messages verbatim. As messages fall
    out of the window, they are merged into a rolling summary via an LLM call.
    The output is: [keep_first prefix] + [summary] + [last window_size messages].

    Args:
        window_size: Number of recent messages to keep verbatim. Default: 20.
        keep_first: Number of initial messages to always preserve (e.g., system prompt). Default: 1.
        summarization_model: LLM for incremental summary merges (any litellm provider). Default: "gpt-4o-mini".
        max_event_length: Max characters per event in the merge prompt. Default: 10_000.
        env_type: Environment type for prompt selection ("swe", "tau2", "generic"). Default: "swe".
        max_tokens: Token limit (inherited, used for tracking). Default: 4000.
    """

    def __init__(
        self,
        window_size: int = 20,
        keep_first: int = 1,
        summarization_model: str = "gpt-4o-mini",
        max_event_length: int = 10_000,
        env_type: str = "swe",
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(max_tokens, tokenizer_name, config)

        self.window_size = window_size
        self.keep_first = keep_first
        self.summarization_model = summarization_model
        self.max_event_length = max_event_length
        self.env_type = env_type

        # Rolling summary state
        self._summary: Optional[str] = None
        self._summary_frontier: int = 0  # Everything before this has been absorbed
        self._condensation_requested: bool = False
        self._last_summarization_time: float = 0

    def request_condensation(self) -> None:
        """Force a tighter window on next manage_context() call.

        Used when the LLM hits a context window error — temporarily shrinks
        the effective window to reduce token count.
        """
        self._condensation_requested = True

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """
        Fixed sliding window + incremental rolling summary.

        Algorithm:
        1. Build full message list
        2. Compute window boundary
        3. Identify newly-fallen-off messages (between old frontier and new window start)
        4. If any fell off, merge them into rolling summary via LLM
        5. Build output: [keep_first] + [summary_msg] + [window tail]
        """
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)
        total = len(messages)

        # Determine effective window size (may be halved on emergency condensation)
        effective_window = self.window_size
        if self._condensation_requested:
            effective_window = max(3, self.window_size // 2)
            self._condensation_requested = False

        # Compute window boundary
        window_start = max(self.keep_first, total - effective_window)

        # If everything fits in window + keep_first, no summarization needed
        if window_start <= self.keep_first:
            self._update_token_tracking(original_tokens)
            return FilteredContext(
                content=messages,
                metadata={
                    "tokens": original_tokens,
                    "original_tokens": original_tokens,
                    "was_compacted": self._summary is not None,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "sliding_window",
                    "has_summary": self._summary is not None,
                    "summarization_count": self._compaction_count,
                    "window_size": effective_window,
                    "total_messages": total,
                    "window_messages": total,
                    "newly_absorbed": 0,
                    "step": self._step_count,
                },
            )

        # Identify newly fallen-off messages
        absorb_start = max(self._summary_frontier, self.keep_first)
        absorb_end = window_start
        newly_fallen = []

        if absorb_end > absorb_start:
            newly_fallen = messages[absorb_start:absorb_end]
            self._summary_frontier = absorb_end

        # Merge into summary if messages fell off
        was_compacted = False
        if newly_fallen:
            self._merge_into_summary(newly_fallen)
            was_compacted = True

        # Build output
        head = messages[:self.keep_first]
        tail = messages[window_start:]

        if self._summary:
            summary_msg = {
                "role": "user",
                "content": f"{SUMMARY_MARKER} {self._summary}",
            }
            result = head + [summary_msg] + tail
        else:
            result = head + tail

        filtered_tokens = self.count_tokens(result)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=result,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": was_compacted or self._summary is not None,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "sliding_window",
                "has_summary": self._summary is not None,
                "summarization_count": self._compaction_count,
                "window_size": effective_window,
                "total_messages": total,
                "window_messages": len(tail),
                "newly_absorbed": len(newly_fallen),
                "step": self._step_count,
            },
        )

    def _merge_into_summary(self, newly_fallen: List[Any]) -> None:
        """Merge newly-fallen-off messages into the rolling summary."""
        current_summary = self._summary or "(No previous summary — this is the first summarization)"

        # Format fallen messages
        fallen_text = ""
        for i, msg in enumerate(newly_fallen):
            content = _get_message_content(msg)
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            truncated = _truncate_content(content, self.max_event_length)
            fallen_text += f"<MESSAGE role={role} index={i}>\n{truncated}\n</MESSAGE>\n"

        merge_prompt = MERGE_PROMPTS.get(self.env_type, MERGE_PROMPTS["generic"])

        prompt = f"""{merge_prompt}

<CURRENT SUMMARY>
{_truncate_content(current_summary, self.max_event_length)}
</CURRENT SUMMARY>

<NEW MESSAGES>
{fallen_text}
</NEW MESSAGES>

Produce the updated summary now."""

        try:
            t0 = time.time()
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
            )
            self._last_summarization_time = time.time() - t0
            self._summary = response.choices[0].message.content.strip()
            self._compaction_count += 1

            # Safety: truncate summary if it grows too large
            if len(self._summary) > self.max_event_length:
                self._summary = self._summary[:self.max_event_length]

        except Exception as e:
            self._last_summarization_time = 0
            print(f"Warning: Incremental merge failed: {e}")
            # Fallback: append truncated message text
            fallback = "\n".join(
                _get_message_content(m)[:200] for m in newly_fallen
            )
            if self._summary:
                self._summary += f"\n\n[Unmerged]: {fallback[:500]}"
            else:
                self._summary = fallback[:1000]
            self._compaction_count += 1

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._summary = None
        self._summary_frontier = 0
        self._condensation_requested = False
        self._last_summarization_time = 0
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            "has_summary": self._summary is not None,
            "summarization_model": self.summarization_model,
            "window_size": self.window_size,
            "keep_first": self.keep_first,
            "env_type": self.env_type,
            "max_event_length": self.max_event_length,
            "summary_frontier": self._summary_frontier,
            "summary_preview": (
                self._summary[:100] + "..."
                if self._summary and len(self._summary) > 100
                else self._summary
            ),
        })
        return base_stats


# Register in memory model registry
register_memory_model("sliding_window", SlidingWindowSummaryMemory)
register_memory_model("sliding_window_summary", SlidingWindowSummaryMemory)
