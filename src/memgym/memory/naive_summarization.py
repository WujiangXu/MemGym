"""
Naive context management: summarize when over N tokens.

Simple strategy for testing:
- Accumulates all observations
- When total tokens > max_tokens, summarizes everything
- Returns summary + latest observation

This is the simplest possible context management strategy, useful as a
baseline or for testing with small token thresholds.
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseMemoryManager, FilteredContext, register_memory_model
from .summarizer_backend import LitellmSummarizerBackend, SummarizerBackend


# H.1.5 — strip Qwen3 <think> blocks from agent messages BEFORE they reach
# the summarizer. The 582-pair SFT dataset comes from non-thinking teachers
# (haiku45/sonnet45/gptoss120b); the trained summarizer never saw <think>
# tokens in SFT, so at deployment we keep its input on-distribution by
# removing them here. Also shrinks the summarizer context — reasoning traces
# eat 3-5K tokens per turn.
_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks (and dangling unclosed openings)."""
    text = _THINK_CLOSED.sub("", text)
    return _THINK_OPEN_UNCLOSED.sub("", text)


def _strip_thinking_in_message(msg: Any) -> Any:
    """Apply _strip_thinking to a message's content without mutating input.

    Messages arrive as dicts ({"role": ..., "content": ...}) for chat-format
    scaffolds and as bare strings for legacy ones. Handle both; pass through
    unchanged types (no surprises if a caller evolves the schema)."""
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        stripped = _strip_thinking(msg["content"])
        if stripped == msg["content"]:
            return msg
        return {**msg, "content": stripped}
    if isinstance(msg, str):
        return _strip_thinking(msg)
    return msg


# =============================================================================
# Environment-Specific Summarization Prompts
# Based on: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
# =============================================================================

SUMMARIZATION_PROMPTS = {
    # =========================================================================
    # SWE-bench / Coding Agents
    # =========================================================================
    "swe": """You are summarizing context for a coding agent working on a software engineering task.

PRESERVE (high priority):
- Architectural decisions and design patterns chosen
- Unresolved bugs, errors, and technical blockers
- Implementation details that affect future work
- File paths and code locations referenced
- Dependencies between components
- Test results and their implications

DISCARD:
- Redundant command outputs already processed
- Duplicate file contents
- Raw stack traces (keep only key error message)
- Verbose tool outputs once insights captured

Output a concise summary maintaining all critical technical context.""",

    # =========================================================================
    # Tau2-bench / Dialogue Agents
    # =========================================================================
    "tau2": """You are summarizing context for a customer service dialogue agent.

PRESERVE (high priority):
- User's original intent and goal
- Key decisions made during conversation
- Current task state and progress
- Important constraints or requirements mentioned
- Tool calls made and their results
- Any commitments or promises made to user
- Unresolved issues or pending actions

DISCARD:
- Repetitive pleasantries and acknowledgments
- Redundant confirmations
- Tool outputs already acted upon
- Verbose API responses (keep key data only)

Output a concise summary that enables the agent to continue the conversation seamlessly.""",

    # =========================================================================
    # Generic / Default
    # =========================================================================
    "generic": """Summarize the following context concisely.

PRESERVE:
- Key facts and decisions
- Current state and progress
- Important details for continuing the task
- Any unresolved issues

DISCARD:
- Redundant information
- Verbose outputs already processed

Keep the summary focused and actionable."""
}


# =============================================================================
# Naive Summarization Memory Manager
# =============================================================================

class NaiveSummarizationMemory(BaseMemoryManager):
    """
    Naive context management: summarize when over N tokens.

    Simple strategy for testing:
    - Accumulates all observations
    - When total tokens > max_tokens, summarizes everything
    - Returns summary + latest observation

    Args:
        max_tokens: Threshold for triggering summarization. Default: 16000 tokens
            (~120 messages worth, similar to OpenHands' max_size=120 events).
        tokenizer_name: Tokenizer model name for token counting
        config: Optional additional configuration
        summarization_model: LLM model for summarization
        env_type: Environment type for prompt selection ("swe", "tau2", or "generic")
    """

    def __init__(
        self,
        max_tokens: int = 16000,  # ~120 messages worth, matching OpenHands
        keep_recent: int = 5,  # Keep recent messages after summarization
        keep_first: int = 1,  # Preserve system prompt (index 0)
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
        summarization_model: str = "gpt-4o-mini",
        env_type: str = "generic",  # "swe", "tau2", or "generic"
        summarizer_backend: Optional[SummarizerBackend] = None,
    ):
        config = config or {}
        config["env_type"] = env_type  # Store for prompt selection
        super().__init__(max_tokens, tokenizer_name, config)

        self.summarization_model = summarization_model
        self.env_type = env_type
        self.keep_recent = keep_recent
        self.keep_first = keep_first

        # Pluggable summarizer. Default mirrors pre-refactor litellm behavior;
        # (see paper) RL loop passes a LocalHFSummarizerBackend instead to
        # hot-swap the trained 8B checkpoint per rollout.
        self.summarizer_backend: SummarizerBackend = (
            summarizer_backend if summarizer_backend is not None
            else LitellmSummarizerBackend(summarization_model)
        )

        # Summarization state
        self._summary: Optional[str] = None
        # Count of messages already absorbed into self._summary. On the
        # next compaction we feed only messages past this index to the
        # summarizer, not the whole history — without this, every call
        # resends the full conversation and overruns the summarizer's
        # context window after a handful of steps.
        self._summarized_upto: int = 0
        # Hard ceiling on tokens sent to the summarizer in a single call.
        # Keeps this manager safe even for summarizers with small context
        # windows (self-hosted Qwen3.6-35B at 32k). When the incremental
        # batch is still too large, take the tail — recent context is
        # more load-bearing for coding agents than the oldest messages.
        self._summarize_input_budget_tokens: int = 12000
        # H.1.5 telemetry — populated per compaction by _summarize_all.
        self._last_think_chars_stripped: int = 0
        self._last_think_strip_ratio: float = 0.0

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Manage context with naive summarization strategy.

        Logic:
        1. Build full message list from original_context + current_observation
        2. Count total tokens
        3. If tokens > threshold N, summarize content into one summary
        4. Return: summary + recent tail (if summarized) or all content

        Args:
            original_context: Complete conversation history
            current_observation: Latest observation from environment
            metadata: Optional step/task info

        Returns:
            FilteredContext with processed content and token/compaction stats
        """
        # 1. Build full message list (no accumulation — original_context already has history)
        messages = list(original_context) + [current_observation]

        # 2. Count tokens
        total_tokens = self.count_tokens(messages)

        # 3. If over threshold, summarize
        was_compacted = False
        if total_tokens > self.max_tokens:
            self._summarize_all(messages)
            was_compacted = True

        # 4. Build output
        if self._summary:
            # Keep head (system prompt) + summary + recent tail
            # Matches LLMSummarizing/StructuredSummary/SlidingWindow pattern
            head = messages[:self.keep_first]
            summary_msg = {"role": "user", "content": f"[Summary] {self._summary}"}
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
                "strategy": "naive_summarization",
                "has_summary": self._summary is not None,
                "summarization_count": self._compaction_count,
                "env_type": self.env_type
            }
        )

    def _summarize_all(self, messages: List[Any]) -> None:
        """Summarize messages using environment-specific prompt.

        Incremental: when a previous summary exists, we feed only the
        messages added since that summary — not the whole history — so
        input size stays O(delta) instead of O(full-trajectory).
        """
        new_start = self._summarized_upto if self._summary else 0
        new_msgs = messages[new_start:]
        # Keep the summarizer on-distribution: H.1.5 strips <think>…</think>
        # from every message before concatenation. Record pre/post sizes so
        # downstream telemetry can flag reasoning-dominated trajectories
        # (think_strip_ratio > 0.7 on > 20% of events is the sanity gate).
        raw_content = "\n".join(str(msg) for msg in new_msgs)
        stripped_msgs = [_strip_thinking_in_message(m) for m in new_msgs]
        new_content = "\n".join(str(msg) for msg in stripped_msgs)
        self._last_think_chars_stripped = len(raw_content) - len(new_content)
        self._last_think_strip_ratio = (
            self._last_think_chars_stripped / len(raw_content)
            if raw_content else 0.0
        )
        if self._summary:
            content = f"Previous summary: {self._summary}\n\nNew content:\n{new_content}"
        else:
            content = new_content

        # Tail-truncate if still over the per-call budget. We keep the
        # tail because recent messages are more load-bearing than the
        # oldest for a coding agent, and the Previous summary carries
        # forward the older context anyway.
        if self.count_tokens(content) > self._summarize_input_budget_tokens:
            max_chars = self._summarize_input_budget_tokens * 4  # ~4 chars/token
            if len(content) > max_chars:
                content = "[... earlier content truncated ...]\n" + content[-max_chars:]

        # Get environment-specific prompt
        system_prompt = self._get_summarization_prompt()

        # Backend owns its own error handling and reasoning_content fallback
        # (see LitellmSummarizerBackend). An empty return means the call
        # failed or emitted no usable text — keep the prior summary and
        # do NOT advance _summarized_upto so the next compaction retries
        # the same window instead of silently dropping unsummarized steps.
        # Budget = 1024 so reasoning models have room for <think> + summary.
        text = self.summarizer_backend.summarize(system_prompt, content, max_tokens=1024)
        if text:
            self._summary = text
            self._compaction_count += 1
            self._summarized_upto = len(messages)
        else:
            print("Warning: Summarization returned empty content; keeping previous summary.")

    def _get_summarization_prompt(self) -> str:
        """Return environment-specific summarization prompt."""
        env_type = self.config.get("env_type", "generic")
        return SUMMARIZATION_PROMPTS.get(env_type, SUMMARIZATION_PROMPTS["generic"])

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._summary = None
        self._summarized_upto = 0
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._last_think_chars_stripped = 0
        self._last_think_strip_ratio = 0.0
        # Legacy support
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            "has_summary": self._summary is not None,
            "summarization_model": self.summarization_model,
            "env_type": self.env_type,
            "keep_recent": self.keep_recent,
            "keep_first": self.keep_first,
            "summary_preview": self._summary[:100] + "..." if self._summary and len(self._summary) > 100 else self._summary,
            "last_think_chars_stripped": self._last_think_chars_stripped,
            "last_think_strip_ratio": self._last_think_strip_ratio,
        })
        return base_stats


# Legacy alias
NaiveSummarizationMemoryModel = NaiveSummarizationMemory

# Register in memory model registry
register_memory_model("naive", NaiveSummarizationMemory)
register_memory_model("naive_summarization", NaiveSummarizationMemory)
