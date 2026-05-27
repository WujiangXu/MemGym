"""
LLM summarizing memory manager for MemGym.

Port of OpenHands LLMSummarizingCondenser — their production default.
Maintains a rolling summary of forgotten events using an LLM, preserving
task-tracking, code state, and version control context.

Architecture matches OpenHands exactly:
- CondensationRecord ≡ CondensationAction (persists condensation decisions)
- build_view() ≡ View.from_events() (rebuilds condensed view each step)
- should_condense checks VIEW length, not raw history length
- request_condensation() ≡ CondensationRequestAction (fallback trigger)

OpenHands benchmarks: 54% SWE-bench Verified (vs 53% baseline), 50% cost reduction.

Source: openhands/memory/condenser/impl/llm_summarizing_condenser.py
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from ..base import BaseMemoryManager, FilteredContext, register_memory_model


# =============================================================================
# Exact prompt template from OpenHands LLMSummarizingCondenser
# =============================================================================

SUMMARIZATION_PROMPT = """You are maintaining a context-aware state summary for an interactive agent.
You will be given a list of events corresponding to actions taken by the agent, and the most recent previous summary if one exists.
If the events being summarized contain ANY task-tracking, you MUST include a TASK_TRACKING section to maintain continuity.
When referencing tasks make sure to preserve exact task IDs and statuses.

Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

TASK_TRACKING: {Active tasks, their IDs and statuses - PRESERVE TASK IDs}

COMPLETED: (Tasks completed so far, with brief results)
PENDING: (Tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant state)

For code-specific tasks, also include:
CODE_STATE: {File paths, function signatures, data structures}
TESTS: {Failing cases, error messages, outputs}
CHANGES: {Code edits, variable updates}
DEPS: {Dependencies, imports, external calls}
VERSION_CONTROL_STATUS: {Repository state, current branch, PR status, commit history}

PRIORITIZE:
1. Adapt tracking format to match the actual task type
2. Capture key user requirements and goals
3. Distinguish between completed and pending tasks
4. Keep all sections concise and relevant

SKIP: Tracking irrelevant details for the current task type

Example formats:

For code tasks:
USER_CONTEXT: Fix FITS card float representation issue
COMPLETED: Modified mod_float() in card.py, all tests passing
PENDING: Create PR, update documentation
CODE_STATE: mod_float() in card.py updated
TESTS: test_format() passed
CHANGES: str(val) replaces f"{val:.16G}"
DEPS: None modified
VERSION_CONTROL_STATUS: Branch: fix-float-precision, Latest commit: a1b2c3d

For other tasks:
USER_CONTEXT: Write 20 haikus based on coin flip results
COMPLETED: 15 haikus written for results [T,H,T,H,T,H,T,T,H,T,H,T,H,T,H]
PENDING: 5 more haikus needed
CURRENT_STATE: Last flip: Heads, Haiku count: 15/20"""


# Summary marker used to identify summary messages in the view
# (equivalent to OpenHands AgentCondensationObservation)
SUMMARY_MARKER = "[Summary]"


# =============================================================================
# CondensationRecord — equivalent of OpenHands CondensationAction
# =============================================================================

@dataclass
class CondensationRecord:
    """Record of a condensation event, stored in memory manager state.

    Equivalent of OpenHands CondensationAction stored in the event stream.
    Each record describes which original message indices were forgotten
    and what summary replaced them.
    """
    forgotten_events_start: int   # First original message index to forget (inclusive)
    forgotten_events_end: int     # Last original message index to forget (exclusive)
    summary: str                  # LLM-generated summary text
    summary_offset: int           # Where to insert summary in kept events (= keep_first)


# =============================================================================
# build_view — equivalent of OpenHands View.from_events()
# =============================================================================

def build_view(
    messages: List[Any],
    condensations: List[CondensationRecord]
) -> Tuple[List[Any], List[int]]:
    """Build condensed view from full message history + stored condensation records.

    Equivalent of OpenHands View.from_events(events):
    1. Collect all forgotten indices from all CondensationRecords
    2. Filter out forgotten messages
    3. Insert summary from most recent condensation at summary_offset

    Args:
        messages: Full (unfiltered) message history
        condensations: List of stored condensation records

    Returns:
        Tuple of (view_messages, original_indices) where original_indices[i]
        is the index in `messages` that view_messages[i] came from, or -1
        for inserted summary messages.
    """
    # 1. Collect all forgotten indices
    forgotten = set()
    for record in condensations:
        forgotten.update(range(record.forgotten_events_start, record.forgotten_events_end))

    # 2. Filter out forgotten messages, tracking original indices
    kept_indices = [i for i in range(len(messages)) if i not in forgotten]
    kept_messages = [messages[i] for i in kept_indices]

    # 3. Insert summary from most recent condensation at summary_offset
    if condensations:
        last = condensations[-1]
        summary_msg = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {last.summary}"
        }
        offset = last.summary_offset
        view_messages = kept_messages[:offset] + [summary_msg] + kept_messages[offset:]
        view_indices = kept_indices[:offset] + [-1] + kept_indices[offset:]
        return view_messages, view_indices

    return kept_messages, kept_indices


# =============================================================================
# Helper functions
# =============================================================================

def _truncate_content(content: str, max_chars: int) -> str:
    """Truncate content to fit within max_chars.

    Matches OpenHands truncate_content from events/serialization/event.py.
    """
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]


def _is_summary_message(msg: Any) -> bool:
    """Check if a message is a summary message (AgentCondensationObservation equivalent)."""
    if not isinstance(msg, dict):
        return False
    content = msg.get("content", "")
    return isinstance(content, str) and SUMMARY_MARKER in content


def _get_message_content(msg: Any) -> str:
    """Extract string content from a message dict or raw value."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
        return str(content) if content else ""
    return str(msg)


# =============================================================================
# LLMSummarizingMemory — exact port of OpenHands LLMSummarizingCondenser
# =============================================================================

class LLMSummarizingMemory(BaseMemoryManager):
    """
    A memory manager that summarizes forgotten events using an LLM.

    Port of OpenHands LLMSummarizingCondenser — their production default.

    Architecture (matches OpenHands exactly):
    - Stores CondensationRecords (≡ CondensationAction in event stream)
    - build_view() rebuilds condensed view each step (≡ View.from_events())
    - should_condense checks VIEW length, not raw history length
    - Supports request_condensation() fallback (≡ CondensationRequestAction)

    Algorithm:
    1. Build view from full history + stored condensation records
    2. Check should_condense: len(view) > max_size OR condensation_requested
    3. If condensing: head + [summary] + tail from the VIEW
    4. Store CondensationRecord with original message indices
    5. Return condensed view for LLM

    Args:
        max_size: Maximum number of events in view before condensation. Default: 100.
        keep_first: Number of initial events to always keep. Default: 1.
        max_event_length: Maximum character length per event in prompt. Default: 10_000.
        summarization_model: LLM model for summarization, any litellm provider. Default: "gpt-4o-mini".
        max_token_size: If set, trigger condensation based on token count instead of
            message count. When view tokens exceed this threshold, condensation fires.
            Default: None (use max_size message-count trigger).
    """

    def __init__(
        self,
        max_size: int = 100,
        keep_first: int = 1,
        max_event_length: int = 20_000,
        summarization_model: str = "gpt-4o-mini",
        condensation_ratio: float = 0.75,
        max_tokens: int = 4000,
        max_token_size: Optional[int] = None,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        # Validation (same as OpenHands)
        if keep_first >= int(max_size * condensation_ratio):
            raise ValueError(
                f"keep_first ({keep_first}) must be less than half of max_size ({max_size})"
            )
        if keep_first < 0:
            raise ValueError(f"keep_first ({keep_first}) cannot be negative")
        if max_size < 1:
            raise ValueError(f"max_size ({max_size}) cannot be non-positive")

        super().__init__(max_tokens, tokenizer_name, config)

        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.max_token_size = max_token_size

        # Condensation state (equivalent to CondensationActions in OpenHands event stream)
        self._condensations: List[CondensationRecord] = []
        self._condensation_requested: bool = False

        # Summarizer LLM token tracking (cumulative across all condensation calls)
        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0
        self._last_summarizer_prompt_tokens: int = 0
        self._last_summarizer_completion_tokens: int = 0

    def request_condensation(self) -> None:
        """Force condensation on next manage_context() call.

        Equivalent of OpenHands CondensationRequestAction — used when the LLM
        hits a context window error to force compression on the next step.
        """
        self._condensation_requested = True

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Manage context with rolling LLM summarization.

        Matches OpenHands flow exactly:
        1. Build view from full history + stored condensations (≡ View.from_events)
        2. Check should_condense against VIEW (not raw history)
        3. If condensing, work on the VIEW and store CondensationRecord
        """
        messages = list(original_context) + [current_observation]

        # Step 1: Build VIEW from full history + stored condensations
        # (equivalent to state.view → View.from_events(state.history))
        view, view_to_original = build_view(messages, self._condensations)

        view_tokens = self.count_tokens(view)

        # Step 2: should_condense(view)
        # Token-based trigger: count_tokens(view) > max_token_size
        # Message-based trigger (OpenHands default): len(view) > max_size
        if self.max_token_size is not None:
            needs_condensation = (
                view_tokens > self.max_token_size
                or self._condensation_requested
            )
        else:
            needs_condensation = (
                len(view) > self.max_size
                or self._condensation_requested
            )

        if not needs_condensation:
            self._update_token_tracking(view_tokens)
            return FilteredContext(
                content=view,
                metadata={
                    "tokens": view_tokens,
                    "original_tokens": self.count_tokens(messages),
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": self.count_tokens(messages) / max(1, view_tokens),
                    "strategy": "llm_summarizing",
                    "view_size": len(view),
                    "raw_size": len(messages),
                    "condensation_count": len(self._condensations),
                    "step": self._step_count
                }
            )

        # Reset request flag
        self._condensation_requested = False

        # Step 3: get_condensation(view) — work on the VIEW
        head = view[:self.keep_first]
        target_size = int(self.max_size * self.condensation_ratio)

        # Number of events to keep from tail: target_size - head - 1 (for summary slot)
        events_from_tail = target_size - len(head) - 1
        if events_from_tail < 1:
            events_from_tail = 1
        tail = view[-events_from_tail:]

        # Step 4: Detect previous summary in view
        # OpenHands: isinstance(view[keep_first], AgentCondensationObservation)
        summary_event_content = ""
        if self.keep_first < len(view) and _is_summary_message(view[self.keep_first]):
            raw = _get_message_content(view[self.keep_first])
            summary_event_content = raw.replace(SUMMARY_MARKER, "").strip()

        # Step 5: Identify forgotten events (middle of VIEW, skip summary messages)
        middle_start = self.keep_first
        middle_end = len(view) - events_from_tail
        forgotten_events = []
        forgotten_view_indices = []
        for vi in range(middle_start, middle_end):
            if not _is_summary_message(view[vi]):
                forgotten_events.append(view[vi])
                forgotten_view_indices.append(vi)

        if not forgotten_events:
            self._update_token_tracking(view_tokens)
            return FilteredContext(
                content=view,
                metadata={
                    "tokens": view_tokens,
                    "original_tokens": self.count_tokens(messages),
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "llm_summarizing",
                    "step": self._step_count
                }
            )

        # Step 6: Call LLM for summary (exact OpenHands prompt)
        summary = self._call_llm_summarize(forgotten_events, summary_event_content)

        # Step 7: Store CondensationRecord
        # Map forgotten view indices back to original message indices
        original_forgotten_indices = [
            view_to_original[vi] for vi in forgotten_view_indices
            if vi < len(view_to_original) and view_to_original[vi] >= 0
        ]

        if original_forgotten_indices:
            self._condensations.append(CondensationRecord(
                forgotten_events_start=min(original_forgotten_indices),
                forgotten_events_end=max(original_forgotten_indices) + 1,
                summary=summary,
                summary_offset=self.keep_first,
            ))

        self._compaction_count += 1

        # Step 8: Build result: head + [summary_message] + tail
        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {summary}"
        }
        result = head + [summary_message] + tail

        filtered_tokens = self.count_tokens(result)
        original_tokens = self.count_tokens(messages)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=result,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": True,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "llm_summarizing",
                "summarization_count": self._compaction_count,
                "condensation_count": len(self._condensations),
                "messages_before": len(view),
                "messages_after": len(result),
                "forgotten_events": len(forgotten_events),
                "view_size": len(result),
                "raw_size": len(messages),
                "step": self._step_count,
                "condensation_event": {
                    "summary_text": summary,
                    "forgotten_count": len(forgotten_events),
                    "summarization_model": self.summarization_model,
                    "summarization_time_seconds": getattr(self, '_last_summarization_time', 0),
                    "summarizer_prompt_tokens": self._last_summarizer_prompt_tokens,
                    "summarizer_completion_tokens": self._last_summarizer_completion_tokens,
                }
            }
        )

    def _call_llm_summarize(
        self,
        forgotten_events: List[Any],
        previous_summary: str
    ) -> str:
        """Call LLM to summarize forgotten events.

        Uses exact OpenHands prompt format with <PREVIOUS SUMMARY> and <EVENT> tags.
        """
        prompt = SUMMARIZATION_PROMPT + "\n\n"

        # Add previous summary
        truncated_summary = _truncate_content(previous_summary, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n"
        prompt += "\n\n"

        # Add all events being forgotten
        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        prompt += "Now summarize the events using the rules above."

        try:
            t0 = time.time()
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}]
            )
            self._last_summarization_time = time.time() - t0

            # Extract summarizer token usage (provider-compatible)
            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                self._last_summarizer_prompt_tokens = (
                    getattr(response.usage, 'prompt_tokens', 0)
                    or getattr(response.usage, 'input_tokens', 0) or 0
                )
                self._last_summarizer_completion_tokens = (
                    getattr(response.usage, 'completion_tokens', 0)
                    or getattr(response.usage, 'output_tokens', 0) or 0
                )
                self._summarizer_prompt_tokens += self._last_summarizer_prompt_tokens
                self._summarizer_completion_tokens += self._last_summarizer_completion_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._last_summarization_time = 0
            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
            print(f"Warning: LLM summarization failed: {e}")
            return previous_summary or "Summary unavailable due to error."

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._condensations.clear()
        self._condensation_requested = False
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._summarizer_prompt_tokens = 0
        self._summarizer_completion_tokens = 0
        self._last_summarizer_prompt_tokens = 0
        self._last_summarizer_completion_tokens = 0

    def get_condensation_history(self) -> List[Dict[str, Any]]:
        """Export condensation records for trajectory logging."""
        return [
            {
                "forgotten_events_start": r.forgotten_events_start,
                "forgotten_events_end": r.forgotten_events_end,
                "summary": r.summary,
                "summary_offset": r.summary_offset,
            }
            for r in self._condensations
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            "summarization_model": self.summarization_model,
            "max_size": self.max_size,
            "keep_first": self.keep_first,
            "max_event_length": self.max_event_length,
            "condensation_count": len(self._condensations),
            "summarizer_total_prompt_tokens": self._summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": self._summarizer_completion_tokens,
            "summarizer_total_tokens": self._summarizer_prompt_tokens + self._summarizer_completion_tokens,
        })
        return base_stats


# Register in memory model registry
register_memory_model("llm_summarizing", LLMSummarizingMemory)
