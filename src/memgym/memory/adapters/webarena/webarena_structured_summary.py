"""
Structured summary memory manager for WebArena (Web GUI) trajectories.

Architecture mirrors `memory/structured_summary.py` (the proven port of
OpenHands StructuredSummaryCondenser) line-for-line:

- Same control-flow logic (head + summary + tail)
- Same condensation triggers (max_size OR max_token_size)
- Same litellm function-calling approach for structured output
- Same fallback handling when the tool call can't be parsed

**Only the field schema and prompt change**: the SWE-bench schema has 17
code-edit-oriented fields (files_modified, function_changes, tests_passing,
branch_created, pr_status...). Those are useless for browsing trajectories.
This adapter replaces them with a ~10-field schema capturing the critical
state of a web GUI agent: browser location, navigation history, form values,
interaction log, extracted data, task progress, blockers, next step.

Constructor adds `task_instruction` and `app_name` fields (following the IR
and webarena_summarizing precedent) so the structured summarizer sees the
original task when deciding which subtasks are DONE/IN_PROGRESS/TODO.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from ...base import BaseMemoryManager, FilteredContext, register_memory_model
from ...strategies.llm_summarizing import (
    SUMMARY_MARKER,
    _truncate_content,
    _is_summary_message,
    _get_message_content,
)
from .webarena_summarizing import CondensationRecord, build_view


# =============================================================================
# WebArena state-summary fields — replaces OpenHands StateSummary
# =============================================================================

# Field definitions: {name: description} — web-task-tailored schema.
# Flat string-typed fields so litellm function-calling JSON-Schema validation
# behaves the same way as the SWE-bench version (where all 17 fields are
# also strings). List/dict-shaped content is encoded as newline-delimited
# or JSON text inside the string value.
WEBARENA_STATE_SUMMARY_FIELDS = {
    "task": "The original task instruction given to the agent.",
    "app": "Which webarena-infinity app this trajectory runs in (e.g. gmail, gitlab, reddit).",
    "current_url": "Current page URL the browser is on.",
    "current_page_title": "Current page title as shown in the browser tab.",
    "focused_element": "The currently focused element id/selector if any, else 'none'.",
    "navigation_history": (
        "Newline-delimited list of pages visited in causal order. "
        "One entry per line, formatted as 'URL — brief purpose'."
    ),
    "form_state": (
        "Newline-delimited list of form fields filled so far. "
        "One entry per line, formatted as 'element_id → value'."
    ),
    "interaction_log": (
        "Newline-delimited log of actions taken. One entry per line, formatted as "
        "'step N: action element_id → result'. Keep exact IDs; do not paraphrase."
    ),
    "extracted_data": (
        "Facts observed from page content that are needed downstream. "
        "Newline-delimited 'key: value' pairs — search results, account state, "
        "table rows, error messages, prices, dates, names, etc."
    ),
    "blockers": (
        "Newline-delimited list of errors, missing elements, timeouts, captchas, "
        "permission denials, dead-ends encountered. Include what was tried and why it failed."
    ),
    "task_progress_summary": (
        "Completion counter: 'X of Y items/steps complete'. "
        "For batch operations on multiple items, state the count explicitly."
    ),
    "task_progress_done": (
        "Newline-delimited list of subtasks already completed. "
        "For batch operations, list EACH item individually (e.g. 'Deleted Amazon email', "
        "'Deleted Nike email')."
    ),
    "task_progress_in_progress": "The subtask currently being worked on, if any.",
    "task_progress_todo": (
        "Newline-delimited list of subtasks still to do. "
        "For batch operations, list EACH remaining item individually. "
        "This field is CRITICAL — the agent uses it to decide whether to continue or stop."
    ),
    "next_step": (
        "What the agent should do next based on task_progress_todo. "
        "Must be a concrete action — 'click [ID]', 'type [ID] value', etc."
    ),
}

WEBARENA_STATE_SUMMARY_REQUIRED = [
    "task",
    "current_url",
    "navigation_history",
    "interaction_log",
    "task_progress_summary",
    "task_progress_done",
    "task_progress_todo",
]


def _build_webarena_tool_description() -> Dict[str, Any]:
    """Build the litellm tool description for web-GUI function-calling summarization."""
    properties = {}
    for field_name, description in WEBARENA_STATE_SUMMARY_FIELDS.items():
        properties[field_name] = {"type": "string", "description": description}

    return {
        "type": "function",
        "function": {
            "name": "create_webarena_state_summary",
            "description": (
                "Creates a comprehensive summary of the current state of a web GUI "
                "agent's browsing session, to preserve context when history grows "
                "too large. You MUST include non-empty values for task, current_url, "
                "navigation_history, interaction_log, task_progress_summary, "
                "task_progress_done, and task_progress_todo. For batch operations "
                "on multiple items, track EACH item individually in progress fields. "
                "Preserve exact URLs, element IDs, field names, and values — "
                "do NOT paraphrase or hallucinate."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": WEBARENA_STATE_SUMMARY_REQUIRED,
            },
        },
    }


def _format_webarena_state_summary(fields: Dict[str, str]) -> str:
    """Format a web-GUI state summary dict into markdown string for agent context."""
    sections = [
        "# Web Agent State Summary",
        "## Task",
        f"**Task**: {fields.get('task', '')}",
        f"**App**: {fields.get('app', '')}",
        "## Browser Location",
        f"**Current URL**: {fields.get('current_url', '')}",
        f"**Current Page Title**: {fields.get('current_page_title', '')}",
        f"**Focused Element**: {fields.get('focused_element', '')}",
        "## Navigation History",
        fields.get("navigation_history", "") or "_(none)_",
        "## Form State",
        fields.get("form_state", "") or "_(none)_",
        "## Interaction Log",
        fields.get("interaction_log", "") or "_(none)_",
        "## Extracted Data",
        fields.get("extracted_data", "") or "_(none)_",
        "## Blockers",
        fields.get("blockers", "") or "_(none)_",
        "## Task Progress",
        f"**{fields.get('task_progress_summary', '')}**",
        f"**Done**:\n{fields.get('task_progress_done', '') or '_(none)_'}",
        f"**In Progress**: {fields.get('task_progress_in_progress', '') or '_(none)_'}",
        f"**Todo**:\n{fields.get('task_progress_todo', '') or '_(none)_'}",
        "## Next Step",
        fields.get("next_step", "") or "_(not set)_",
    ]
    return "\n\n".join(sections)


# =============================================================================
# Web-GUI-tailored structured summary prompt
# =============================================================================

WEBARENA_STRUCTURED_SUMMARY_PROMPT = """You are maintaining a context-aware state summary for a web GUI agent.
The agent is controlling a browser (via Playwright/Chromium) to complete a task.

This summary is critical because it:
1. Preserves essential browsing state when conversation history grows too large
2. Prevents lost work when the session exceeds the agent's context window
3. Helps maintain continuity across many browsing steps

You will be given:
- The task the agent was asked to complete (prepended below)
- A list of events (observations and actions the agent has taken)
- The most recent previous state summary (if one exists)

Capture ALL relevant information, especially:
- Exact URLs and page titles for every page visited
- Exact element IDs for every button clicked, field filled, link followed
- Exact values typed or selected into form fields
- Data extracted from pages that the agent needs later (search results, account balances, table rows, error messages)
- Subtasks already completed (so the agent does not redo them)
- Errors and dead-ends encountered (so the agent does not re-try failed approaches)

Use exact identifiers, not prose. "clicked #submit-btn" is correct;
"submitted the form" is not. Do NOT hallucinate pages, elements, or values
that were not in the events."""


class WebArenaStructuredSummary(BaseMemoryManager):
    """
    Structured-summary memory manager for web GUI agent trajectories.

    Architecture identical to `StructuredSummaryMemory` (the proven OpenHands
    StructuredSummaryCondenser port). Only the field schema and prompt are
    web-task-tailored, plus two extra constructor fields (task_instruction,
    app_name) used to format the prompt header.

    Args:
        task_instruction: The user's task description, formatted into the
            prompt header so the structured summarizer can classify subtasks.
        app_name: Which webarena-infinity app (gmail/gitlab/etc.).
        max_size: Maximum number of messages before condensation.
        keep_first: Number of initial messages to always keep.
        max_event_length: Maximum character length per event in the prompt.
        summarization_model: LLM model for summarization.
        condensation_ratio: Fraction of max_size the view is compressed to.
        max_tokens: Token budget for the final view.
        max_token_size: If set, trigger condensation by token count.
        tokenizer_name: Tokenizer for counting.
    """

    def __init__(
        self,
        task_instruction: str = "",
        app_name: str = "",
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
        if keep_first >= int(max_size * condensation_ratio):
            raise ValueError(
                f"keep_first ({keep_first}) must be less than half of max_size ({max_size})"
            )
        if keep_first < 0:
            raise ValueError(f"keep_first ({keep_first}) cannot be negative")
        if max_size < 1:
            raise ValueError(f"max_size ({max_size}) cannot be non-positive")

        super().__init__(max_tokens, tokenizer_name, config)

        self.task_instruction = task_instruction
        self.app_name = app_name
        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.max_token_size = max_token_size

        # Persistent condensation records — mirrors summarizing memory.
        # Without these, the structured strategy re-summarizes the ENTIRE
        # history on every condensation step (O(n²) summarizer tokens).
        # With build_view(), each condensation only processes ~5-7 NEW events.
        self._condensations: List[CondensationRecord] = []

        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """Manage context with web-GUI-tailored structured summary condensation.

        Uses build_view() + persistent condensation records (matching the
        summarizing strategy) so that each condensation only processes NEW
        forgotten events, not the entire history. This reduces summarizer
        token cost from O(n²) to O(n) over the episode.
        """
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)

        # Build compressed view using stored condensation records
        view, view_to_original = build_view(messages, self._condensations)
        view_tokens = self.count_tokens(view)

        if self.max_token_size is not None:
            needs_condensation = view_tokens > self.max_token_size
        else:
            needs_condensation = len(view) > self.max_size

        if not needs_condensation:
            self._update_token_tracking(view_tokens)
            return FilteredContext(
                content=view,
                metadata={
                    "tokens": view_tokens,
                    "original_tokens": original_tokens,
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": original_tokens / max(1, view_tokens),
                    "strategy": "webarena_structured",
                    "view_size": len(view),
                    "raw_size": len(messages),
                    "condensation_count": len(self._condensations),
                    "step": self._step_count
                }
            )

        head = view[:self.keep_first]
        target_size = int(self.max_size * self.condensation_ratio)
        events_from_tail = target_size - len(head) - 1
        if events_from_tail < 1:
            events_from_tail = 1
        tail = view[-events_from_tail:]

        # Extract previous summary if present
        summary_event_content = ""
        if self.keep_first < len(view) and _is_summary_message(view[self.keep_first]):
            raw_content = _get_message_content(view[self.keep_first])
            summary_event_content = raw_content.replace(SUMMARY_MARKER, "").strip()

        # Collect forgotten events from the middle of the VIEW (not raw history)
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
                    "original_tokens": original_tokens,
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "webarena_structured",
                    "step": self._step_count
                }
            )

        # Build prompt with web-tailored header (task + app)
        prompt = WEBARENA_STRUCTURED_SUMMARY_PROMPT + "\n\n"
        prompt += f"TASK: {self.task_instruction or '(not provided)'}\n"
        prompt += f"APP: {self.app_name or '(unknown)'}\n\n"

        truncated_summary = _truncate_content(summary_event_content, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n"
        prompt += "\n\n"

        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        tool_desc = _build_webarena_tool_description()
        call_prompt_tokens = 0
        call_completion_tokens = 0
        t0 = time.time()
        try:
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_desc],
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_webarena_state_summary"},
                },
            )
            summarization_time = time.time() - t0

            if hasattr(response, 'usage') and response.usage:
                call_prompt_tokens = (
                    getattr(response.usage, 'prompt_tokens', 0)
                    or getattr(response.usage, 'input_tokens', 0) or 0
                )
                call_completion_tokens = (
                    getattr(response.usage, 'completion_tokens', 0)
                    or getattr(response.usage, 'output_tokens', 0) or 0
                )
                self._summarizer_prompt_tokens += call_prompt_tokens
                self._summarizer_completion_tokens += call_completion_tokens

            message = response.choices[0].message
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                raise ValueError("No tool calls found in response")

            summary_tool_call = None
            for tool_call in message.tool_calls:
                if tool_call.function.name == "create_webarena_state_summary":
                    summary_tool_call = tool_call
                    break

            if not summary_tool_call:
                raise ValueError("create_webarena_state_summary tool call not found")

            args_json = summary_tool_call.function.arguments
            args_dict = json.loads(args_json)

            if not args_dict.get("task"):
                args_dict["task"] = self.task_instruction
            if not args_dict.get("app"):
                args_dict["app"] = self.app_name

            summary_text = _format_webarena_state_summary(args_dict)

        except (ValueError, AttributeError, KeyError, json.JSONDecodeError) as e:
            summarization_time = time.time() - t0
            print(f"Warning: Failed to parse webarena summary tool call: {e}. Using empty summary.")
            summary_text = _format_webarena_state_summary({
                "task": self.task_instruction,
                "app": self.app_name,
            })

        except Exception as e:
            summarization_time = time.time() - t0
            print(f"Warning: WebArena structured summary LLM call failed: {e}")
            summary_text = summary_event_content or _format_webarena_state_summary({
                "task": self.task_instruction,
                "app": self.app_name,
            })

        # Store condensation record for incremental compression
        original_forgotten_indices = [
            view_to_original[vi] for vi in forgotten_view_indices
            if vi < len(view_to_original) and view_to_original[vi] >= 0
        ]
        if original_forgotten_indices:
            self._condensations.append(CondensationRecord(
                forgotten_events_start=min(original_forgotten_indices),
                forgotten_events_end=max(original_forgotten_indices) + 1,
                summary=summary_text,
                summary_offset=self.keep_first,
            ))

        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {summary_text}"
        }
        result_messages = head + [summary_message] + tail

        self._compaction_count += 1
        filtered_tokens = self.count_tokens(result_messages)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=result_messages,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": True,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "webarena_structured",
                "summarization_count": self._compaction_count,
                "condensation_count": len(self._condensations),
                "messages_before": len(view),
                "messages_after": len(result_messages),
                "forgotten_events": len(forgotten_events),
                "view_size": len(result_messages),
                "raw_size": len(messages),
                "step": self._step_count,
                "summarizer_prompt_tokens": call_prompt_tokens,
                "summarizer_completion_tokens": call_completion_tokens,
                "summarization_time_seconds": summarization_time,
            }
        )

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._condensations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._summarizer_prompt_tokens = 0
        self._summarizer_completion_tokens = 0
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()

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
            "task_instruction": self.task_instruction,
            "app_name": self.app_name,
        })
        return base_stats


# Register in memory model registry
register_memory_model("webarena_structured", WebArenaStructuredSummary)
