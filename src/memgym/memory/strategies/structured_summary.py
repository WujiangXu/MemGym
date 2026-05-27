"""
Structured summary memory manager for MemGym.

Exact port of OpenHands StructuredSummaryCondenser + StateSummary.
Uses LLM function calling to produce structured summaries with exact 17 fields,
ensuring consistent, machine-parseable context preservation.

Source: openhands/memory/condenser/impl/structured_summary_condenser.py
"""

import json
from typing import Any, Dict, List, Optional

from litellm import completion

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from .llm_summarizing import SUMMARY_MARKER, _truncate_content, _is_summary_message, _get_message_content


# =============================================================================
# StateSummary — exact 17 fields from OpenHands
# =============================================================================

# Field definitions: {name: description} — exact match of OpenHands StateSummary
STATE_SUMMARY_FIELDS = {
    "user_context": "Essential user requirements, goals, and clarifications in concise form.",
    "completed_tasks": "List of tasks completed so far with brief results.",
    "pending_tasks": "List of tasks that still need to be done.",
    "current_state": "Current variables, data structures, or other relevant state information.",
    "files_modified": "List of files that have been created or modified.",
    "function_changes": "List of functions that have been created or modified.",
    "data_structures": "List of key data structures in use or modified.",
    "tests_written": "Whether tests have been written for the changes. True, false, or unknown.",
    "tests_passing": "Whether all tests are currently passing. True, false, or unknown.",
    "failing_tests": "List of names or descriptions of any failing tests.",
    "error_messages": "List of key error messages encountered.",
    "branch_created": "Whether a branch has been created for this work. True, false, or unknown.",
    "branch_name": "Name of the current working branch if known.",
    "commits_made": "Whether any commits have been made. True, false, or unknown.",
    "pr_created": "Whether a pull request has been created. True, false, or unknown.",
    "pr_status": "Status of any pull request: 'draft', 'open', 'merged', 'closed', or 'unknown'.",
    "dependencies": "List of dependencies or imports that have been added or modified.",
    "other_relevant_context": "Any other important information that doesn't fit into the categories above.",
}

STATE_SUMMARY_REQUIRED = ["user_context", "completed_tasks", "pending_tasks"]


def _build_tool_description() -> Dict[str, Any]:
    """
    Build the tool description for function calling.

    Exact port of OpenHands StateSummary.tool_description().
    """
    properties = {}
    for field_name, description in STATE_SUMMARY_FIELDS.items():
        properties[field_name] = {"type": "string", "description": description}

    return {
        "type": "function",
        "function": {
            "name": "create_state_summary",
            "description": (
                "Creates a comprehensive summary of the current state of the interaction "
                "to preserve context when history grows too large. You must include "
                "non-empty values for user_context, completed_tasks, and pending_tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": STATE_SUMMARY_REQUIRED,
            },
        },
    }


def _format_state_summary(fields: Dict[str, str]) -> str:
    """
    Format a state summary dict into markdown string.

    Exact port of OpenHands StateSummary.__str__().
    """
    sections = [
        "# State Summary",
        "## Core Information",
        f"**User Context**: {fields.get('user_context', '')}",
        f"**Completed Tasks**: {fields.get('completed_tasks', '')}",
        f"**Pending Tasks**: {fields.get('pending_tasks', '')}",
        f"**Current State**: {fields.get('current_state', '')}",
        "## Code Changes",
        f"**Files Modified**: {fields.get('files_modified', '')}",
        f"**Function Changes**: {fields.get('function_changes', '')}",
        f"**Data Structures**: {fields.get('data_structures', '')}",
        f"**Dependencies**: {fields.get('dependencies', '')}",
        "## Testing Status",
        f"**Tests Written**: {fields.get('tests_written', '')}",
        f"**Tests Passing**: {fields.get('tests_passing', '')}",
        f"**Failing Tests**: {fields.get('failing_tests', '')}",
        f"**Error Messages**: {fields.get('error_messages', '')}",
        "## Version Control",
        f"**Branch Created**: {fields.get('branch_created', '')}",
        f"**Branch Name**: {fields.get('branch_name', '')}",
        f"**Commits Made**: {fields.get('commits_made', '')}",
        f"**PR Created**: {fields.get('pr_created', '')}",
        f"**PR Status**: {fields.get('pr_status', '')}",
        "## Additional Context",
        f"**Other Relevant Context**: {fields.get('other_relevant_context', '')}",
    ]
    return "\n\n".join(sections)


# =============================================================================
# Exact prompt from OpenHands StructuredSummaryCondenser
# =============================================================================

STRUCTURED_SUMMARY_PROMPT = """You are maintaining a context-aware state summary for an interactive software agent. This summary is critical because it:
1. Preserves essential context when conversation history grows too large
2. Prevents lost work when the session length exceeds token limits
3. Helps maintain continuity across multiple interactions

You will be given:
- A list of events (actions taken by the agent)
- The most recent previous summary (if one exists)

Capture all relevant information, especially:
- User requirements that were explicitly stated
- Work that has been completed
- Tasks that remain pending
- Current state of code, variables, and data structures
- The status of any version control operations"""


class StructuredSummaryMemory(BaseMemoryManager):
    """
    A memory manager that produces structured summaries via function calling.

    Exact port of OpenHands StructuredSummaryCondenser.

    Same trigger/head/tail/forgotten logic as LLMSummarizingCondenser, but uses
    function calling to produce a structured StateSummary with 17 fields instead
    of free-text summary.

    The StateSummary is formatted as markdown and inserted as a system message.

    Args:
        max_size: Maximum number of messages before condensation. Default: 100.
        keep_first: Number of initial messages to always keep. Default: 1.
        max_event_length: Maximum character length per event in the prompt. Default: 10_000.
        summarization_model: LLM model for summarization, any litellm provider. Default: "gpt-4o-mini".
        max_token_size: If set, trigger condensation based on token count instead of
            message count. Default: None (use max_size message-count trigger).
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

        # Summarizer LLM token tracking (cumulative across all condensation calls)
        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0

        # litellm handles API key routing automatically

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Manage context with structured summary condensation.

        Same algorithm as LLMSummarizingMemory but uses function calling
        to produce structured output.
        """
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)

        # Check if condensation needed
        # Token-based trigger: original_tokens > max_token_size
        # Message-based trigger (OpenHands default): len(messages) > max_size
        if self.max_token_size is not None:
            needs_condensation = original_tokens > self.max_token_size
        else:
            needs_condensation = len(messages) > self.max_size

        if not needs_condensation:
            self._update_token_tracking(original_tokens)
            return FilteredContext(
                content=messages,
                metadata={
                    "tokens": original_tokens,
                    "original_tokens": original_tokens,
                    "was_compacted": False,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "structured_summary",
                    "step": self._step_count
                }
            )

        # Condensation needed — exact OpenHands get_condensation logic
        head = messages[:self.keep_first]
        target_size = int(self.max_size * self.condensation_ratio)
        events_from_tail = target_size - len(head) - 1
        tail = messages[-events_from_tail:]

        # Check for previous summary
        summary_event_content = ""
        if self.keep_first < len(messages) and _is_summary_message(messages[self.keep_first]):
            raw_content = _get_message_content(messages[self.keep_first])
            summary_event_content = raw_content.replace(SUMMARY_MARKER, "").strip()

        # Identify events to forget
        forgotten_events = []
        middle = messages[self.keep_first:-events_from_tail]
        for msg in middle:
            if not _is_summary_message(msg):
                forgotten_events.append(msg)

        if not forgotten_events:
            self._update_token_tracking(original_tokens)
            return FilteredContext(
                content=messages,
                metadata={
                    "tokens": original_tokens,
                    "original_tokens": original_tokens,
                    "was_compacted": False,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "structured_summary",
                    "step": self._step_count
                }
            )

        # Build prompt (exact OpenHands StructuredSummaryCondenser format)
        prompt = STRUCTURED_SUMMARY_PROMPT + "\n\n"

        truncated_summary = _truncate_content(summary_event_content, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n"

        prompt += "\n\n"

        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        # Call LLM with function calling (exact OpenHands approach)
        tool_desc = _build_tool_description()
        call_prompt_tokens = 0
        call_completion_tokens = 0
        try:
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_desc],
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_state_summary"},
                },
            )

            # Extract summarizer token usage (provider-compatible)
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

            # Parse function call response (same as OpenHands)
            message = response.choices[0].message

            if not hasattr(message, "tool_calls") or not message.tool_calls:
                raise ValueError("No tool calls found in response")

            summary_tool_call = None
            for tool_call in message.tool_calls:
                if tool_call.function.name == "create_state_summary":
                    summary_tool_call = tool_call
                    break

            if not summary_tool_call:
                raise ValueError("create_state_summary tool call not found")

            args_json = summary_tool_call.function.arguments
            args_dict = json.loads(args_json)

            # Format as markdown string (exact OpenHands StateSummary.__str__())
            summary_text = _format_state_summary(args_dict)

        except (ValueError, AttributeError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to parse summary tool call: {e}. Using empty summary.")
            # Fall back to empty summary (same as OpenHands)
            summary_text = _format_state_summary({})

        except Exception as e:
            print(f"Warning: Structured summary LLM call failed: {e}")
            summary_text = summary_event_content or _format_state_summary({})

        # Build result: head + [summary_message] + tail
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
                "strategy": "structured_summary",
                "summarization_count": self._compaction_count,
                "messages_before": len(messages),
                "messages_after": len(result_messages),
                "forgotten_events": len(forgotten_events),
                "step": self._step_count,
                "summarizer_prompt_tokens": call_prompt_tokens,
                "summarizer_completion_tokens": call_completion_tokens
            }
        )

    def reset(self) -> None:
        """Reset memory state for new episode."""
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
            "summarizer_total_prompt_tokens": self._summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": self._summarizer_completion_tokens,
            "summarizer_total_tokens": self._summarizer_prompt_tokens + self._summarizer_completion_tokens,
        })
        return base_stats


# Register in memory model registry
register_memory_model("structured_summary", StructuredSummaryMemory)
