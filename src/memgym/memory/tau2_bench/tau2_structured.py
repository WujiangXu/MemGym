"""
Structured-summary memory manager for tau2-bench dialogue trajectories.

JSON-schema variant of ``Tau2SummarizingMemory``. Same control flow
(``head + summary + tail`` with explicit ``keep_first`` / ``keep_last``
anchors), same trigger logic (``len > max_size`` or token-budget), same
condensation_ratio handling — but the summarizer is called via litellm
function calling so the output is structured JSON conforming to the
8-field dialogue schema. The compressed message is then deterministically
rendered to text from the validated JSON object, so the agent always sees
consistent formatting (no LLM-style prose drift between calls).

Schema (role-aware):
    USER_GOAL or USER_PERSONA
    USER_CONSTRAINTS  (list)
    DECISIONS_MADE  (list)
    TOOL_CALLS_EXECUTED  (list)
    INFORMATION_GATHERED  (list)
    OUTSTANDING_COMMITMENTS  (list)
    OPEN_QUESTIONS  (list)
    NEXT_ACTION  (string)

When the LLM call fails or returns invalid JSON, we fall back to the
free-form ``Tau2SummarizingMemory`` summary text and bump
``stats["fallback_count"]`` so we can see how often the structured
adapter degrades to free-form during a run.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from .tau2_summarizing import (
    SUMMARY_MARKER,
    TAU2_SUMMARIZATION_PROMPT_AGENT,
    TAU2_SUMMARIZATION_PROMPT_USER,
    CondensationRecord,
    _adjust_tail_for_tool_groups,
    _get_message_content,
    _is_summary_message,
    _truncate_content,
    build_view,
)


# =============================================================================
# Tau2 dialogue state-summary fields (the 8-field schema)
# =============================================================================

# Field definitions: {name: description}.
# All fields are strings — list-shaped fields are encoded as newline-delimited
# text inside the string value, matching the WebArena structured summarizer's
# schema shape (litellm function calling validates better with flat strings
# than nested arrays across providers).
TAU2_STATE_SUMMARY_FIELDS_AGENT = {
    "user_goal": (
        "One-sentence restatement of the user's task in their own words. "
        "The agent must honor this across the entire conversation."
    ),
    "user_constraints": (
        "Newline-delimited list of hard constraints the user has stated "
        "(must depart before noon, can only use credit card ending 1234, "
        "won't pay over $X, prefers aisle seats, etc). One per line."
    ),
    "decisions_made": (
        "Newline-delimited list of concrete decisions already taken. "
        "One per line — selected flight AA123, chose refund over exchange, etc."
    ),
    "tool_calls_executed": (
        "Newline-delimited list of tools the agent has already called and "
        "what came back. Format: tool_name(args) → result_summary. One per line. "
        "Do NOT list tools the agent plans to call — only ones already executed."
    ),
    "information_gathered": (
        "Newline-delimited list of facts learned from tool calls or user "
        "statements. user_id=U12345, account_balance=$340, frequent_flyer=gold, "
        "etc. One per line."
    ),
    "outstanding_commitments": (
        "Newline-delimited list of things the agent promised to do but has "
        "not done yet. One per line."
    ),
    "open_questions": (
        "Newline-delimited list of things the agent asked the user that "
        "remain unanswered. One per line."
    ),
    "next_action": (
        "One sentence describing what the agent should do on the very next "
        "turn, inferred from the most recent events."
    ),
}

TAU2_STATE_SUMMARY_FIELDS_USER = {
    **TAU2_STATE_SUMMARY_FIELDS_AGENT,
    # Drop the agent-side user_goal field and replace with a persona field.
    # Python dicts preserve insertion order so the JSON schema lists
    # user_persona in the same slot user_goal would have occupied.
}
del TAU2_STATE_SUMMARY_FIELDS_USER["user_goal"]
TAU2_STATE_SUMMARY_FIELDS_USER = {
    "user_persona": (
        "One-sentence restatement of the character the user simulator is "
        "playing — 'a frequent flyer demanding a same-day refund', 'a "
        "budget-conscious shopper buying their first phone'. The simulator "
        "must stay in character across the entire conversation."
    ),
    **TAU2_STATE_SUMMARY_FIELDS_USER,
}

TAU2_STATE_SUMMARY_REQUIRED_AGENT = [
    "user_goal",
    "user_constraints",
    "decisions_made",
    "tool_calls_executed",
    "next_action",
]

TAU2_STATE_SUMMARY_REQUIRED_USER = [
    "user_persona",
    "user_constraints",
    "decisions_made",
    "next_action",
]


def _build_tau2_tool_description(role: str) -> Dict[str, Any]:
    """Build the litellm tool description for tau2 structured summarization."""
    fields = (
        TAU2_STATE_SUMMARY_FIELDS_USER if role == "user"
        else TAU2_STATE_SUMMARY_FIELDS_AGENT
    )
    required = (
        TAU2_STATE_SUMMARY_REQUIRED_USER if role == "user"
        else TAU2_STATE_SUMMARY_REQUIRED_AGENT
    )

    properties = {
        name: {"type": "string", "description": desc}
        for name, desc in fields.items()
    }

    return {
        "type": "function",
        "function": {
            "name": "create_tau2_dialogue_summary",
            "description": (
                "Create a comprehensive summary of the current state of a "
                "tau2-bench dialogue, to preserve context when conversation "
                "history grows too large. Preserve exact identifiers (user_id, "
                "account numbers, flight numbers, ticket ids) and exact "
                "constraints. Do NOT paraphrase or hallucinate."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _format_tau2_state_summary(fields: Dict[str, str], role: str) -> str:
    """Render a validated state-summary dict into the schema's text form."""
    goal_or_persona_label = "USER_PERSONA" if role == "user" else "USER_GOAL"
    goal_or_persona_value = (
        fields.get("user_persona", "") if role == "user"
        else fields.get("user_goal", "")
    )

    sections = [
        "[COMPRESSED HISTORY]",
        f"{goal_or_persona_label}: {goal_or_persona_value or '(unknown)'}",
        f"USER_CONSTRAINTS:\n{fields.get('user_constraints', '') or '  - (none)'}",
        f"DECISIONS_MADE:\n{fields.get('decisions_made', '') or '  - (none)'}",
        f"TOOL_CALLS_EXECUTED:\n{fields.get('tool_calls_executed', '') or '  - (none)'}",
        f"INFORMATION_GATHERED:\n{fields.get('information_gathered', '') or '  - (none)'}",
        f"OUTSTANDING_COMMITMENTS:\n{fields.get('outstanding_commitments', '') or '  - (none)'}",
        f"OPEN_QUESTIONS:\n{fields.get('open_questions', '') or '  - (none)'}",
        f"NEXT_ACTION: {fields.get('next_action', '') or '(unspecified)'}",
    ]
    return "\n\n".join(sections)


# =============================================================================
# Tau2StructuredSummary
# =============================================================================

class Tau2StructuredSummary(BaseMemoryManager):
    """Structured-summary memory manager for tau2-bench dialogue trajectories.

    Args:
        task_instruction: User's task description / persona.
        domain: tau2 domain ("airline", "retail", "telecom", "mock").
        solo_mode: Whether the agent runs without a user simulator.
        role: ``"agent"`` or ``"user"`` — controls schema (USER_GOAL vs
            USER_PERSONA) and which prompt template is used.
        max_size: Maximum number of events in view before condensation.
        keep_first: Number of initial events to always keep (top anchor).
        keep_last: Number of most-recent events to always keep verbatim
            (bottom anchor). Same dialogue-specific floor as
            Tau2SummarizingMemory.
        max_event_length: Maximum character length per event in the prompt.
        summarization_model: LLM model id for summarization.
        condensation_ratio: Fraction of max_size the view is compressed to.
        max_tokens: Token budget for the final view.
        max_token_size: If set, trigger condensation by token count.
        tokenizer_name: Tokenizer name for counting.
    """

    def __init__(
        self,
        task_instruction: str = "",
        domain: str = "",
        solo_mode: bool = False,
        role: str = "agent",
        max_size: int = 30,
        keep_first: int = 1,
        keep_last: int = 4,
        max_event_length: int = 20_000,
        summarization_model: str = "gpt-4o-mini",
        condensation_ratio: float = 0.6,
        max_tokens: int = 4000,
        max_token_size: Optional[int] = None,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
    ):
        if keep_first < 0:
            raise ValueError(f"keep_first ({keep_first}) cannot be negative")
        if keep_last < 1:
            raise ValueError(f"keep_last ({keep_last}) must be >= 1 — dialogue agents need fresh tail context")
        if max_size < keep_first + keep_last + 1:
            raise ValueError(
                f"max_size ({max_size}) must be > keep_first ({keep_first}) + keep_last ({keep_last})"
            )
        if role not in ("agent", "user"):
            raise ValueError(f"role must be 'agent' or 'user', got {role!r}")

        super().__init__(max_tokens, tokenizer_name, config)

        self.task_instruction = task_instruction
        self.domain = domain
        self.solo_mode = solo_mode
        self.role = role
        self.max_size = max_size
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.max_event_length = max_event_length
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.max_token_size = max_token_size

        self._condensations: List[CondensationRecord] = []
        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0
        self._fallback_count: int = 0
        # Per-call deltas reset at the start of every _call_structured_summarize
        # invocation so the compacted-return metadata can carry an accurate
        # ``condensation_event`` block (matches tau2_summarizing's contract).
        self._last_summarizer_prompt_tokens: int = 0
        self._last_summarizer_completion_tokens: int = 0
        self._last_summarization_time: float = 0.0

    def set_episode_state(self, **state: Any) -> None:
        super().set_episode_state(**state)
        if "instruction" in state and state["instruction"]:
            self.task_instruction = str(state["instruction"])
        if "domain" in state and state["domain"]:
            self.domain = str(state["domain"])
        if "solo_mode" in state:
            self.solo_mode = bool(state["solo_mode"])

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Compress with structured JSON output, fall back to free-form on error."""
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)

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
                    "strategy": f"tau2_structured[{self.role}]",
                    "view_size": len(view),
                    "raw_size": len(messages),
                    "step": self._step_count,
                },
            )

        # ---- Anchor + middle-compress (same logic as tau2_summarizing) ----
        head = view[: self.keep_first]
        target_size = int(self.max_size * self.condensation_ratio)
        events_from_tail = max(self.keep_last, target_size - len(head) - 1)
        events_from_tail = min(events_from_tail, max(1, len(view) - len(head) - 1))

        # Extend tail boundary backward to avoid splitting tool-call groups
        # (same fix as tau2_summarizing — see _adjust_tail_for_tool_groups).
        events_from_tail = _adjust_tail_for_tool_groups(
            view, events_from_tail, self.keep_first
        )

        tail = view[-events_from_tail:] if events_from_tail > 0 else []

        previous_summary_text = ""
        if self.keep_first < len(view) and _is_summary_message(view[self.keep_first]):
            raw = _get_message_content(view[self.keep_first])
            previous_summary_text = raw.replace(SUMMARY_MARKER, "").strip()

        middle_start = self.keep_first
        middle_end = len(view) - events_from_tail
        forgotten_events: List[Any] = []
        forgotten_view_indices: List[int] = []
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
                    "strategy": f"tau2_structured[{self.role}]",
                    "step": self._step_count,
                },
            )

        # ---- Build the structured-output prompt ----
        prompt_template = (
            TAU2_SUMMARIZATION_PROMPT_USER if self.role == "user"
            else TAU2_SUMMARIZATION_PROMPT_AGENT
        )
        prompt = prompt_template.format(
            domain=self.domain or "(unknown)",
            role=self.role,
            solo_mode="true" if self.solo_mode else "false",
            task_instruction=self.task_instruction or "(not provided)",
        ) + "\n\n"

        truncated_summary = _truncate_content(previous_summary_text, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n\n"

        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        prompt += (
            "\nReturn the summary by calling the create_tau2_dialogue_summary "
            "tool with all required fields."
        )

        summary_text, fallback = self._call_structured_summarize(prompt, previous_summary_text)
        if fallback:
            self._fallback_count += 1

        # ---- Persist condensation record + build the new view ----
        original_forgotten_indices = [
            view_to_original[vi] for vi in forgotten_view_indices
            if vi < len(view_to_original) and view_to_original[vi] >= 0
        ]
        if original_forgotten_indices:
            self._condensations.append(
                CondensationRecord(
                    forgotten_events_start=min(original_forgotten_indices),
                    forgotten_events_end=max(original_forgotten_indices) + 1,
                    summary=summary_text,
                    summary_offset=self.keep_first,
                )
            )
        self._compaction_count += 1

        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {summary_text}",
        }
        result = head + [summary_message] + tail

        filtered_tokens = self.count_tokens(result)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=result,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": True,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": f"tau2_structured[{self.role}]",
                "summarization_count": self._compaction_count,
                "fallback_count": self._fallback_count,
                "messages_before": len(view),
                "messages_after": len(result),
                "forgotten_events": len(forgotten_events),
                "step": self._step_count,
                "keep_first": self.keep_first,
                "keep_last": self.keep_last,
                "events_from_tail": events_from_tail,
                "structured_fallback": fallback,
                # Mirrors tau2_summarizing's contract so the runner can consume
                # either adapter uniformly when writing {task_id}_training.json.
                "condensation_event": {
                    "summary_text": summary_text,
                    "forgotten_count": len(forgotten_events),
                    "summarization_model": self.summarization_model,
                    "summarization_time_seconds": self._last_summarization_time,
                    "summarizer_prompt_tokens": self._last_summarizer_prompt_tokens,
                    "summarizer_completion_tokens": self._last_summarizer_completion_tokens,
                    "structured_fallback": fallback,
                },
            },
        )

    def _call_structured_summarize(
        self,
        prompt: str,
        previous_summary_text: str,
    ) -> Tuple[str, bool]:
        """Call litellm with function calling, return (summary_text, used_fallback)."""
        tool_desc = _build_tau2_tool_description(self.role)
        self._last_summarizer_prompt_tokens = 0
        self._last_summarizer_completion_tokens = 0
        self._last_summarization_time = 0.0
        try:
            t0 = time.time()
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_desc],
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_tau2_dialogue_summary"},
                },
            )
            self._last_summarization_time = time.time() - t0

            usage = getattr(response, "usage", None)
            if usage:
                self._last_summarizer_prompt_tokens = (
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "input_tokens", 0) or 0
                )
                self._last_summarizer_completion_tokens = (
                    getattr(usage, "completion_tokens", 0)
                    or getattr(usage, "output_tokens", 0) or 0
                )
                self._summarizer_prompt_tokens += self._last_summarizer_prompt_tokens
                self._summarizer_completion_tokens += self._last_summarizer_completion_tokens

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            summary_call = next(
                (tc for tc in tool_calls
                 if getattr(tc, "function", None)
                 and tc.function.name == "create_tau2_dialogue_summary"),
                None,
            )
            if summary_call is None:
                raise ValueError("create_tau2_dialogue_summary tool call not found")

            args_dict = json.loads(summary_call.function.arguments)
            return _format_tau2_state_summary(args_dict, self.role), False

        except (ValueError, AttributeError, KeyError, json.JSONDecodeError) as exc:
            print(f"Warning: tau2 structured summary parse failed: {exc}. Falling back to previous summary.")
            return previous_summary_text or _format_tau2_state_summary({}, self.role), True
        except Exception as exc:
            print(f"Warning: tau2 structured summary LLM call failed: {exc}")
            return previous_summary_text or _format_tau2_state_summary({}, self.role), True

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._condensations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._summarizer_prompt_tokens = 0
        self._summarizer_completion_tokens = 0
        self._fallback_count = 0
        self._last_summarizer_prompt_tokens = 0
        self._last_summarizer_completion_tokens = 0
        self._last_summarization_time = 0.0

    def get_condensation_history(self) -> List[Dict[str, Any]]:
        """Export condensation records for trajectory logging.

        Mirrors ``Tau2SummarizingMemory.get_condensation_history`` so the
        runner can iterate over both adapters uniformly when writing
        ``{task_id}_training.json``.
        """
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
            "keep_last": self.keep_last,
            "max_event_length": self.max_event_length,
            "summarizer_total_prompt_tokens": self._summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": self._summarizer_completion_tokens,
            "summarizer_total_tokens": (
                self._summarizer_prompt_tokens + self._summarizer_completion_tokens
            ),
            "fallback_count": self._fallback_count,
            "task_instruction": self.task_instruction,
            "domain": self.domain,
            "solo_mode": self.solo_mode,
            "role": self.role,
        })
        return base_stats


# Register
register_memory_model("tau2_structured", Tau2StructuredSummary)
register_memory_model("tau2-structured", Tau2StructuredSummary)
