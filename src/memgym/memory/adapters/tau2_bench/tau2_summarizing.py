"""
LLM summarizing memory manager for tau2-bench dialogue trajectories.

Architecture mirrors ``memory/webarena/webarena_summarizing.py`` line-for-line:

- ``CondensationRecord`` dataclass persists condensation decisions
- ``build_view()`` rebuilds the condensed view from full history + records
- Trigger checks the VIEW length, not raw history
- ``manage_context()`` triggers when ``len(view) > max_size`` OR ``condensation_requested``
- Persistence: stored records survive across ``manage_context()`` calls

**Two things change** vs the WebArena adapter:

1. **Schema and prompt**: Replaces the 7-field web schema with the 8-field
   dialogue schema documented in the τ²-bench plan: USER_GOAL,
   USER_CONSTRAINTS, DECISIONS_MADE, TOOL_CALLS_EXECUTED,
   INFORMATION_GATHERED, OUTSTANDING_COMMITMENTS, OPEN_QUESTIONS, NEXT_ACTION.
   When ``role="user"``, USER_GOAL is swapped for USER_PERSONA — the user
   simulator needs to remember the character it's playing, not the task
   it's pursuing.

2. **Explicit ``keep_last`` parameter**: WebArena derives the tail anchor
   from ``int(max_size * condensation_ratio) - keep_first - 1``. For
   dialogue, fresh turns are critical for next-action grounding (the
   agent has to remember "the user just said her credit card ends in
   1234" verbatim), so we make the bottom anchor explicit and enforce
   it as a hard floor on ``events_from_tail`` independent of
   ``condensation_ratio``.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from ...base import BaseMemoryManager, FilteredContext, register_memory_model


# =============================================================================
# Dialogue-tailored summarization prompts (agent + user variants)
# =============================================================================
#
# We keep two prompt variants — agent and user — instead of branching
# per-call, so the only field that differs (USER_GOAL vs USER_PERSONA)
# stays in one obvious place. Everything else about the schema is shared.

_TAU2_PROMPT_HEADER_COMMON = """You are maintaining a context-aware state summary for a tau2-bench dialogue agent.
The agent is having a tool-augmented conversation with a user (or user simulator) to complete a customer-service task.
You will be given a list of events corresponding to messages and tool calls in the conversation,
and the most recent previous summary if one exists.

Some of the earlier messages and tool calls are being EVICTED from the agent's context window —
your summary is the only record the agent will have of them going forward. The summary must
preserve the critical dialogue state so the agent can continue seamlessly without forgetting
constraints, decisions, or facts the user already provided.

DOMAIN: {domain}
ROLE: {role}
SOLO_MODE: {solo_mode}
TASK: {task_instruction}

Track the following sections. Preserve exact identifiers, account numbers, dates, prices,
flight numbers, ticket ids, and order ids. Do NOT hallucinate facts that were not in the events."""

_TAU2_SCHEMA_BODY_AGENT = """

USER_GOAL: (One-sentence restatement of the user's task in their own words. The agent must
honor this across the entire conversation, so this is the most important field.)

USER_CONSTRAINTS: (Hard constraints the user has stated — must depart before noon, can only
use credit card ending 1234, won't pay over $X, prefers aisle seats, etc. One per line.
The agent MUST honor every constraint. List even constraints stated many turns ago.)

DECISIONS_MADE: (Concrete decisions already taken — selected flight AA123, chose refund over
exchange, agreed to upgrade to economy plus, picked Tuesday departure. One per line.)

TOOL_CALLS_EXECUTED: (Tools the agent has already called and what came back. Format as:
tool_name(args) → result_summary. One per line. CRITICAL: do not list tools the agent
plans to call — only ones already executed.)

INFORMATION_GATHERED: (Facts about the user / account / domain learned from tool calls or
user statements. user_id=U12345, account_balance=$340, frequent_flyer=gold, has 2 active
reservations, etc. One per line.)

OUTSTANDING_COMMITMENTS: (Things the agent promised to do but has not done yet. "agent
said it would email the receipt", "agent promised to apply 10% discount before final
charge". One per line.)

OPEN_QUESTIONS: (Things the agent asked the user that remain unanswered. The agent should
re-ask or wait for these rather than proceeding. One per line.)

NEXT_ACTION: (One sentence describing what the agent should do on the very next turn,
inferred from the most recent events.)

PRIORITIZE:
1. Exact identifiers and amounts over prose
2. Constraints — they MUST be re-honored even after compression
3. State that cannot be re-derived by re-asking the user
4. Tool calls already executed — re-running them wastes turns and may double-charge

SKIP: Pleasantries, repeated greetings, tool call boilerplate, hallucinated history."""

_TAU2_SCHEMA_BODY_USER = """

USER_PERSONA: (One-sentence restatement of the character the user simulator is playing —
"a frequent flyer demanding a same-day refund", "a budget-conscious shopper buying their
first phone". Replaces USER_GOAL on the user side because the simulator must stay in
character across the entire conversation.)

USER_CONSTRAINTS: (Hard constraints the user has stated and must continue to enforce —
budget caps, schedule windows, payment methods. One per line. The simulator MUST keep
asserting these even if the agent tries to negotiate them away.)

DECISIONS_MADE: (Concrete decisions the user has already accepted from the agent. The
simulator should not re-litigate these.)

TOOL_CALLS_EXECUTED: (Tools the user has called via the user-side tool API, if any.
Format as: tool_name(args) → result_summary. One per line.)

INFORMATION_GATHERED: (Facts the user has learned from the agent or from their own
tool calls. The simulator should not pretend to be ignorant of these on later turns.)

OUTSTANDING_COMMITMENTS: (Things the user promised the agent — "I'll send the photo",
"I'll wait on hold". One per line.)

OPEN_QUESTIONS: (Things the user asked the agent that remain unanswered. The simulator
should re-ask or escalate rather than letting the question drop.)

NEXT_ACTION: (One sentence describing what the user should do on the very next turn,
inferred from the persona + most recent events.)

PRIORITIZE:
1. The persona — staying in character is the simulator's whole job
2. Constraints — the user must keep enforcing them
3. Information already learned — do not pretend to be a fresh user

SKIP: Agent boilerplate, repeated apologies, anything that contradicts the persona."""


TAU2_SUMMARIZATION_PROMPT_AGENT = _TAU2_PROMPT_HEADER_COMMON + _TAU2_SCHEMA_BODY_AGENT
TAU2_SUMMARIZATION_PROMPT_USER = _TAU2_PROMPT_HEADER_COMMON + _TAU2_SCHEMA_BODY_USER


# Marker used to identify summary messages in the view (kept identical to
# WebArena so trajectory recorders that grep for the marker keep working).
SUMMARY_MARKER = "[Summary]"


# =============================================================================
# CondensationRecord — mirrors webarena_summarizing.CondensationRecord
# =============================================================================

@dataclass
class CondensationRecord:
    """Record of a condensation event, stored in memory manager state."""
    forgotten_events_start: int
    forgotten_events_end: int
    summary: str
    summary_offset: int


def build_view(
    messages: List[Any],
    condensations: List[CondensationRecord],
) -> Tuple[List[Any], List[int]]:
    """Build condensed view from full message history + condensation records.

    Identical to webarena_summarizing.build_view — kept inline rather than
    imported because the SUMMARY_MARKER format is the same and we don't
    want a cross-track import dependency.
    """
    forgotten = set()
    for record in condensations:
        forgotten.update(range(record.forgotten_events_start, record.forgotten_events_end))

    kept_indices = [i for i in range(len(messages)) if i not in forgotten]
    kept_messages = [messages[i] for i in kept_indices]

    if condensations:
        last = condensations[-1]
        summary_msg = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {last.summary}",
        }
        offset = last.summary_offset
        view_messages = kept_messages[:offset] + [summary_msg] + kept_messages[offset:]
        view_indices = kept_indices[:offset] + [-1] + kept_indices[offset:]
        return view_messages, view_indices

    return kept_messages, kept_indices


# =============================================================================
# Helpers
# =============================================================================

def _truncate_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]


def _is_summary_message(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    content = msg.get("content", "")
    return isinstance(content, str) and SUMMARY_MARKER in content


def _get_message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content", "")
        return str(content) if content else ""
    return str(msg)


def _adjust_tail_for_tool_groups(
    view: List[Any],
    events_from_tail: int,
    keep_first: int,
) -> int:
    """Extend ``events_from_tail`` so the tail boundary doesn't split a tool-call group.

    When the boundary (``view[-events_from_tail]``) lands on a tool message,
    the preceding assistant's ``tool_calls`` end up in the compressed middle
    while the tool results stay in the tail.  After compression the middle
    becomes a ``[Summary]`` message (role=user), so the tail's leading tool
    messages have no matching ``toolUse`` in the immediately preceding
    assistant turn → Bedrock ``BadRequestError``.

    Fix: walk backward from the boundary past consecutive tool messages.
    If we find an assistant message with ``tool_calls``, extend the tail to
    include it (and the intervening tool messages).  If the assistant can't
    be found (compressed in a prior round), return the original boundary and
    let ``repair_tool_call_pairs`` strip the orphans.
    """
    if events_from_tail <= 0 or events_from_tail >= len(view):
        return events_from_tail

    tail_start = len(view) - events_from_tail

    if tail_start <= keep_first:
        return events_from_tail  # already at or past the head boundary

    # Check whether the first tail message is a tool message.
    msg = view[tail_start]
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return events_from_tail  # boundary is clean — no split

    # Walk backward past consecutive tool messages.
    scan = tail_start - 1
    while scan >= keep_first:
        m = view[scan]
        role = m.get("role", "") if isinstance(m, dict) else ""
        if role != "tool":
            break
        scan -= 1

    # If we landed on an assistant with tool_calls, include it in the tail.
    if scan >= keep_first:
        m = view[scan]
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            return len(view) - scan

    # Couldn't find the initiating assistant (may be in a prior condensation).
    # repair_tool_call_pairs will strip the orphaned tool results.
    return events_from_tail


# =============================================================================
# Tau2SummarizingMemory
# =============================================================================

class Tau2SummarizingMemory(BaseMemoryManager):
    """LLM-summarizing memory manager for tau2-bench dialogue trajectories.

    Args:
        task_instruction: The user's task description, formatted into the
            summarization prompt header.
        domain: tau2 domain ("airline", "retail", "telecom", "mock").
        solo_mode: Whether the agent runs without a user simulator.
        role: ``"agent"`` (default) or ``"user"`` — controls schema and
            prompt template (USER_GOAL vs USER_PERSONA).
        max_size: Maximum number of events in view before condensation.
        keep_first: Number of initial events to always keep (top anchor).
        keep_last: Number of most-recent events to always keep verbatim
            (bottom anchor). NEW vs WebArena — explicit floor on the
            tail anchor independent of condensation_ratio.
        max_event_length: Maximum character length per event in the
            summarizer prompt.
        summarization_model: LLM model id for summarization.
        condensation_ratio: Fraction of max_size the view is compressed
            to during condensation.
        max_tokens: Token budget for the final view.
        max_token_size: If set, trigger condensation by token count
            instead of message count.
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
                f"max_size ({max_size}) must be > keep_first ({keep_first}) + keep_last ({keep_last}) "
                f"otherwise compression has nothing to compress"
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
        self._condensation_requested: bool = False

        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0
        self._last_summarizer_prompt_tokens: int = 0
        self._last_summarizer_completion_tokens: int = 0
        self._last_summarization_time: float = 0.0

    def request_condensation(self) -> None:
        """Force condensation on next manage_context() call."""
        self._condensation_requested = True

    def set_episode_state(self, **state: Any) -> None:
        """Update task / domain / solo_mode from runner-supplied state.

        Stays compatible with the WebArena pattern of capturing the
        instruction at episode start, but also lets the runner override
        ``domain`` / ``solo_mode`` if a single memory instance is reused
        across domains (e.g. inside a worker pool).
        """
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
        """Manage context with rolling LLM summarization (dialogue prompt)."""
        messages = list(original_context) + [current_observation]

        view, view_to_original = build_view(messages, self._condensations)
        view_tokens = self.count_tokens(view)

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
                    "strategy": f"tau2_summarizing[{self.role}]",
                    "view_size": len(view),
                    "raw_size": len(messages),
                    "condensation_count": len(self._condensations),
                    "step": self._step_count,
                },
            )

        self._condensation_requested = False

        # ---- Anchor + middle-compress ----
        # Top anchor: keep_first events.
        head = view[: self.keep_first]

        # Bottom anchor: HARD floor of self.keep_last.
        # We also let condensation_ratio extend the tail if it computes
        # a larger value, but never below keep_last.
        target_size = int(self.max_size * self.condensation_ratio)
        events_from_tail = max(self.keep_last, target_size - len(head) - 1)
        # Don't request more tail events than there are slots after the head + summary.
        events_from_tail = min(events_from_tail, max(1, len(view) - len(head) - 1))

        # Extend tail boundary backward to avoid splitting tool-call groups.
        # Without this, a boundary landing between an assistant's tool_calls
        # and their tool results produces orphaned toolResult blocks that
        # crash Bedrock's Converse API.
        events_from_tail = _adjust_tail_for_tool_groups(
            view, events_from_tail, self.keep_first
        )

        tail = view[-events_from_tail:] if events_from_tail > 0 else []

        # Pull a previous summary out of the head region (if it landed there).
        summary_event_content = ""
        if self.keep_first < len(view) and _is_summary_message(view[self.keep_first]):
            raw = _get_message_content(view[self.keep_first])
            summary_event_content = raw.replace(SUMMARY_MARKER, "").strip()

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
                    "original_tokens": self.count_tokens(messages),
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": f"tau2_summarizing[{self.role}]",
                    "step": self._step_count,
                },
            )

        summary = self._call_llm_summarize(forgotten_events, summary_event_content)

        original_forgotten_indices = [
            view_to_original[vi] for vi in forgotten_view_indices
            if vi < len(view_to_original) and view_to_original[vi] >= 0
        ]

        if original_forgotten_indices:
            self._condensations.append(
                CondensationRecord(
                    forgotten_events_start=min(original_forgotten_indices),
                    forgotten_events_end=max(original_forgotten_indices) + 1,
                    summary=summary,
                    summary_offset=self.keep_first,
                )
            )

        self._compaction_count += 1

        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {summary}",
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
                "strategy": f"tau2_summarizing[{self.role}]",
                "summarization_count": self._compaction_count,
                "condensation_count": len(self._condensations),
                "messages_before": len(view),
                "messages_after": len(result),
                "forgotten_events": len(forgotten_events),
                "view_size": len(result),
                "raw_size": len(messages),
                "step": self._step_count,
                "keep_first": self.keep_first,
                "keep_last": self.keep_last,
                "events_from_tail": events_from_tail,
                "condensation_event": {
                    "summary_text": summary,
                    "forgotten_count": len(forgotten_events),
                    "summarization_model": self.summarization_model,
                    "summarization_time_seconds": self._last_summarization_time,
                    "summarizer_prompt_tokens": self._last_summarizer_prompt_tokens,
                    "summarizer_completion_tokens": self._last_summarizer_completion_tokens,
                },
            },
        )

    def _call_llm_summarize(
        self,
        forgotten_events: List[Any],
        previous_summary: str,
    ) -> str:
        """Call LLM to summarize forgotten events with the dialogue-tailored prompt."""
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

        truncated_summary = _truncate_content(previous_summary, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n\n"

        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        prompt += "\nNow summarize the events using the rules above."

        try:
            t0 = time.time()
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
            )
            self._last_summarization_time = time.time() - t0

            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
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

            return response.choices[0].message.content
        except Exception as exc:
            self._last_summarization_time = 0.0
            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
            print(f"Warning: tau2 LLM summarization failed: {exc}")
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
        self._last_summarization_time = 0.0

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
            "keep_last": self.keep_last,
            "max_event_length": self.max_event_length,
            "condensation_count": len(self._condensations),
            "summarizer_total_prompt_tokens": self._summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": self._summarizer_completion_tokens,
            "summarizer_total_tokens": (
                self._summarizer_prompt_tokens + self._summarizer_completion_tokens
            ),
            "task_instruction": self.task_instruction,
            "domain": self.domain,
            "solo_mode": self.solo_mode,
            "role": self.role,
        })
        return base_stats


# Register in memory model registry — both legacy + new aliases for parity
# with the SWE / WebArena adapters.
register_memory_model("tau2_summarizing", Tau2SummarizingMemory)
register_memory_model("tau2-summarizing", Tau2SummarizingMemory)
