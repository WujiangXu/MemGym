"""
Adaptive token-budget memory manager.

Keeps pinned task-critical messages and a recent raw tail. When the estimated
prompt token count exceeds a threshold, it summarizes the middle/history block
instead of relying on message-count thresholds.
"""

from typing import Any, Dict, List, Optional

from litellm import completion

from .base import BaseMemoryManager, FilteredContext, register_memory_model
from .naive_summarization import SUMMARIZATION_PROMPTS


SUMMARY_MARKER = "[Summary]"


def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    return ""


def _message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content", "")
        return "" if content is None else str(content)
    return str(msg)


def _has_tool_calls(msg: Any) -> bool:
    """Check if an assistant message contains tool_calls."""
    return (isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls"))


def _get_tool_call_ids(msg: Any) -> set:
    """Extract tool_call IDs from an assistant message."""
    if not _has_tool_calls(msg):
        return set()
    return {tc.get("id") or tc.get("function", {}).get("name", "")
            for tc in msg.get("tool_calls", [])
            if isinstance(tc, dict)}


def _get_tool_response_id(msg: Any) -> Optional[str]:
    """Get tool_call_id from a tool response message."""
    if isinstance(msg, dict) and msg.get("role") == "tool":
        return msg.get("tool_call_id")
    return None


def _fix_tool_call_pairing(
    pinned_indices: List[int],
    middle_indices: List[int],
    tail_indices: List[int],
    messages: List[Any],
) -> tuple:
    """Ensure tool_call/tool_response pairs are never split across groups.

    Fixes three boundary cases:
    1. Pinned assistant with tool_calls → tool responses in middle must be pinned too
    2. Tail assistant with tool_calls → tool responses in middle must move to tail
    3. Middle assistant with tool_calls → if responses in tail, move call to tail
    """
    # Collect tool_call IDs from each group
    pinned_call_ids = set()
    for idx in pinned_indices:
        pinned_call_ids.update(_get_tool_call_ids(messages[idx]))

    tail_call_ids = set()
    for idx in tail_indices:
        tail_call_ids.update(_get_tool_call_ids(messages[idx]))

    # 1. Fix pinned/middle boundary: move tool responses from middle → pinned
    move_to_pinned = []
    for idx in middle_indices:
        resp_id = _get_tool_response_id(messages[idx])
        if resp_id and resp_id in pinned_call_ids:
            move_to_pinned.append(idx)

    # 2. Fix tail/middle boundary: move tool responses from middle → tail
    move_to_tail = []
    for idx in middle_indices:
        if idx in move_to_pinned:
            continue
        resp_id = _get_tool_response_id(messages[idx])
        if resp_id and resp_id in tail_call_ids:
            move_to_tail.append(idx)

    # 3. Fix middle assistant → tail: if any response is in tail, move call to tail
    for idx in middle_indices:
        if idx in move_to_pinned or idx in move_to_tail:
            continue
        if _has_tool_calls(messages[idx]):
            call_ids = _get_tool_call_ids(messages[idx])
            for tidx in tail_indices:
                resp_id = _get_tool_response_id(messages[tidx])
                if resp_id and resp_id in call_ids:
                    move_to_tail.append(idx)
                    # Also move ALL responses for this call to tail
                    for midx in middle_indices:
                        if midx == idx:
                            continue
                        r = _get_tool_response_id(messages[midx])
                        if r and r in call_ids and midx not in move_to_tail:
                            move_to_tail.append(midx)
                    break

    # Apply moves
    move_to_pinned_set = set(move_to_pinned)
    move_to_tail_set = set(move_to_tail)
    new_pinned = sorted(set(pinned_indices) | move_to_pinned_set)
    new_middle = [i for i in middle_indices if i not in move_to_pinned_set and i not in move_to_tail_set]
    new_tail = sorted(set(tail_indices) | move_to_tail_set)

    return new_pinned, new_middle, new_tail


class AdaptiveTokenBudgetMemory(BaseMemoryManager):
    """
    Token-aware memory strategy that summarizes the middle/history block.

    Output shape:
      [pinned prefix/critical msgs] + [summary?] + [recent raw tail]
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_recent: int = 8,
        keep_first: int = 2,
        preserve_first_user: bool = True,
        summarization_model: str = "gpt-4o-mini",
        env_type: str = "swe",
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        config.update(
            {
                "env_type": env_type,
                "keep_recent": keep_recent,
                "keep_first": keep_first,
                "preserve_first_user": preserve_first_user,
            }
        )
        super().__init__(max_tokens=max_tokens, tokenizer_name=tokenizer_name, config=config)

        self.keep_recent = keep_recent
        self.keep_first = keep_first
        self.preserve_first_user = preserve_first_user
        self.summarization_model = summarization_model
        self.env_type = env_type

        self._summary: Optional[str] = None
        self._condensation_requested = False

        self._raw_tokens_before_history: List[int] = []
        self._raw_tokens_after_history: List[int] = []
        self._compression_history: List[float] = []
        self._summary_trigger_count = 0

    def request_condensation(self) -> None:
        """Force compaction on the next call, used after context-window errors."""
        self._condensation_requested = True

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        messages = list(original_context) + [current_observation]
        raw_tokens_before = self.count_tokens(messages)

        pinned_indices = self._pinned_indices(messages)
        tail_indices = self._tail_indices(messages, pinned_indices)
        middle_indices = [
            idx for idx in range(len(messages))
            if idx not in pinned_indices and idx not in tail_indices
        ]

        # Fix tool_call/tool_response pairing — ensure pairs aren't split
        pinned_indices, middle_indices, tail_indices = _fix_tool_call_pairing(
            pinned_indices, middle_indices, tail_indices, messages
        )

        needs_summary = raw_tokens_before > self.max_tokens or self._condensation_requested
        was_compacted = False
        condensation_event = None

        if needs_summary and middle_indices:
            middle_messages = [messages[idx] for idx in middle_indices]
            self._summary = self._summarize_middle(
                pinned_messages=[messages[idx] for idx in pinned_indices],
                middle_messages=middle_messages,
                recent_messages=[messages[idx] for idx in tail_indices],
            )
            self._compaction_count += 1
            self._summary_trigger_count += 1
            was_compacted = True
            condensation_event = {
                "summary_text": self._summary,
                "forgotten_count": len(middle_messages),
                "summarization_model": self.summarization_model,
                "strategy": "adaptive_token_budget",
            }

        self._condensation_requested = False

        content = self._build_output(messages, pinned_indices, tail_indices)
        raw_tokens_after = self.count_tokens(content)
        compression_ratio = raw_tokens_before / max(1, raw_tokens_after)

        self._update_token_tracking(raw_tokens_after)
        self._raw_tokens_before_history.append(raw_tokens_before)
        self._raw_tokens_after_history.append(raw_tokens_after)
        self._compression_history.append(compression_ratio)

        metadata_out = {
            "tokens": raw_tokens_after,
            "original_tokens": raw_tokens_before,
            "raw_tokens_before": raw_tokens_before,
            "raw_tokens_after": raw_tokens_after,
            "was_compacted": was_compacted,
            "compression_ratio": compression_ratio,
            "direct_messages": True,
            "strategy": "adaptive_token_budget",
            "has_summary": self._summary is not None,
            "num_summaries": self._compaction_count,
            "summarization_count": self._compaction_count,
            "summary_trigger_count": self._summary_trigger_count,
            "keep_first": self.keep_first,
            "keep_recent": self.keep_recent,
            "preserve_first_user": self.preserve_first_user,
            "pinned_messages": len(pinned_indices),
            "recent_messages": len(tail_indices),
            "summarized_messages": len(middle_indices) if was_compacted else 0,
            "step": self._step_count,
        }
        if condensation_event:
            metadata_out["condensation_event"] = condensation_event

        return FilteredContext(content=content, metadata=metadata_out)

    def _pinned_indices(self, messages: List[Any]) -> List[int]:
        pinned = list(range(min(self.keep_first, len(messages))))
        if self.preserve_first_user:
            for idx, msg in enumerate(messages):
                if _message_role(msg) == "user" and idx not in pinned:
                    pinned.append(idx)
                    break
        return sorted(set(pinned))

    def _tail_indices(self, messages: List[Any], pinned_indices: List[int]) -> List[int]:
        if self.keep_recent <= 0:
            return []

        pinned = set(pinned_indices)
        tail: List[int] = []
        for idx in range(len(messages) - 1, -1, -1):
            if idx in pinned:
                continue
            tail.append(idx)
            if len(tail) >= self.keep_recent:
                break
        return sorted(reversed(tail))

    def _build_output(
        self,
        messages: List[Any],
        pinned_indices: List[int],
        tail_indices: List[int],
    ) -> List[Any]:
        output: List[Any] = []
        pinned_set = set(pinned_indices)
        tail_set = set(tail_indices)

        for idx in pinned_indices:
            output.append(messages[idx])

        middle_exists = any(
            idx not in pinned_set and idx not in tail_set
            for idx in range(len(messages))
        )
        if self._summary and middle_exists:
            output.append({"role": "user", "content": f"{SUMMARY_MARKER} {self._summary}"})

        for idx in tail_indices:
            output.append(messages[idx])

        if not output:
            return messages
        return output

    def _summarize_middle(
        self,
        pinned_messages: List[Any],
        middle_messages: List[Any],
        recent_messages: List[Any],
    ) -> str:
        system_prompt = self._get_summarization_prompt()
        pinned_text = "\n".join(
            f"<PINNED role={_message_role(msg)}>\n{_message_content(msg)}\n</PINNED>"
            for msg in pinned_messages
        )
        middle_text = "\n".join(
            f"<HISTORY role={_message_role(msg)}>\n{_message_content(msg)}\n</HISTORY>"
            for msg in middle_messages
        )
        recent_text = "\n".join(
            f"<RECENT role={_message_role(msg)}>\n{_message_content(msg)}\n</RECENT>"
            for msg in recent_messages
        )
        previous_summary = self._summary or "(none)"

        user_prompt = f"""Summarize the history block for an agent that is nearing its token budget.

Pinned messages that remain verbatim:
{pinned_text or "(none)"}

Existing rolling summary:
{previous_summary}

Recent raw messages that remain verbatim:
{recent_text or "(none)"}

History block to compress:
{middle_text}

Requirements:
- Preserve technical decisions, constraints, blockers, file paths, and test results.
- Avoid repeating the pinned or recent raw messages unless needed for continuity.
- Preserve the user task/issue statement if relevant.
- Output only the updated summary text."""

        try:
            response = completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Warning: Adaptive summarization failed: {exc}")
            fallback = "\n".join(_message_content(msg)[:200] for msg in middle_messages[:5]).strip()
            return self._summary or fallback or "Summary unavailable due to error."

    def _get_summarization_prompt(self) -> str:
        return SUMMARIZATION_PROMPTS.get(self.env_type, SUMMARIZATION_PROMPTS["generic"])

    def reset(self) -> None:
        self._summary = None
        self._condensation_requested = False
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._raw_tokens_before_history.clear()
        self._raw_tokens_after_history.clear()
        self._compression_history.clear()
        self._summary_trigger_count = 0
        if hasattr(self, "_legacy_observations"):
            self._legacy_observations.clear()

    def get_stats(self) -> Dict[str, Any]:
        base_stats = super().get_stats()
        avg_before = (
            sum(self._raw_tokens_before_history) / len(self._raw_tokens_before_history)
            if self._raw_tokens_before_history else 0
        )
        avg_after = (
            sum(self._raw_tokens_after_history) / len(self._raw_tokens_after_history)
            if self._raw_tokens_after_history else 0
        )
        avg_ratio = (
            sum(self._compression_history) / len(self._compression_history)
            if self._compression_history else 1.0
        )
        base_stats.update(
            {
                "summarization_model": self.summarization_model,
                "env_type": self.env_type,
                "keep_recent": self.keep_recent,
                "keep_first": self.keep_first,
                "preserve_first_user": self.preserve_first_user,
                "has_summary": self._summary is not None,
                "summary_preview": (
                    self._summary[:100] + "..."
                    if self._summary and len(self._summary) > 100
                    else self._summary
                ),
                "raw_tokens_before": (
                    self._raw_tokens_before_history[-1]
                    if self._raw_tokens_before_history else 0
                ),
                "raw_tokens_after": (
                    self._raw_tokens_after_history[-1]
                    if self._raw_tokens_after_history else 0
                ),
                "avg_raw_tokens_before": avg_before,
                "avg_raw_tokens_after": avg_after,
                "peak_raw_tokens_before": max(self._raw_tokens_before_history) if self._raw_tokens_before_history else 0,
                "peak_raw_tokens_after": max(self._raw_tokens_after_history) if self._raw_tokens_after_history else 0,
                "avg_compression_ratio": avg_ratio,
                "num_summaries": self._compaction_count,
                "summary_trigger_count": self._summary_trigger_count,
            }
        )
        return base_stats


AdaptiveTokenBudgetMemoryModel = AdaptiveTokenBudgetMemory

register_memory_model("adaptive_token_budget", AdaptiveTokenBudgetMemory)
