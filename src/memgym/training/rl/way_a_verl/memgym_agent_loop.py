"""Self-summarizing tool agent loop for Way A on verl 0.7.1.

Subclasses ``ToolAgentLoop``. When ``len(agent_data.messages)`` exceeds
``multi_turn.summarize_max_messages``, the loop drops the middle of the
conversation, runs the SAME trainable policy with a summarization prompt to
produce a ``[Summary]`` assistant message, and re-renders the prompt so
summary tokens stay in the trainable region (``response_mask=1``).

Mirrors ``src/memgym/memory/llm_summarizing.py`` — the trajectory-collection
condenser. Same trigger (message count, not token count: ``max_token_size``
is None in trajectory collection per ``swe_bench/env.py:1200-1240``), same
head/tail split (``head + [Summary] + tail`` with
``target_size = max_messages * target_ratio``,
``events_from_tail = target_size - keep_first - 1``). Only difference:
instead of calling ``litellm`` against an external model, we call
``self.server_manager.generate`` against the Qwen3-8B policy under training.
That makes summarization itself a learnable behavior — GRPO's terminal
reward propagates through summary tokens via ``response_mask=1``.

SFT-distribution parity: the entire 14K-trajectory SFT corpus the H.2
summarizer warm-start is trained on was produced under message-count
triggering with ``max_size=100``. RL-time triggers must match or the
policy gets a different invocation distribution at train vs SFT time.

Registered as ``memgym_tool``. Select it in YAML via
``actor_rollout_ref.rollout.agent.agent_name: memgym_tool``.
"""

from __future__ import annotations

import logging
from typing import Any, List

from verl.experimental.agent_loop.agent_loop import register  # type: ignore
from verl.experimental.agent_loop.tool_agent_loop import (  # type: ignore
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.utils.profiler import simple_timer  # type: ignore

logger = logging.getLogger(__name__)


# Lifted verbatim from ``src/memgym/memory/llm_summarizing.py`` so the
# RL self-summarize and the trajectory-collection condenser share format.
# Friend-cluster checkpoints generalize across both paths because the
# prompt the policy learns to answer is the same.
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

SKIP: Tracking irrelevant details for the current task type"""

SUMMARY_MARKER = "[Summary]"
_MAX_EVENT_CHARS = 20_000


def _safe_int(env_value: Any, default: int, var_name: str) -> int:
    """Audit M2: tolerate misconfigured env vars without crashing the actor.

    A user export like ``MEMGYM_SUMMARIZE_MAX_MESSAGES=auto`` would otherwise
    raise ``ValueError`` at ``__init__`` and kill the rollout silently — the
    Ray actor crashes outside any try/except the trainer can see.
    """
    if env_value is None or env_value == "":
        return default
    try:
        return int(env_value)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not an int; falling back to %d",
            var_name, env_value, default,
        )
        return default


def _safe_float(env_value: Any, default: float, var_name: str) -> float:
    if env_value is None or env_value == "":
        return default
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a float; falling back to %f",
            var_name, env_value, default,
        )
        return default


def _truncate(text: str, max_chars: int = _MAX_EVENT_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... truncated ...]\n\n" + text[-half:]


def _is_summary_message(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    content = msg.get("content", "")
    return isinstance(content, str) and SUMMARY_MARKER in content


def _stringify_content(content: Any) -> str:
    """Flatten chat-template content (which may be a list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text", "")))
            else:
                out.append(str(part))
        return " ".join(out)
    return str(content)


@register("memgym_tool")
class MemGymToolAgentLoop(ToolAgentLoop):
    """``ToolAgentLoop`` + self-summarize over the same trainable policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # verl 0.7.1's MultiTurnConfig is a strict dataclass and rejects
        # unknown kwargs (TypeError("unexpected keyword 'summarize_enable'"))
        # when Hydra calls ``omega_conf_to_dataclass(cfg.rollout)``. We can't
        # stash the summarize knobs in the YAML under multi_turn, so train.py
        # sets them in env vars before launching ppo_trainer.main(); we read
        # them here. Defaults below mirror trajectory-collection production
        # (OpenHands LLMSummarizingCondenser): max_size=100, ratio=0.75,
        # keep_first=1 — the same trigger the H.2 SFT corpus was produced
        # under (see llm_summarizing.py:290-330).
        import os as _os
        self.summarize_enable = _os.environ.get(
            "MEMGYM_SUMMARIZE_ENABLE", "1"
        ).lower() not in {"0", "false", "no"}
        self.summarize_max_messages = _safe_int(
            _os.environ.get("MEMGYM_SUMMARIZE_MAX_MESSAGES"),
            100, "MEMGYM_SUMMARIZE_MAX_MESSAGES",
        )
        self.summarize_target_ratio = _safe_float(
            _os.environ.get("MEMGYM_SUMMARIZE_TARGET_RATIO"),
            0.75, "MEMGYM_SUMMARIZE_TARGET_RATIO",
        )
        self.summarize_keep_first = _safe_int(
            _os.environ.get("MEMGYM_SUMMARIZE_KEEP_FIRST"),
            1, "MEMGYM_SUMMARIZE_KEEP_FIRST",
        )
        self.summarize_max_tokens = _safe_int(
            _os.environ.get("MEMGYM_SUMMARIZE_MAX_TOKENS"),
            1024, "MEMGYM_SUMMARIZE_MAX_TOKENS",
        )

    # ------------------------------------------------------------------
    # State-machine extension points
    # ------------------------------------------------------------------

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        state = await super()._handle_pending_state(agent_data, sampling_params)
        agent_data.extra_fields.setdefault("_mgym_summarize_count", 0)
        return state

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        # Pre-generation: run a self-summarize pass if context is too long.
        # Doing it here (rather than after PROCESSING_TOOLS) means the next
        # ``server_manager.generate`` call uses the compacted prompt directly.
        if self._should_summarize(agent_data):
            try:
                await self._maybe_summarize(agent_data, sampling_params)
            except Exception:  # noqa: BLE001
                # Compaction is best-effort: if rendering or generation fails
                # for one trajectory, keep going on the un-compacted prompt
                # rather than aborting the whole episode.
                logger.exception("self-summarize step failed; continuing un-compacted")

        pre_msg_count = len(agent_data.messages)
        next_state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )

        # The base class only appends assistant messages to ``messages`` when
        # ``interaction_config_file`` is set. We need a faithful list for the
        # *next* compaction's re-render, so append it ourselves here.
        if (
            not self.interaction_config_file
            and len(agent_data.messages) == pre_msg_count
            and agent_data.response_ids
        ):
            assistant_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    agent_data.response_ids, skip_special_tokens=True
                ),
            )
            agent_data.messages.append(
                {"role": "assistant", "content": assistant_text}
            )

        return next_state

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _should_summarize(self, agent_data: AgentData) -> bool:
        if not self.summarize_enable:
            return False
        msg_count = len(agent_data.messages)
        if msg_count <= self.summarize_max_messages:
            return False
        # Mirror llm_summarizing.py:322-330 exactly: the post-compaction layout
        # is head + [Summary] + tail, where target_size = max * ratio and the
        # tail length absorbs whatever's left after head + the summary slot.
        target_size = int(self.summarize_max_messages * self.summarize_target_ratio)
        events_from_tail = max(
            1, target_size - self.summarize_keep_first - 1
        )
        # Need at least one message strictly between head and tail to drop.
        return msg_count > self.summarize_keep_first + events_from_tail

    async def _maybe_summarize(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> None:
        msgs = agent_data.messages
        head_count = self.summarize_keep_first
        target_size = int(self.summarize_max_messages * self.summarize_target_ratio)
        events_from_tail = max(1, target_size - head_count - 1)

        head = msgs[:head_count]
        tail = msgs[-events_from_tail:] if events_from_tail else []
        middle = msgs[head_count : len(msgs) - events_from_tail]
        if not middle:
            return

        previous_summary = ""
        if _is_summary_message(middle[0]):
            previous_summary = (
                _stringify_content(middle[0].get("content", ""))
                .replace(SUMMARY_MARKER, "")
                .strip()
            )
            middle = middle[1:]
        if not middle:
            return

        # Audit H11: match the trajectory-collection condenser exactly —
        # ``llm_summarizing.py`` calls litellm with
        # ``[{"role":"system","content":SUMMARIZATION_PROMPT},
        #   {"role":"user","content":<events>}]``. The H.2 summarizer was SFT'd
        # under that two-message shape; folding both into one ``user`` turn
        # produces a different chat-template trace (Qwen3's template wraps
        # each role in ``<|im_start|>``/``<|im_end|>``) and shifts the policy
        # off-distribution at RL time.
        user_text = self._build_summarize_prompt(middle, previous_summary)
        summarize_prompt_ids = await self.apply_chat_template(
            [
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tools=None,
        )

        summary_sampling = dict(sampling_params)
        summary_sampling["max_new_tokens"] = self.summarize_max_tokens
        # The summary call's logprobs aren't used by the GRPO update; verl
        # recomputes old_logprobs at training time. Drop the request so vLLM
        # doesn't waste cycles returning them.
        summary_sampling.pop("logprobs", None)

        idx = int(agent_data.extra_fields.get("_mgym_summarize_count", 0))
        agent_data.extra_fields["_mgym_summarize_count"] = idx + 1

        with simple_timer("memgym_summarize", agent_data.metrics):
            summary_output = await self.server_manager.generate(
                request_id=f"{agent_data.request_id}_sum_{idx}",
                prompt_ids=summarize_prompt_ids,
                sampling_params=summary_sampling,
                image_data=None,
                video_data=None,
            )

        summary_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                summary_output.token_ids, skip_special_tokens=True
            ),
        )
        summary_text = (summary_text or "").strip()
        if not summary_text:
            # Empty summary — skip this round; threshold will trip again next turn.
            return

        summary_msg = {
            "role": "assistant",
            "content": f"{SUMMARY_MARKER} {summary_text}",
        }
        agent_data.messages = list(head) + [summary_msg] + list(tail)

        await self._rebuild_token_state(agent_data)

        agent_data.metrics["memgym/summarize_count"] = (
            agent_data.metrics.get("memgym/summarize_count", 0) + 1
        )
        agent_data.metrics["memgym/last_summary_input_msgs"] = len(middle)
        agent_data.metrics["memgym/last_summary_tokens"] = len(summary_output.token_ids)

    def _build_summarize_prompt(
        self, forgotten: List[dict], previous_summary: str
    ) -> str:
        # Audit H11: ``SUMMARIZATION_PROMPT`` now ships as a separate system
        # message (see ``_maybe_summarize``). This builder only emits the
        # user-side payload — previous summary + event blocks + closing line.
        chunks: list[str] = [
            f"<PREVIOUS SUMMARY>\n{_truncate(previous_summary)}\n</PREVIOUS SUMMARY>",
            "",
        ]
        for i, msg in enumerate(forgotten):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = _stringify_content(msg.get("content", ""))
            else:
                role = "user"
                content = str(msg)
            chunks.append(
                f"<EVENT id={i} role={role}>\n{_truncate(content)}\n</EVENT>"
            )
        chunks.append("")
        chunks.append("Now summarize the events using the rules above.")
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Token-state rebuild (re-render messages → prompt_ids + response_mask)
    # ------------------------------------------------------------------

    async def _rebuild_token_state(self, agent_data: AgentData) -> None:
        """Re-render ``agent_data.messages`` and recompute ``response_mask``.

        Strategy: incrementally render ``messages[:k]`` with
        ``add_generation_prompt=False`` to recover per-message token spans
        (Jinja templates are strictly prefix-extending in that mode), then do
        one final render with ``add_generation_prompt=True`` to get the
        prompt that the next ``server_manager.generate`` call will consume.

        Mask assignment: ``assistant`` messages — including the synthetic
        ``[Summary]`` message — get ``mask=1``. Everything else (system, user,
        tool, trailing generation-prompt header) gets ``mask=0``. The head
        region (the original system+user_task tokens, including their gen
        prompt) is excluded from ``response_mask`` entirely so the partition
        ``len(prompt_ids) == head_len + len(response_mask)`` holds — same
        invariant the base class maintains.
        """
        msgs = agent_data.messages
        head_count = self.summarize_keep_first
        tools = self.tool_schemas if self.tool_schemas else None
        chat_kwargs = dict(self.apply_chat_template_kwargs)

        def _render(message_slice: list[dict], gen: bool) -> list[int]:
            return list(
                self.tokenizer.apply_chat_template(
                    message_slice,
                    tools=tools,
                    add_generation_prompt=gen,
                    tokenize=True,
                    **chat_kwargs,
                )
            )

        # Compute head_len from a head-only render with gen=True. This matches
        # how the base class first builds ``prompt_ids`` in ``_handle_pending_state``.
        head_only_render = await self.loop.run_in_executor(
            None, lambda: _render(msgs[:head_count], gen=True)
        )
        head_len = len(head_only_render)

        # Incremental renders with gen=False to span-tag each message.
        spans: list[tuple[int, int, str]] = []
        prev_ids: list[int] = []
        for k in range(1, len(msgs) + 1):
            rendered = await self.loop.run_in_executor(
                None, lambda k=k: _render(msgs[:k], gen=False)
            )
            # Standard chat templates extend strictly. If a non-standard
            # template diverges mid-stream, clamp: treat the divergence point
            # as the end of the previous message's span.
            i = 0
            limit = min(len(prev_ids), len(rendered))
            while i < limit and prev_ids[i] == rendered[i]:
                i += 1
            if i < len(prev_ids) and spans:
                start, _end, role = spans[-1]
                spans[-1] = (start, i, role)
            role = (
                msgs[k - 1].get("role", "user")
                if isinstance(msgs[k - 1], dict)
                else "user"
            )
            spans.append((max(i, 0), len(rendered), role))
            prev_ids = rendered

        # Final render with gen=True — this is what the next generate() call sees.
        final_ids = await self.loop.run_in_executor(
            None, lambda: _render(msgs, gen=True)
        )
        if len(final_ids) > len(prev_ids):
            spans.append((len(prev_ids), len(final_ids), "_gen_prompt"))

        # Build response_mask covering final_ids[head_len:].
        new_mask: list[int] = []
        cursor = head_len
        for start, end, role in spans:
            if end <= head_len:
                continue
            seg_start = max(start, head_len)
            seg_len = end - seg_start
            if seg_len <= 0:
                continue
            # Pad gap (defensive — shouldn't trigger for well-behaved templates).
            if seg_start > cursor:
                new_mask.extend([0] * (seg_start - cursor))
                cursor = seg_start
            mask_val = 1 if role == "assistant" else 0
            new_mask.extend([mask_val] * seg_len)
            cursor = end

        # Trim/pad to exactly len(final_ids) - head_len in case of edge cases.
        target = len(final_ids) - head_len
        if len(new_mask) > target:
            new_mask = new_mask[:target]
        elif len(new_mask) < target:
            new_mask.extend([0] * (target - len(new_mask)))

        agent_data.prompt_ids = list(final_ids)
        agent_data.response_mask = new_mask
        # Rollout-time logprobs no longer align with the new prompt_ids; verl
        # recomputes old_logprobs at training time via
        # ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``,
        # so clearing here is safe.
        agent_data.response_logprobs = []
        # Audit H10: ``response_ids`` still holds the just-generated assistant
        # text that we just folded into the summary. The post-base-class
        # fallback in ``_handle_generating_state`` (lines 185-198) appends
        # ``messages.append({"role":"assistant","content":decode(response_ids)})``
        # iff ``response_ids`` is non-empty AND no message was appended this
        # turn. After compaction, ``messages`` ends with the *new* tail so we
        # would otherwise re-append the pre-compaction assistant text as a
        # duplicate of what the summary already absorbed. Clear it.
        agent_data.response_ids = []


__all__ = ["MemGymToolAgentLoop", "SUMMARIZATION_PROMPT", "SUMMARY_MARKER"]
