"""Reward function for verl 0.7.1's PPO trainer in Way A.

verl 0.7.1's ``custom_reward_function`` is invoked per rollout. Its
calling convention is positional+keyword, and the exact shape varies
across verl versions. We accept several call shapes and pick whichever
one fires:

1. ``compute_reward(rollout_outcome: dict)`` — verl 0.4.x legacy.
2. ``compute_reward(data_source, solution_str, ground_truth, extra_info=...,
   **kwargs)`` — the v0.7.1 ``RewardManager.compute_score`` signature.
3. ``compute_reward(prompt=..., response=..., extra_info=..., **kwargs)``
   — alt signature seen in some verl recipes.

In all cases the source of truth is the per-tool-call reward emitted by
``MemGymSWETool.execute`` when the model submits: that value is already
``1.0`` for resolved / ``0.0`` for unresolved, and it lands in
``AgentLoopOutput.extra_fields["tool_rewards"]`` (set by
``ToolAgentLoop.run``). When verl hands us ``extra_info`` carrying that
list, we sum it. When verl gives us only the trajectory dict (legacy),
we read ``episode_resolved`` directly.

Compression-floor penalty is kept for cross-comparability with Way B's
``grpo_loop._episode_reward`` even though Way A typically runs without
explicit memory compaction.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

COMPRESSION_FLOOR = 0.2
R_COMPRESSION_PENALTY = -1.0


def _terminal_from_tool_rewards(rewards: Optional[Iterable[float]]) -> float:
    """Sum tool-step rewards. ``MemGymSWETool.execute`` only emits a non-zero
    value on the submit step (``1.0`` resolved / ``0.0`` not), so the sum is
    equivalent to the terminal reward for the smoke. Once we surface dense
    intermediate rewards (e.g. per-compaction shaping), the sum is what the
    docstring of this module promises — keep them aligned."""
    total = 0.0
    if not rewards:
        return total
    for r in rewards:
        try:
            total += float(r)
        except (TypeError, ValueError):
            continue
    return total


def _compression_penalty(extra_info: Optional[Dict[str, Any]]) -> float:
    if not extra_info:
        return 0.0
    ratio = extra_info.get("max_compression_ratio")
    if ratio is None:
        meta = extra_info.get("metadata") or {}
        ratio = meta.get("max_compression_ratio")
    try:
        ratio_f = float(ratio) if ratio is not None else 1.0
    except (TypeError, ValueError):
        return 0.0
    return R_COMPRESSION_PENALTY if ratio_f < COMPRESSION_FLOOR else 0.0


def compute_reward(*args: Any, **kwargs: Any) -> float:
    """Score one rollout. Accepts every verl call shape we've seen."""
    extra_info: Optional[Dict[str, Any]] = kwargs.get("extra_info")
    rollout_outcome: Optional[Dict[str, Any]] = None

    # Shape 1: legacy single-dict.
    if len(args) == 1 and isinstance(args[0], dict):
        rollout_outcome = args[0]
    # Shape 2: v0.7.1 RewardManager — (data_source, solution_str, ground_truth, ...)
    elif len(args) >= 3 and extra_info is None:
        extra_info = kwargs.get("extra_info") or (args[3] if len(args) >= 4 else None)

    # Try the v0.7.1 path first: extra_info carries tool_rewards.
    if extra_info is not None:
        tool_rewards = extra_info.get("tool_rewards") if isinstance(extra_info, dict) else None
        if tool_rewards is not None:
            R = _terminal_from_tool_rewards(tool_rewards)
            return float(R + _compression_penalty(extra_info))
        if isinstance(extra_info, dict) and "episode_resolved" in extra_info:
            rollout_outcome = extra_info

    if rollout_outcome is not None:
        R = 1.0 if rollout_outcome.get("episode_resolved") else 0.0
        return float(R + _compression_penalty(rollout_outcome))

    # Last resort — verl gave us nothing recognizable. Don't crash the
    # trainer; log once and return zero so the rollout doesn't get an
    # accidental positive reward.
    logger.debug("compute_reward: unknown call shape args=%s kwargs=%s",
                 [type(a).__name__ for a in args], list(kwargs.keys()))
    return 0.0


__all__ = ["compute_reward", "COMPRESSION_FLOOR", "R_COMPRESSION_PENALTY"]
