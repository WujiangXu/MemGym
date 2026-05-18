"""Env-reward wrapper for the (see paper) RL summarizer loop.

The RL loop rolls out episodes of `SWEMemoryEnv` with the 8B policy
plugged in as the summarizer. GRPO needs, per rollout:

    (prompt, completion) pairs for each summary the policy emitted
    +
    one scalar terminal reward (did the episode resolve).

Rather than hacking per-manager metadata to surface summary text, we
decorate the pluggable `summarizer_backend` ((see paper) plumbing
in `memgym/memory/summarizer_backend.py`). Every `summarize()` call is
logged at the source — the exact (system, user) strings the policy saw
— so GRPO can recompute log-probs for the advantage without needing to
replay the whole episode.

Credit assignment: REINFORCE-style attribution of the terminal reward
to every summary event in the episode (plan §H.3.2). GRPO then
normalizes across rollouts per `instance_id` group.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memgym.memory.summarizer_backend import SummarizerBackend


# Mirrors `eval_memory_ab.py:CONTEXT_LIMIT_KEYWORDS` so EpisodeResult can flag
# context-window crashes without depending on the eval script.
_CONTEXT_LIMIT_KEYWORDS = (
    "context length",
    "context window",
    "prompt is too long",
    "maximum context",
    "please reduce the length",
    "token limit",
    "too many tokens",
)


def _is_context_limit_error(err: Optional[str]) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(k in low for k in _CONTEXT_LIMIT_KEYWORDS)


@dataclass
class SummaryCall:
    """One summarize() invocation captured during a rollout."""

    system: str
    user: str
    max_tokens: int
    completion: str


class RecordingSummarizerBackend:
    """Decorator that records every `.summarize()` call on an inner backend.

    Implements the `SummarizerBackend` Protocol by duck-typing: no base
    class needed, the runtime-checkable Protocol accepts any object with
    a `.summarize(...)` method that returns a string.
    """

    def __init__(self, inner: SummarizerBackend):
        self.inner = inner
        self.calls: List[SummaryCall] = []

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        text = self.inner.summarize(system, user, max_tokens=max_tokens)
        self.calls.append(
            SummaryCall(
                system=system,
                user=user,
                max_tokens=max_tokens,
                completion=text,
            )
        )
        return text


@dataclass
class EpisodeResult:
    """One rollout's payload for GRPO.

    `episode_resolved` and `hit_context_limit` are surfaced as first-class
    fields so the GRPO outer loop and tier-2 eval can branch on them
    without re-parsing strings each rollout. `R_terminal` remains the
    canonical reward; `episode_resolved` is just `R_terminal > 0.5` (the
    SWE eval emits 1.0 / 0.0).
    """

    instance_id: str
    R_terminal: float
    episode_summaries: List[SummaryCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_resolved: bool = False
    hit_context_limit: bool = False


def run_episode_with_reward(
    env,
    summarizer_backend: SummarizerBackend,
    verbose: bool = False,
) -> EpisodeResult:
    """Run one SWEMemoryEnv episode, capture summaries + terminal reward.

    The caller is responsible for env construction and reset. We hot-swap
    the recording decorator into the memory manager's `summarizer_backend`
    slot, then invoke `run_episode()` and unpack the result.

    Raises TypeError if the env's memory manager does not expose a
    `summarizer_backend` attribute — that's the contract the RL loop
    relies on (currently only `NaiveSummarizationMemory`; H.3 does not
    support the legacy hard-coded-litellm managers).
    """
    mm = getattr(env, "_memory_model", None)
    if mm is None or not hasattr(mm, "summarizer_backend"):
        raise TypeError(
            f"env memory manager {type(mm).__name__!r} has no "
            "`summarizer_backend` attribute; H.3 RL requires the "
            "pluggable-backend refactor (see summarizer_backend.py)."
        )

    recorder = RecordingSummarizerBackend(summarizer_backend)
    mm.summarizer_backend = recorder

    result = env.run_episode(verbose=verbose)
    R_terminal = float(result.get("reward", 0.0))

    # Surface the signals GRPO + mid-loop eval consume. We keep this
    # slim — the caller can grab the full `result` via `env` if needed;
    # this dict is the stable contract.
    metadata: Dict[str, Any] = {
        "status": result.get("status"),
        "patch_size": result.get("patch_size", 0),
        "token_stats": result.get("token_stats", {}),
        "wrapper_stats": result.get("wrapper_stats", {}),
        "num_summaries": len(recorder.calls),
        "error": result.get("error"),
    }
    wrapper_stats = result.get("wrapper_stats", {}) or {}
    agent_stats = wrapper_stats.get("agent", {}) if isinstance(wrapper_stats, dict) else {}
    if isinstance(agent_stats, dict):
        metadata["avg_compression_ratio"] = agent_stats.get("avg_compression_ratio", 1.0)
        metadata["max_compression_ratio"] = agent_stats.get("max_compression_ratio", 1.0)
        metadata["max_original_tokens"] = agent_stats.get("max_original_tokens", 0)
        metadata["max_filtered_tokens"] = agent_stats.get("max_filtered_tokens", 0)

    return EpisodeResult(
        instance_id=getattr(env, "instance_id", "<unknown>"),
        R_terminal=R_terminal,
        episode_summaries=recorder.calls,
        metadata=metadata,
        episode_resolved=R_terminal > 0.5,
        hit_context_limit=_is_context_limit_error(result.get("error")),
    )


__all__ = [
    "SummaryCall",
    "RecordingSummarizerBackend",
    "EpisodeResult",
    "run_episode_with_reward",
]
