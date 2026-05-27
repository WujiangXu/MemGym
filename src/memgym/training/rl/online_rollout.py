"""Async batch rollout pool for the GRPO loop.

The GRPO outer loop wants `N rollouts × M instances` `EpisodeResult`s
per batch. Each rollout is one full SWEMemoryEnv episode (Docker
container + agent calls + summarizer calls). On EC2 we run at low
concurrency (4 default) for code-correctness validation; friends raise
this for scale.

Concurrency model:
- ThreadPoolExecutor with `max_workers=concurrency`.
- Each worker calls a user-supplied `env_factory(instance, rollout_idx)`
  that returns a freshly-built SWEMemoryEnv (the env's own __init__
  starts the Docker container; close() tears it down).
- Each worker also receives a `summarizer_factory(rollout_idx)` so the
  GRPO loop can pin a per-rollout sampling temperature / seed without
  threading those through the env construction call.

Why threads, not asyncio: SWEMemoryEnv runs synchronous Docker / litellm
code with no async surface. Wrapping it in asyncio buys nothing and
duplicates the threadpool inside `litellm.completion`. Threads also
play nicely with the GIL here because the bottleneck is I/O wait on
the agent endpoint, not Python compute.

Why a single `_LOCAL_HF_LOCK`: when the summarizer backend is a
`LocalHFSummarizerBackend` (Way B's policy under training), multiple
threads would race for the same GPU. We serialize summarize() calls
through a module-level lock; the agent API calls (the bulk of episode
wall-clock) parallelize freely. For Way A or API-summarizer setups
the lock is a no-op (zero contention).
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

from memgym.memory.summarizer_backend import LocalHFSummarizerBackend, SummarizerBackend
from memgym.training.rl.env_reward import (
    EpisodeResult,
    RecordingSummarizerBackend,
    run_episode_with_reward,
)

logger = logging.getLogger(__name__)


# Shared across the process. Only contended when a LocalHFSummarizerBackend
# is in use; otherwise lock acquire is sub-microsecond.
_LOCAL_HF_LOCK = threading.Lock()


def _wrap_for_gpu_safety(backend: SummarizerBackend) -> SummarizerBackend:
    """Serialize summarize() calls when the backend uses a shared GPU.

    Returns a thin proxy that takes the module lock around each call.
    Plain API-backed backends (litellm) skip this — they're I/O-bound
    and parallelize fine.
    """
    if not isinstance(backend, LocalHFSummarizerBackend):
        return backend

    class _LockedBackend:
        def __init__(self, inner: SummarizerBackend):
            self.inner = inner

        def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
            with _LOCAL_HF_LOCK:
                return self.inner.summarize(system, user, max_tokens=max_tokens)

    return _LockedBackend(backend)


@dataclass
class RolloutRequest:
    """One rollout to schedule.

    `rollout_idx` is the position within the GRPO group (0..N-1) for an
    instance — used by the summarizer_factory to pin a per-rollout
    sampling temperature so the group has variance.
    """

    instance: dict
    rollout_idx: int


@dataclass
class RolloutOutcome:
    """`EpisodeResult` plus the originating request metadata.

    Carrying `instance_id` + `rollout_idx` separately from the
    EpisodeResult means the caller can recover the GRPO grouping even
    if the env failed mid-rollout and we synthesized a zero-reward
    `EpisodeResult` (`error` field set).
    """

    instance_id: str
    rollout_idx: int
    result: Optional[EpisodeResult]
    error: Optional[str] = None


def _run_one(
    request: RolloutRequest,
    env_factory: Callable[[dict, int], Any],
    summarizer_factory: Callable[[int], SummarizerBackend],
    verbose: bool,
) -> RolloutOutcome:
    iid = request.instance.get("instance_id", "<unknown>")
    env = None
    try:
        env = env_factory(request.instance, request.rollout_idx)
        backend = _wrap_for_gpu_safety(summarizer_factory(request.rollout_idx))
        result = run_episode_with_reward(env, backend, verbose=verbose)
        return RolloutOutcome(
            instance_id=iid,
            rollout_idx=request.rollout_idx,
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the whole batch
        logger.exception("rollout failed: instance=%s rollout=%d", iid, request.rollout_idx)
        return RolloutOutcome(
            instance_id=iid,
            rollout_idx=request.rollout_idx,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.warning("env.close() failed for instance=%s", iid, exc_info=True)


def run_batch(
    requests: Iterable[RolloutRequest],
    env_factory: Callable[[dict, int], Any],
    summarizer_factory: Callable[[int], SummarizerBackend],
    concurrency: int = 4,
    verbose: bool = False,
) -> List[RolloutOutcome]:
    """Execute a batch of rollouts in parallel.

    Returns one `RolloutOutcome` per request, in completion order. The
    GRPO outer loop re-groups by `instance_id` to compute group-relative
    advantages.

    `concurrency` should match the bottleneck of the slowest backend
    in use:
    - GPT-4o-mini / Bedrock Sonnet API agents → 4–50, limited by Docker
      container slots and provider rate limits.
    - Self-hosted vLLM agent → 1–8, limited by vLLM batch size.
    - LocalHFSummarizerBackend in the loop → effectively 1 for the
      summarize-call critical section, but agent calls still
      parallelize because the lock is summarize-only.
    """
    requests_list = list(requests)
    if not requests_list:
        return []

    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    outcomes: List[RolloutOutcome] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_one, req, env_factory, summarizer_factory, verbose)
            for req in requests_list
        ]
        for fut in as_completed(futures):
            outcomes.append(fut.result())
    return outcomes


__all__ = [
    "RolloutRequest",
    "RolloutOutcome",
    "run_batch",
]
