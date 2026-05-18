"""GRPO update step for the (see paper) online summarizer RL loop.

Reusable across Way A (agent-RL) and Way B (memory-RL):

- Way A: advantages flow through the agent policy's own forward pass;
  the summaries are also drawn from the same model, so its logprobs
  serve both roles.
- Way B: the agent is frozen; only the summarizer policy receives the
  gradient. Same advantage math, applied to the summarizer's logprobs.

Three primitives, each independently testable:

1. `prepare_advantages(outcomes)` — group rollouts by instance_id,
   compute reward, group-mean / group-std normalize. Returns one
   `RolloutAdvantage` per (rollout, summary call); each advantage is
   tagged with its (system, user, completion) so the update loop has
   everything it needs.
2. `snapshot_old_logprobs(advantages, backend)` — one-time forward pass
   to capture log p_old(completion | prompt). The PPO ratio in step (3)
   uses these as the fixed denominator for K update epochs over the
   same rollout batch. Without this snapshot, the ratio drifts and the
   clip becomes meaningless.
3. `grpo_step(advantages, old_logprobs, backend, ref_backend, ...)` —
   one optimizer step. Loss = -clip(ratio, 1-ε, 1+ε) · A + β · KL(π||π_ref).

The R_compression guardrail (plan §H.3.2) is a method on
`RolloutAdvantage` rather than baked into the loss — that way the
training script can log skip-compression rate as a histogram without
re-deriving it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from memgym.memory.summarizer_backend import LocalHFSummarizerBackend
from memgym.training.rl.env_reward import EpisodeResult, SummaryCall

logger = logging.getLogger(__name__)


# Plan §H.3.2: skip-compression hack penalty when ratio<0.2.
COMPRESSION_FLOOR = 0.2
R_COMPRESSION_PENALTY = -1.0


@dataclass
class RolloutAdvantage:
    """One (rollout, summary_call) pair with its advantage attached.

    The same advantage is replicated across every summary in a rollout
    (REINFORCE-style credit assignment per plan §H.3.2). When a rollout
    has multiple summaries, each one gets the same `advantage` value
    but its own `summary_call` payload — the optimizer sees them as
    independent gradient contributions.

    `instance_id` and `rollout_idx` survive into here so the training
    log can group and re-derive metrics post-hoc.
    """

    instance_id: str
    rollout_idx: int
    summary_call: SummaryCall
    reward: float
    advantage: float
    metadata: dict = field(default_factory=dict)


def _episode_reward(
    result: EpisodeResult,
    use_compression_guardrail: bool,
) -> float:
    """R_combined per plan §H.3.2 — terminal + compression guardrail."""
    R = float(result.R_terminal)
    if use_compression_guardrail:
        ratio = float(result.metadata.get("max_compression_ratio") or 1.0)
        if ratio < COMPRESSION_FLOOR:
            R = R + 0.1 * R_COMPRESSION_PENALTY
    return R


def prepare_advantages(
    outcomes: Sequence[EpisodeResult],
    rollout_indices: Optional[Sequence[int]] = None,
    use_compression_guardrail: bool = True,
    advantage_eps: float = 1e-6,
) -> List[RolloutAdvantage]:
    """Group rollouts by instance_id, normalize advantages within group.

    `rollout_indices` is parallel to `outcomes` and tags each rollout
    with its position in the GRPO group (0..N-1). When omitted, we use
    a within-instance counter — fine for the EC2 smoke, but for real
    runs the caller should pass the indices it submitted to the
    rollout pool so every rollout in the same group sees the same
    `n_rollouts` for advantage normalization.

    Skips rollouts with zero summaries (`times_filtered=0`) — no
    summary means no gradient target. Plan §H.3'' explicitly excludes
    these.
    """
    if rollout_indices is None:
        # Reconstruct group indices defensively; preserves stable order.
        seen: dict = {}
        rollout_indices_list: List[int] = []
        for r in outcomes:
            seen[r.instance_id] = seen.get(r.instance_id, -1) + 1
            rollout_indices_list.append(seen[r.instance_id])
        rollout_indices = rollout_indices_list

    if len(rollout_indices) != len(outcomes):
        raise ValueError(
            f"rollout_indices length {len(rollout_indices)} does not match "
            f"outcomes length {len(outcomes)}"
        )

    # Group rewards by instance_id for normalization.
    rewards_by_instance: dict = {}
    rewards_per_outcome: List[float] = []
    for r in outcomes:
        R = _episode_reward(r, use_compression_guardrail)
        rewards_per_outcome.append(R)
        rewards_by_instance.setdefault(r.instance_id, []).append(R)

    # Per-group mean / std → group-relative advantage.
    group_stats: dict = {}
    for iid, group_rewards in rewards_by_instance.items():
        n = len(group_rewards)
        if n == 0:
            continue
        mean = sum(group_rewards) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in group_rewards) / n
            std = max(var ** 0.5, advantage_eps)
        else:
            std = 1.0  # n=1 group: no relative signal; advantage falls to 0
        group_stats[iid] = (mean, std)

    advantages: List[RolloutAdvantage] = []
    for r, rollout_idx, R in zip(outcomes, rollout_indices, rewards_per_outcome):
        if not r.episode_summaries:
            continue
        mean, std = group_stats[r.instance_id]
        adv = (R - mean) / std
        for call in r.episode_summaries:
            advantages.append(
                RolloutAdvantage(
                    instance_id=r.instance_id,
                    rollout_idx=rollout_idx,
                    summary_call=call,
                    reward=R,
                    advantage=adv,
                    metadata={
                        "episode_resolved": r.episode_resolved,
                        "hit_context_limit": r.hit_context_limit,
                    },
                )
            )
    return advantages


def snapshot_old_logprobs(
    advantages: Sequence[RolloutAdvantage],
    backend: LocalHFSummarizerBackend,
) -> List[Any]:
    """One-time forward pass to capture log p_old(completion | prompt).

    Returns one detached 1-D tensor per advantage (parallel list). The
    GRPO step uses these as the fixed denominator for the PPO ratio
    over K update epochs.

    Detaching is critical — these tensors must NOT carry grad. We're
    just memoizing the policy's view at rollout time.
    """
    import torch

    old_logprobs: List[Any] = []
    backend.model.eval()
    with torch.no_grad():
        for adv in advantages:
            call = adv.summary_call
            # mode="rollout" routes the forward through the per-rank
            # non-sharded inference copy. We're in the snapshot phase
            # *between* GRPO updates: per-rank rollout call counts have
            # already diverged here, so any FSDP collective on
            # `self.model` would NCCL-time-out.
            lp = backend.compute_logprobs_for_completion(
                call.system, call.user, call.completion, mode="rollout",
            )
            old_logprobs.append(lp.detach())
    return old_logprobs


def grpo_step(
    advantages: Sequence[RolloutAdvantage],
    old_logprobs: Sequence[Any],
    backend: LocalHFSummarizerBackend,
    ref_backend: Optional[LocalHFSummarizerBackend],
    optimizer: Any,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.01,
    grad_accum_steps: int = 1,
) -> dict:
    """One PPO-clipped policy gradient step with KL anchor.

    Returns a dict of scalar metrics for logging:
        loss_total, loss_pg, loss_kl, mean_ratio, frac_clipped,
        mean_advantage, num_updates.

    `ref_backend` is the SFT-frozen model used for the KL anchor. Pass
    `None` to disable the KL term (plain PPO). At our scale we set
    `kl_coef=0.01` with decay; friends may schedule.

    Gradient accumulation: caller passes `grad_accum_steps>1` to spread
    one optimizer step across multiple `grpo_step` calls. The function
    scales loss by `1/grad_accum_steps` and skips `optimizer.step()` —
    the caller invokes the optimizer once per accumulation window.
    """
    import torch

    if len(advantages) != len(old_logprobs):
        raise ValueError(
            f"advantages length {len(advantages)} does not match "
            f"old_logprobs length {len(old_logprobs)}"
        )
    if not advantages:
        return {"loss_total": 0.0, "num_updates": 0}

    backend.model.train()
    if ref_backend is not None:
        ref_backend.model.eval()

    # Park `_inference_model` on CPU during the GRPO step. It's a full-base
    # deepcopy (~16 GiB bf16 for 8B) used only for rollout-side `summarize()`
    # calls, so it sits idle here. On A100-40GB the FSDP all-gather of the
    # trainable params + activations + Adam state collides with the resident
    # inference copy and trips an OOM that surfaces as
    # `ProcessGroupNCCL.cpp:3754 unhandled cuda error`. V4l died here.
    _inference_device = None
    if getattr(backend, "_inference_model", None) is not None and backend._inference_model is not backend.model:
        _inference_device = next(backend._inference_model.parameters()).device
        if _inference_device.type == "cuda":
            backend._inference_model = backend._inference_model.to("cpu")
            torch.cuda.empty_cache()

    # Pre-filter so we know n_calls before the loop. The loss normalization
    # divides by n_calls, and we need the divisor up front because we now do
    # per-call backward (see comment below). Drop empty-completion entries
    # since they contribute zero gradient.
    valid_pairs = [
        (adv, old_lp)
        for adv, old_lp in zip(advantages, old_logprobs)
        if adv.summary_call.completion
    ]
    n_calls = len(valid_pairs)
    if n_calls == 0:
        # Restore `_inference_model` to its original device on the empty-batch
        # early-return path. Otherwise the next rollout would call `summarize()`
        # against a CPU-resident inference copy.
        if _inference_device is not None and _inference_device.type == "cuda" and getattr(backend, "_inference_model", None) is not None:
            backend._inference_model = backend._inference_model.to(_inference_device)
        return {"loss_total": 0.0, "num_updates": 0}

    sum_loss_pg = 0.0
    sum_loss_kl = 0.0
    sum_ratio = 0.0
    n_clipped = 0
    n_updates = 0
    norm = float(n_calls) * float(max(1, grad_accum_steps))

    optimizer.zero_grad()

    # Per-call forward + backward. The previous version accumulated forward
    # graphs across all calls and ran one backward at the end, so peak
    # activation memory was N × per-call (~30 GiB at 8B-bf16 with 4-8 calls).
    # Per-call backward frees each call's activations immediately and reaches
    # the same gradient as the single big backward when each call's loss is
    # divided by n_calls (and grad_accum_steps for the outer accumulator).
    for adv, old_lp in valid_pairs:
        call = adv.summary_call

        new_lp = backend.compute_logprobs_for_completion(
            call.system, call.user, call.completion
        )
        if new_lp.numel() == 0:
            continue

        # Sum over completion tokens — sequence-level log-prob is the
        # standard GRPO formulation. Mean would over-weight short
        # completions; sum keeps per-token gradient magnitude consistent.
        new_lp_sum = new_lp.sum()
        old_lp_sum = old_lp.sum().detach().to(new_lp_sum.device)
        ratio = torch.exp(new_lp_sum - old_lp_sum)

        adv_t = torch.tensor(adv.advantage, device=ratio.device, dtype=ratio.dtype)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t
        # Negative because we minimize the loss but maximize the
        # advantage-weighted log-prob. PPO's pessimistic min selects
        # the more conservative of the two terms.
        loss_pg = -torch.min(unclipped, clipped)

        if ref_backend is not None:
            with torch.no_grad():
                ref_lp = ref_backend.compute_logprobs_for_completion(
                    call.system, call.user, call.completion
                )
            # Sequence-level KL approximation: log p_new - log p_ref.
            # This is the "k1" estimator; for our scale it's adequate
            # and avoids the variance of the unbiased "k3" form.
            kl = (new_lp.sum() - ref_lp.sum().to(new_lp.device)).abs()
            loss_kl = kl_coef * kl
            call_loss = (loss_pg + loss_kl) / norm
            sum_loss_kl += float(loss_kl.detach().item())
        else:
            call_loss = loss_pg / norm

        call_loss.backward()

        sum_loss_pg += float(loss_pg.detach().item())
        sum_ratio += float(ratio.detach().item())
        if abs(float(ratio.detach().item()) - 1.0) > clip_ratio:
            n_clipped += 1
        n_updates += 1

    if n_updates == 0:
        if _inference_device is not None and _inference_device.type == "cuda" and getattr(backend, "_inference_model", None) is not None:
            backend._inference_model = backend._inference_model.to(_inference_device)
        return {"loss_total": 0.0, "num_updates": 0}

    loss_total_value = (sum_loss_pg + sum_loss_kl) / norm

    if grad_accum_steps == 1:
        optimizer.step()
        # Post-step sync: gather full FSDP weights and copy into the per-rank
        # inference copy. This is the one place in the loop where every rank
        # is guaranteed to be at the same point — perfect for the all-gather
        # collective inside `sync_inference_from_fsdp`. After the sync, the
        # next rollout sees the just-optimized weights.
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.barrier()
        except Exception:  # noqa: BLE001 — torch.distributed missing is fine
            pass
        # Restore `_inference_model` to GPU before sync. `sync_inference_from_fsdp`
        # writes the gathered FSDP weights into it via `load_state_dict`, which
        # would silently leave it on CPU (and slow down the next rollout's
        # `summarize()` calls) if we skip this.
        if _inference_device is not None and _inference_device.type == "cuda":
            backend._inference_model = backend._inference_model.to(_inference_device)
        backend.sync_inference_from_fsdp()

    return {
        "loss_total": loss_total_value,
        "loss_pg": sum_loss_pg / max(1, n_updates),
        "loss_kl": sum_loss_kl / max(1, n_updates),
        "mean_ratio": sum_ratio / n_updates,
        "frac_clipped": n_clipped / n_updates,
        "mean_advantage": sum(a.advantage for a in advantages) / max(1, len(advantages)),
        "num_updates": n_updates,
    }


__all__ = [
    "COMPRESSION_FLOOR",
    "R_COMPRESSION_PENALTY",
    "RolloutAdvantage",
    "prepare_advantages",
    "snapshot_old_logprobs",
    "grpo_step",
]
