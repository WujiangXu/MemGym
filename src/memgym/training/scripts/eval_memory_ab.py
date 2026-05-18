"""(see paper) memory A/B smoke: memory-OFF vs memory-ON + gate.

Runs each coding-env instance (SWE-Smith / SWE-Gym / SWE-Bench) twice
with the **same frozen** agent LLM:

  * Arm A  memory-OFF: `PassThroughMemory`, no gate. Agent sees raw
                       history; context-limit overflow surfaces as
                       episode error (a legitimate data point).
  * Arm B  memory-ON : `NaiveSummarizationMemory` + `WorldModelGate`
                       at t*=0.78 (the class-weighted CE checkpoint champion contract).

The research claim this harness tests is the (see paper) thesis: a
frozen, strong reasoning model **with** the MemGym memory module
beats the **same frozen model without memory**. If the claim holds
we ship the memory contribution; no RL / no summarizer-training is
required to justify it.

The sibling script `eval_gym_gate.py` is gate-vs-no-gate (both arms
run SummarizationMemory); this script is memory-vs-no-memory (one
arm has no compaction at all). The ship criteria therefore differ —
`eval_gym_gate.py` reports reject-rate / FA-HARM / p95 latency, while
this script reports resolve-rate delta + token cost + context-limit
failure rate as the primary axes.

Writes `<output>/eval_memory_ab.json` with per-instance per-arm
results, aggregates, and the plan-G ship verdict.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "SWE-bench/SWE-smith"
DEFAULT_AGENT_LLM = "openai/qwen3-14b-thinking"
DEFAULT_MAX_TOKENS = 4000

ARM_A = "memory-off"
ARM_B = "memory-on"
ARM_C = "memory-on-no-gate"

CONTEXT_LIMIT_KEYWORDS = (
    "context length",
    "context window",
    "prompt is too long",
    "maximum context",
    "please reduce the length",
    "token limit",
    "too many tokens",
)


@dataclass
class InstanceResult:
    """Per-instance outcome in one arm."""

    instance_id: str
    arm: str
    repo: str
    status: str
    reward: float
    patch_size: int
    elapsed_seconds: float
    num_compactions: int
    max_context_tokens: int  # raw pre-filter size — what the memory module receives
    max_effective_context_tokens: int  # post-filter size — what the LLM actually consumes
    avg_context_tokens: float
    hit_context_limit: bool
    gate_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _load_gate(checkpoint: Path, threshold: float, base_model: str):
    """Import lazily so pure-Arm-A runs do not require torch."""
    from memgym.training.models.world_model_gate import WorldModelGate

    return WorldModelGate(
        checkpoint=str(checkpoint),
        threshold=threshold,
        base_model=base_model,
    )


def _is_context_limit_error(err: Optional[str]) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(k in low for k in CONTEXT_LIMIT_KEYWORDS)


def _build_memory_model(args: argparse.Namespace):
    """Construct the memory manager implied by `--memory-strategy` + `--summarizer-backend`.

    Two orthogonal axes:

    * `--memory-strategy` selects the *compaction algorithm* (how/when to fold
      old turns into a single message): `naive` (default, threshold + full
      re-summarize), `structured` (OpenHands 17-field function-call schema),
      `llm` (OpenHands rolling-summary default), `sliding` (fixed window +
      incremental rolling summary).
    * `--summarizer-backend` is only meaningful for `naive`, which accepts a
      pluggable `SummarizerBackend`: `litellm` (legacy (see paper)/G.2b path),
      `local-hf` ((see paper) SFT smoke-eval), `llmlingua2` (H.4' literature
      baseline — token pruning, not generative).

    The non-naive strategies call `litellm.completion` internally; they do NOT
    accept a pluggable backend and are therefore incompatible with
    `--summarizer-backend {local-hf,llmlingua2}`. We raise rather than silently
    ignoring the backend flag — a user who sets `--memory-strategy structured
    --summarizer-backend local-hf` expects their SFT checkpoint to run, and
    would get the base model instead.
    """
    from memgym.memory import (
        LLMSummarizingMemory,
        NaiveSummarizationMemory,
        SlidingWindowSummaryMemory,
        StructuredSummaryMemory,
    )

    strategy = getattr(args, "memory_strategy", "naive")
    backend_kind = getattr(args, "summarizer_backend", "litellm")

    if strategy != "naive" and backend_kind != "litellm":
        raise ValueError(
            f"--memory-strategy {strategy} uses litellm.completion internally and "
            f"does not accept a pluggable summarizer backend; got "
            f"--summarizer-backend {backend_kind}. Use --memory-strategy naive "
            f"for local-hf / llmlingua2 backends."
        )

    summarization_model = args.summarization_model or args.agent_llm

    if strategy == "structured":
        return StructuredSummaryMemory(
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
            max_token_size=args.max_tokens,
        )

    if strategy == "llm":
        return LLMSummarizingMemory(
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
            max_token_size=args.max_tokens,
        )

    if strategy == "sliding":
        return SlidingWindowSummaryMemory(
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
            env_type="swe",
        )

    common = dict(
        max_tokens=args.max_tokens,
        env_type="swe",
    )
    if backend_kind == "local-hf":
        from memgym.memory.summarizer_backend import LocalHFSummarizerBackend

        if not args.summarization_model:
            raise ValueError(
                "--summarizer-backend local-hf requires --summarization-model "
                "to be a path or HF id for the checkpoint to load."
            )
        hf_backend = LocalHFSummarizerBackend(
            model_name_or_path=args.summarization_model,
            torch_dtype=args.summarizer_dtype,
            max_prompt_tokens=args.summarizer_max_prompt_tokens,
        )
        return NaiveSummarizationMemory(
            summarization_model=args.summarization_model,
            summarizer_backend=hf_backend,
            **common,
        )

    if backend_kind == "llmlingua2":
        # H.4' literature baseline. LLMLingua-2 is token-pruning, not
        # generative — `summarization_model` is unused, but we still
        # pass a label so get_stats() reports something meaningful.
        from memgym.memory.llmlingua_backend import LLMLingua2SummarizerBackend

        lingua_backend = LLMLingua2SummarizerBackend(
            model_name=args.llmlingua2_model,
            rate=args.llmlingua2_rate,
            device_map=args.llmlingua2_device,
        )
        return NaiveSummarizationMemory(
            summarization_model=f"llmlingua2@rate={args.llmlingua2_rate}",
            summarizer_backend=lingua_backend,
            **common,
        )

    return NaiveSummarizationMemory(
        summarization_model=summarization_model,
        **common,
    )


def _run_one(
    inst: Dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    gate,
) -> InstanceResult:
    """Run one instance in one arm. Never re-raises — records error on the result."""
    from memgym.envs import SWEMemoryEnv
    from memgym.memory import NaiveSummarizationMemory, PassThroughMemoryModel

    iid = inst["instance_id"]
    repo = inst.get("repo", "")
    t0 = datetime.now()
    env = None
    try:
        if arm == ARM_A:
            memory_model = PassThroughMemoryModel(max_tokens=args.max_tokens)
            memory_gate = None
        elif arm == ARM_B:
            memory_model = _build_memory_model(args)
            memory_gate = gate
        elif arm == ARM_C:
            memory_model = _build_memory_model(args)
            memory_gate = None
        else:
            raise ValueError(f"unknown arm: {arm}")

        # Qwen3 thinking-mode sampling (model card recipe). top_k / min_p are
        # NOT OpenAI-chat-completions standard params — with drop_params=true
        # (set by swebench*.yaml) litellm silently drops them on the openai/*
        # route, so they never reach vLLM. Route them through extra_body,
        # which the OpenAI SDK forwards verbatim to vLLM's JSON payload.
        # Real OpenAI validates extra_body strictly and 400s on unknown keys,
        # so only attach extras when an OPENAI_API_BASE override is present
        # (i.e. we're talking to vLLM, not api.openai.com).
        agent_model_kwargs = {
            "max_tokens": args.agent_max_tokens,
            "temperature": args.agent_temperature,
            "top_p": args.agent_top_p,
        }
        if os.environ.get("OPENAI_API_BASE"):
            agent_model_kwargs["extra_body"] = {
                "top_k": args.agent_top_k,
                "min_p": args.agent_min_p,
            }
        env = SWEMemoryEnv(
            dataset=args.dataset,
            split=args.split,
            instance_id=iid,
            agent_llm=args.agent_llm,
            agent_llm_args={"model_kwargs": agent_model_kwargs},
            memory_model=memory_model,
            use_context_management=True,
            use_docker=args.env_type in ("docker", "swerex_docker"),
            environment_class=args.env_type,
            docker_image_dir=args.docker_image_dir,
            docker_image_source=args.docker_image_source,
            max_steps=args.step_limit,
            agent_type="mini-swe",
            skip_eval=args.skip_eval,
            memory_gate=memory_gate,
        )
        result = env.run_episode(verbose=args.verbose)
        patch = result.get("patch") or ""
        wrapper_stats = (result.get("wrapper_stats") or {}).get("agent") or {}
        gate_events = wrapper_stats.get("gate_events", []) or [] if arm == ARM_B else []
        token_stats = result.get("token_stats") or {}
        err = result.get("error")
        # LLM-context high-water. Track two distinct quantities:
        # - raw: max pre-filter size the memory module saw. Grows unbounded
        #   as the episode runs; same in both arms when memory is off.
        # - effective: max post-filter size the LLM actually consumed.
        #   In memory-off this equals raw (no filter). In memory-on this
        #   is the compacted payload — the metric the "cheaper at matched
        #   rate" ship criterion depends on.
        # token_stats.max_tokens is a fallback for older runs (pre
        # wrapper_stats tracking) when the env step() path was used.
        raw_ctx = int(
            wrapper_stats.get("max_original_tokens", 0)
            or token_stats.get("max_tokens", 0)
            or 0
        )
        eff_ctx = int(
            wrapper_stats.get("max_filtered_tokens", 0)
            or token_stats.get("max_tokens", 0)
            or 0
        )
        return InstanceResult(
            instance_id=iid,
            arm=arm,
            repo=repo,
            status=result.get("status", "Unknown"),
            reward=float(result.get("reward", 0) or 0),
            patch_size=len(patch),
            elapsed_seconds=(datetime.now() - t0).total_seconds(),
            num_compactions=int(wrapper_stats.get("times_filtered", 0) or 0),
            max_context_tokens=raw_ctx,
            max_effective_context_tokens=eff_ctx,
            avg_context_tokens=float(token_stats.get("avg_tokens", 0.0) or 0.0),
            hit_context_limit=_is_context_limit_error(err),
            gate_events=gate_events,
            error=err,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("instance %s failed in arm=%s", iid, arm)
        msg = f"{type(exc).__name__}: {exc}"
        return InstanceResult(
            instance_id=iid,
            arm=arm,
            repo=repo,
            status="Error",
            reward=0.0,
            patch_size=0,
            elapsed_seconds=(datetime.now() - t0).total_seconds(),
            num_compactions=0,
            max_context_tokens=0,
            max_effective_context_tokens=0,
            avg_context_tokens=0.0,
            hit_context_limit=_is_context_limit_error(msg),
            error=msg,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (p / 100) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _resolve_rate(results: List[InstanceResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.reward > 0) / len(results)


def _log_result(
    result: InstanceResult,
    arm: str,
    idx: int,
    total: int,
    running: List[InstanceResult],
) -> None:
    """Emit a per-instance result line so running resolve-rate is visible in stdout.

    Complements `_write_partial` (which is a disk-only snapshot) by making the
    signal grep-able from a streamed log — the monitor on a remote run then
    surfaces resolve status live instead of only at terminal state.
    """
    logger.info(
        "[%s %d/%d] %s resolved=%s reward=%.2f ctx=%d/%d rate=%.2f%%",
        arm, idx, total, result.instance_id,
        result.reward > 0, result.reward,
        result.max_effective_context_tokens, result.max_context_tokens,
        100.0 * _resolve_rate(running),
    )


def _ctx_fail_rate(results: List[InstanceResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit_context_limit) / len(results)


def _token_summary(results: List[InstanceResult]) -> Dict[str, float]:
    """Raw (pre-filter) context tokens — what the memory module receives."""
    vals = [r.max_context_tokens for r in results if r.max_context_tokens > 0]
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": float(statistics.median(vals)),
        "p95": float(_percentile(list(map(float, vals)), 95)),
        "mean": float(statistics.mean(vals)),
    }


def _effective_token_summary(results: List[InstanceResult]) -> Dict[str, float]:
    """Effective (post-filter) context tokens — what the LLM actually consumes.
    This is the load-bearing metric for 'cheaper at matched rate' — a
    memory module reduces this but not necessarily raw context."""
    vals = [
        r.max_effective_context_tokens
        for r in results
        if r.max_effective_context_tokens > 0
    ]
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": float(statistics.median(vals)),
        "p95": float(_percentile(list(map(float, vals)), 95)),
        "mean": float(statistics.mean(vals)),
    }


def _wallclock_summary(results: List[InstanceResult]) -> Dict[str, float]:
    vals = [r.elapsed_seconds for r in results]
    if not vals:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(statistics.median(vals)),
        "p95": float(_percentile(vals, 95)),
    }


def _gate_summary_b(arm_b: List[InstanceResult]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for r in arm_b:
        events.extend(r.gate_events)
    if not events:
        return {
            "gate_events_total": 0,
            "reject_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_prob_safe": 0.0,
        }
    latencies = [ev["latency_ms"]["total"] for ev in events]
    prob_safe = [ev["prob_safe"] for ev in events]
    rejects = sum(1 for ev in events if not ev["decision"])
    return {
        "gate_events_total": len(events),
        "reject_rate": rejects / len(events),
        "p50_latency_ms": float(statistics.median(latencies)),
        "p95_latency_ms": float(_percentile(latencies, 95)),
        "avg_prob_safe": float(statistics.mean(prob_safe)),
    }


def _per_repo_breakdown(
    arm_a: List[InstanceResult],
    arm_b: List[InstanceResult],
) -> Dict[str, Dict[str, Any]]:
    by_repo: Dict[str, Dict[str, List[InstanceResult]]] = defaultdict(
        lambda: {ARM_A: [], ARM_B: []}
    )
    for r in arm_a:
        by_repo[r.repo or "unknown"][ARM_A].append(r)
    for r in arm_b:
        by_repo[r.repo or "unknown"][ARM_B].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for repo, groups in by_repo.items():
        a, b = groups[ARM_A], groups[ARM_B]
        out[repo] = {
            "n_a": len(a),
            "n_b": len(b),
            "resolve_a": _resolve_rate(a),
            "resolve_b": _resolve_rate(b),
            "delta": _resolve_rate(b) - _resolve_rate(a),
        }
    return out


def _summarize(
    arm_a: List[InstanceResult],
    arm_b: List[InstanceResult],
    args: argparse.Namespace,
    arm_c: Optional[List[InstanceResult]] = None,
) -> Dict[str, Any]:
    arm_c = arm_c or []
    resolve_a = _resolve_rate(arm_a)
    resolve_b = _resolve_rate(arm_b)
    resolve_c = _resolve_rate(arm_c)
    delta = resolve_b - resolve_a
    delta_c = resolve_c - resolve_a
    tok_a = _token_summary(arm_a)
    tok_b = _token_summary(arm_b)
    tok_c = _token_summary(arm_c)
    tok_a_eff = _effective_token_summary(arm_a)
    tok_b_eff = _effective_token_summary(arm_b)
    tok_c_eff = _effective_token_summary(arm_c)

    # Ship criteria for the memory A/B ablation ((see paper) § Staged rollout).
    # Either a meaningful resolve-rate lift, or matched rate at lower token
    # cost, counts as validation. "Token cost" = what the LLM consumes
    # (effective / post-filter), not what the memory module received.
    # The pre-filter metric grows unbounded in both arms as episodes
    # lengthen, so it cannot be the axis on which we compare cost.
    delta_non_negative = delta >= 0.0
    meaningful_lift = delta >= 0.05
    matched_rate = abs(delta) < 0.02
    token_relief = (
        tok_a_eff["p50"] > 0
        and tok_b_eff["p50"] > 0
        and tok_b_eff["p50"] <= 0.80 * tok_a_eff["p50"]
    )
    cheaper_at_matched = matched_rate and token_relief
    validated = delta_non_negative and (meaningful_lift or cheaper_at_matched)

    # An N<20 smoke is too small to call; still print numbers.
    n_matched = min(len(arm_a), len(arm_b))
    verdict = "validated" if validated else (
        "insufficient_n" if n_matched < 20 else "not_validated"
    )

    # Memory contribution isolated from gate: A vs C delta.
    # If gate is unusable (e.g. OOD), this is the load-bearing claim.
    delta_c_non_negative = delta_c >= 0.0
    meaningful_lift_c = delta_c >= 0.05
    matched_rate_c = abs(delta_c) < 0.02
    token_relief_c = (
        tok_a_eff["p50"] > 0
        and tok_c_eff["p50"] > 0
        and tok_c_eff["p50"] <= 0.80 * tok_a_eff["p50"]
    )
    cheaper_at_matched_c = matched_rate_c and token_relief_c
    validated_c = delta_c_non_negative and (meaningful_lift_c or cheaper_at_matched_c) if arm_c else False

    return {
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "docker_image_source": args.docker_image_source,
            "agent_llm": args.agent_llm,
            "summarization_model": args.summarization_model,
            "memory_strategy": getattr(args, "memory_strategy", "naive"),
            "summarizer_backend": getattr(args, "summarizer_backend", "litellm"),
            "llmlingua2_model": getattr(args, "llmlingua2_model", None)
                if getattr(args, "summarizer_backend", "") == "llmlingua2" else None,
            "llmlingua2_rate": getattr(args, "llmlingua2_rate", None)
                if getattr(args, "summarizer_backend", "") == "llmlingua2" else None,
            "max_tokens": args.max_tokens,
            "agent_max_tokens": args.agent_max_tokens,
            "agent_temperature": args.agent_temperature,
            "agent_top_p": args.agent_top_p,
            "agent_top_k": args.agent_top_k,
            "agent_min_p": args.agent_min_p,
            "n_instances_requested": args.n_instances,
            "n_instances_matched": n_matched,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "threshold": args.threshold,
            "base_model": args.base_model,
            "diverse_by_repo": bool(getattr(args, "diverse_by_repo", False)),
            "timestamp": datetime.now().isoformat(),
        },
        "resolve_rate": {
            "memory_off": resolve_a,
            "memory_on": resolve_b,
            "memory_on_no_gate": resolve_c,
            "delta_b_minus_a": delta,
            "delta_c_minus_a": delta_c,
        },
        "context_limit_fail_rate": {
            "memory_off": _ctx_fail_rate(arm_a),
            "memory_on": _ctx_fail_rate(arm_b),
            "memory_on_no_gate": _ctx_fail_rate(arm_c),
        },
        "max_context_tokens": {
            "memory_off": tok_a,
            "memory_on": tok_b,
            "memory_on_no_gate": tok_c,
            "_note": "raw pre-filter size; grows with episode length, not LLM cost",
        },
        "max_effective_context_tokens": {
            "memory_off": tok_a_eff,
            "memory_on": tok_b_eff,
            "memory_on_no_gate": tok_c_eff,
            "_note": "post-filter size = what the LLM actually consumes",
        },
        "wall_clock_seconds": {
            "memory_off": _wallclock_summary(arm_a),
            "memory_on": _wallclock_summary(arm_b),
            "memory_on_no_gate": _wallclock_summary(arm_c),
        },
        "compactions_memory_on": {
            "mean": float(statistics.mean([r.num_compactions for r in arm_b])) if arm_b else 0.0,
            "max": max((r.num_compactions for r in arm_b), default=0),
        },
        "compactions_memory_on_no_gate": {
            "mean": float(statistics.mean([r.num_compactions for r in arm_c])) if arm_c else 0.0,
            "max": max((r.num_compactions for r in arm_c), default=0),
        },
        "gate_metrics": _gate_summary_b(arm_b),
        "per_repo": _per_repo_breakdown(arm_a, arm_b),
        "ship_verdict": {
            "delta_non_negative": delta_non_negative,
            "meaningful_lift": meaningful_lift,
            "cheaper_at_matched_rate": cheaper_at_matched,
            "verdict": verdict,
        },
        "ship_verdict_no_gate": {
            "delta_non_negative": delta_c_non_negative,
            "meaningful_lift": meaningful_lift_c,
            "cheaper_at_matched_rate": cheaper_at_matched_c,
            "verdict": "validated" if validated_c else (
                "insufficient_n" if min(len(arm_a), len(arm_c)) < 20 else "not_validated"
            ),
            "_note": "memory-off vs memory-on-no-gate — isolates compression benefit from the gate-reject-loop confound",
        },
        "arm_memory_off_results": [r.__dict__ for r in arm_a],
        "arm_memory_on_results": [r.__dict__ for r in arm_b],
        "arm_memory_on_no_gate_results": [r.__dict__ for r in arm_c],
    }


def _write_partial(
    out_dir: Path,
    arm_a: List[InstanceResult],
    arm_b: List[InstanceResult],
    args: argparse.Namespace,
    arm_c: Optional[List[InstanceResult]] = None,
) -> None:
    """Incremental dump so a crashed long run still preserves progress."""
    arm_c = arm_c or []
    tmp_path = out_dir / "eval_memory_ab.partial.json"
    with tmp_path.open("w") as fh:
        json.dump(
            {
                "arm_memory_off_results": [r.__dict__ for r in arm_a],
                "arm_memory_on_results": [r.__dict__ for r in arm_b],
                "arm_memory_on_no_gate_results": [r.__dict__ for r in arm_c],
                "config": {
                    "dataset": args.dataset,
                    "agent_llm": args.agent_llm,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "threshold": args.threshold,
                },
            },
            fh, indent=2, default=str,
        )


def _filter_by_subcategory(
    instances: List[Dict[str, Any]],
    allowed: List[str],
) -> List[Dict[str, Any]]:
    """Keep only instances whose `instance_id` names one of `allowed` subcategories.

    SWE-Smith `instance_id` shape: `<repo>__<repo>.<sha>.<subcategory>__<hash>`
    (e.g. `bottlepy__bottle.a8dfef30.combine_file__ahufnyv5`). Subcategory is
    the segment between the last `.` and the `__<hash>` suffix. Matching by
    `prefix` lets "func_basic" capture "func_basic_modify" / "func_basic_rm"
    and related variants without enumerating every suffix.

    Defensive: if the filter returns zero, log a warning and return the
    original list. Callers should treat that as a config error, but we
    don't want a silent empty sampler downstream.
    """
    if not allowed:
        return instances
    pattern = re.compile(r"\.(" + "|".join(re.escape(a) for a in allowed) + r")[_.]")
    filtered = [inst for inst in instances if pattern.search(inst.get("instance_id", ""))]
    if not filtered:
        logger.warning(
            "--subcategory-include %s filtered out every instance; returning "
            "the unfiltered source (check subcategory names).",
            allowed,
        )
        return instances
    logger.info(
        "subcategory filter %s: %d -> %d instances",
        allowed, len(instances), len(filtered),
    )
    return filtered


def _select_diverse_by_repo(instances: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pick `n` instances spread across distinct repos.

    Walks the source order, takes the first instance from each new repo
    until we either hit `n` or exhaust new repos. If we exhaust repos
    first, fall back to round-robin within repos in source order.

    This is the fix for the G.2 sampling bug where --n-instances 20 on
    SWE-Smith picked 20 instances from the same repo because the first
    20 lexicographic IDs share a repo prefix. Diverse-by-repo is the
    *minimum* fix for an ablation result to generalize.
    """
    by_repo: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for inst in instances:
        repo = inst.get("repo", "") or ""
        by_repo.setdefault(repo, []).append(inst)

    selected: List[Dict[str, Any]] = []
    # Round 0: one instance per distinct repo, in source order.
    for repo, group in by_repo.items():
        if len(selected) >= n:
            break
        selected.append(group[0])
    # Round 1+: next instance per distinct repo, until n.
    offset = 1
    while len(selected) < n:
        progressed = False
        for repo, group in by_repo.items():
            if len(selected) >= n:
                break
            if offset < len(group):
                selected.append(group[offset])
                progressed = True
        if not progressed:
            break  # exhausted
        offset += 1
    return selected


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="WorldModelGate LoRA checkpoint (the class-weighted CE checkpoint champion). "
                             "Required only when the run touches the memory-on (gate) "
                             "arm; Arm-A-only and Arm-C-only runs ignore it.")
    parser.add_argument("--threshold", type=float, default=0.78,
                        help="Gate threshold (the class-weighted CE checkpoint constrained best = 0.78).")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base",
                        help="Classifier base model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--agent-llm", default=DEFAULT_AGENT_LLM,
                        help="LLM used for BOTH arms. (see paper) assumes a self-hosted vLLM "
                             "endpoint; set --agent-llm openai/<served-name> and export "
                             "OPENAI_API_BASE=http://<host>:8000/v1 (dummy OPENAI_API_KEY ok).")
    parser.add_argument("--summarization-model", default=None,
                        help="Summarizer model. Interpretation depends on "
                             "--summarizer-backend: for 'litellm' this is a "
                             "litellm model id (defaults to --agent-llm); for "
                             "'local-hf' this is an HF checkpoint path/id "
                             "loaded directly via transformers (required).")
    parser.add_argument("--summarizer-backend",
                        choices=["litellm", "local-hf", "llmlingua2"],
                        default="litellm",
                        help="Which SummarizerBackend to plug into NaiveSummarizationMemory. "
                             "'litellm' (default) matches (see paper)/G.2b behavior. "
                             "'local-hf' loads the (see paper) SFT checkpoint directly "
                             "via transformers. 'llmlingua2' is the H.4' literature "
                             "baseline: token-pruning via llmlingua.PromptCompressor. "
                             "Only meaningful when --memory-strategy=naive.")
    parser.add_argument("--memory-strategy",
                        choices=["naive", "structured", "llm", "sliding"],
                        default="naive",
                        help="Compaction algorithm. 'naive' (default) is the (see paper)/H "
                             "path: threshold-triggered full re-summarize, accepts any "
                             "--summarizer-backend. 'structured' is OpenHands' 17-field "
                             "StateSummary via function calling (H.4' baseline row). "
                             "'llm' is OpenHands' rolling-summary default. 'sliding' is a "
                             "fixed window + incrementally-updated rolling summary. "
                             "Non-naive strategies call litellm internally and REQUIRE "
                             "--summarizer-backend=litellm.")
    parser.add_argument("--summarizer-dtype",
                        choices=["bfloat16", "float16", "float32"],
                        default="bfloat16",
                        help="Torch dtype for the local-hf summarizer (ignored for litellm).")
    parser.add_argument("--summarizer-max-prompt-tokens",
                        type=int, default=6144,
                        help="Left-truncation budget for the local-hf summarizer input "
                             "(ignored for litellm).")
    # H.4' LLMLingua-2 literature baseline. Unused outside --summarizer-backend
    # llmlingua2; default rate 0.33 matches the H.1.1 datacard's SFT-pair
    # completion-to-input char-ratio p50 so the comparison is "matched
    # compression" per the H.4' ship criterion.
    parser.add_argument("--llmlingua2-model",
                        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                        help="HF id of the LLMLingua-2 compressor checkpoint.")
    parser.add_argument("--llmlingua2-rate", type=float, default=0.33,
                        help="Target fraction of tokens retained. Tune until "
                             "observed char-ratio p50 matches the SFT row's p50.")
    parser.add_argument("--llmlingua2-device",
                        choices=["cpu", "cuda"],
                        default="cpu",
                        help="Device for the LLMLingua-2 RoBERTa compressor. "
                             "CPU is fine (small model); use cuda only if "
                             "the agent GPU has spare memory.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Compaction trigger threshold (Arm B) / tracked max (Arm A).")
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument("--slice", default=None,
                        help="Override n-instances with an explicit 'start:end' slice.")
    parser.add_argument("--env-type",
                        choices=["docker", "swerex_docker", "singularity", "bubblewrap", "local"],
                        default="docker")
    parser.add_argument("--docker-image-dir",
                        default="${HOME}/swe-bench/dockers")
    parser.add_argument("--docker-image-source",
                        choices=["swebench", "epoch", "swe-gym", "swe-smith"],
                        default="swe-smith")
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--agent-max-tokens", type=int, default=32768,
                        help="Cap the agent LLM's max output tokens per turn "
                             "(model_kwargs.max_tokens). Reasoning models like "
                             "Qwen3.6-Thinking eat 3-5K tokens in the <think> "
                             "block alone; the Qwen3 model card recommends 32K+ "
                             "so there is headroom left for the actual bash "
                             "action after the reasoning trace finishes.")
    # Qwen3 thinking-mode sampling recipe (model card). DO NOT set temperature=0
    # in thinking mode — it causes repetition collapse per the official guide.
    parser.add_argument("--agent-temperature", type=float, default=0.6,
                        help="Sampling temperature for the agent LLM. 0.6 is the "
                             "Qwen3 thinking-mode recommendation. Never 0 in "
                             "thinking mode (causes repetition collapse).")
    parser.add_argument("--agent-top-p", type=float, default=0.95,
                        help="Top-p nucleus sampling for the agent LLM. 0.95 "
                             "matches the Qwen3 thinking-mode recipe.")
    parser.add_argument("--agent-top-k", type=int, default=20,
                        help="Top-k sampling for the agent LLM. 20 matches the "
                             "Qwen3 thinking-mode recipe.")
    parser.add_argument("--agent-min-p", type=float, default=0.0,
                        help="Min-p sampling for the agent LLM. 0 matches the "
                             "Qwen3 thinking-mode recipe.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip Docker patch evaluation (for non-Docker env-types).")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for eval_memory_ab.json.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--arm",
                        choices=["both", "all", "memory-off", "memory-on", "memory-on-no-gate"],
                        default="both",
                        help="'both' = A+B (historical default). 'all' = A+B+C for the "
                             "three-arm ablation that isolates memory compression (C) "
                             "from the gate-reject-loop (B vs C). Single-arm values "
                             "run just that arm for node-parallel execution.")
    parser.add_argument("--diverse-by-repo", action="store_true",
                        help="Sample --n-instances across distinct repos (one per repo "
                             "until repos exhausted, then round-robin), instead of the "
                             "default first-N-lexicographic. Required for A/B generalizability "
                             "on repo-grouped datasets like SWE-Smith.")
    parser.add_argument("--subcategory-include", default=None,
                        help="Comma-separated SWE-Smith subcategory prefixes to keep "
                             "(e.g. 'func_basic,lm_modify,lm_rewrite'). Matches the "
                             "segment between the last '.' and '__<hash>' in each "
                             "instance_id; prefix-matched so 'func_basic' also captures "
                             "'func_basic_modify' etc. Applied BEFORE --diverse-by-repo "
                             "or --slice sampling. Used by (see paper) to restrict the "
                             "capability-floor run to easy single-function bugs.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    instances = list(dataset)
    if args.subcategory_include:
        allowed = [s.strip() for s in args.subcategory_include.split(",") if s.strip()]
        instances = _filter_by_subcategory(instances, allowed)
    if args.slice:
        start, end = map(int, args.slice.split(":"))
        instances = instances[start:end]
    elif args.diverse_by_repo:
        instances = _select_diverse_by_repo(instances, args.n_instances)
    else:
        instances = instances[: args.n_instances]
    unique_repos = len({inst.get("repo", "") for inst in instances})
    logger.info("running %d instances across %d unique repo(s), arm=%s, dataset=%s/%s",
                len(instances), unique_repos, args.arm, args.dataset, args.split)

    wants_gate = args.arm in ("both", "all", "memory-on")
    gate = None
    if wants_gate:
        if args.checkpoint is None:
            parser.error(
                "--checkpoint is required when --arm includes the memory-on (gate) "
                "arm. Use --arm memory-off or --arm memory-on-no-gate to skip the gate."
            )
        logger.info("loading gate checkpoint from %s (threshold=%.3f)",
                    args.checkpoint, args.threshold)
        gate = _load_gate(args.checkpoint, args.threshold, args.base_model)

    arm_a: List[InstanceResult] = []
    arm_b: List[InstanceResult] = []
    arm_c: List[InstanceResult] = []

    run_a = args.arm in ("both", "all", "memory-off")
    run_b = args.arm in ("both", "all", "memory-on")
    run_c = args.arm in ("all", "memory-on-no-gate")

    if run_a:
        for i, inst in enumerate(instances):
            logger.info("[memory-off %d/%d] %s", i + 1, len(instances), inst["instance_id"])
            arm_a.append(_run_one(inst, ARM_A, args, None))
            _log_result(arm_a[-1], ARM_A, i + 1, len(instances), arm_a)
            _write_partial(args.output, arm_a, arm_b, args, arm_c)

    if run_b:
        for i, inst in enumerate(instances):
            logger.info("[memory-on %d/%d] %s", i + 1, len(instances), inst["instance_id"])
            arm_b.append(_run_one(inst, ARM_B, args, gate))
            _log_result(arm_b[-1], ARM_B, i + 1, len(instances), arm_b)
            _write_partial(args.output, arm_a, arm_b, args, arm_c)

    if run_c:
        for i, inst in enumerate(instances):
            logger.info("[memory-on-no-gate %d/%d] %s",
                        i + 1, len(instances), inst["instance_id"])
            arm_c.append(_run_one(inst, ARM_C, args, None))
            _log_result(arm_c[-1], ARM_C, i + 1, len(instances), arm_c)
            _write_partial(args.output, arm_a, arm_b, args, arm_c)

    summary = _summarize(arm_a, arm_b, args, arm_c)
    out_path = args.output / "eval_memory_ab.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(
        {
            "step": "eval_memory_ab",
            "out_path": str(out_path),
            "n_matched": summary["config"]["n_instances_matched"],
            "resolve_rate": summary["resolve_rate"],
            "context_limit_fail_rate": summary["context_limit_fail_rate"],
            "max_context_tokens": summary["max_context_tokens"],
            "max_effective_context_tokens": summary["max_effective_context_tokens"],
            "gate_metrics": summary["gate_metrics"],
            "ship_verdict": summary["ship_verdict"],
        },
        indent=2,
    ), file=sys.stderr)
    return 0 if summary["ship_verdict"]["verdict"] == "validated" else 1


if __name__ == "__main__":
    sys.exit(main())
