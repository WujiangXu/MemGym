"""Online smoke: SWE-Bench with vs without WorldModelGate.

Runs a class-weighted CE checkpoint (default t*=0.78) on N SWE-Bench
instances in two modes:

  * `--no-gate`  control: MemoryAwareSWEAgent without memory_gate.
  * `--with-gate` treatment: same agent, same memory strategy, same LLM,
    but every compaction is scored by the gate. Rejections trigger
    harder condensation on the next step (fail-soft this step; the
    integration point is `memgym.gym.swe_bench.agent`).

Aggregates the online hard-ship metrics:

  * reject_rate           — gate rejections / gate events  ∈ (0.10, 0.40)
  * fa_harm_proxy         — instances where gate-on accepts compressions
                            that gate-off does not, AND gate-on regresses
                            task success vs gate-off  ≤ 0.03
  * p50 / p95 gate latency    — total end-to-end per-call  ≤ 200 ms (p95)
  * success_delta         — gate-on resolve_rate − gate-off resolve_rate  ≥ 0
  * context_overhead_tokens — extra tokens sent under gate-on (extra
                            agent call on full history). Reported, no
                            hard floor.

Writes `<output>/eval_gym_gate.json` with per-instance results, per-mode
aggregates, and pass/fail on every hard criterion.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_AGENT_LLM = "openai/gpt-5-mini"
DEFAULT_MEMORY_TYPE = "naive"
DEFAULT_MAX_TOKENS = 4000


@dataclass
class InstanceResult:
    """Per-instance outcome in one mode (gate on or off)."""

    instance_id: str
    mode: str  # "no-gate" or "with-gate"
    status: str
    reward: float
    patch_size: int
    elapsed_seconds: float
    num_compactions: int
    gate_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _load_gate(checkpoint: Path, threshold: float, base_model: str):
    """Import + load lazily so `--no-gate-only` runs don't need torch."""
    from memgym.training.models.world_model_gate import WorldModelGate

    return WorldModelGate(
        checkpoint=str(checkpoint),
        threshold=threshold,
        base_model=base_model,
    )


def _run_one(
    inst: Dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    gate,
) -> InstanceResult:
    """Run one SWE-Bench instance in one mode; never re-raise."""
    from memgym.envs import SWEMemoryEnv
    from memgym.memory import NaiveSummarizationMemory

    iid = inst["instance_id"]
    t0 = datetime.now()
    env = None
    try:
        memory_model = NaiveSummarizationMemory(
            max_tokens=args.max_tokens,
            env_type="swe",
            summarization_model=args.summarization_model or args.agent_llm,
        )
        env = SWEMemoryEnv(
            dataset=args.dataset,
            split=args.split,
            instance_id=iid,
            agent_llm=args.agent_llm,
            memory_model=memory_model,
            use_context_management=True,
            use_docker=args.env_type in ("docker", "swerex_docker"),
            environment_class=args.env_type,
            docker_image_dir=args.docker_image_dir,
            docker_image_source=args.docker_image_source,
            max_steps=args.step_limit,
            agent_type="mini-swe",
            skip_eval=args.skip_eval,
            memory_gate=gate if mode == "with-gate" else None,
        )
        result = env.run_episode(verbose=args.verbose)
        patch = result.get("patch") or ""
        wrapper_stats = (result.get("wrapper_stats") or {}).get("agent") or {}
        gate_events = wrapper_stats.get("gate_events", []) or []
        return InstanceResult(
            instance_id=iid,
            mode=mode,
            status=result.get("status", "Unknown"),
            reward=float(result.get("reward", 0) or 0),
            patch_size=len(patch),
            elapsed_seconds=(datetime.now() - t0).total_seconds(),
            num_compactions=int(wrapper_stats.get("times_filtered", 0) or 0),
            gate_events=gate_events,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("instance %s failed in mode=%s", iid, mode)
        return InstanceResult(
            instance_id=iid,
            mode=mode,
            status="Error",
            reward=0.0,
            patch_size=0,
            elapsed_seconds=(datetime.now() - t0).total_seconds(),
            num_compactions=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _aggregate_gate_metrics(results: List[InstanceResult]) -> Dict[str, Any]:
    """Roll per-step gate events into deployment-shaped metrics."""
    events: List[Dict[str, Any]] = []
    for r in results:
        events.extend(r.gate_events)
    if not events:
        return {
            "gate_events_total": 0,
            "reject_rate": 0.0,
            "fail_soft_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_prob_safe": 0.0,
        }
    latencies = [ev["latency_ms"]["total"] for ev in events]
    prob_safe = [ev["prob_safe"] for ev in events]
    rejects = sum(1 for ev in events if not ev["decision"])
    fail_softs = sum(1 for ev in events if ev.get("fail_soft"))
    return {
        "gate_events_total": len(events),
        "reject_rate": rejects / len(events),
        "fail_soft_rate": fail_softs / len(events),
        "p50_latency_ms": float(statistics.median(latencies)),
        "p95_latency_ms": float(_percentile(latencies, 95)),
        "avg_prob_safe": float(statistics.mean(prob_safe)),
    }


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (p / 100) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _fa_harm_proxy(
    no_gate: List[InstanceResult],
    with_gate: List[InstanceResult],
) -> Dict[str, Any]:
    """Trajectory-regression proxy for online false-accept rate on HARMFUL.

    A compression-instance is flagged `fa_harm_candidate` when gate-on
    accepted at least one compression that gate-off's parallel run did
    NOT reject OR errored before compaction. It becomes a
    `fa_harm_event` if gate-on subsequently regressed task success
    relative to gate-off on the same instance (resolved-by-gate-off,
    failed-by-gate-on).

    This is a proxy — not a label — because we have no ground-truth
    HARMFUL tag at deployment. The ship criterion uses the rate.
    """
    by_iid_off = {r.instance_id: r for r in no_gate}
    by_iid_on = {r.instance_id: r for r in with_gate}
    shared = set(by_iid_off) & set(by_iid_on)

    candidates = 0
    events = 0
    per_instance: List[Dict[str, Any]] = []
    for iid in sorted(shared):
        off = by_iid_off[iid]
        on = by_iid_on[iid]
        on_accepted = sum(
            1 for ev in on.gate_events
            if ev.get("decision") and not ev.get("fail_soft")
        )
        if on_accepted == 0:
            continue
        candidates += 1
        regressed = off.reward > 0 and on.reward <= 0
        if regressed:
            events += 1
        per_instance.append({
            "instance_id": iid,
            "gate_accepted": on_accepted,
            "off_reward": off.reward,
            "on_reward": on.reward,
            "regressed": regressed,
        })
    return {
        "fa_harm_candidates": candidates,
        "fa_harm_events": events,
        "fa_harm_rate": events / candidates if candidates else 0.0,
        "per_instance": per_instance,
    }


def _resolve_rate(results: List[InstanceResult]) -> float:
    if not results:
        return 0.0
    resolved = sum(1 for r in results if r.reward > 0)
    return resolved / len(results)


def _summarize(
    no_gate: List[InstanceResult],
    with_gate: List[InstanceResult],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    gate_metrics = _aggregate_gate_metrics(with_gate)
    fa_metrics = _fa_harm_proxy(no_gate, with_gate)
    off_resolve = _resolve_rate(no_gate)
    on_resolve = _resolve_rate(with_gate)

    # Hard online ship criteria from the internal handoff doc.
    ship = {
        "reject_rate_in_band": 0.10 < gate_metrics["reject_rate"] < 0.40,
        "fa_harm_within_gate": fa_metrics["fa_harm_rate"] <= 0.03,
        "p95_latency_within_gate": gate_metrics["p95_latency_ms"] <= 200.0,
        "success_delta_non_negative": (on_resolve - off_resolve) >= 0.0,
    }
    ship["all_clear"] = all(ship.values())

    return {
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "agent_llm": args.agent_llm,
            "memory_type": DEFAULT_MEMORY_TYPE,
            "max_tokens": args.max_tokens,
            "n_instances": args.n_instances,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "threshold": args.threshold,
            "base_model": args.base_model,
            "timestamp": datetime.now().isoformat(),
        },
        "resolve_rate": {
            "no_gate": off_resolve,
            "with_gate": on_resolve,
            "delta": on_resolve - off_resolve,
        },
        "gate_metrics": gate_metrics,
        "fa_harm": fa_metrics,
        "ship_criteria": ship,
        "no_gate_results": [r.__dict__ for r in no_gate],
        "with_gate_results": [r.__dict__ for r in with_gate],
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="WorldModelGate LoRA checkpoint (the class-weighted CE checkpoint champion).")
    parser.add_argument("--threshold", type=float, default=0.78,
                        help="Gate threshold (the class-weighted CE checkpoint constrained best = 0.78).")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base",
                        help="Classifier base model.")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--agent-llm", default=DEFAULT_AGENT_LLM)
    parser.add_argument("--summarization-model", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
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
                        default="swebench")
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip Docker patch evaluation (for non-Docker env-types).")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for eval_gym_gate.json.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--mode", choices=["both", "no-gate", "with-gate"], default="both",
                        help="Run both modes (default), or just one. Used for parallel "
                             "dispatch — e.g. gate-off on CPU node, gate-on on GPU node.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    instances = list(dataset)
    if args.slice:
        start, end = map(int, args.slice.split(":"))
        instances = instances[start:end]
    else:
        instances = instances[: args.n_instances]
    logger.info("running %d instances in mode=%s", len(instances), args.mode)

    gate = None
    if args.mode in ("both", "with-gate"):
        logger.info("loading gate checkpoint from %s (threshold=%.3f)",
                    args.checkpoint, args.threshold)
        gate = _load_gate(args.checkpoint, args.threshold, args.base_model)

    no_gate: List[InstanceResult] = []
    with_gate: List[InstanceResult] = []

    if args.mode in ("both", "no-gate"):
        for i, inst in enumerate(instances):
            logger.info("[no-gate %d/%d] %s", i + 1, len(instances), inst["instance_id"])
            no_gate.append(_run_one(inst, "no-gate", args, None))
            _write_partial(args.output, no_gate, with_gate, args)

    if args.mode in ("both", "with-gate"):
        for i, inst in enumerate(instances):
            logger.info("[with-gate %d/%d] %s", i + 1, len(instances), inst["instance_id"])
            with_gate.append(_run_one(inst, "with-gate", args, gate))
            _write_partial(args.output, no_gate, with_gate, args)

    summary = _summarize(no_gate, with_gate, args)
    out_path = args.output / "eval_gym_gate.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(
        {
            "step": "eval_gym_gate",
            "out_path": str(out_path),
            "n_instances": len(instances),
            "resolve_rate": summary["resolve_rate"],
            "gate_metrics": summary["gate_metrics"],
            "fa_harm": {k: v for k, v in summary["fa_harm"].items() if k != "per_instance"},
            "ship_criteria": summary["ship_criteria"],
        },
        indent=2,
    ), file=sys.stderr)
    return 0 if summary["ship_criteria"].get("all_clear") else 1


def _write_partial(
    out_dir: Path,
    no_gate: List[InstanceResult],
    with_gate: List[InstanceResult],
    args: argparse.Namespace,
) -> None:
    """Incremental dump so a long smoke run doesn't lose partial results."""
    tmp_path = out_dir / "eval_gym_gate.partial.json"
    with tmp_path.open("w") as fh:
        json.dump(
            {
                "no_gate_results": [r.__dict__ for r in no_gate],
                "with_gate_results": [r.__dict__ for r in with_gate],
                "config": {
                    "dataset": args.dataset,
                    "agent_llm": args.agent_llm,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "threshold": args.threshold,
                },
            },
            fh, indent=2, default=str,
        )


if __name__ == "__main__":
    sys.exit(main())
