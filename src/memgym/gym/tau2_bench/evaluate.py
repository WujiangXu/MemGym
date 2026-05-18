"""
Tau2-bench evaluation CLI — two-phase (baseline → memory) runner.

Usage:
    python -m memgym.gym.tau2_bench \
        --domains mock,telecom,airline,retail \
        --agent_llm bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --memory_model tau2_summarizing \
        --mode both \
        --result_dir results/tau2_bench/ \
        --workers 4

Design — why two phases
=======================
The paper claim for the τ²-bench track is *symmetric memory wrapping*
improves performance. To make that measurement fair:

    1. the training phase (baseline): run the unwrapped tau2 ``LLMAgent`` /
       ``UserSimulator`` pair with ``memory_agent=None, memory_user=None``.
    2. the training phase (memory):  run the SAME agent class + SAME task list +
       SAME seed, but pass the configured memory manager(s) into
       ``env.run_tau2_episode``.

The task set, task order, and seed are pinned BEFORE phase 1 starts, so
phase 2 sees the exact same task configurations. Any reward delta is
then attributable to memory alone (agent, model, tools held constant).

Each phase writes its own per-task ``trajectory.json`` + ``result.json``
and its own ``summary.json``. The top-level ``summary.json`` contains
the cross-phase comparison built by ``compare_baseline_vs_memory``.

Trajectory directory layout
===========================
``<result_dir>/<run_id>/baseline/<domain>/<task_id>/`` and
``<result_dir>/<run_id>/memory/<domain>/<task_id>/``.

The ``run_id`` encodes ``{YYYYMMDD-HHMMSS}_{domains}_{memory_key}_{hparams}``
so concurrent runs never collide and you can eyeball a directory listing
to recover the provenance of any on-disk trajectory. If the resolved
run dir somehow already exists, an integer suffix is appended instead
of overwriting.

Tau2 lazy imports
=================
All ``tau2.*`` imports are deferred into worker-process functions so the
CLI still `--help`s, `--list-domains` and `--list-memory-models` without
having `third_party/tau2-bench` installed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.memory import get_memory_model, list_memory_models

logger = logging.getLogger("memgym.tau2_bench.evaluate")


# =============================================================================
# CLI config
# =============================================================================

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tau2-bench dialogue agents with MemGym memory integration",
    )

    # ---- Task selection ----
    parser.add_argument(
        "--domains", type=str, default="mock",
        help="Comma-separated tau2 domains. Use 'all' for "
             "mock,telecom,airline,retail. Default: mock.",
    )
    parser.add_argument(
        "--task_ids", type=str, default="",
        help="Comma-separated task ids to run (applied within each domain). "
             "Empty = all tasks for the domain.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Per-domain task cap (0 = no limit). Useful for smoke tests.",
    )
    parser.add_argument(
        "--task_split_name", type=str, default="base",
        help="Task split: 'base' (all tasks, default), 'train', or 'test'. "
             "Matches tau2's split_tasks.json per domain.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed passed into tau2 orchestrator. Same value is used for "
             "baseline and memory phases to keep the user-sim deterministic.",
    )
    parser.add_argument("--max_steps", type=int, default=100)

    # ---- Agent / user LLM ----
    parser.add_argument(
        "--agent_llm", type=str, required=False,
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="litellm model id for the agent.",
    )
    parser.add_argument(
        "--user_llm", type=str, default="",
        help="litellm model id for the user simulator (defaults to --agent_llm).",
    )
    parser.add_argument(
        "--solo_mode", type=str, default="auto",
        choices=["auto", "true", "false"],
        help="Force single-agent mode. 'auto' picks per-domain from "
             "DOMAIN_SOLO_MODE_SUPPORT (mock/telecom → solo).",
    )

    # ---- Memory ----
    parser.add_argument(
        "--memory_model", type=str, default="none",
        help="Memory strategy: 'none', 'tau2_summarizing', 'tau2_structured', "
             "or any name in the registry. Use --list_memory_models to enumerate.",
    )
    parser.add_argument(
        "--memory_side", type=str, default="both",
        choices=["both", "agent", "user"],
        help="Symmetric wrapping control: 'both' (default) wraps agent AND "
             "user simulator; 'agent' wraps only the agent; 'user' wraps only "
             "the user simulator (for the 'whose memory matters more' ablation).",
    )
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--keep_first", type=int, default=1)
    parser.add_argument("--keep_last", type=int, default=4)
    parser.add_argument("--condensation_ratio", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument(
        "--summarization_model", type=str, default="",
        help="Model used for memory summarization. Defaults to --agent_llm.",
    )
    parser.add_argument(
        "--nl_judge_model", type=str, default="",
        help="LLM for NL-assertion evaluation (retail tasks need this). "
             "Defaults to --agent_llm. Tau2 upstream defaults to gpt-4.1 "
             "which requires OPENAI_API_KEY; set this to a Bedrock model "
             "when running on EC2.",
    )

    # ---- Two-phase orchestration ----
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["baseline", "memory", "both"],
        help="Phase selector. 'baseline' = no memory; 'memory' = memory on; "
             "'both' (default) = baseline then memory, emitting a comparison.",
    )

    # ---- Output ----
    parser.add_argument(
        "--result_dir", type=str, default="results/tau2_bench",
        help="Parent directory for the run. A unique run_id subdir is created "
             "underneath so repeated runs never clobber.",
    )
    parser.add_argument(
        "--run_name", type=str, default="",
        help="Optional human-readable tag appended to the run_id. "
             "Use it to label experiments like 'ablate_user_side'.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of ProcessPool workers. Defaults to 1 (serial) to keep "
             "Bedrock rate-limiting predictable; bump up for mock/telecom.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Per-episode progress logging inside the runner.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ---- Introspection ----
    parser.add_argument(
        "--list_domains", action="store_true",
        help="Print supported tau2 domains and exit.",
    )
    parser.add_argument(
        "--list_memory_models", action="store_true",
        help="Print the memory-model registry and exit.",
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="Print task ids for --domains and exit (no execution).",
    )

    return parser.parse_args()


# =============================================================================
# Result-directory management
# =============================================================================

_INVALID_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str, max_len: int = 40) -> str:
    """Make a string safe for use as a directory-name component."""
    s = _INVALID_PATH_CHARS.sub("-", value).strip("-")
    return s[:max_len] or "unnamed"


def _build_run_id(args: argparse.Namespace) -> str:
    """Derive a human-readable, collision-resistant run id.

    Shape: ``{YYYYMMDD-HHMMSS}_{domains}_{mode}_{memory}_{hparams}[_{tag}]``.

    Keeping the hyperparameters in the path means a ``ls`` on the parent
    directory is enough to recover "which run was this" without cracking
    open args.json.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    domains = _slugify(args.domains.replace(",", "-"))
    memory_tag = _slugify(args.memory_model)
    hparams = f"ms{args.max_size}kf{args.keep_first}kl{args.keep_last}r{args.condensation_ratio}"
    hparams = _slugify(hparams, max_len=30)

    parts = [timestamp, domains, args.mode, memory_tag, hparams]
    if args.run_name:
        parts.append(_slugify(args.run_name))
    return "_".join(parts)


def _resolve_result_dir(args: argparse.Namespace) -> Path:
    """Create a unique run subdirectory and return it.

    If the generated path exists (e.g. same-second timestamp, reused tag),
    append ``_N`` until we find a free slot rather than overwriting.
    """
    parent = Path(args.result_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)

    run_id = _build_run_id(args)
    candidate = parent / run_id
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{run_id}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


# =============================================================================
# Memory factory
# =============================================================================

def build_memory_manager(
    args: argparse.Namespace,
    role: str,
    task_instruction: str,
    domain: str,
    solo_mode: bool,
) -> Optional[Any]:
    """Instantiate a BaseMemoryManager for one side (agent or user).

    Returns ``None`` when memory is disabled or the side is excluded by
    ``--memory_side``. Each side gets its OWN instance — sharing one
    manager between agent and user would be incorrect because their
    condensation prompts diverge (USER_GOAL vs USER_PERSONA).
    """
    if args.memory_model in ("none", "None", ""):
        return None
    if args.memory_side == "agent" and role == "user":
        return None
    if args.memory_side == "user" and role == "agent":
        return None

    summarization_model = args.summarization_model or args.agent_llm
    name = args.memory_model

    # Tau2-specific memory adapters accept the full episode-state kwargs
    # in the constructor so their prompt template resolves immediately;
    # registry models that don't understand them are called without.
    if name in ("tau2_summarizing", "tau2-summarizing", "tau2_structured", "tau2-structured"):
        return get_memory_model(
            name,
            task_instruction=task_instruction,
            domain=domain,
            solo_mode=solo_mode,
            role=role,
            max_size=args.max_size,
            keep_first=args.keep_first,
            keep_last=args.keep_last,
            condensation_ratio=args.condensation_ratio,
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
        )

    # Generic fallback — pass the subset of kwargs the base summarizers know.
    return get_memory_model(
        name,
        max_size=args.max_size,
        keep_first=args.keep_first,
        condensation_ratio=args.condensation_ratio,
        summarization_model=summarization_model,
        max_tokens=args.max_tokens,
    )


# =============================================================================
# Task enumeration
# =============================================================================

ALL_DOMAINS = ["mock", "telecom", "airline", "retail"]


def _resolve_domains(domains_arg: str) -> List[str]:
    if domains_arg.strip().lower() == "all":
        return list(ALL_DOMAINS)
    return [d.strip() for d in domains_arg.split(",") if d.strip()]


def enumerate_tasks(
    domains: List[str],
    task_id_filter: str,
    limit: int,
    task_split: str = "base",
) -> List[Tuple[str, str, str]]:
    """Return list of ``(domain, task_id, instruction)`` tuples.

    Loads each domain once via tau2's registry. If ``task_id_filter`` is
    set, only the named ids are kept (same filter applied per-domain).
    ``limit`` trims each domain's list after filtering.
    """
    from .env import load_domain_tasks  # defers tau2 import

    requested_ids: Optional[set] = None
    if task_id_filter:
        requested_ids = {t.strip() for t in task_id_filter.split(",") if t.strip()}

    out: List[Tuple[str, str, str]] = []
    for domain in domains:
        try:
            tasks = load_domain_tasks(domain, task_split=task_split)
        except Exception as e:
            logger.error("failed to load tasks for domain=%s: %s", domain, e)
            continue

        domain_out = []
        for task in tasks:
            tid = str(task.id)
            if requested_ids is not None and tid not in requested_ids:
                continue
            instruction = _extract_task_instruction(task)
            domain_out.append((domain, tid, instruction))
            if limit and len(domain_out) >= limit:
                break
        out.extend(domain_out)
        logger.info("enumerated domain=%s → %d tasks", domain, len(domain_out))
    return out


def _extract_task_instruction(task: Any) -> str:
    """Pull the user-facing instruction out of a tau2 Task object.

    Tau2 Task has ``user_scenario`` (dict with 'instructions') and
    ``description``. We prefer the scenario so the memory manager sees the
    same text the user simulator is role-playing.
    """
    scenario = getattr(task, "user_scenario", None)
    if isinstance(scenario, dict) and scenario.get("instructions"):
        return str(scenario["instructions"])
    if scenario is not None and hasattr(scenario, "instructions"):
        return str(scenario.instructions)
    description = getattr(task, "description", "") or ""
    return str(description)


# =============================================================================
# Worker function (serializable for ProcessPoolExecutor)
# =============================================================================

def run_single_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one (domain, task_id, phase) triple and return a result dict.

    Fully self-contained so it can be dispatched to a ProcessPool worker —
    all tau2 + memgym imports happen INSIDE this function so the parent
    process doesn't pay the tau2 import cost and worker crashes don't
    corrupt the parent's import cache.
    """
    # Re-hydrate args inside the worker so the Namespace survives pickling.
    args = argparse.Namespace(**payload["args"])
    domain = payload["domain"]
    task_id = payload["task_id"]
    task_instruction = payload["task_instruction"]
    phase = payload["phase"]  # "baseline" | "memory"
    result_dir = Path(payload["result_dir"])

    _configure_logging(args.log_level, tag=f"{phase}/{domain}/{task_id}")

    from .env import Tau2BenchMemoryEnv, DOMAIN_SOLO_MODE_SUPPORT
    from .agent import Tau2BenchAgent
    from memgym.runners.tau2_bench_runner import Tau2BenchRunner

    # Bedrock requires litellm to auto-fix consecutive system messages
    import litellm
    litellm.modify_params = True

    # ---- Resolve solo_mode ----
    if args.solo_mode == "auto":
        solo_mode = DOMAIN_SOLO_MODE_SUPPORT.get(domain, False)
    else:
        solo_mode = (args.solo_mode == "true")

    # ---- Build env ----
    env = Tau2BenchMemoryEnv(
        domain=domain,
        task_id=task_id,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm or None,
        solo_mode=solo_mode,
        max_steps=args.max_steps,
        nl_judge_model=args.nl_judge_model or args.agent_llm,
    )

    # ---- Build memory managers (None in baseline phase) ----
    memory_agent = None
    memory_user = None
    if phase == "memory":
        memory_agent = build_memory_manager(
            args, role="agent",
            task_instruction=task_instruction,
            domain=domain, solo_mode=solo_mode,
        )
        if not solo_mode:
            memory_user = build_memory_manager(
                args, role="user",
                task_instruction=task_instruction,
                domain=domain, solo_mode=solo_mode,
            )

    # ---- Build tracking agent + runner ----
    agent = Tau2BenchAgent(agent_llm=args.agent_llm)
    runner = Tau2BenchRunner(
        env=env,
        agent=agent,
        memory_agent=memory_agent,
        memory_user=memory_user,
        output_dir=str(result_dir / phase),
        verbose=args.verbose,
    )

    # ---- Run ----
    try:
        result = runner.run_episode(
            task_config={"task_instruction": task_instruction},
            seed=args.seed,
        )
    except Exception as exc:
        logger.exception("phase=%s domain=%s task=%s failed", phase, domain, task_id)
        result = {
            "domain": domain,
            "task_id": task_id,
            "phase": phase,
            "error": str(exc),
            "reward": 0.0,
            "success": False,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass

    result["phase"] = phase
    return result


# =============================================================================
# Phase orchestration
# =============================================================================

def run_phase(
    args: argparse.Namespace,
    tasks: List[Tuple[str, str, str]],
    phase: str,
    result_dir: Path,
) -> List[Dict[str, Any]]:
    """Run every (domain, task_id) in ``tasks`` under the given phase."""
    logger.info("starting phase=%s over %d tasks (workers=%d)", phase, len(tasks), args.workers)

    payloads = [
        {
            "args": vars(args),
            "domain": domain,
            "task_id": task_id,
            "task_instruction": instruction,
            "phase": phase,
            "result_dir": str(result_dir),
        }
        for domain, task_id, instruction in tasks
    ]

    results: List[Dict[str, Any]] = []
    start = time.time()

    if args.workers <= 1:
        # Serial path — easier to debug + avoids pickling-failures on exotic
        # memory adapters. Prefer for smoke tests.
        for p in payloads:
            results.append(run_single_task(p))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_single_task, p) for p in payloads]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.exception("worker future raised: %s", exc)
                    results.append({"phase": phase, "error": str(exc),
                                    "reward": 0.0, "success": False})

    elapsed = time.time() - start
    _write_phase_summary(result_dir, phase, results, args, elapsed)
    return results


def _write_phase_summary(
    result_dir: Path,
    phase: str,
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    """Write ``<result_dir>/<phase>/summary.json`` with per-phase aggregates."""
    phase_dir = result_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)

    n = len(results)
    succ = sum(1 for r in results if r.get("success"))
    rewards = [float(r.get("reward", 0.0)) for r in results]
    avg_reward = sum(rewards) / n if n else 0.0

    agent_comp = sum(int(r.get("agent_compaction_count", 0)) for r in results)
    user_comp = sum(int(r.get("user_compaction_count", 0)) for r in results)
    coercion_fail = sum(
        int(r.get("agent_coercion_failures", 0)) + int(r.get("user_coercion_failures", 0))
        for r in results
    )

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": phase,
        "domains": args.domains,
        "memory_model": args.memory_model if phase == "memory" else "none",
        "memory_side": args.memory_side if phase == "memory" else "none",
        "agent_llm": args.agent_llm,
        "n_tasks": n,
        "n_success": succ,
        "success_rate": (succ / n) if n else 0.0,
        "avg_reward": avg_reward,
        "agent_compaction_count": agent_comp,
        "user_compaction_count": user_comp,
        "total_coercion_failures": coercion_fail,
        "wall_time_seconds": round(elapsed, 1),
        "results": results,
    }

    with (phase_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "phase=%s done: %d/%d success (%.1f%%) avg_reward=%.3f agent_comp=%d user_comp=%d coerce_fail=%d %.1fs",
        phase, succ, n, 100 * succ / max(1, n), avg_reward,
        agent_comp, user_comp, coercion_fail, elapsed,
    )


# =============================================================================
# Cross-phase comparison — contributed by user (see main())
# =============================================================================

def compare_baseline_vs_memory(
    baseline_results: List[Dict[str, Any]],
    memory_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-task deltas between the two phases.

    This function shapes the headline number of the experiment. The
    implementation is intentionally left to the user because the right
    answer depends on experimental-design decisions: which metrics to
    delta, how to handle tasks missing from one phase, how to filter
    down to "memory actually mattered here" cases, etc.

    See ``main()`` for where this is called. A placeholder implementation
    is provided so the pipeline runs end-to-end; replace it with your own.

    Args:
        baseline_results: Per-task results from the baseline phase (no memory).
        memory_results: Per-task results from the memory phase.

    Returns:
        A dict that will be merged into the top-level summary.json under
        the 'comparison' key.
    """
    # ------------------------------------------------------------------
    # TODO(user): Replace this placeholder with your preferred aggregation.
    #
    # Things to decide:
    #   * Match keys — probably ``(domain, task_id)`` — to diff per task.
    #   * What to report: reward delta, success-rate delta, matched-task
    #     coverage, tokens-saved ratio, number of tasks where memory
    #     STRICTLY helped / STRICTLY hurt / tied.
    #   * How to handle missing tasks (baseline had them, memory didn't
    #     or vice versa). Skip? Count as failure?
    #   * Whether to filter out tasks where memory never triggered
    #     (``agent_compaction_count == 0 and user_compaction_count == 0``)
    #     — those aren't "memory experiments", they're noise.
    #
    # The placeholder below is intentionally minimal: it only reports
    # top-line success-rate numbers. Not enough for a paper.
    # ------------------------------------------------------------------
    def _rate(results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.get("success")) / len(results)

    return {
        "baseline_success_rate": _rate(baseline_results),
        "memory_success_rate": _rate(memory_results),
        "delta_success_rate": _rate(memory_results) - _rate(baseline_results),
        "n_baseline": len(baseline_results),
        "n_memory": len(memory_results),
        # TODO(user): add per-task breakdown, compaction-filtered deltas,
        # and tokens-saved once you've picked your methodology.
        "TODO": "replace this stub with the full comparison — see function body",
    }


# =============================================================================
# Top-level summary writer
# =============================================================================

def _write_combined_summary(
    result_dir: Path,
    baseline: List[Dict[str, Any]],
    memory: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": result_dir.name,
        "domains": args.domains,
        "mode": args.mode,
        "agent_llm": args.agent_llm,
        "memory_model": args.memory_model,
        "memory_side": args.memory_side,
        "memory_hparams": {
            "max_size": args.max_size,
            "keep_first": args.keep_first,
            "keep_last": args.keep_last,
            "condensation_ratio": args.condensation_ratio,
            "max_tokens": args.max_tokens,
        },
        "comparison": comparison,
        "n_baseline_tasks": len(baseline),
        "n_memory_tasks": len(memory),
        "wall_time_seconds": round(elapsed, 1),
    }
    with (result_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)


# =============================================================================
# Entry point
# =============================================================================

def _configure_logging(level_name: str, tag: str = "main") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = f"[%(asctime)s tau2 {tag} %(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def main() -> None:
    args = config()
    _configure_logging(args.log_level)

    if args.list_memory_models:
        print("Registered memory models:")
        for n in sorted(list_memory_models()):
            print(f"  {n}")
        return

    if args.list_domains:
        print("Supported tau2 domains:")
        for d in ALL_DOMAINS:
            print(f"  {d}")
        return

    domains = _resolve_domains(args.domains)
    tasks = enumerate_tasks(domains, args.task_ids, args.limit,
                            task_split=args.task_split_name)
    if not tasks:
        raise SystemExit(
            f"No tasks found for domains={domains} task_ids={args.task_ids!r}"
        )

    if args.list_tasks:
        for domain, task_id, instruction in tasks:
            print(f"{domain}\t{task_id}\t{instruction[:80]}")
        return

    # Create unique run dir and persist args.json BEFORE running anything —
    # if the workers crash, we still have provenance on disk.
    result_dir = _resolve_result_dir(args)
    with (result_dir / "args.json").open("w") as f:
        json.dump({
            **vars(args),
            "tasks": [(d, t) for d, t, _ in tasks],
        }, f, indent=2)
    logger.info("run_dir=%s", result_dir)

    start = time.time()
    baseline_results: List[Dict[str, Any]] = []
    memory_results: List[Dict[str, Any]] = []

    if args.mode in ("baseline", "both"):
        baseline_results = run_phase(args, tasks, "baseline", result_dir)

    if args.mode in ("memory", "both"):
        memory_results = run_phase(args, tasks, "memory", result_dir)

    elapsed = time.time() - start

    if args.mode == "both":
        comparison = compare_baseline_vs_memory(baseline_results, memory_results)
    elif args.mode == "baseline":
        comparison = {"note": "baseline-only run; no memory phase executed"}
    else:
        comparison = {"note": "memory-only run; no baseline phase executed"}

    _write_combined_summary(result_dir, baseline_results, memory_results,
                            comparison, args, elapsed)
    logger.info("all done: run_dir=%s elapsed=%.1fs", result_dir, elapsed)


if __name__ == "__main__":
    main()
