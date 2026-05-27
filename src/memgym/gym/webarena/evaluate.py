"""
WebArena (Web GUI) evaluation CLI.

Usage:
    python -m memgym.gym.webarena \
        --app_name gitlab \
        --task_ids 0,1,2 \
        --policy_model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --memory_model webarena_summarizing \
        --max_size 20 --keep_first 1 --condensation_ratio 0.75 \
        --max_steps 30 --workers 4 \
        --observation_mode text \
        --result_dir results/webarena/smoke_test/

Design:

- One agent + one memory manager + one env are created per worker process,
  mirroring the SWE-bench / old-CUA pattern. The env is re-used across tasks
  within a worker when possible; `close()` tears down browser + server.
- Parallelism is implemented via `multiprocessing.Process` + a task queue.
  The `WebArenaServerPool` is created inside each worker so each worker
  has its own pool (and its own port range slice).
- Trajectories and result JSON are written under
  `<result_dir>/<app>/<task_id>/` — one subdir per episode.
- Memory strategies: `none`, `webarena_summarizing`, `webarena_structured`.
  Legacy adapters (`llm_summarizing`, `structured_summary`) are also
  available via the generic registry path.

This script deliberately mirrors the shape of `gym/cua/evaluate.py` so the
launch.py endpoint can follow the same command-building conventions.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from memgym.memory import get_memory_model, list_memory_models

logger = logging.getLogger("memgym.webarena.evaluate")


# =============================================================================
# Config
# =============================================================================

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WebArena (web GUI) agents with MemGym memory integration",
    )

    # Environment
    parser.add_argument(
        "--webarena_dir", type=str, default="",
        help="Path to the webarena-infinity checkout. "
             "Defaults to src/memgym/envs/webarena_infinity/ inside the repo.",
    )
    parser.add_argument(
        "--app_name", type=str, required=True,
        help="WebArena app name (e.g. gmail, gitlab, reddit). "
             "Use --list_apps to enumerate.",
    )
    parser.add_argument(
        "--task_ids", type=str, default="",
        help="Comma-separated task ids to run (e.g. '0,1,2'). "
             "Empty = run all tasks found for the app.",
    )
    parser.add_argument(
        "--task_filter", type=str, default="",
        help="Optional filter expression applied to task.instruction (substring match).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=50,
        help="Maximum steps per episode. Matches the upstream webarena-infinity "
             "canonical (evaluation/run_eval_parallel.py:526 and agents.py:93). "
             "Caps below 50 silently depress hard-tier pass rates because hard "
             "tasks routinely need 30-40 steps; see docs/webarena/methodology.md §5.7.",
    )
    parser.add_argument(
        "--observation_mode", type=str, default="text",
        choices=["text", "vision"],
        help="Observation modality: text (accessibility tree, default) or vision (screenshot).",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run Chromium headless (default).",
    )
    parser.add_argument(
        "--no_headless", dest="headless", action="store_false",
        help="Disable headless mode (requires a display).",
    )

    # Agent
    parser.add_argument(
        "--agent_type", type=str, default="webarena",
        choices=["webarena"],
        help="Agent implementation (only 'webarena' supported in the initial track).",
    )
    parser.add_argument("--policy_model", type=str, required=True,
                        help="litellm model id (e.g. 'bedrock/us.anthropic.claude-...').")
    parser.add_argument("--policy_max_tokens", type=int, default=256,
                        help="Max tokens per agent response.")
    parser.add_argument("--policy_temperature", type=float, default=0.0)

    # Memory
    parser.add_argument(
        "--memory_model", type=str, default="none",
        help="Memory strategy name: 'none', 'webarena_summarizing', 'webarena_structured', "
             "or any name in the registry. Use --list_memory_models to enumerate.",
    )
    parser.add_argument("--max_size", type=int, default=20)
    parser.add_argument("--keep_first", type=int, default=1)
    parser.add_argument("--condensation_ratio", type=float, default=0.75)
    parser.add_argument("--summarization_model", type=str, default="",
                        help="Model used for summarization (default: same as --policy_model).")
    parser.add_argument("--max_tokens", type=int, default=16000,
                        help="Token budget for the memory manager.")

    # Server pool / parallelism
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes (each owns a browser + server pool).")
    parser.add_argument("--port_range", type=str, default="18000-19000",
                        help="Port range for the server pool (inclusive-exclusive).")
    parser.add_argument("--spawn_timeout_sec", type=float, default=30.0)
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Hard ceiling on workers (defends against accidentally swamping the server pool).")

    # Observation replay (offline memory-strategy evaluation)
    parser.add_argument(
        "--replay_from", type=str, default="",
        help="Offline replay mode: path to a baseline trajectory directory "
             "(layout: <dir>/<task_id>/trajectory.json). When set, the runner "
             "loads recorded observations and replays them through the memory "
             "strategy without a browser. Only re-queries the policy model at "
             "steps where memory compresses the context. ~10x cheaper than live.",
    )

    # Smart replay (live — follow baseline in browser until memory diverges)
    parser.add_argument(
        "--smart_replay_from", type=str, default="",
        help="Live smart-replay mode: path to a baseline trajectory directory "
             "(layout: <dir>/<task_id>/trajectory.json). Follows baseline "
             "actions in the real browser while running memory at each step. "
             "At non-compaction steps, uses baseline action (no policy call). "
             "At compaction steps, queries the policy model and checks for "
             "divergence. On divergence, switches to fully live mode. Gives "
             "real browser-evaluated success rates while saving 60-90%% of "
             "policy calls vs full live runs.",
    )

    # Prefix-injection (replay-first-K-steps experiment)
    parser.add_argument(
        "--prefix_trajectory_dir", type=str, default="",
        help="Prefix-injection mode: directory containing recorded baseline trajectories "
             "(layout: <dir>/<task_id>/trajectory.json). When set, the runner "
             "deterministically replays the first K recorded actions per task "
             "before handing off to the live agent + memory strategy. "
             "K is controlled by --prefix_steps (absolute) or --prefix_fraction. "
             "Gated by the replay-determinism probe result: every "
             "replayed trajectory reaches the same verifier reward as its "
             "recording, even when the a11y tree drifts slightly. See "
             "replay_probe.py for the determinism check.",
    )
    parser.add_argument(
        "--prefix_steps", type=int, default=-1,
        help="Prefix-injection mode: absolute number of recorded actions to replay before "
             "handing off. -1 (default) means use --prefix_fraction instead.",
    )
    parser.add_argument(
        "--prefix_fraction", type=float, default=0.5,
        help="Prefix-injection mode: fraction of the recorded trajectory length to replay as "
             "prefix. Ignored when --prefix_steps >= 0. Default 0.5 gives a "
             "mid-trajectory handoff that is task-length-normalized.",
    )

    # Output + discovery
    parser.add_argument("--result_dir", type=str, default="results/webarena/default")
    parser.add_argument("--list_apps", action="store_true",
                        help="List available webarena apps and exit.")
    parser.add_argument("--list_memory_models", action="store_true",
                        help="List registered memory strategies and exit.")
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    if args.workers > args.max_workers:
        logger.warning(
            "clamping --workers from %d to --max_workers=%d", args.workers, args.max_workers
        )
        args.workers = args.max_workers
    return args


# =============================================================================
# Factories
# =============================================================================

def _resolve_webarena_dir(explicit: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    # Default: src/memgym/envs/webarena_infinity relative to this file
    here = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = here / "envs" / "webarena_infinity"
        if candidate.exists():
            return candidate
        candidate = here / "src" / "memgym" / "envs" / "webarena_infinity"
        if candidate.exists():
            return candidate
        here = here.parent
    raise FileNotFoundError(
        "Could not locate webarena-infinity directory. "
        "Pass --webarena_dir explicitly or run `python -m memgym.gym.webarena.install`."
    )


def _parse_port_range(spec: str) -> tuple[int, int]:
    a, _, b = spec.partition("-")
    start, end = int(a), int(b)
    if start >= end:
        raise ValueError(f"bad --port_range {spec!r}")
    return start, end


def create_memory(args: argparse.Namespace, task_instruction: str):
    """Create the memory manager based on --memory_model, passing task context."""
    from memgym.memory import PassThroughMemory

    summarization_model = args.summarization_model or args.policy_model
    model_name = args.memory_model

    if model_name in ("none", "passthrough"):
        return PassThroughMemory(max_tokens=args.max_tokens)

    if model_name == "webarena_summarizing":
        return get_memory_model(
            "webarena_summarizing",
            task_instruction=task_instruction,
            app_name=args.app_name,
            max_size=args.max_size,
            keep_first=args.keep_first,
            condensation_ratio=args.condensation_ratio,
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
        )

    if model_name == "webarena_structured":
        return get_memory_model(
            "webarena_structured",
            task_instruction=task_instruction,
            app_name=args.app_name,
            max_size=args.max_size,
            keep_first=args.keep_first,
            condensation_ratio=args.condensation_ratio,
            summarization_model=summarization_model,
            max_tokens=args.max_tokens,
        )

    # Fallback to the generic registry (e.g. llm_summarizing / structured_summary)
    return get_memory_model(
        model_name,
        max_size=args.max_size,
        keep_first=args.keep_first,
        condensation_ratio=args.condensation_ratio,
        summarization_model=summarization_model,
        max_tokens=args.max_tokens,
    )


def create_agent(args: argparse.Namespace):
    from .agent import WebArenaAgent
    return WebArenaAgent(
        policy_model=args.policy_model,
        observation_mode=args.observation_mode,
        max_tokens=args.policy_max_tokens,
        temperature=args.policy_temperature,
    )


# =============================================================================
# Task enumeration
# =============================================================================

def enumerate_task_ids(webarena_dir: Path, app_name: str,
                       requested: str, task_filter: str) -> List[str]:
    """Return the list of task ids to run for an app.

    Same candidate-paths fallback as env.load_task() — if the user gave us
    `--task_ids 0,1,2` we honor that verbatim without touching the disk.
    """
    if requested:
        return [t.strip() for t in requested.split(",") if t.strip()]

    app_dir = webarena_dir / "apps" / app_name
    ids: List[str] = []

    # Flat tasks/ directory of JSON files
    tasks_dir = app_dir / "tasks"
    if tasks_dir.exists():
        for p in sorted(tasks_dir.glob("*.json")):
            ids.append(p.stem)

    # JSONL manifest
    for manifest in (app_dir / "tasks.jsonl", app_dir / "benchmark.jsonl"):
        if manifest.exists():
            with manifest.open() as f:
                for line in f:
                    row = json.loads(line)
                    ids.append(str(row.get("id") or row.get("task_id")))

    # WebArena-Infinity layout: array manifests. Every app under
    # webarena-infinity/apps/<name>/ ships one or both of these, containing a
    # JSON list of {id, instruction, verify, ...} dicts.
    for manifest in (app_dir / "real-tasks.json", app_dir / "function-tasks.json"):
        if not manifest.exists():
            continue
        with manifest.open() as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        for row in rows:
            tid = str(row.get("id") or row.get("task_id") or "")
            if tid:
                ids.append(tid)

    # Apply instruction filter if requested
    if task_filter:
        from .env import load_task
        filtered: List[str] = []
        for tid in ids:
            try:
                t = load_task(webarena_dir, app_name, tid)
                if task_filter.lower() in (t.instruction or "").lower():
                    filtered.append(tid)
            except Exception:
                continue
        ids = filtered

    return ids


# =============================================================================
# Worker process
# =============================================================================

def run_worker(
    worker_id: int,
    task_queue: "mp.Queue[tuple[str, str]]",
    result_list: "mp.managers.ListProxy",
    args_dict: Dict[str, Any],
    webarena_dir_str: str,
    port_start: int,
    port_end: int,
) -> None:
    """Each worker pulls (app_name, task_id) pairs from the queue and runs them.

    A brand-new WebArenaServerPool is created per worker — each worker has
    its own slice of the port range to avoid cross-worker contention.
    """
    args = argparse.Namespace(**args_dict)
    _configure_logging(args.log_level, tag=f"worker-{worker_id}")

    from .env import WebArenaMemoryEnv
    from .server_pool import WebArenaServerPool
    from memgym.runners.webarena_runner import WebArenaRunner

    webarena_dir = Path(webarena_dir_str)
    pool = WebArenaServerPool(
        webarena_dir=webarena_dir,
        port_range=(port_start, port_end),
        spawn_timeout_sec=args.spawn_timeout_sec,
        release_kill=True,
    )

    agent = create_agent(args)

    try:
        while True:
            try:
                item = task_queue.get(timeout=2.0)
            except Exception:
                break
            if item is None:
                break
            app_name, task_id = item
            logger.info("worker %d picked up %s/%s", worker_id, app_name, task_id)

            # One env + one memory manager per episode (task-instruction is
            # per-task so memory must be reconstructed with the right prompt).
            from .env import load_task
            try:
                task = load_task(webarena_dir, app_name, task_id)
            except Exception as e:
                logger.error("failed to load task %s/%s: %s", app_name, task_id, e)
                result_list.append({
                    "example_id": task_id, "app_name": app_name,
                    "error": f"load_task: {e}", "reward": 0.0, "success": False,
                })
                continue

            memory = create_memory(args, task_instruction=task.instruction)

            env = WebArenaMemoryEnv(
                webarena_dir=webarena_dir,
                app_name=app_name,
                task_id=task_id,
                server_pool=pool,
                observation_mode=args.observation_mode,
                headless=args.headless,
            )

            # Prefix-injection: thread args from CLI -> runner.
            # `prefix_steps < 0` is the CLI "unset" sentinel; we hand None
            # to the runner so it falls back to `prefix_fraction`.
            prefix_dir = Path(args.prefix_trajectory_dir) / app_name if args.prefix_trajectory_dir else None
            prefix_steps_arg = args.prefix_steps if args.prefix_steps >= 0 else None

            # Smart replay: trajectory dir contains <task_id>/trajectory.json
            smart_replay_dir = Path(args.smart_replay_from) if args.smart_replay_from else None

            runner = WebArenaRunner(
                env=env,
                agent=agent,
                memory=memory,
                output_dir=str(Path(args.result_dir) / app_name),
                max_steps=args.max_steps,
                verbose=True,
                prefix_trajectory_dir=prefix_dir,
                prefix_steps=prefix_steps_arg,
                prefix_fraction=args.prefix_fraction,
                smart_replay_trajectory_dir=smart_replay_dir,
            )

            try:
                result = runner.run_episode(task_config={"task_id": task_id, "app_name": app_name})
                result_list.append(result)
            except Exception as e:
                logger.exception("episode failed for %s/%s", app_name, task_id)
                result_list.append({
                    "example_id": task_id, "app_name": app_name,
                    "error": str(e), "reward": 0.0, "success": False,
                })
            finally:
                try:
                    env.close()
                except Exception:
                    pass
    finally:
        pool.shutdown()
        logger.info("worker %d done", worker_id)


# =============================================================================
# Replay worker (offline — no browser/server needed)
# =============================================================================

def run_replay_worker(
    worker_id: int,
    task_queue: "mp.Queue[tuple[str, str]]",
    result_list: "mp.managers.ListProxy",
    args_dict: Dict[str, Any],
) -> None:
    """Worker for offline replay mode.  No browser or server pool needed."""
    args = argparse.Namespace(**args_dict)
    _configure_logging(args.log_level, tag=f"worker-{worker_id}")

    from memgym.runners.webarena_replay_runner import ObservationReplayRunner

    agent = create_agent(args)

    try:
        while True:
            try:
                item = task_queue.get(timeout=2.0)
            except Exception:
                break
            if item is None:
                break
            app_name, task_id = item
            logger.info("replay worker %d picked up %s/%s", worker_id, app_name, task_id)

            # Create a fresh memory manager per task (needs task instruction
            # from the trajectory metadata, but we may not have it yet — use
            # a placeholder; the replay runner will set_episode_state from
            # the trajectory metadata).
            memory = create_memory(args, task_instruction="")

            trajectory_dir = Path(args.replay_from)
            runner = ObservationReplayRunner(
                agent=agent,
                memory=memory,
                trajectory_dir=trajectory_dir,
                output_dir=str(Path(args.result_dir) / app_name),
                verbose=True,
            )

            try:
                r = runner.replay_episode(task_id)
                if r is not None:
                    result_list.append({
                        "example_id": task_id,
                        "app_name": app_name,
                        "success": r.baseline_success,
                        "reward": 1.0 if r.baseline_success else 0.0,
                        "steps": r.total_steps,
                        "compaction_steps": r.compaction_steps,
                        "divergence_steps": r.divergence_steps,
                        "divergence_rate": r.divergence_rate,
                        "compression_ratio": r.compression_ratio,
                        "policy_calls": r.policy_calls,
                        "wall_time_seconds": r.wall_time_seconds,
                    })
            except Exception as e:
                logger.exception("replay failed for %s/%s", app_name, task_id)
                result_list.append({
                    "example_id": task_id, "app_name": app_name,
                    "error": str(e), "reward": 0.0, "success": False,
                })
    finally:
        logger.info("replay worker %d done", worker_id)


# =============================================================================
# Orchestration
# =============================================================================

def _configure_logging(level_name: str, tag: str = "main") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = f"[%(asctime)s {tag} %(name)s %(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)


def main() -> None:
    args = config()
    _configure_logging(args.log_level)

    # ------------------------------------------------------------------
    # Replay mode: offline memory evaluation, no browser/server needed
    # ------------------------------------------------------------------
    if args.replay_from:
        replay_dir = Path(args.replay_from)
        if not replay_dir.exists():
            raise SystemExit(f"--replay_from directory does not exist: {replay_dir}")

        # In replay mode, task_ids come from the trajectory directory
        # (each subdirectory is a task_id), unless explicitly provided.
        if args.task_ids:
            task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        else:
            task_ids = sorted([
                d.name for d in replay_dir.iterdir()
                if d.is_dir() and (d / "trajectory.json").exists()
            ])

        if not task_ids:
            raise SystemExit(f"No trajectories found in {replay_dir}")

        logger.info(
            "REPLAY mode: %d task(s) from %s, memory=%s, policy=%s",
            len(task_ids), replay_dir, args.memory_model, args.policy_model,
        )

        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.result_dir) / "args.json").open("w") as f:
            json.dump({**vars(args), "task_ids": task_ids}, f, indent=2)

        mgr = mp.Manager()
        result_list = mgr.list()
        task_queue: "mp.Queue[tuple[str, str]]" = mgr.Queue()
        for tid in task_ids:
            task_queue.put((args.app_name, tid))

        args_dict = vars(args)
        start = time.time()
        processes: List[mp.Process] = []
        try:
            for i in range(args.workers):
                p = mp.Process(
                    target=run_replay_worker,
                    args=(i, task_queue, result_list, args_dict),
                    name=f"replay-worker-{i}",
                )
                p.daemon = False
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — terminating replay workers")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise
        finally:
            elapsed = time.time() - start
            results = list(result_list)
            _write_summary(args, results, elapsed)
        return

    # ------------------------------------------------------------------
    # Live mode: normal WebArena evaluation with browser
    # ------------------------------------------------------------------
    webarena_dir = _resolve_webarena_dir(args.webarena_dir)
    logger.info("using webarena-infinity at %s", webarena_dir)

    if args.list_memory_models:
        print("Registered memory models:")
        for n in sorted(list_memory_models()):
            print(f"  {n}")
        return

    if args.list_apps:
        from .server_pool import WebArenaServerPool
        pool = WebArenaServerPool(
            webarena_dir=webarena_dir,
            port_range=_parse_port_range(args.port_range),
        )
        print("Available webarena apps:")
        for name in pool.list_apps():
            print(f"  {name}")
        return

    # Resolve task ids
    task_ids = enumerate_task_ids(
        webarena_dir, args.app_name,
        requested=args.task_ids,
        task_filter=args.task_filter,
    )
    if not task_ids:
        raise SystemExit(
            f"No tasks found for app {args.app_name}. Check --task_ids or the app directory."
        )

    logger.info(
        "running %d task(s) from app %s with memory=%s workers=%d obs=%s",
        len(task_ids), args.app_name, args.memory_model, args.workers, args.observation_mode,
    )

    # Persist the args used for this run (debugging aid)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.result_dir) / "args.json").open("w") as f:
        json.dump({**vars(args), "task_ids": task_ids,
                   "webarena_dir": str(webarena_dir)}, f, indent=2)

    # Parallelize per-task via a multiprocessing queue
    port_start, port_end = _parse_port_range(args.port_range)
    per_worker = max(10, (port_end - port_start) // max(1, args.workers))

    mgr = mp.Manager()
    result_list = mgr.list()
    task_queue: "mp.Queue[tuple[str, str]]" = mgr.Queue()
    for tid in task_ids:
        task_queue.put((args.app_name, tid))

    args_dict = vars(args)
    webarena_dir_str = str(webarena_dir)

    start = time.time()
    processes: List[mp.Process] = []
    try:
        for i in range(args.workers):
            worker_port_start = port_start + i * per_worker
            worker_port_end = min(port_end, worker_port_start + per_worker)
            p = mp.Process(
                target=run_worker,
                args=(i, task_queue, result_list, args_dict, webarena_dir_str,
                      worker_port_start, worker_port_end),
                name=f"webarena-worker-{i}",
            )
            p.daemon = False
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — terminating workers")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        elapsed = time.time() - start
        results = list(result_list)
        _write_summary(args, results, elapsed)


def _write_summary(args: argparse.Namespace, results: List[Dict[str, Any]], elapsed: float) -> None:
    n = len(results)
    succ = sum(1 for r in results if r.get("success"))
    avg_steps = (sum(r.get("steps", 0) for r in results) / n) if n else 0

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "app_name": args.app_name,
        "policy_model": args.policy_model,
        "memory_model": args.memory_model,
        "observation_mode": args.observation_mode,
        "n_tasks": n,
        "n_success": succ,
        "success_rate": (succ / n) if n else 0,
        "avg_steps": avg_steps,
        "wall_time_seconds": round(elapsed, 1),
        "results": results,
    }
    summary_path = Path(args.result_dir) / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("summary: %d/%d success (%.1f%%), avg_steps=%.1f, %.1fs",
                succ, n, 100 * succ / max(1, n), avg_steps, elapsed)


if __name__ == "__main__":
    main()
