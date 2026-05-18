"""Scan every τ2 run dir under a root and report which format each task
uses: rich (`0_training.json` + `0_replay.json`) vs flat (`trajectory.json`)
vs missing.

Used to determine whether ANY τ2 run on EC2 has the rich format that
would let the long-context builder populate per-step `active_summary`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def classify_task(task_dir: Path) -> str:
    if (task_dir / "0_training.json").is_file() and (task_dir / "0_replay.json").is_file():
        return "rich"
    if (task_dir / "trajectory.json").is_file():
        return "flat"
    return "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Parent dir containing the τ2 run subdirs.")
    ap.add_argument("--max-tasks-per-run", type=int, default=10,
                    help="Sample only up to N tasks per run (0=all).")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"missing root: {args.root}", file=sys.stderr)
        return 2

    summary: list = []
    for run in sorted(p for p in args.root.iterdir() if p.is_dir()):
        # Skip non-_both_ runs — they don't have the baseline+memory pair we need
        is_both = "_both_" in run.name
        # Walk one or two levels deep: <run>/{baseline,memory}/<domain>/<task>/
        per_run_counts = defaultdict(int)
        n_tasks_seen = 0
        for sub in run.iterdir():
            if not sub.is_dir():
                continue
            if sub.name not in ("baseline", "memory") and not is_both:
                # For non-_both_ runs, scan children directly
                if sub.is_dir():
                    for dom in sub.iterdir():
                        if dom.is_dir():
                            for task in dom.iterdir():
                                if task.is_dir():
                                    n_tasks_seen += 1
                                    if args.max_tasks_per_run and n_tasks_seen > args.max_tasks_per_run:
                                        break
                                    per_run_counts[classify_task(task)] += 1
                continue
            # _both_ format: <run>/{baseline,memory}/<domain>/<task>/
            for dom in sub.iterdir():
                if not dom.is_dir():
                    continue
                for task in dom.iterdir():
                    if not task.is_dir():
                        continue
                    n_tasks_seen += 1
                    if args.max_tasks_per_run and n_tasks_seen > args.max_tasks_per_run:
                        break
                    per_run_counts[classify_task(task)] += 1
                if args.max_tasks_per_run and n_tasks_seen > args.max_tasks_per_run:
                    break
            if args.max_tasks_per_run and n_tasks_seen > args.max_tasks_per_run:
                break

        summary.append({
            "run": run.name,
            "is_both": is_both,
            "sampled_tasks": n_tasks_seen,
            "format_counts": dict(per_run_counts),
        })

    # Aggregate: which runs have ANY rich tasks?
    rich_runs = [s for s in summary if s["format_counts"].get("rich", 0) > 0]
    flat_runs = [s for s in summary if s["format_counts"].get("flat", 0) > 0]

    print(f"Total runs scanned: {len(summary)}")
    print(f"Runs with ≥1 rich task: {len(rich_runs)}")
    print(f"Runs with ≥1 flat task: {len(flat_runs)}\n")

    print("=== RUNS WITH RICH FORMAT ===")
    for s in rich_runs:
        print(f"  [{'_both_' if s['is_both'] else 'single'}] {s['run']}")
        print(f"     sampled {s['sampled_tasks']} tasks: {s['format_counts']}")
    if not rich_runs:
        print("  (none)")

    print("\n=== RUNS WITH FLAT FORMAT (sample of first 10) ===")
    for s in flat_runs[:10]:
        print(f"  [{'_both_' if s['is_both'] else 'single'}] {s['run']}")
        print(f"     sampled {s['sampled_tasks']} tasks: {s['format_counts']}")

    print("\n=== RUNS WITH NEITHER (missing data) ===")
    missing_runs = [s for s in summary
                    if not s["format_counts"].get("rich") and not s["format_counts"].get("flat")]
    for s in missing_runs[:20]:
        print(f"  {s['run']}: {s['format_counts']}  sampled={s['sampled_tasks']}")
    if len(missing_runs) > 20:
        print(f"  ... and {len(missing_runs) - 20} more")

    print(f"\nFull summary as JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
