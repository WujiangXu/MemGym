"""Inventory raw trajectories on EC2 for τ²/WebArena long-context rebuild.

For each candidate run dir under ``${REPO_ROOT}/results/{tau2_bench,webarena}``:
  - list paired tasks (baseline ∩ memory),
  - read ``0_training.json::steps[].memory`` (τ²) or
    ``<step>/training_step.json::memory`` (WebArena) to find the first
    step where ``summary`` is non-empty (the first-compaction step k*),
  - report per-run stats:
      n_tasks_paired, n_with_compaction (k* != None),
      median k*, max steps, frac that would survive the gate.

Run via ``POST /run-script`` with module=
  ``memgym.training.scripts.inventory_longctx_sources``.
Output is a single JSON object printed to stdout — captured by the
launch.py job log; the agent reads it back to plan rebuild slate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _first_compaction_step_tau2(task_dir: Path) -> Tuple[Optional[int], int, int]:
    """Return (first_compact_k, n_steps, n_compactions) for one τ² task dir.

    Reads ``0_training.json::steps[].memory.summary``. Returns
    (None, n_steps, 0) if the episode never compacted.
    """
    training = _read_json(task_dir / "0_training.json")
    if training is None:
        return None, 0, 0
    steps = training.get("steps") or []
    first_k: Optional[int] = None
    n_compact = 0
    for s in steps:
        try:
            k = int(s.get("step", -1))
        except (TypeError, ValueError):
            continue
        mem = s.get("memory") or {}
        summ = (mem.get("summary") or "").strip()
        if summ:
            n_compact += 1
            if first_k is None:
                first_k = k
    return first_k, len(steps), n_compact


def _inventory_tau2(root: Path) -> Dict[str, Any]:
    """Walk ``<tau2_root>/<run>/{baseline,memory}/<domain>/<task>/``."""
    if not root.is_dir():
        return {"error": "missing_root", "root": str(root)}
    out: Dict[str, Any] = {"root": str(root), "runs": []}
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        base = run / "baseline"
        mem = run / "memory"
        if not base.is_dir() or not mem.is_dir():
            continue

        per_run: Dict[str, Any] = {
            "name": run.name,
            "domains": {},
            "n_paired": 0,
            "n_compacted": 0,
            "first_k_p50": None,
            "first_k_max": None,
            "n_steps_p50": None,
        }
        ks: List[int] = []
        ns: List[int] = []
        for dom_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            domain = dom_dir.name
            mem_dom = mem / domain
            if not mem_dom.is_dir():
                continue
            base_tids = {p.name for p in dom_dir.iterdir() if p.is_dir()}
            mem_tids = {p.name for p in mem_dom.iterdir() if p.is_dir()}
            paired = sorted(base_tids & mem_tids)
            n_paired = len(paired)
            n_compact_paired = 0
            for tid in paired:
                # We use the OOD/memory side to decide first_compact_k —
                # that's the side whose triggers gate the rebuild.
                k_star, n_steps, n_c = _first_compaction_step_tau2(mem_dom / tid)
                if n_steps > 0:
                    ns.append(n_steps)
                if k_star is not None:
                    ks.append(k_star)
                    n_compact_paired += 1
            per_run["domains"][domain] = {
                "n_paired": n_paired,
                "n_compacted": n_compact_paired,
            }
            per_run["n_paired"] += n_paired
            per_run["n_compacted"] += n_compact_paired

        if ks:
            per_run["first_k_p50"] = statistics.median(ks)
            per_run["first_k_max"] = max(ks)
        if ns:
            per_run["n_steps_p50"] = statistics.median(ns)
        out["runs"].append(per_run)
    return out


def _first_compaction_step_webarena(task_dir: Path) -> Tuple[Optional[int], int, int]:
    """WebArena format: ``<task>/step_<NN>/training_step.json`` per step.

    Some legacy runs use ``<task>/training.json`` with embedded
    ``steps[].memory``; we try both. Returns (first_k, n_steps, n_c).
    """
    # New per-step format
    step_dirs = sorted(p for p in task_dir.iterdir()
                       if p.is_dir() and p.name.startswith("step_"))
    if step_dirs:
        first_k: Optional[int] = None
        n_c = 0
        for sd in step_dirs:
            try:
                k = int(sd.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            ts = _read_json(sd / "training_step.json")
            if ts is None:
                continue
            mem = ts.get("memory") or {}
            summ = (mem.get("summary") or "").strip()
            if summ:
                n_c += 1
                if first_k is None:
                    first_k = k
        return first_k, len(step_dirs), n_c

    # Legacy single-file format
    training = _read_json(task_dir / "training.json")
    if training is None:
        training = _read_json(task_dir / "0_training.json")
    if training is None:
        return None, 0, 0
    steps = training.get("steps") or []
    first_k: Optional[int] = None
    n_c = 0
    for s in steps:
        try:
            k = int(s.get("step", -1))
        except (TypeError, ValueError):
            continue
        mem = s.get("memory") or {}
        summ = (mem.get("summary") or "").strip()
        if summ:
            n_c += 1
            if first_k is None:
                first_k = k
    return first_k, len(steps), n_c


def _inventory_webarena(root: Path) -> Dict[str, Any]:
    """Walk ``<webarena_root>/<phase>/<cohort>/<task>/``.

    Cohorts have names like ``gitlab_struct_ms10``, ``paypal_summ_ms15``,
    ``gmail_none``, etc. The "task" dirname looks like ``run_<id>`` or
    ``<task_id>``; we treat any subdir as a task.
    """
    if not root.is_dir():
        return {"error": "missing_root", "root": str(root)}
    out: Dict[str, Any] = {"root": str(root), "phases": []}
    for phase in sorted(p for p in root.iterdir() if p.is_dir()):
        per_phase: Dict[str, Any] = {"name": phase.name, "cohorts": []}
        for cohort in sorted(p for p in phase.iterdir() if p.is_dir()):
            tasks = [p for p in cohort.iterdir() if p.is_dir()]
            if not tasks:
                continue
            ks: List[int] = []
            ns: List[int] = []
            n_compacted = 0
            for t in tasks:
                k, n, _ = _first_compaction_step_webarena(t)
                if n > 0:
                    ns.append(n)
                if k is not None:
                    ks.append(k)
                    n_compacted += 1
            per_phase["cohorts"].append({
                "name": cohort.name,
                "n_tasks": len(tasks),
                "n_compacted": n_compacted,
                "first_k_p50": statistics.median(ks) if ks else None,
                "first_k_max": max(ks) if ks else None,
                "n_steps_p50": statistics.median(ns) if ns else None,
            })
        if per_phase["cohorts"]:
            out["phases"].append(per_phase)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path,
                        default=Path("${REPO_ROOT}/results/tau2_bench"))
    parser.add_argument("--webarena-root", type=Path,
                        default=Path("${REPO_ROOT}/results/webarena"))
    parser.add_argument("--skip-webarena", action="store_true",
                        help="Skip WebArena scan if too slow.")
    parser.add_argument("--skip-tau2", action="store_true")
    args = parser.parse_args()

    out: Dict[str, Any] = {}
    if not args.skip_tau2:
        out["tau2"] = _inventory_tau2(args.tau2_root)
    if not args.skip_webarena:
        out["webarena"] = _inventory_webarena(args.webarena_root)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
