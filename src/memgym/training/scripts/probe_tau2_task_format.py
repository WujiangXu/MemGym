"""Dump the schema of a single τ² task dir to figure out where the
compaction summary actually lives. The inventory probe assumed
``0_training.json::steps[].memory.summary`` but found 0 compactions
across 1k+ tasks — schema mismatch likely.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("${REPO_ROOT}/results/tau2_bench")
_PROD_RUN_PREFIXES = (
    "20260412-024137_all_both_tau2_summarizing",  # HP_full_summ_ms10  (288)
    "20260412-213744_all_both_tau2_structured",   # HP_full_struct_ms10 (288)
    "20260413-220758_all_both_tau2_summarizing",  # HP_train_summ      (178)
    "20260413-220800_all_both_tau2_structured",   # HP_train_struct    (178)
)


def _peek(d: dict, max_keys: int = 8) -> str:
    if not isinstance(d, dict):
        return f"<{type(d).__name__}>"
    keys = list(d.keys())[:max_keys]
    return "{" + ", ".join(keys) + ("...}" if len(d) > max_keys else "}")


def main() -> int:
    # Prioritize production runs over smoke/mock dirs.
    out: dict = {"runs_inspected": []}
    all_runs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    prod_runs = [r for r in all_runs if any(r.name.startswith(p) for p in _PROD_RUN_PREFIXES)]
    other_runs = [r for r in all_runs if r not in prod_runs]
    for run in prod_runs + other_runs:
        mem = run / "memory"
        if not mem.is_dir():
            continue
        for dom in mem.iterdir():
            if not dom.is_dir():
                continue
            for task in dom.iterdir():
                if not task.is_dir():
                    continue
                # List task contents
                contents = sorted(p.name for p in task.iterdir())
                training_path = task / "0_training.json"
                if not training_path.is_file():
                    out["runs_inspected"].append({
                        "run": run.name,
                        "task_path": str(task),
                        "contents": contents,
                        "missing": "0_training.json",
                    })
                    if len(out["runs_inspected"]) >= 4:
                        print(json.dumps(out, indent=2))
                        return 0
                    continue
                try:
                    with training_path.open() as fh:
                        td = json.load(fh)
                except Exception as e:
                    out["runs_inspected"].append({"run": run.name, "task": task.name, "err": str(e)})
                    continue
                steps = td.get("steps") or []
                step_keys_sample = list((steps[0] or {}).keys())[:20] if steps else []
                # Look at one step's memory dict
                first_mem = steps[0].get("memory") if steps else None
                first_mem_keys = list(first_mem.keys()) if isinstance(first_mem, dict) else None
                # Hunt for a summary across steps
                summary_locations = []
                for i, s in enumerate(steps):
                    if not isinstance(s, dict):
                        continue
                    for k, v in s.items():
                        if "summary" in k.lower() or "compact" in k.lower():
                            summary_locations.append({"step": i, "key": k, "type": type(v).__name__,
                                                      "preview": str(v)[:80]})
                    mem = s.get("memory")
                    if isinstance(mem, dict):
                        for mk, mv in mem.items():
                            if "summary" in mk.lower() or "compact" in mk.lower() or "view" in mk.lower():
                                if isinstance(mv, str) and mv:
                                    summary_locations.append({"step": i, "key": f"memory.{mk}",
                                                              "len": len(mv), "preview": mv[:80]})
                                elif isinstance(mv, (list, dict)) and mv:
                                    summary_locations.append({"step": i, "key": f"memory.{mk}",
                                                              "type": type(mv).__name__})
                out["runs_inspected"].append({
                    "run": run.name,
                    "task_path": str(task),
                    "contents": contents,
                    "training_top_keys": list(td.keys())[:20],
                    "n_steps": len(steps),
                    "step0_keys": step_keys_sample,
                    "step0_memory_keys": first_mem_keys,
                    "summary_locations_first5": summary_locations[:5],
                    "n_summary_hits": len(summary_locations),
                })
                if len(out["runs_inspected"]) >= 3:
                    print(json.dumps(out, indent=2))
                    return 0
                break
            break
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
