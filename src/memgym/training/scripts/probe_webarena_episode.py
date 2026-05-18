"""Deeper WebArena schema probe — inspect inside the `episodes` array.

Outer probe (`probe_webarena_schema.py`) found phase_d_prime files have
`{num_episodes, total_steps, config, episodes, metadata}`. Now we need
the *turn-level* schema so we can write a `WebArenaLoadedTrajectory`
loader and a `webarena_long_context_pairs.py` builder targeting
`phase_d_prime/paypal_none` (baseline) vs `paypal_struct_ms10`
(OOD) cohorts.

Reports for one representative file per cohort:
  - len(episodes)
  - keys of episodes[0]
  - keys of episodes[0]["actions"|"steps"|"trajectory"|...] sample
  - reward / score field locations
  - instance/task identifier (so we can join baseline ↔ OOD by task_id)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("${REPO_ROOT}/results/webarena/phase_d_prime")

CANDIDATES = [
    ROOT / "paypal_none",
    ROOT / "paypal_struct_ms10",
    ROOT / "paypal_summ_ms15",
]


def _shape(v: Any, depth: int = 0) -> Any:
    """Return a compact shape hint for arbitrary JSON values."""
    if depth > 3:
        return "..."
    if isinstance(v, dict):
        return {k: _shape(vv, depth + 1) for k, vv in list(v.items())[:12]}
    if isinstance(v, list):
        if not v:
            return []
        return [_shape(v[0], depth + 1), f"len={len(v)}"]
    if isinstance(v, str):
        return f"str(len={len(v)})"
    return type(v).__name__


def _inspect_episode(ep: dict) -> dict:
    out: dict = {"keys": list(ep.keys())[:30]}
    # Common reward / id fields
    for k in ("reward", "score", "success", "task_id", "task", "intent",
              "config_file", "instance_id", "id"):
        if k in ep:
            v = ep[k]
            out[f"f_{k}"] = (
                v if not isinstance(v, (dict, list)) else
                f"{type(v).__name__}(len={len(v) if hasattr(v,'__len__') else '?'})"
            )
    # Trajectory body candidates
    for k in ("trajectory", "actions", "messages", "steps", "turns",
              "history", "observations"):
        if k in ep and isinstance(ep[k], list):
            arr = ep[k]
            out[f"len_{k}"] = len(arr)
            if arr:
                first = arr[0]
                if isinstance(first, dict):
                    out[f"first_{k}_keys"] = list(first.keys())[:20]
                    # dump shapes for one sample turn
                    out[f"first_{k}_shape"] = _shape(first)
                else:
                    out[f"first_{k}_type"] = type(first).__name__
    # Memory / compaction signals (the field we anchor pair-extraction on)
    for k in ("compactions", "compaction_events", "memory_state",
              "summary", "n_compactions"):
        if k in ep:
            out[f"has_{k}"] = type(ep[k]).__name__
    return out


def _summarize_one(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return {"path": str(p), "error": str(e)}

    summary: dict = {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "top_level_keys": list(d.keys())[:30] if isinstance(d, dict) else None,
    }
    if not isinstance(d, dict):
        return summary

    # Top-level config / metadata snapshot — helps us see the memory strategy
    for k in ("config", "metadata"):
        if k in d and isinstance(d[k], dict):
            summary[f"top_{k}"] = {kk: _shape(vv, 1) for kk, vv in list(d[k].items())[:12]}

    eps = d.get("episodes")
    if isinstance(eps, list):
        summary["n_episodes"] = len(eps)
        if eps:
            summary["episode0"] = _inspect_episode(eps[0])
            # Mid sample to confirm consistency
            if len(eps) > 1:
                summary["episode_mid"] = _inspect_episode(eps[len(eps) // 2])
    return summary


def main() -> int:
    out: dict = {"cohorts": []}
    for cand in CANDIDATES:
        cohort: dict = {"path": str(cand), "exists": cand.is_dir()}
        if cand.is_dir():
            jsons = sorted(p for p in cand.rglob("*.json") if p.stat().st_size > 50_000)
            cohort["n_big_jsons"] = len(jsons)
            if jsons:
                cohort["sample"] = _summarize_one(jsons[0])
        out["cohorts"].append(cohort)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
