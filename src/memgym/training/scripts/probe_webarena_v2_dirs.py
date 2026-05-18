"""Discover the Prompt-Fix V2 sweep root on EC2.

The V2 sweep (April 13, 2026) ran 5 apps x 3 strategies:
  apps:       paypal, gmail, gitlab, linear, superhuman
  strategies: none, struct_ms10, summ_ms10

A directory qualifies as a V2 root when it contains cohort subdirs that match
at least one app AND at least two distinct strategies, with the additional
requirement that both _none AND (_struct_ms10 OR _summ_ms10) are present for
some app.

Algorithm:
  1. Walk ${HOME}/{MemGym/results,results}/webarena up to 2 levels deep.
  2. For every dir at depth 1 (candidate root), scan its immediate children
     looking for cohort dirs named  <app>_<strategy>[_suffix].
  3. Score each candidate. Emit a JSON report plus a flat tree for any
     directory that passes the V2 filter.

Read-only — no writes, no LLM calls. Submit via /run-script:

    {"module": "memgym.training.scripts.probe_webarena_v2_dirs",
     "args": [],
     "venv": "${VENV_DIR}",
     "cwd": "${REPO_ROOT}"}
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ── constants ────────────────────────────────────────────────────────────────

V2_APPS = {"paypal", "gmail", "gitlab", "linear", "superhuman"}
V2_STRATEGIES = {"none", "struct_ms10", "summ_ms10", "summ_ms15"}

# Directories that pre-date the V2 sweep or are known Phase-C / the dataset-augmentation phase'
# experiments. We still list them as fallback candidates if V2 is not found.
SKIP_NAMES: set[str] = set()  # intentionally empty — let the mtime filter do it

SEARCH_ROOTS = [
    Path("${REPO_ROOT}/results/webarena"),
    Path("${HOME}/results/webarena"),
    Path("${HOME}/results"),
]

# April 12, 2026 00:00 UTC  (V2 jobs ran April 13)
V2_CUTOFF_TS: float = datetime(2026, 4, 12).timestamp()


# ── helpers ──────────────────────────────────────────────────────────────────

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _mtime_str(p: Path) -> str:
    ts = _mtime(p)
    if ts == 0.0:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _infer_app(name: str) -> str | None:
    low = name.lower()
    for a in V2_APPS:
        if low.startswith(a) or f"_{a}" in low or f"-{a}" in low:
            return a
    return None


def _infer_strategy(name: str) -> str | None:
    """Return canonical strategy token from a cohort dir name."""
    low = name.lower()
    # struct first so "struct_ms10" doesn't accidentally match "summ"
    if "struct_ms10" in low or ("struct" in low and "ms10" in low):
        return "struct_ms10"
    if "struct_ms15" in low or ("struct" in low and "ms15" in low):
        return "struct_ms15"
    if "struct" in low:
        return "struct"
    if "summ_ms10" in low or ("summ" in low and "ms10" in low):
        return "summ_ms10"
    if "summ_ms15" in low or ("summ" in low and "ms15" in low):
        return "summ_ms15"
    if "summ" in low or "summariz" in low:
        return "summ"
    # _none must come after structured / summarizing checks
    low_parts = low.replace("-", "_").split("_")
    if "none" in low_parts or "baseline" in low_parts:
        return "none"
    return None


def _count_episodes(cohort_dir: Path) -> int:
    """Count leaf episodes: dirs that contain result.json or trajectory.json."""
    count = 0
    try:
        for entry in cohort_dir.iterdir():
            if not entry.is_dir():
                continue
            has_result = (entry / "result.json").exists()
            has_traj = (entry / "trajectory.json").exists()
            if has_result or has_traj:
                count += 1
    except OSError:
        pass
    return count


def _flat_tree(d: Path, depth: int = 2, _cur: int = 0) -> list[str]:
    lines: list[str] = []
    indent = "  " * _cur
    try:
        entries = sorted(d.iterdir(), key=lambda x: x.name)
    except OSError:
        return [f"{indent}<err reading {d}>"]
    for e in entries:
        sz_tag = ""
        if e.is_file():
            try:
                sz_tag = f"  ({e.stat().st_size:,}B)"
            except OSError:
                pass
        lines.append(f"{indent}{e.name}{'/' if e.is_dir() else sz_tag}")
        if e.is_dir() and _cur < depth:
            lines.extend(_flat_tree(e, depth, _cur + 1))
    return lines


def _analyze_candidate(root: Path) -> dict[str, Any] | None:
    """
    Inspect a candidate root dir (depth-1 under a search root).
    Returns a record if it contains any recognisable V2-style cohort dirs,
    or None if it looks completely unrelated.
    """
    try:
        children = sorted(root.iterdir(), key=lambda x: x.name)
    except OSError:
        return None

    # Map (app, strategy) -> cohort dir path
    cohorts: dict[tuple[str, str], Path] = {}
    other_dirs: list[str] = []

    for child in children:
        if not child.is_dir():
            continue
        app = _infer_app(child.name)
        strat = _infer_strategy(child.name)
        if app and strat:
            cohorts[(app, strat)] = child
        else:
            other_dirs.append(child.name)

    if not cohorts:
        return None

    apps_found = sorted({a for a, _ in cohorts})
    strats_found = sorted({s for _, s in cohorts})

    # V2 filter: must have _none AND at least one memory strategy
    has_none = "none" in strats_found
    has_mem = any(s in strats_found for s in ("struct_ms10", "summ_ms10",
                                               "struct_ms15", "summ_ms15",
                                               "struct", "summ"))

    # Episode counts per cohort
    n_episodes: dict[str, int] = {}
    for (app, strat), cdir in cohorts.items():
        key = f"{app}_{strat}"
        n_episodes[key] = _count_episodes(cdir)

    is_v2 = (
        has_none
        and has_mem
        and len(apps_found) >= 2
    )

    # Also accept if it has ≥ 2 distinct V2 apps and ≥ 2 V2 strategies
    v2_app_match = [a for a in apps_found if a in V2_APPS]
    v2_strat_match = [s for s in strats_found if s in V2_STRATEGIES]
    if len(v2_app_match) >= 2 and len(v2_strat_match) >= 2:
        is_v2 = True

    return {
        "root": str(root),
        "mtime": _mtime_str(root),
        "mtime_ts": _mtime(root),
        "apps": apps_found,
        "strategies": strats_found,
        "n_cohorts": len(cohorts),
        "n_episodes_per_cohort": n_episodes,
        "other_dirs": other_dirs,
        "is_v2_candidate": is_v2,
        "has_none": has_none,
        "has_memory_strategy": has_mem,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78, flush=True)
    print("probe_webarena_v2_dirs — searching for Prompt-Fix V2 sweep root", flush=True)
    print("=" * 78, flush=True)

    all_candidates: list[dict[str, Any]] = []
    all_dirs_seen: list[str] = []   # flat inventory for fallback

    for search_root in SEARCH_ROOTS:
        if not search_root.is_dir():
            print(f"\n[skip] not found: {search_root}", flush=True)
            continue

        print(f"\n[scan] {search_root}", flush=True)
        try:
            top_entries = sorted(search_root.iterdir(), key=lambda x: x.name)
        except OSError as e:
            print(f"  err: {e}", flush=True)
            continue

        for entry in top_entries:
            if not entry.is_dir():
                continue
            mt = _mtime_str(entry)
            all_dirs_seen.append(f"{entry}  [{mt}]")
            print(f"  {entry.name}  [{mt}]", flush=True)

            rec = _analyze_candidate(entry)
            if rec is not None:
                all_candidates.append(rec)
                flag = " *** V2 CANDIDATE ***" if rec["is_v2_candidate"] else ""
                print(f"    apps={rec['apps']}  strats={rec['strategies']}"
                      f"  cohorts={rec['n_cohorts']}{flag}", flush=True)

    # ── partition results ───────────────────────────────────────────────────
    v2_hits = [r for r in all_candidates if r["is_v2_candidate"]]
    other_hits = [r for r in all_candidates if not r["is_v2_candidate"]]

    # Sort V2 hits by mtime descending (most recent first)
    v2_hits.sort(key=lambda r: r["mtime_ts"], reverse=True)

    # ── JSON summary ────────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "v2_candidates": v2_hits,
        "other_candidates": other_hits,
        "all_dirs_inventory": all_dirs_seen,
    }
    print("\n" + "=" * 78, flush=True)
    print("JSON REPORT", flush=True)
    print("=" * 78, flush=True)
    print(json.dumps(output, indent=2), flush=True)

    # ── flat tree for V2 hits ───────────────────────────────────────────────
    if v2_hits:
        print("\n" + "=" * 78, flush=True)
        print("FLAT TREE — V2 candidates", flush=True)
        print("=" * 78, flush=True)
        for rec in v2_hits:
            root_path = Path(rec["root"])
            print(f"\n[tree] {root_path}  [{rec['mtime']}]", flush=True)
            for line in _flat_tree(root_path, depth=2):
                print(f"  {line}", flush=True)
    else:
        print("\n*** No V2 candidates found ***", flush=True)
        print("\nFallback: other structured candidates (may include Phase-C / the dataset-augmentation phase'):", flush=True)
        for rec in other_hits:
            print(f"  {rec['root']}  apps={rec['apps']}  strats={rec['strategies']}", flush=True)
        if not other_hits:
            print("  (none — all dirs are unstructured or empty)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
