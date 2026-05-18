"""Plan v3 the baseline-train phase.2/B0.3 probe: inventory τ2-bench + WebArena dirs on EC2.

τ2 run dirs are named:
    YYYYMMDD-HHMMSS_<domain>_<mode>_tau2_<strategy>_ms<X>kf<Y>kl<Z>r<W>_<tag>

  domain   ∈ {airline, retail, telecom, mock, all}
  mode     ∈ {baseline, memory, both}
  strategy ∈ {summarizing, structured}

We need (baseline_dir, memory_dir) pairs that share the same domain +
generation config so the memory-gain label is a clean A/B. Output:

    {
      "tau2": {
        "<domain>__<strategy>__<config>": {
          "baseline": [<dir>...],
          "memory":   [<dir>...],
          "both":     [<dir>...],
        }, ...
      },
      "webarena": {"layout": <find tree>, "n_files": <int>}
    }

Read-only. Submit via /run-script.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TAU2_ROOT = Path("${REPO_ROOT}/results/tau2_bench")
WA_ROOT = Path("${REPO_ROOT}/results/webarena")

# 20260409-213006_all_both_tau2_summarizing_ms30kf1kl4r0.6_T2_pilot
# 20260413-220800_all_both_tau2_structured_ms10kf1kl4r0.6_HP_train_struct_ms10_kl4
TAU2_RE = re.compile(
    r"^(?P<ts>\d{8}-\d{6})_"
    r"(?P<domain>airline|retail|telecom|mock|all)_"
    r"(?P<mode>baseline|memory|both)_"
    r"tau2_(?P<strategy>summarizing|structured)_"
    r"(?P<config>ms\d+kf\d+kl\d+r[\d.]+)"
    r"(?:_(?P<tag>.+))?$"
)


def _count_traj_subdirs(p: Path) -> int:
    """A trajectory dir typically holds per-instance subdirs."""
    if not p.is_dir():
        return 0
    n = 0
    try:
        for c in p.iterdir():
            if c.is_dir():
                n += 1
    except (PermissionError, OSError):
        pass
    return n


def _du(p: Path) -> int:
    try:
        out = subprocess.run(
            ["du", "-sb", str(p)], capture_output=True, text=True, timeout=20,
        )
        return int(out.stdout.split()[0]) if out.returncode == 0 else -1
    except Exception:
        return -1


def _scan_tau2() -> dict:
    if not TAU2_ROOT.exists():
        return {"exists": False}
    by_key: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"baseline": [], "memory": [], "both": []}
    )
    unparsed: list[str] = []
    for child in sorted(TAU2_ROOT.iterdir()):
        if not child.is_dir():
            continue
        m = TAU2_RE.match(child.name)
        if not m:
            unparsed.append(child.name)
            continue
        key = f"{m.group('domain')}__{m.group('strategy')}__{m.group('config')}"
        by_key[key][m.group("mode")].append({
            "name": child.name,
            "ts": m.group("ts"),
            "tag": m.group("tag"),
            "n_subdirs": _count_traj_subdirs(child),
            "size_bytes": _du(child),
        })
    # surface only keys with both a baseline and a memory side
    pairable = {
        k: v for k, v in by_key.items()
        if v["baseline"] and (v["memory"] or v["both"])
    }
    return {
        "exists": True,
        "n_keys_total": len(by_key),
        "n_keys_pairable": len(pairable),
        "pairable": pairable,
        "all_keys": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in by_key.items()},
        "unparsed": unparsed,
    }


def _scan_webarena() -> dict:
    if not WA_ROOT.exists():
        return {"exists": False}
    out: dict = {"exists": True, "size_bytes": _du(WA_ROOT)}
    # full file tree, max depth 4
    layout: list[str] = []
    n_files = 0
    for dirpath, dirs, files in os.walk(WA_ROOT):
        rel = os.path.relpath(dirpath, WA_ROOT)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            dirs.clear()
            continue
        layout.append(f"{rel}: {len(dirs)} dirs, {len(files)} files")
        n_files += len(files)
        if len(layout) > 200:
            layout.append("... (truncated)")
            break
    out["n_files_total"] = n_files
    out["layout"] = layout
    # also probe the broader filesystem for "infinity" / "wa_" / "web-" outside this root
    try:
        find = subprocess.run(
            [
                "find", "${HOME}", "-maxdepth", "6", "-type", "d",
                "(",
                "-iname", "*infinity*", "-o", "-iname", "*wa_*",
                "-o", "-iname", "*web-*", "-o", "-iname", "*webarena*",
                ")",
            ],
            capture_output=True, text=True, timeout=120,
        )
        out["broad_find"] = [l for l in find.stdout.splitlines() if l.strip()]
    except Exception as e:
        out["broad_find"] = [f"<find error: {e}>"]
    return out


def main() -> int:
    print(json.dumps({
        "tau2": _scan_tau2(),
        "webarena": _scan_webarena(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
