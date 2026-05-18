"""
One-shot helper that walks the upstream webarena-infinity clone and reports
per-app task counts by difficulty tier.

Why a dedicated module: we kept getting the hard-tier counts wrong from
memory (claimed "3 x 20 = 60 per env" — true for some apps but several
have 80-200). This reads the source of truth (`apps/*/real-tasks.json`)
and prints one row per app.

Usage:

    python -m memgym.gym.webarena.dataset_stats [--webarena-dir PATH]

Output columns: `app_name`, `easy`, `medium`, `hard`, `total`, sorted by
descending hard-tier count. Also writes `dataset_stats.json` to the
current working directory when `--json-out` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _count_tasks(real_tasks_path: Path) -> Dict[str, int]:
    with real_tasks_path.open() as f:
        payload = json.load(f)
    tasks: List[dict] = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    counts = {"easy": 0, "medium": 0, "hard": 0, "other": 0}
    for t in tasks:
        tid = t.get("id", "")
        if tid.startswith("task_e"):
            counts["easy"] += 1
        elif tid.startswith("task_m"):
            counts["medium"] += 1
        elif tid.startswith("task_h"):
            counts["hard"] += 1
        else:
            counts["other"] += 1
    counts["total"] = sum(counts.values())
    return counts


def _default_webarena_dir() -> Path:
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
        "Could not locate webarena-infinity clone. Pass --webarena-dir."
    )


def collect(webarena_dir: Path) -> List[Dict[str, object]]:
    apps_dir = webarena_dir / "apps"
    if not apps_dir.exists():
        raise FileNotFoundError(f"no apps/ dir under {webarena_dir}")

    rows: List[Dict[str, object]] = []
    for app_dir in sorted(apps_dir.iterdir()):
        real = app_dir / "real-tasks.json"
        if not real.exists():
            continue
        counts = _count_tasks(real)
        rows.append({"app": app_dir.name, **counts})

    rows.sort(key=lambda r: r["hard"], reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webarena-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    webarena_dir = args.webarena_dir or _default_webarena_dir()
    rows = collect(webarena_dir)

    header = f"{'app':<40} {'easy':>5} {'medium':>7} {'hard':>5} {'total':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['app']:<40} {r['easy']:>5} {r['medium']:>7} {r['hard']:>5} {r['total']:>6}")

    totals = {
        "easy": sum(r["easy"] for r in rows),
        "medium": sum(r["medium"] for r in rows),
        "hard": sum(r["hard"] for r in rows),
        "total": sum(r["total"] for r in rows),
    }
    print("-" * len(header))
    print(f"{'TOTAL':<40} {totals['easy']:>5} {totals['medium']:>7} {totals['hard']:>5} {totals['total']:>6}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"rows": rows, "totals": totals}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
