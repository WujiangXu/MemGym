"""One-shot inventory of EC2 run artifacts the world-model pipeline depends on.

Lists sizes and file counts for three artifact classes, so a download
plan can be made with knowledge of the bandwidth cost:

1. `trajectories/` — raw baseline + memory-replay + fork agent runs.
   These are the upstream of everything; each pair in aug_sft_pairs.jsonl
   was carved from a (recorded_action, proposed_action) at some step
   within one of these trajectories.
2. `augmentation_output/` — `pairs_partial.jsonl` + `augmentation_pairs.json`
   per teacher. The distilled pair rows the trainer consumes. Small.
3. `results/` top-level — evaluation artifacts per named run
   (`results/<run_name>/*.json`). Moderate size.

Emits one JSON line per dir (so the downloader can stream-parse) plus
a final `summary` line with totals.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def walk_sizes(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    rows = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        n_files = 0
        size = 0
        for dp, _, fs in os.walk(entry):
            for f in fs:
                try:
                    size += os.path.getsize(os.path.join(dp, f))
                    n_files += 1
                except OSError:
                    pass
        rows.append({
            "class": root.name,
            "dir": entry.name,
            "bytes": size,
            "mb": round(size / 1e6, 1),
            "files": n_files,
        })
    return rows


def main() -> int:
    roots = ["trajectories", "augmentation_output", "results"]
    all_rows: list[dict] = []
    for r in roots:
        rows = walk_sizes(Path(r))
        for row in rows:
            print(json.dumps(row))
        all_rows.extend(rows)

    by_class: dict[str, dict] = {}
    for row in all_rows:
        c = row["class"]
        d = by_class.setdefault(c, {"dirs": 0, "bytes": 0, "files": 0})
        d["dirs"] += 1
        d["bytes"] += row["bytes"]
        d["files"] += row["files"]
    total_bytes = sum(d["bytes"] for d in by_class.values())
    summary = {
        "summary": True,
        "by_class": {c: {**v, "gb": round(v["bytes"] / 1e9, 2)} for c, v in by_class.items()},
        "total_gb": round(total_bytes / 1e9, 2),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
