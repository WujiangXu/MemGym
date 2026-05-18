"""Concatenate JSONL files into one output, preserving line order.

Used by Plan v7 Step 2 to merge the two WebArena V2 OOD pair files
(grid_idfix + prompt_fix_v2 fallback) into one canonical
reward_model_pairs_webarena_v2_combined.jsonl for downstream eval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with args.out.open("w") as out:
        for src in args.inputs:
            if not src.is_file():
                print(f"MISSING: {src}", file=sys.stderr)
                continue
            with src.open() as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out.write(line + "\n")
                    n_total += 1
            print(f"  appended: {src}")
    print(f"\nwrote {n_total} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
