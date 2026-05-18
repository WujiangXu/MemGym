"""Split `reward_model_pairs_v2.jsonl` into `train.jsonl` + `eval.jsonl`.

Drives the assignment from the committed manifest at
`data/world_model/reward_model_v2_split.json`, so consumers who want the
splits as separate files (e.g. for tools that don't filter on a column)
get the exact same partition that the manifest certifies.

Usage:
    # default: split the v2 JSONL in-place under its own directory
    python3 src/memgym/training/scripts/split_jsonl_by_manifest.py

    # override paths
    python3 src/memgym/training/scripts/split_jsonl_by_manifest.py \
        --pairs   path/to/reward_model_pairs_v2.jsonl \
        --manifest path/to/reward_model_v2_split.json \
        --out-dir path/to/output_dir

Sanity check on completion: row counts must match `meta.row_counts` in
the manifest, otherwise we exit non-zero.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DEFAULT_PAIRS = REPO / "data" / "world_model" / "training_output" / "reward_model_v2" / "reward_model_pairs_v2.jsonl"
DEFAULT_MANIFEST = REPO / "data" / "world_model" / "reward_model_v2_split.json"
DEFAULT_OUT = DEFAULT_PAIRS.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",    type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    train_ids = set(manifest["splits"]["train"])
    eval_ids  = set(manifest["splits"]["eval"])
    expected = manifest["meta"]["row_counts"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    eval_path  = args.out_dir / "eval.jsonl"

    written: Counter = Counter()
    skipped_unknown = 0
    print(f"reading  {args.pairs}", flush=True)
    with args.pairs.open() as fin, train_path.open("w") as ftr, eval_path.open("w") as fev:
        for line in fin:
            row = json.loads(line)
            tid = row["trajectory_id"]
            if tid in train_ids:
                ftr.write(line); written["train"] += 1
            elif tid in eval_ids:
                fev.write(line); written["eval"] += 1
            else:
                skipped_unknown += 1

    print(f"  wrote train.jsonl: {written['train']:>6d}  (expected {expected['train']})")
    print(f"  wrote eval.jsonl : {written['eval']:>6d}  (expected {expected['eval']})")
    if skipped_unknown:
        print(f"  ⚠️  skipped {skipped_unknown} rows whose trajectory_id is in neither split")

    if written["train"] != expected["train"] or written["eval"] != expected["eval"]:
        print("  ❌ row-count mismatch — manifest and JSONL are out of sync")
        return 1
    print("  ✅ row counts match manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
