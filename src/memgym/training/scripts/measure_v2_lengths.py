"""Measure tokenized prompt-length distribution of the v2 RM JSONL.

Tokenizes every row's `prompt` with the Qwen tokenizer and reports:
  - raw char length distribution (cheap, no model load)
  - tokenized length distribution (using the actual training tokenizer)
  - share truncated at each candidate --max-length cutoff
  - per-split breakdown so we know how many EVAL rows would lose info

Run via:
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PAIRS = Path(
    "data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl"
)
CANDIDATE_LENGTHS = [4096, 8192, 12288, 16384, 24576, 32768]


def percentile(sorted_xs, q):
    if not sorted_xs:
        return 0
    idx = max(0, min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, sample this many rows uniformly. 0 = all rows.")
    args = ap.parse_args()

    if not args.pairs.exists():
        print(f"ERROR: pairs file missing: {args.pairs}", file=sys.stderr)
        return 1

    print(f"[load] {args.pairs}", flush=True)
    rows = []
    with args.pairs.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[load] {len(rows):,} rows", flush=True)

    if args.sample and args.sample < len(rows):
        # Stratified by split.
        train_rows = [r for r in rows if r.get("split") == "train"]
        eval_rows = [r for r in rows if r.get("split") == "eval"]
        n_train = int(args.sample * len(train_rows) / len(rows))
        n_eval = args.sample - n_train
        step_t = max(1, len(train_rows) // max(1, n_train))
        step_e = max(1, len(eval_rows) // max(1, n_eval))
        rows = train_rows[::step_t][:n_train] + eval_rows[::step_e][:n_eval]
        print(f"[sample] using {len(rows):,} rows", flush=True)

    print(f"[tokenize] loading {args.model} tokenizer", flush=True)
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    by_split: dict[str, list[int]] = {"train": [], "eval": []}
    char_by_split: dict[str, list[int]] = {"train": [], "eval": []}
    for i, r in enumerate(rows):
        prompt = r.get("prompt", "")
        split = r.get("split", "train")
        if split not in by_split:
            split = "train"
        char_by_split[split].append(len(prompt))
        n_tok = len(tok.encode(prompt, add_special_tokens=False))
        by_split[split].append(n_tok)
        if (i + 1) % 1000 == 0:
            print(f"  tokenized {i+1:,}/{len(rows):,}", flush=True)

    print("\n=== Char length distribution ===", flush=True)
    for split, vals in char_by_split.items():
        if not vals:
            continue
        s = sorted(vals)
        print(f"  {split:5s} n={len(vals):5d} "
              f"p10={percentile(s, 0.1):,} p50={percentile(s, 0.5):,} "
              f"p90={percentile(s, 0.9):,} p99={percentile(s, 0.99):,} "
              f"max={max(vals):,}")

    print("\n=== Token length distribution ===", flush=True)
    for split, vals in by_split.items():
        if not vals:
            continue
        s = sorted(vals)
        print(f"  {split:5s} n={len(vals):5d} "
              f"p10={percentile(s, 0.1):,} p50={percentile(s, 0.5):,} "
              f"p90={percentile(s, 0.9):,} p99={percentile(s, 0.99):,} "
              f"max={max(vals):,} mean={sum(vals)//len(vals):,}")

    print("\n=== Truncation share by --max-length ===", flush=True)
    print(f"  {'cutoff':>8s}   {'train_trunc%':>13s}   {'eval_trunc%':>12s}",
          flush=True)
    for cut in CANDIDATE_LENGTHS:
        for split in ("train", "eval"):
            pass
        t_trunc = sum(1 for v in by_split["train"] if v > cut) / max(1, len(by_split["train"]))
        e_trunc = sum(1 for v in by_split["eval"] if v > cut) / max(1, len(by_split["eval"]))
        print(f"  {cut:>8d}   {100*t_trunc:>12.1f}%   {100*e_trunc:>11.1f}%",
              flush=True)

    print("\n=== Recommended max-length cells ===", flush=True)
    print("  Pick the smallest cutoff that keeps eval_trunc% under your")
    print("  tolerance. v2 datacard floor is 8192; preferred 16384.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
