"""One-off diagnostic: print per-split char-length and token-length stats
for the rm_v2 pairs jsonl, plus the eval_results.json truncation block.

Resolves the discrepancy where eval_aug_sft.py's truncation report on the
QLoRA-32K retrain showed `prompt_length_max=314` while local file
inspection shows eval-split prompts are 73K chars at p50. We need to
know which file the EC2 eval actually consumed and whether the tokenizer
saw something unexpected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(arr: list[int], q: float) -> int:
    if not arr:
        return 0
    arr = sorted(arr)
    idx = min(len(arr) - 1, int(round(q * (len(arr) - 1))))
    return int(arr[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval-results", default=None,
                    help="Optional eval_results.json to print truncation block from.")
    ap.add_argument("--max-rows", type=int, default=20000)
    args = ap.parse_args()

    p = Path(args.pairs)
    print(f"file: {p}")
    print(f"size_bytes: {p.stat().st_size}")

    splits: dict[str, list[int]] = {}
    keys_seen: list[str] | None = None
    with p.open() as fh:
        for i, line in enumerate(fh):
            if i >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if keys_seen is None:
                keys_seen = sorted(r.keys())
            s = r.get("split", "train")
            splits.setdefault(s, []).append(len(r.get("prompt") or ""))

    print(f"keys_on_first_row: {keys_seen}")
    for s, lens in splits.items():
        print(
            f"split={s} n={len(lens):<6} "
            f"prompt_chars p50={_pct(lens, 0.50):>7} "
            f"p95={_pct(lens, 0.95):>7} max={max(lens) if lens else 0:>7}"
        )

    if args.eval_results:
        er = json.loads(Path(args.eval_results).read_text())
        print("\neval_results.truncation:")
        print(json.dumps(er.get("truncation", {}), indent=2))
        print("eval_results.pairs:", er.get("pairs"))
        print("eval_results.n_eval_rows:", er.get("n_eval_rows"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
