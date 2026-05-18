"""Quick length / coverage probe for any pairs JSONL.

Reports:
- file size, row count, avg bytes/row
- prompt char/token-approx percentiles
- fraction of rows under common context-length thresholds
- schema keys + sample first row's compaction-block fields

Used to verify whether the τ2 long-context build matches the v2 RM
training prompt distribution (~22-38K tokens) before scoring it with
``eval_aug_sft.py --max-length 32768``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--char-per-token", type=float, default=4.0,
                    help="Approx chars per token (Qwen3 BPE ~4 chars/tok).")
    ap.add_argument("--show-sample", action="store_true",
                    help="Print head/tail of first row's prompt.")
    args = ap.parse_args()

    if not args.path.is_file():
        print(f"missing: {args.path}", file=sys.stderr)
        return 2

    n = 0
    plens: List[int] = []
    n_compactions: List[int] = []
    summ_chars: List[int] = []
    n_msgs: List[int] = []
    label_counts: dict = {}
    src_counts: dict = {}
    domain_counts: dict = {}
    schema_keys = None
    sample = None

    with args.path.open() as fh:
        for line in fh:
            row = json.loads(line)
            n += 1
            if schema_keys is None:
                schema_keys = sorted(row.keys())
                sample = row
            plens.append(len(row.get(args.prompt_key) or ""))
            n_compactions.append(int(row.get("n_compactions_active") or 0))
            summ_chars.append(int(row.get("active_summary_chars") or 0))
            n_msgs.append(int(row.get("n_messages_in_view") or 0))
            lab = row.get("label")
            label_counts[lab] = label_counts.get(lab, 0) + 1
            src = row.get("source_model") or "?"
            src_counts[src] = src_counts.get(src, 0) + 1
            dom = row.get("domain") or "?"
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

    plens.sort()
    sc = sorted(summ_chars)
    nc = sorted(n_compactions)

    def pct(arr, p):
        return arr[int(p * (len(arr) - 1))] if arr else 0

    cpt = args.char_per_token
    print(f"file: {args.path}")
    print(f"size_bytes: {args.path.stat().st_size:,}  n_rows: {n}  avg_bytes_per_row: {args.path.stat().st_size // max(n,1):,}")

    print(f"\nprompt char-length:")
    for p in (0, 0.05, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
        v = pct(plens, p)
        print(f"  p{int(p*100):3d}  chars={v:>9,}  ~tokens={int(v/cpt):>7,}")

    # Coverage at common eval --max-length thresholds
    print(f"\ncoverage (≤ N tokens):")
    for cap in (1024, 4096, 8192, 16384, 32768):
        cap_chars = int(cap * cpt)
        n_at = sum(1 for x in plens if x <= cap_chars)
        print(f"  ≤{cap:>5,} tokens (~{cap_chars:,} chars): {n_at:>5,}/{n} = {n_at/max(n,1)*100:5.1f}%")

    print(f"\ncompaction signal:")
    print(f"  active_summary_chars  p50={pct(sc,.5):,}  p95={pct(sc,.95):,}  max={(sc[-1] if sc else 0):,}")
    print(f"  n_compactions_active  p50={pct(nc,.5)}  p95={pct(nc,.95)}  max={nc[-1] if nc else 0}")
    print(f"  n_messages_in_view    p50={pct(sorted(n_msgs),.5)}  p95={pct(sorted(n_msgs),.95)}  max={max(n_msgs) if n_msgs else 0}")
    print(f"  rows with active_summary_chars > 0:  {sum(1 for x in summ_chars if x>0):,}/{n}")
    print(f"  rows with n_compactions_active > 0:  {sum(1 for x in n_compactions if x>0):,}/{n}")

    print(f"\nlabel counts: {label_counts}")
    print(f"source_model counts ({len(src_counts)}): {dict(sorted(src_counts.items(), key=lambda kv: -kv[1]))}")
    print(f"domain counts: {dict(sorted(domain_counts.items(), key=lambda kv: -kv[1]))}")

    print(f"\nschema keys ({len(schema_keys or [])}): {schema_keys}")

    if args.show_sample and sample is not None:
        prm = sample.get(args.prompt_key) or ""
        print(f"\nfirst row prompt head (300 chars):\n{prm[:300]!r}")
        print(f"\nfirst row prompt tail (300 chars):\n{prm[-300:]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
