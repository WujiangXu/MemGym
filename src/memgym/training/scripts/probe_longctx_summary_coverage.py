"""Probe how many rows in a long-context pairs JSONL have an active
compaction summary in their reconstructed view (i.e. genuinely contain
all three components: history, task description, AND compressed memory).

The training distribution always has compressed memory active in the
view; if a long-context OOD eval pair file emits rows where every
``n_compactions_active == 0``, the RM is being scored on prompts that
match the SHAPE but not the COMPONENT MIX of training, and the eval
is structurally biased.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    if not args.pairs.is_file():
        print(json.dumps({"error": "missing", "path": str(args.pairs)}))
        return 2

    n = 0
    n_summ = 0
    n_summ_ge_2 = 0
    n_view_gt1 = 0
    char_lens = []
    summ_chars = []
    by_source: dict = {}

    with args.pairs.open() as fh:
        for line in fh:
            row = json.loads(line)
            n += 1
            n_compact = int(row.get("n_compactions_active", 0))
            n_view = int(row.get("n_messages_in_view", 0))
            summ_c = int(row.get("active_summary_chars", 0))
            char_lens.append(len(row.get("prompt") or ""))
            if n_compact > 0:
                n_summ += 1
                summ_chars.append(summ_c)
            if n_compact >= 2:
                n_summ_ge_2 += 1
            if n_view > 1:
                n_view_gt1 += 1
            src = row.get("source_model", "?")
            by_source.setdefault(src, {"n": 0, "n_with_summary": 0})
            by_source[src]["n"] += 1
            if n_compact > 0:
                by_source[src]["n_with_summary"] += 1

    char_lens.sort()
    summ_chars.sort()

    out = {
        "path": str(args.pairs),
        "n": n,
        "n_with_active_summary": n_summ,
        "frac_with_active_summary": round(n_summ / max(1, n), 4),
        "n_with_2plus_summaries": n_summ_ge_2,
        "n_with_view_gt1": n_view_gt1,
        "char_len": {
            "p10": char_lens[int(0.10 * (n - 1))] if n else 0,
            "p50": char_lens[n // 2] if n else 0,
            "p90": char_lens[int(0.90 * (n - 1))] if n else 0,
            "p99": char_lens[int(0.99 * (n - 1))] if n else 0,
            "max": char_lens[-1] if n else 0,
        },
        "active_summary_chars": (
            {
                "p50": summ_chars[len(summ_chars) // 2],
                "p90": summ_chars[int(0.90 * (len(summ_chars) - 1))],
                "max": summ_chars[-1],
            }
            if summ_chars else None
        ),
        "by_source": {
            k: {
                "n": v["n"],
                "n_with_summary": v["n_with_summary"],
                "frac": round(v["n_with_summary"] / max(1, v["n"]), 4),
            }
            for k, v in by_source.items()
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
