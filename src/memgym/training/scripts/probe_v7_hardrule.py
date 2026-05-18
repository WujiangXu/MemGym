"""HARD RULE verifier for Plan v7's rebuilt WebArena V2 OOD pairs.

Per-row a row MUST contain:
  1. History — at least one [User] or [Assistant] block in the rendered prompt
  2. Task framing — "App:" + "Goal:" (the framing line emitted by
     webarena_long_context_pairs._flatten_traj_messages at line 175-187)
  3. Summary body — the literal "[Summary]" marker injected by
     reward_model_pairs._reconstruct_view at line 324

Reads N rows from each JSONL passed via --jsonl (repeatable),
samples up to --sample-size per file, prints per-file pass/fail
counts + a per-block presence breakdown.

Exits with code 1 if any row in any file fails any of the three
checks — the binding HARD RULE acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _check_row(prompt: str) -> dict:
    has_history = ("[User]" in prompt) or ("[Assistant]" in prompt) or \
                  ("[USER]" in prompt) or ("[ASSISTANT]" in prompt)
    has_task = ("App:" in prompt) and ("Goal:" in prompt)
    has_summary = "[Summary]" in prompt
    has_action_tmpl = ("Recorded action" in prompt) or ("recorded_action" in prompt) \
                      or ("Predicted action" in prompt)
    return {
        "history": has_history,
        "task_framing": has_task,
        "summary": has_summary,
        "action_template": has_action_tmpl,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", action="append", required=True,
                   help="Path to a JSONL file (repeatable).")
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-failing-head", type=int, default=300)
    args = p.parse_args()

    rng = random.Random(args.seed)
    overall_fail = 0
    for path_str in args.jsonl:
        path = Path(path_str)
        if not path.is_file():
            print(f"\n=== {path} ===")
            print(f"  MISSING FILE")
            overall_fail += 1
            continue

        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        n_total = len(rows)
        sample = rows if n_total <= args.sample_size else rng.sample(rows, args.sample_size)

        per_block = {"history": 0, "task_framing": 0, "summary": 0, "action_template": 0}
        n_pass = 0
        first_fail = None
        for r in sample:
            prompt = r.get("prompt", "")
            checks = _check_row(prompt)
            for k, v in checks.items():
                per_block[k] += int(v)
            ok = checks["history"] and checks["task_framing"] and checks["summary"]
            if ok:
                n_pass += 1
            elif first_fail is None:
                first_fail = (checks, prompt)

        print(f"\n=== {path} ===")
        print(f"  total_rows={n_total}  sampled={len(sample)}")
        print(f"  HARD RULE pass: {n_pass}/{len(sample)}")
        print(f"  per-block presence counts (sampled):")
        for k, v in per_block.items():
            print(f"    {k:18s}: {v}/{len(sample)}")

        if n_pass < len(sample):
            overall_fail += 1
            if first_fail is not None:
                checks, prompt = first_fail
                print(f"\n  FIRST FAILING ROW checks: {checks}")
                print(f"  prompt head[{args.show_failing_head}]:")
                print(f"    {prompt[:args.show_failing_head]!r}")

    print()
    if overall_fail:
        print(f"FAIL: {overall_fail} file(s) failed the HARD RULE gate.")
        return 1
    print("OK: all sampled rows contain history + task framing + [Summary].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
