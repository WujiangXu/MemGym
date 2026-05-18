"""List rm_v2_run1 training-side artifacts on EC2 to see what got logged."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path("${REPO_ROOT}")


def main() -> int:
    candidates = [
        REPO / "training_output" / "rm_v2_run1",
        REPO / "checkpoints" / "rm_v2_8b_yn_cw3",
        REPO / "wandb",
        REPO / "runs",
    ]
    for c in candidates:
        print(f"\n=== {c} ===", flush=True)
        if not c.exists():
            print("  (missing)", flush=True)
            continue
        for p in sorted(c.rglob("*")):
            if p.is_file():
                rel = p.relative_to(c)
                print(f"  {p.stat().st_size:>12d}  {rel}", flush=True)
            elif p.is_dir():
                rel = p.relative_to(c)
                print(f"  {'<DIR>':>12s}  {rel}/", flush=True)

    print("\n=== probe trainer logs ===", flush=True)
    for name in ("trainer_log.jsonl", "trainer_state.json", "training_args.bin", "training_args.json", "all_results.json"):
        for c in candidates:
            f = c / name
            if f.exists():
                print(f"  HIT: {f}  ({f.stat().st_size} bytes)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
