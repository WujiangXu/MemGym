"""One-shot: print specific lines from a source file.

Used to diagnose stack-trace line-number references on the EC2 box
without relying on the /cat HTTP endpoint (which can mangle newlines).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--from-line", type=int, default=1)
    parser.add_argument("--to-line", type=int, default=200)
    args = parser.parse_args()

    if not args.file.exists():
        print(f"NOT FOUND: {args.file}", file=sys.stderr)
        return 2

    lines = args.file.read_text().splitlines()
    print(f"file: {args.file} ({len(lines)} lines)")
    lo = max(1, args.from_line)
    hi = min(len(lines), args.to_line)
    for i in range(lo - 1, hi):
        print(f"{i+1:4d}: {lines[i][:240]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
