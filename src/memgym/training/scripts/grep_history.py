"""One-shot: grep a directory tree for one or more patterns, head N matches.

Used to surface launch.py job_logs for the 1.7B QLoRA run.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pattern", action="append", required=True,
                   help="Repeatable; case-insensitive substring match")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--ext", default="",
                   help="Filter to files with this extension (e.g. .log)")
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"NOT A DIR: {args.root}", file=sys.stderr)
        return 2

    pats = [re.compile(re.escape(s), re.IGNORECASE) for s in args.pattern]
    matches: list[tuple[Path, int, str]] = []
    for fp in args.root.rglob("*"):
        if not fp.is_file():
            continue
        if args.ext and fp.suffix != args.ext:
            continue
        try:
            with fp.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if any(p.search(line) for p in pats):
                        matches.append((fp, i, line.rstrip()))
                        if len(matches) >= args.limit:
                            break
        except OSError:
            continue
        if len(matches) >= args.limit:
            break

    print(f"# {len(matches)} matches in {args.root}")
    for fp, i, line in matches:
        print(f"{fp}:{i}: {line[:240]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
