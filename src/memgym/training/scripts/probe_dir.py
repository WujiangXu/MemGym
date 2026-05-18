"""One-shot: list a directory's contents (depth 1 by default)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--depth", type=int, default=1)
    args = p.parse_args()

    if not args.path.exists():
        print(f"NOT FOUND: {args.path}", file=sys.stderr)
        return 2

    base = str(args.path).rstrip("/")
    base_depth = base.count(os.sep)
    for dirpath, dirs, files in os.walk(args.path):
        rel = dirpath.count(os.sep) - base_depth
        if rel > args.depth:
            dirs.clear()
            continue
        print(f"{dirpath}/  ({len(dirs)} dirs, {len(files)} files)")
        if rel < args.depth:
            for f in sorted(files):
                fp = Path(dirpath) / f
                try:
                    sz = fp.stat().st_size
                except OSError:
                    sz = -1
                print(f"  {f}  ({sz} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
