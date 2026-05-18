"""One-shot: find files/dirs matching name patterns under a root."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="${HOME}")
    p.add_argument("--name", action="append", required=True,
                   help="Repeatable -name glob (case-insensitive)")
    p.add_argument("--type", default="d", choices=["d", "f"])
    p.add_argument("--maxdepth", type=int, default=6)
    args = p.parse_args()

    patterns: list[str] = []
    for i, n in enumerate(args.name):
        if i:
            patterns.append("-o")
        patterns += ["-iname", n]
    cmd = [
        "find", args.root, "-maxdepth", str(args.maxdepth),
        "-type", args.type, "(", *patterns, ")",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(out.stdout)
    if out.returncode != 0:
        print(f"--- find stderr ---\n{out.stderr}", file=sys.stderr)
    return out.returncode


if __name__ == "__main__":
    sys.exit(main())
