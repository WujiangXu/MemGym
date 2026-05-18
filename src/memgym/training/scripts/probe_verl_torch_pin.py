"""Print verl 0.5.0's declared torch dependency range.

We need to downgrade torch 2.11+cu130 to 2.10+cu128 in the way-a venv
to match the EC2 driver. This probe parses verl's metadata to confirm
2.10 is in-range — otherwise we have to bisect a different verl pin.
"""
from __future__ import annotations

import importlib.metadata as md
import sys


def main() -> int:
    pkg = "verl"
    try:
        meta = md.metadata(pkg)
    except md.PackageNotFoundError:
        print(f"{pkg} not installed")
        return 1

    print(f"{pkg} {meta['Version']}")
    print()
    print("=== Requires-Dist (filtered to torch / numpy / vllm / ray / transformers) ===")
    for r in (meta.get_all("Requires-Dist") or []):
        low = r.lower()
        if any(t in low for t in ("torch", "numpy", "vllm", "ray", "transformers", "tensordict")):
            print(f"  {r}")

    print()
    print("=== currently installed ===")
    for name in ("torch", "numpy", "vllm", "ray", "transformers", "tensordict"):
        try:
            v = md.version(name)
            print(f"  {name:18s} {v}")
        except md.PackageNotFoundError:
            print(f"  {name:18s} (not installed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
