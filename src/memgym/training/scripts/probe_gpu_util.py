"""One-shot: query nvidia-smi for live GPU utilization + memory.used."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    print("=== nvidia-smi utilization ===")
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        return out.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
