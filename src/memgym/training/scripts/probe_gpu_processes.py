"""List GPU processes + per-GPU free memory.

Way B FSDP smoke job e01db8c0 OOM'd because GPU 3 had only 538 MiB free
— another process (PID 158620) was holding 38.5 GiB. This probe tells us
what's holding which GPU so we can decide whether to kill it. Uses
`nvidia-smi` JSON-ish CSV output, parsed into a per-GPU table plus a
per-process table.
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def _smi(query: str, scope: str) -> list[list[str]]:
    cmd = ["nvidia-smi", f"--query-{scope}={query}", "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd, timeout=10).decode().strip()
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def main() -> int:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found")
        return 1

    print("=== Per-GPU memory ===")
    print(f"{'idx':>3}  {'used_MiB':>10}  {'free_MiB':>10}  {'total_MiB':>10}")
    for row in _smi("index,memory.used,memory.free,memory.total", "gpu"):
        idx, used, free, total = row
        print(f"{idx:>3}  {used:>10}  {free:>10}  {total:>10}")

    print()
    print("=== Compute processes ===")
    print(f"{'pid':>8}  {'gpu':>3}  {'mem_MiB':>10}  process")
    rows = _smi("pid,gpu_uuid,used_memory,process_name", "compute-apps")
    if not rows:
        print("(none)")
    else:
        # Map gpu_uuid -> index for readability
        idx_rows = _smi("index,gpu_uuid", "gpu")
        uuid2idx = {u: i for i, u in idx_rows}
        for r in rows:
            pid, uuid, mem, pname = r
            print(f"{pid:>8}  {uuid2idx.get(uuid, '?'):>3}  {mem:>10}  {pname}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
