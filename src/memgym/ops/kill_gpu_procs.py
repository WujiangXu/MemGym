"""Kill all processes currently using a CUDA device.

Used between back-to-back GPU jobs on the EC2 box to recover the full
8-GPU pool — leftover Python workers from a crashed `torchrun` or vLLM
server hold the entire CUDA context (~38 GB/rank) and starve the next
job. `nvidia-smi --query-compute-apps=pid` lists every process with a
live CUDA context; we SIGKILL them, then print the after-state so the
caller can confirm GPUs are free before launching.

Safe to run when the GPUs are already idle (the PID list is empty).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time


def _list_compute_pids() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def _free_mem_per_gpu() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main() -> int:
    pids = _list_compute_pids()
    print(f"Found {len(pids)} GPU-using PIDs: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  killed {pid}")
        except ProcessLookupError:
            print(f"  {pid} already gone")
        except PermissionError:
            print(f"  {pid} not ours (skipped)")

    if pids:
        time.sleep(3)

    remaining = _list_compute_pids()
    print(f"Remaining: {remaining}")
    print("Free memory per GPU:")
    print(_free_mem_per_gpu())
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
