"""Kill leftover torchrun/train_sft worker processes holding GPU memory.

After a cancelled `/run-script` job, the elastic-launch agent's children
sometimes survive their parent's SIGTERM and keep CUDA contexts alive.
This module pkills any python process whose cmdline mentions the train
entrypoint, then prints `nvidia-smi` so we can confirm the GPUs are free.
"""
from __future__ import annotations

import shlex
import subprocess
import sys


def _run(cmd: str) -> tuple[int, str]:
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    return res.returncode, (res.stdout + res.stderr)


def main() -> int:
    targets = [
        "memgym.training.scripts.train_sft",
        "torch.distributed.run",
        "torchrun",
    ]
    for t in targets:
        rc, out = _run(f"pkill -9 -f {t}")
        print(f"pkill -9 -f {t!r} -> rc={rc}")

    # Find any remaining python processes holding GPU memory and SIGKILL
    # them by PID directly — pkill -f can miss processes whose cmdline
    # was rewritten or which are stuck mid-CUDA-kernel.
    rc, out = _run("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    pids = [p.strip() for p in out.splitlines() if p.strip().isdigit()]
    print(f"GPU-holding PIDs after pkill: {pids}")
    for pid in pids:
        rc, out = _run(f"kill -9 {pid}")
        print(f"kill -9 {pid} -> rc={rc} {out.strip()}")

    # Wait a moment and re-poll. Stuck CUDA kernels can take a few
    # seconds to relinquish the context after the process is reaped.
    import time
    time.sleep(8)
    rc, out = _run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    print("--- post-cleanup nvidia-smi ---")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
