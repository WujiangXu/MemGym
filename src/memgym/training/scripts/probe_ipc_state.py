"""One-shot: dump POSIX/SysV IPC + rendezvous-port state.

Hypothesis: prior cancelled torchrun jobs leave shared-memory segments,
semaphores, or rendezvous-port binds that block subsequent NCCL init.
This probe is read-only — it does NOT clean anything.

What we look at:
  * /dev/shm — POSIX shared memory (NCCL, PyTorch DataLoader workers)
  * ipcs -m / -s — SysV shm + semaphores (some NCCL/CUDA paths)
  * ss -tlnp on torchrun rendezvous ports (29400-29600)
  * /tmp/*nccl* /tmp/*torch* /tmp/torchelastic*

If any of these surfaces show stale ubuntu-owned entries from prior
runs, that's the leak — and the next run's NCCL init will collide.
"""
from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str], header: str) -> None:
    print(f"=== {header} ===")
    print(f"$ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = r.stdout if r.stdout else "(empty)"
        print(out)
        if r.stderr:
            print("STDERR:", r.stderr, file=sys.stderr)
    except Exception as e:
        print(f"FAILED: {e!r}", file=sys.stderr)
    print()


def main() -> int:
    _run(["bash", "-lc", "ls -la /dev/shm/ | head -100"],
         "/dev/shm contents")
    _run(["bash", "-lc", "du -sh /dev/shm/* 2>/dev/null | sort -hr | head -30"],
         "/dev/shm largest entries")
    _run(["ipcs", "-m"], "SysV shared memory")
    _run(["ipcs", "-s"], "SysV semaphores")
    _run(["bash", "-lc",
          "ss -tlnp 2>/dev/null | head -50 || netstat -tlnp 2>/dev/null | head -50"],
         "TCP listen sockets (any rendezvous bind)")
    _run(["bash", "-lc",
          "ls -la /tmp/ 2>/dev/null | grep -iE '(nccl|torch|elastic)' | head -50"],
         "/tmp/*torch* /*nccl* /*elastic* leftovers")
    _run(["bash", "-lc", "ls -la /tmp/torchelastic* 2>/dev/null | head -20"],
         "/tmp/torchelastic* (rendezvous state dirs)")
    _run(["bash", "-lc",
          "ps -eo pid,etime,comm,cmd | grep -iE '(torchrun|train_sft|nccl)' | grep -v grep | head -20"],
         "live torchrun/train_sft processes")
    _run(["bash", "-lc",
          "find / -name '*.lock' -newer /tmp -mmin -120 2>/dev/null | head -30"],
         "recent .lock files (last 2h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
