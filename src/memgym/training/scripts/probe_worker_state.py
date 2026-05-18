"""One-shot: dump worker process %CPU / state to distinguish slow vs hung.

If train_sft workers are at %CPU near 100% with STAT=R, they're
CPU-bound (slow but progressing). If %CPU=0 with STAT=S/D, they're
stuck in a syscall (futex wait, mutex lock, etc.) — a real deadlock.

Also peeks at /proc/<pid>/stack on a couple of pids if available, to
see what kernel function they're sleeping in.
"""
from __future__ import annotations

import subprocess
import sys


def _run(cmd, header):
    print(f"=== {header} ===")
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        r = subprocess.run(
            cmd if isinstance(cmd, list) else ["bash", "-lc", cmd],
            capture_output=True, text=True, timeout=15,
        )
        print(r.stdout if r.stdout else "(empty)")
        if r.stderr:
            print("STDERR:", r.stderr, file=sys.stderr)
    except Exception as e:
        print(f"FAILED: {e!r}", file=sys.stderr)
    print()


def main() -> int:
    # 1. Identify GPU-holding train_sft workers
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    pids = sorted({line.strip() for line in out.stdout.splitlines() if line.strip().isdigit()})
    if not pids:
        print("(no GPU-holding processes)")
        return 0
    print(f"GPU-holding PIDs: {pids}\n")

    # 2. Per-process state + CPU
    _run(
        ["ps", "-o", "pid,etime,pcpu,pmem,stat,wchan,comm"] + pids,
        "ps -o pid,etime,pcpu,stat,wchan,comm (NO command line)",
    )

    # 3. Kernel stack on first PID — what syscall is it sleeping in?
    pid0 = pids[0]
    _run(
        f"cat /proc/{pid0}/stack 2>/dev/null || echo '(no perm — try sudo)'",
        f"/proc/{pid0}/stack",
    )

    # 4. Open file count + last syscall via wchan
    _run(
        f"ls /proc/{pid0}/fd 2>/dev/null | wc -l",
        f"/proc/{pid0}/fd count",
    )
    _run(
        "cat /proc/" + pid0 + "/wchan 2>/dev/null && echo",
        f"/proc/{pid0}/wchan (kernel function it's blocked in)",
    )

    # 5. Aggregate: how many active vs idle workers?
    _run(
        "ps -o pid,pcpu,stat --no-headers " + " ".join(pids) + " | "
        "awk '{ if ($2+0 > 50) busy++; else idle++ } END { print \"busy(>50%CPU)=\"busy, \"idle=\"idle }'",
        "busy vs idle workers",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
