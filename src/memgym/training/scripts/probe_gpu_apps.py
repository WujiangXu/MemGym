"""One-shot: list per-GPU compute apps (pid, used_memory) via nvidia-smi.

Used to identify zombie processes after a cancelled torchrun job.
Read-only; does NOT kill anything.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    print("=== nvidia-smi compute apps (per-GPU PIDs holding memory) ===")
    print(out.stdout if out.stdout else "(no compute apps reported)")
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        return out.returncode

    # Cross-reference: how old are these processes?
    pids = set()
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.add(parts[1])
    if pids:
        ps = subprocess.run(
            ["ps", "-o", "pid,etime,user,comm,cmd", "--no-headers"] + sorted(pids),
            capture_output=True, text=True, timeout=20,
        )
        print("\n=== ps -o pid,etime,user,comm,cmd for those PIDs ===")
        print(ps.stdout if ps.stdout else "(none — processes may have exited)")
        if ps.stderr:
            print(ps.stderr, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
