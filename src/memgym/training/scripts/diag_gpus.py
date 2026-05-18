"""Quick GPU-occupancy diagnostic — wraps `nvidia-smi` + `ps` so we can
see which process holds GPU memory on a shared box via `/run-script`.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    print("=== nvidia-smi --query-gpu ===")
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(r.stdout or "(no output)")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    print("\n=== nvidia-smi --query-compute-apps ===")
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(r.stdout or "(no compute apps)")

    # Map pids -> command lines via ps so we can tell which job owns them.
    pids = []
    for line in (r.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(parts[1])

    if pids:
        print("\n=== ps for compute PIDs ===")
        ps = subprocess.run(
            ["ps", "-o", "pid,user,etime,cmd", "-p", ",".join(pids)],
            capture_output=True, text=True,
        )
        print(ps.stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
