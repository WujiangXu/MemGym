"""Kill stale torchrun / train_sft worker processes.

Cancelled jobs sometimes leave zombie children that keep port 29500
bound, blocking subsequent torchrun rendezvous with EADDRINUSE.
This module shells out to `pkill -9` for the well-known process
patterns and reports what was killed.

Run as a module:
"""
from __future__ import annotations

import shutil
import subprocess
import sys


PATTERNS = [
    "bin/torchrun",
    "memgym.training.scripts.train_sft",
    "torch.distributed.launch",
    "torch.distributed.run",
]


def _ps_grep(pattern: str) -> list[str]:
    rc = subprocess.run(
        ["pgrep", "-af", pattern], capture_output=True, text=True
    )
    return [ln for ln in rc.stdout.splitlines() if ln.strip()]


def main() -> int:
    print("[before] live processes matching patterns:", flush=True)
    any_alive = False
    for pat in PATTERNS:
        lines = _ps_grep(pat)
        for ln in lines:
            print(f"  [{pat}] {ln}", flush=True)
            any_alive = True
    if not any_alive:
        print("  (none)", flush=True)

    print("\n[kill] sending SIGKILL via pkill -9", flush=True)
    for pat in PATTERNS:
        rc = subprocess.run(
            ["pkill", "-9", "-f", pat], capture_output=True, text=True
        )
        # pkill exit 0=match+kill, 1=no match, >=2=error
        print(f"  pkill -9 -f {pat!r} -> rc={rc.returncode}", flush=True)

    if shutil.which("fuser"):
        rc = subprocess.run(
            ["fuser", "-k", "29500/tcp"], capture_output=True, text=True
        )
        print(f"\n[fuser] -k 29500/tcp -> rc={rc.returncode} "
              f"stdout={rc.stdout.strip()!r} stderr={rc.stderr.strip()!r}",
              flush=True)
    else:
        print("\n[fuser] not available, skipping port-release", flush=True)

    print("\n[after] live processes matching patterns:", flush=True)
    any_alive = False
    for pat in PATTERNS:
        lines = _ps_grep(pat)
        for ln in lines:
            print(f"  [{pat}] {ln}", flush=True)
            any_alive = True
    if not any_alive:
        print("  (none)", flush=True)

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
