"""One-shot: remove stale /tmp/torchelastic_* dirs older than 1 hour.

Three consecutive 8-rank torchrun jobs hung at the post-model-load,
pre-training-loop NCCL barrier. The probe_ipc_state run found 3,312
stale `torchelastic_*` rendezvous-state dirs in `/tmp` going back to
April 16 — every cancelled torchrun leaves one behind. This is a
known torchelastic bug: SIGTERM doesn't trigger the cleanup hook.

The 1-hour age cutoff is deliberate: it's longer than any plausible
training-job startup, so we can't accidentally remove the dir of an
in-flight run. We don't touch /dev/shm/* or anything Ray-owned.

Safe to run between training-job attempts. Idempotent.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    tmp = Path("/tmp")
    cutoff = time.time() - 3600  # only dirs untouched for >= 1 hour

    candidates = sorted(tmp.glob("torchelastic_*"))
    print(f"Found {len(candidates)} torchelastic_* dirs in /tmp")

    removed = 0
    skipped_recent = 0
    failed = 0
    for d in candidates:
        try:
            mtime = d.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > cutoff:
            skipped_recent += 1
            continue
        try:
            shutil.rmtree(d, ignore_errors=False)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"  failed {d}: {e!r}", file=sys.stderr)

    print(f"removed={removed}  skipped_recent_(<1h)={skipped_recent}  failed={failed}")

    # Quick sanity on what's left
    remaining = sorted(tmp.glob("torchelastic_*"))
    print(f"remaining torchelastic_* dirs: {len(remaining)}")
    for d in remaining[:10]:
        try:
            mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
        except FileNotFoundError:
            mt = "(gone)"
        print(f"  {d.name}  ({mt})")

    # Inode budget check (high inode use is the most plausible knock-on)
    df = subprocess.run(["df", "-i", "/tmp"], capture_output=True, text=True, timeout=10)
    print("\n=== df -i /tmp ===")
    print(df.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
