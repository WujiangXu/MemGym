"""Inspect a τ2 trajectory.json to understand its full key inventory.

The flat-fallback τ2 builder (``tau2_long_context_pairs._load_flat``)
synthesizes empty memory dicts because we assumed trajectory.json
doesn't carry per-step compaction summaries. This probe verifies that
assumption by enumerating every key path in a real trajectory.json.

If summaries are hiding in some other field (e.g.
``episodes[0].steps[i].memory_state``, ``condensation_history``), we
can salvage them without re-collecting the rollouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def walk(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 6, key_set=None):
    if depth > max_depth:
        return
    if key_set is None:
        key_set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else k
            t = type(v).__name__
            extra = ""
            if isinstance(v, str):
                extra = f"  (len={len(v)})"
            elif isinstance(v, list):
                extra = f"  (n={len(v)})"
            elif isinstance(v, dict):
                extra = f"  (n_keys={len(v)})"
            key_set.add(f"{kp:<60s} {t}{extra}")
            walk(v, kp, depth + 1, max_depth, key_set)
    elif isinstance(obj, list) and obj:
        # Walk just the first element to learn schema, but tag with [*]
        walk(obj[0], f"{prefix}[*]", depth + 1, max_depth, key_set)
    return key_set


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--scan-summary-fields", action="store_true",
                    help="Search for any field name containing 'summ' or 'memory' or 'condense'.")
    args = ap.parse_args()

    if not args.path.is_file():
        print(f"missing: {args.path}", file=sys.stderr)
        return 2

    traj = json.loads(args.path.read_text())
    keys = walk(traj, max_depth=args.max_depth)
    print(f"file: {args.path}  size={args.path.stat().st_size:,}")
    print(f"\nkey paths (sorted):")
    for kp in sorted(keys):
        print(f"  {kp}")

    if args.scan_summary_fields:
        print(f"\nfields matching summ/memory/condense:")
        for kp in sorted(keys):
            low = kp.lower()
            if any(t in low for t in ("summ", "memor", "condens", "filter", "compact")):
                print(f"  {kp}")

    # Bonus: print a sample of episodes[0].steps[0] if present, since that's
    # where memory state per step would live.
    eps = traj.get("episodes") or []
    if eps and isinstance(eps[0], dict):
        steps = eps[0].get("steps") or []
        if steps:
            s0 = steps[0]
            print(f"\nepisodes[0].steps[0] keys: {sorted(s0.keys()) if isinstance(s0, dict) else type(s0).__name__}")
            if isinstance(s0, dict):
                for k, v in s0.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        rep = repr(v)[:200]
                    elif isinstance(v, dict):
                        rep = f"<dict len={len(v)} keys={sorted(v.keys())[:8]}>"
                    elif isinstance(v, list):
                        rep = f"<list n={len(v)}>"
                        if v and isinstance(v[0], dict):
                            rep += f" first_keys={sorted(v[0].keys())[:8]}"
                    else:
                        rep = type(v).__name__
                    print(f"  {k}: {rep}")
            # Also any later step that might have memory rebuilt
            if len(steps) > 5:
                s5 = steps[5]
                print(f"\nepisodes[0].steps[5] keys: {sorted(s5.keys()) if isinstance(s5, dict) else type(s5).__name__}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
