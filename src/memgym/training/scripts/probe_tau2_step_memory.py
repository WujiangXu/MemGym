"""Scan a τ2 trajectory.json for any step where memory_action / context /
related fields are populated — i.e. whether per-step summary state is
recoverable from trajectory.json without the rich _training.json companion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    traj = json.loads(args.path.read_text())
    eps = traj.get("episodes") or []
    print(f"file: {args.path}")
    print(f"n_episodes: {len(eps)}")
    if not eps:
        return 0
    md = eps[0].get("metadata", {}) or {}
    print(f"episode metadata.memory_agent_strategy={md.get('memory_agent_strategy')}")
    # Top-level compaction/summarization counters
    top_md = traj.get("metadata", {}) or {}
    ms = top_md.get("memory_stats", {}).get("agent", {})
    print(f"agent_compaction_count={top_md.get('agent_compaction_count')}  "
          f"summarization_count={ms.get('summarization_count')}  "
          f"condensation_count={ms.get('condensation_count')}  "
          f"wrapper_compaction_count={ms.get('wrapper_compaction_count')}")

    steps = eps[0].get("steps") or []
    print(f"\nn_steps: {len(steps)}")
    candidate_keys = ("memory_action", "context", "reasoning_action")
    populated = {k: 0 for k in candidate_keys}
    samples = {k: None for k in candidate_keys}
    for i, s in enumerate(steps):
        for k in candidate_keys:
            v = s.get(k)
            if v is not None and v != "":
                populated[k] += 1
                if samples[k] is None:
                    samples[k] = (i, v)

    print(f"per-step field population (out of {len(steps)} steps):")
    for k in candidate_keys:
        print(f"  {k}: populated_in {populated[k]} steps")

    print(f"\nsamples (step_idx, value-head):")
    for k, hit in samples.items():
        if hit is None:
            print(f"  {k}: (always None)")
            continue
        i, v = hit
        if isinstance(v, dict):
            head = f"<dict keys={sorted(v.keys())[:8]} len={len(v)}>"
            # If it has 'content' or 'summary', show that
            for sk in ("content", "summary", "text", "compaction_summary"):
                if sk in v:
                    sv = v[sk]
                    if isinstance(sv, str):
                        head += f" {sk}[:200]={sv[:200]!r}"
        elif isinstance(v, list):
            head = f"<list n={len(v)}>"
            if v and isinstance(v[0], dict):
                head += f" first_keys={sorted(v[0].keys())[:8]}"
        elif isinstance(v, str):
            head = f"<str len={len(v)}> {v[:200]!r}"
        else:
            head = repr(v)[:200]
        print(f"  step {i}: {k} = {head}")

    # Bonus: dump full first populated memory_action if any
    if samples["memory_action"] is not None:
        i, v = samples["memory_action"]
        print(f"\nfull memory_action at step {i} (truncated to 1500 chars):")
        print(json.dumps(v, indent=2, default=str)[:1500])

    return 0


if __name__ == "__main__":
    sys.exit(main())
