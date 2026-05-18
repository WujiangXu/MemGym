"""Definitive check: does the [Summary] marker live in raw V2
trajectory.json::episodes[].steps[k].context?

Code-trace finding (webarena_runner.py:288, 408, 467) shows the
runner writes `_json_safe(filtered.content)` — which contains the
[Summary]-prefixed user message from structured_summary.py:337 — into
Step.context whenever `was_compacted=True`. Earlier probes scanned
DERIVED pairs (webarena_v2.jsonl) for the marker and got 0 hits.
That doesn't prove absence in the raw trajectory; it only proves the
pair-builder isn't reading step.context.

This probe confirms by opening the raw V2 trajectory.json directly:
  1. Top-3 V2 cohort tasks by trajectory size (proxy for compaction
     pressure).
  2. For each: count `was_compacted=True` steps, count steps with
     non-null `step.context`, count steps whose `step.context`
     contains `[Summary]`.
  3. Print the (head-200, tail-200) of the first such context found,
     so we see the actual injected summary text the runtime kept.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("${REPO_ROOT}/results/webarena")

CANDIDATES = [
    "prompt_fix_v2/gmail_struct_ms10/gmail/task_h11",
    "grid_idfix_gmail_struct_ms10/task_h20",
    "phaseC_ms10",       # whole cohort — find largest task
    "phaseDp_struct_ms10",
]


def _resolve(c: str) -> Path | None:
    p = ROOT / c
    if (p / "trajectory.json").is_file():
        return p
    if p.is_dir():
        # find largest trajectory under it
        bests = sorted(p.rglob("trajectory.json"),
                       key=lambda x: x.stat().st_size, reverse=True)
        return bests[0].parent if bests else None
    return None


def _scan(task: Path) -> None:
    print(f"\n{'='*78}\nTASK: {task}\n{'='*78}")
    tj = task / "trajectory.json"
    print(f"  size: {tj.stat().st_size:,} B")
    try:
        traj = json.loads(tj.read_text())
    except Exception as e:
        print(f"  parse error: {e}")
        return

    md = traj.get("metadata") or {}
    print(f"  metadata.compaction_count: {md.get('compaction_count')}")
    print(f"  metadata.memory_stats.compaction_count: {(md.get('memory_stats') or {}).get('compaction_count')}")

    eps = traj.get("episodes") or []
    if not eps:
        print("  no episodes")
        return
    steps = (eps[0] or {}).get("steps") or []
    n_steps = len(steps)

    n_was_compacted = 0
    n_ctx_nonempty = 0
    n_ctx_has_marker = 0
    first_marker_step: int | None = None

    for k, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        ma = s.get("memory_action") or {}
        was = bool(ma.get("was_compacted")) if isinstance(ma, dict) else False
        if was:
            n_was_compacted += 1
        ctx = s.get("context")
        if ctx not in (None, "", [], {}):
            n_ctx_nonempty += 1
            blob = json.dumps(ctx, default=str)
            if "[Summary]" in blob or "[SUMMARY:" in blob:
                n_ctx_has_marker += 1
                if first_marker_step is None:
                    first_marker_step = k

    print(f"  n_steps={n_steps}")
    print(f"  was_compacted=True count: {n_was_compacted}")
    print(f"  context non-null count:   {n_ctx_nonempty}")
    print(f"  context with [Summary]:   {n_ctx_has_marker}")

    if first_marker_step is not None:
        print(f"\n  --- step[{first_marker_step}].context (first marker) ---")
        ctx = steps[first_marker_step]["context"]
        # Find the message containing [Summary]
        if isinstance(ctx, list):
            for j, m in enumerate(ctx):
                if not isinstance(m, dict):
                    continue
                content = m.get("content") or ""
                if "[Summary]" in content or "[SUMMARY:" in content:
                    print(f"  msg[{j}] role={m.get('role')!r}  content_len={len(content)}")
                    print(f"    head[:400]: {content[:400]!r}")
                    if len(content) > 800:
                        print(f"    tail[-400:]: {content[-400:]!r}")
                    break
            else:
                print(f"  (marker present in blob, but no message content held it — full ctx[0]: {json.dumps(ctx[0], default=str)[:400] if ctx else '?'!r})")
        else:
            print(f"  context type={type(ctx).__name__}")
            print(f"    str: {str(ctx)[:600]!r}")


def main() -> int:
    print(f"# probe_v2_summary_in_context")
    for c in CANDIDATES:
        task = _resolve(c)
        if task is None:
            print(f"\n(skip: {c} → not found)")
            continue
        _scan(task)
    print("\n# done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
