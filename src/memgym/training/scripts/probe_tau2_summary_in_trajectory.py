"""Probe whether ``[Summary]`` markers can be recovered from the
``trajectory.json`` files we DO have.

The runtime didn't persist ``0_training.json`` for τ² runs, but the
summarizer injects ``[Summary] {body}`` into the message stream
(``tau2_summarizing.py:184``). If those markers survive into
``trajectory.json``, we can reconstruct first-compaction-step + active
summary content per step from existing artefacts (no re-run needed).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("${REPO_ROOT}/results/tau2_bench")
TARGETS = (
    "20260412-024137_all_both_tau2_summarizing_ms10kf1kl2r0.6_HP_full_summ_ms10_kl2",
    "20260412-213744_all_both_tau2_structured_ms10kf1kl4r0.6_HP_full_struct_ms10_kl4",
)


def _scan_trajectory(traj_path: Path) -> Dict[str, Any]:
    try:
        with traj_path.open() as fh:
            traj = json.load(fh)
    except Exception as e:
        return {"error": str(e)}

    eps = traj.get("episodes") or []
    if not eps:
        return {"n_episodes": 0}
    ep = eps[0]
    steps = ep.get("steps") or []

    summary_steps: List[int] = []
    summary_lens: List[int] = []
    sample_summary: str = ""
    for k, step in enumerate(steps):
        # Each step has 'action' and 'observation' message dicts.
        # The summarizer rewrites the existing message stream, so the
        # marker may live inside the action's content OR inside an
        # earlier message (the summary block sits in the view, not in
        # the action). We scan the full JSON of the step for the marker.
        blob = json.dumps(step, default=str)
        if "[Summary]" in blob:
            summary_steps.append(k)
            # Extract the summary body for length measurement
            i = blob.find("[Summary]")
            j = min(len(blob), i + 4000)
            slice_ = blob[i:j]
            summary_lens.append(len(slice_))
            if not sample_summary:
                sample_summary = slice_[:500]

    # Also scan top-level episode messages (some runners place messages
    # at episodes[0]['messages'] in addition to inside steps).
    msgs = ep.get("messages") or []
    msg_summary_pos: List[int] = []
    for i, m in enumerate(msgs):
        c = (m or {}).get("content") if isinstance(m, dict) else None
        if isinstance(c, str) and "[Summary]" in c:
            msg_summary_pos.append(i)

    return {
        "n_steps": len(steps),
        "n_steps_with_summary": len(summary_steps),
        "first_summary_step": summary_steps[0] if summary_steps else None,
        "last_summary_step": summary_steps[-1] if summary_steps else None,
        "summary_len_p50": statistics.median(summary_lens) if summary_lens else None,
        "n_messages_top_level": len(msgs),
        "n_messages_with_summary_top_level": len(msg_summary_pos),
        "first_summary_msg_idx": msg_summary_pos[0] if msg_summary_pos else None,
        "sample_summary_head_500": sample_summary,
    }


def main() -> int:
    out: Dict[str, Any] = {}
    for run_name in TARGETS:
        run = ROOT / run_name
        if not run.is_dir():
            out[run_name] = {"error": "missing"}
            continue
        mem = run / "memory"
        if not mem.is_dir():
            out[run_name] = {"error": "no_memory_subdir"}
            continue
        # Pick the first task across any domain that has a trajectory.json.
        scanned = 0
        per_run: List[Dict[str, Any]] = []
        for dom in sorted(mem.iterdir()):
            if not dom.is_dir():
                continue
            for task in sorted(dom.iterdir()):
                if not task.is_dir():
                    continue
                traj_p = task / "trajectory.json"
                if not traj_p.is_file():
                    continue
                rec = _scan_trajectory(traj_p)
                rec["task"] = f"{dom.name}/{task.name}"
                per_run.append(rec)
                scanned += 1
                if scanned >= 5:
                    break
            if scanned >= 5:
                break
        # Aggregate
        n_with = sum(1 for r in per_run if (r.get("n_steps_with_summary") or 0) > 0
                     or (r.get("n_messages_with_summary_top_level") or 0) > 0)
        out[run_name] = {
            "n_scanned": scanned,
            "n_with_any_summary_marker": n_with,
            "samples": per_run,
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
