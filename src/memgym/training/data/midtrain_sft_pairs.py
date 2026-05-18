"""Memory-world-model mid-train SFT pairs (option B).

Built from the same SAFE-filtered the training phase compactions as
``summarizer_pairs.py`` (label=1 in ``aug_sft_pairs.jsonl``), but the
user-body rendering is reshaped to make the pre/post-compaction
boundary explicit and to always include the task header.

Differences from ``summarizer_pairs.py``:

1. Problem statement (the runtime ``keep_first`` head) is included on
   *every* compaction, not only the first. This matches what
   ``LLMSummarizingMemory._summarize_all`` actually shows the teacher
   at compaction time (head messages stay verbatim across compactions).
2. The user body uses three structural sections instead of the prose
   ``"Previous summary: ... New content: ..."`` wrapper:

       === Task ===
       {problem_statement}

       === Compressed history ===   (omitted on first compaction)
       {previous_summary}

       === Recent context ===
       {delta_step_triples}

   Explicit section headers give the student a learnable signal for
   "this part was already compacted — don't re-summarize it; this part
   is new and is what the new summary needs to fold in."

Everything else — SAFE filter, D2 (multi-compaction), tail-truncation
budget, repo-grouped split — is reused verbatim from
``summarizer_pairs``.

Schema is identical to ``summarizer_pairs.jsonl`` so the existing
trainer code (``models.sft``) can ingest it without changes.

    python -m memgym.training.data.midtrain_sft_pairs \\
        --trajectories data/world_model/trajectories \\
        --pairs data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl \\
        --output data/world_model/training_output/midtrain_sft/midtrain_pairs.jsonl

    python -m memgym.training.data.midtrain_sft_pairs \\
        --inspect data/world_model/training_output/midtrain_sft/midtrain_pairs.jsonl \\
        --n 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.training.data.aug_pairs import assign_splits
from memgym.training.data.summarizer_pairs import (
    CompactionSnapshot,
    SWE_SYSTEM_PROMPT,
    _extract_compactions,
    _load_safe_pairs,
    _load_steps_for_triples,
    _render_step_triple,
    _trajectory_file,
    inspect_sample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option-B rendering — explicit structural sections, head always included.
# ---------------------------------------------------------------------------

def _render_recent_steps(steps: List[Dict[str, Any]]) -> str:
    """Stringify the delta step triples with no problem-statement prefix
    (the prefix is rendered separately as the Task section).
    """
    if not steps:
        return "(no new steps)"
    return "\n\n".join(_render_step_triple(s) for s in steps)


def _build_user_body(
    problem_statement: str,
    prev_summary: Optional[str],
    delta_steps: List[Dict[str, Any]],
) -> str:
    """Compose the option-B user body.

    `=== Task ===`              — runtime keep_first head, always present.
    `=== Compressed history ===`— prior summary, only on subsequent
                                  compactions; absent on the first.
    `=== Recent context ===`    — delta step triples since prev compaction
                                  (or since trajectory start on first).
    """
    sections: List[str] = []

    task = (problem_statement or "").strip()
    if task:
        sections.append(f"=== Task ===\n{task}")

    if prev_summary:
        sections.append(f"=== Compressed history ===\n{prev_summary.strip()}")

    sections.append(f"=== Recent context ===\n{_render_recent_steps(delta_steps)}")

    return "\n\n".join(sections)


def _build_prompt(user_body: str) -> str:
    """Same flat ``[System]/[User]/[Assistant]`` envelope as
    ``summarizer_pairs._build_prompt``. Kept local so the rendering
    contract for this dataset is self-contained.
    """
    return (
        f"[System]\n{SWE_SYSTEM_PROMPT}\n\n"
        f"[User]\n{user_body}\n\n"
        f"[Assistant]\n"
    )


# ---------------------------------------------------------------------------
# Main builder — structurally identical to summarizer_pairs.build_pairs;
# the only divergence is the _build_user_body call (option B).
# ---------------------------------------------------------------------------

def build_pairs(
    trajectories_root: Path,
    pairs_path: Path,
    output_path: Path,
    eval_frac: float = 0.10,
    max_input_chars: int = 48_000,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_pairs = _load_safe_pairs(pairs_path)
    logger.info("SAFE pair keys (iid, source_dir, step): %d", len(safe_pairs))

    pairs_by_traj: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for key, row in safe_pairs.items():
        iid, source_dir, step = key
        pairs_by_traj[(iid, source_dir)].append({**row, "_step": step})

    rows: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    missing_training_json = 0
    missing_step_match = 0
    tail_truncated = 0

    for (iid, source_dir), traj_pair_rows in pairs_by_traj.items():
        traj_path = _trajectory_file(trajectories_root, source_dir, iid)
        if traj_path is None:
            missing_training_json += 1
            continue

        snapshots, problem = _extract_compactions(traj_path)
        steps_triples = _load_steps_for_triples(traj_path)
        if not snapshots or not steps_triples:
            missing_training_json += 1
            continue

        by_step: Dict[int, CompactionSnapshot] = {s.step: s for s in snapshots}
        traj_pair_rows.sort(key=lambda r: r["_step"])

        for pair_row in traj_pair_rows:
            target_step = pair_row["_step"]
            snap = by_step.get(target_step)
            if snap is None:
                missing_step_match += 1
                continue

            prev_snap: Optional[CompactionSnapshot] = (
                snapshots[snap.compaction_index - 1]
                if snap.compaction_index > 0
                else None
            )
            prev_summary = prev_snap.summary if prev_snap else None
            prev_step = prev_snap.step if prev_snap else -1

            start = prev_step + 1 if prev_snap else 0
            delta_steps = [s for s in steps_triples if start <= s["step"] < target_step]

            user_body = _build_user_body(
                problem_statement=problem,
                prev_summary=prev_summary,
                delta_steps=delta_steps,
            )

            if len(user_body) > max_input_chars:
                # Tail-truncate: runtime's budget guard keeps recent
                # context. Preserve that semantics. Note the head/Task
                # section may itself be lost on extremely long bodies —
                # that's acceptable, this only fires on pathological rows.
                user_body = "[... earlier content truncated ...]\n" + user_body[-max_input_chars:]
                tail_truncated += 1

            prompt = _build_prompt(user_body)
            completion = snap.summary.strip()

            rows.append({
                "trajectory_id": iid,
                "fork_event_id": pair_row.get("fork_event_id", f"{iid}_{target_step}"),
                "source_model": pair_row.get("source_model", ""),
                "source_dir": source_dir,
                "step": target_step,
                "compaction_index": snap.compaction_index,
                "prev_compaction_step": prev_step,
                "problem_statement": problem[:3000],
                "prompt": prompt,
                "completion": completion,
                "num_input_chars": len(user_body),
                "completion_chars": len(completion),
                "compression_ratio": snap.compression_ratio,
                "original_tokens": snap.original_tokens,
                "filtered_tokens": snap.filtered_tokens,
                "forgotten": snap.forgotten,
            })
            source_counts[pair_row.get("source_model", "")] += 1

    if not rows:
        raise RuntimeError(
            "emitted zero rows; check --trajectories root and SOURCE_DIR_TO_TRAJ_ROOT mapping"
        )

    splits = assign_splits((r["trajectory_id"] for r in rows), eval_frac)
    with output_path.open("w") as out:
        for r in rows:
            r["split"] = splits[r["trajectory_id"].split("__")[0]]
            out.write(json.dumps(r) + "\n")

    split_counts = Counter(r["split"] for r in rows)
    return {
        "written": len(rows),
        "by_source_model": dict(source_counts),
        "by_split": dict(split_counts),
        "missing_training_json_instances": missing_training_json,
        "missing_step_match_pairs": missing_step_match,
        "tail_truncated_rows": tail_truncated,
        "out_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path,
                        default=Path("data/world_model/trajectories"))
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/world_model/training_output/midtrain_sft/midtrain_pairs.jsonl"))
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--max-input-chars", type=int, default=48_000)
    parser.add_argument("--inspect", type=Path, default=None)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    if args.inspect is not None:
        inspect_sample(args.inspect, n=args.n)
        return 0

    stats = build_pairs(
        trajectories_root=args.trajectories,
        pairs_path=args.pairs,
        output_path=args.output,
        eval_frac=args.eval_frac,
        max_input_chars=args.max_input_chars,
    )
    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
