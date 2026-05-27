"""SAFE-filtered summarizer training pairs for the summarizer SFT.

Emits `(prompt, completion)` rows where:

- `completion` is the teacher's summary text (`memory.summary`) recorded in
  a the training phase fork `_training.json` at some compaction step.
- `prompt` matches what ``NaiveSummarizationMemory._summarize_all`` would
  have fed the teacher at that step — the same system prompt, and a user
  body that is incremental in exactly the same way (raw step-history on
  the first compaction; ``"Previous summary: ...\\n\\nNew content:\\n..."``
  on subsequent compactions).

Filtering: we only keep rows from the training phase pairs with ``label=1`` (SAFE).
Rationale is in ``the internal handoff doc`` §the baseline-train phase — SAFE
means the agent took the same next action on the post-compaction
history as on the full history, so the summary was load-bearing enough
not to regress task resolution.

Schema of each emitted row (one JSON per line):

    {
        "trajectory_id": str,
        "fork_event_id": str,          # same as aug_sft_pairs (iid_step_pert)
        "source_model": "sonnet45"|"haiku45"|"gptoss120b",
        "source_dir": str,
        "step": int,
        "compaction_index": int,       # 0-based index of this compaction
                                       # within the trajectory (0 = first)
        "prev_compaction_step": int,   # step of the preceding summary, or -1
        "problem_statement": str,      # preserved for dataset-card inspection
        "prompt": str,                 # full "system\\n\\nuser" rendered text
        "completion": str,             # teacher's summary
        "num_input_chars": int,
        "completion_chars": int,
        "compression_ratio": float,    # original_tokens / filtered_tokens
        "original_tokens": int,
        "filtered_tokens": int,
        "forgotten": int,
        "split": "train" | "eval"      # repo-grouped, same grouping as pairs
    }

The script has two subcommands, selected by flag rather than positional:

    python -m memgym.training.data.summarizer_pairs \
        --trajectories data/world_model/trajectories \
        --pairs data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl \
        --output data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl

    python -m memgym.training.data.summarizer_pairs \
        --inspect data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl \
        --n 20
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from memgym.training.data.aug_pairs import assign_splits

# Copied verbatim from ``memgym.memory.naive_summarization.SUMMARIZATION_PROMPTS["swe"]``
# to keep this data-prep script self-contained — importing the memory
# package pulls in optional runtime deps (sentence-transformers, sklearn)
# via the `amem` sub-package that this pipeline does not need. If the
# runtime prompt changes, update this constant and `SUMMARIZATION_PROMPTS`
# in the same commit.
SWE_SYSTEM_PROMPT = """You are summarizing context for a coding agent working on a software engineering task.

PRESERVE (high priority):
- Architectural decisions and design patterns chosen
- Unresolved bugs, errors, and technical blockers
- Implementation details that affect future work
- File paths and code locations referenced
- Dependencies between components
- Test results and their implications

DISCARD:
- Redundant command outputs already processed
- Duplicate file contents
- Raw stack traces (keep only key error message)
- Verbose tool outputs once insights captured

Output a concise summary maintaining all critical technical context."""

logger = logging.getLogger(__name__)


# the training phase pairs were generated from these three fork runs. Each entry maps a
# `source_dir` field (as it appears in `aug_sft_pairs.jsonl`) to the
# trajectory root directory (under `data/world_model/trajectories/<root>/`)
# that contains per-instance `_training.json` files with the teacher's
# summaries. Only these three contain `label=1` (SAFE) rows — the scopedB
# runs never produced SAFE pairs in the full yn export.
SOURCE_DIR_TO_TRAJ_ROOT: Dict[str, str] = {
    "gptoss_fork_gap554_scaleA_v1": "fork_gptoss_llmsumm_gap554_v2",
    "haiku45_fork_diverse500_scaleA_v1": "fork_haiku45_llmsumm_diverse500_haikusum_v1",
    "sonnet_fork_gap554_scaleA_v1": "fork_sonnet45_llmsumm_gap554_v2",
}


# ---------------------------------------------------------------------------
# Step-triple serialization — this is the single lever for how much task-env
# context the student summarizer sees. The runtime stringifies actual
# assistant/tool messages; we only have (thought, action, observation)
# per step in `_training.json`, so we reconstruct a content-equivalent
# view. Including `action` is a no-op for mini-swe-agent (all reasoning
# goes into `thought` and the bash command is embedded there), but we
# keep the slot so this is trivially extensible if future trajectories
# emit non-empty `action` fields.
# ---------------------------------------------------------------------------

def _render_step_triple(step: Dict[str, Any]) -> str:
    """Render one (thought, action, observation) into a readable block.

    Intentionally verbose so the student sees the same kind of reasoning
    + environment feedback the agent saw before compaction. No labels
    (SAFE/HARMFUL), no post-compaction steps, no episode outcome — those
    would be future-information leakage.
    """
    parts = [f"=== Step {step.get('step', '?')} ==="]
    thought = (step.get("thought") or "").strip()
    action = (step.get("action") or "").strip()
    obs = (step.get("observation") or "").strip()
    if thought:
        parts.append(f"Thought: {thought}")
    if action:
        parts.append(f"Action: {action}")
    if obs:
        parts.append(f"Observation: {obs}")
    return "\n".join(parts)


def _render_history_slice(
    steps: List[Dict[str, Any]],
    problem_statement: str,
    include_problem: bool,
) -> str:
    """Concat a slice of step triples. The problem prefix is added only
    on the first compaction — on subsequent ones, the problem has been
    folded into the previous summary and repeating it would waste budget.
    """
    blocks: List[str] = []
    if include_problem and problem_statement:
        blocks.append(f"=== Problem ===\n{problem_statement.strip()}")
    for s in steps:
        blocks.append(_render_step_triple(s))
    return "\n\n".join(blocks)


def _build_prompt(user_body: str) -> str:
    """Assemble the full "system + user" prompt as a single string.

    We flatten to text rather than a messages list because TRL SFT with
    `completion_only_loss=True` wants `prompt`/`completion` strings. At
    H.2 time the Qwen3-8B-Base chat template will re-wrap this prompt
    with its own role tokens; the content is what matters.
    """
    return (
        f"[System]\n{SWE_SYSTEM_PROMPT}\n\n"
        f"[User]\n{user_body}\n\n"
        f"[Assistant]\n"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class CompactionSnapshot:
    """One compaction event within a trajectory's `_training.json`."""
    step: int
    compaction_index: int
    summary: str
    original_tokens: int
    filtered_tokens: int
    original_msgs: int
    filtered_msgs: int
    forgotten: int
    compression_ratio: float


def _load_safe_pairs(pairs_path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Load SAFE (label=1) rows from `aug_sft_pairs.jsonl`, keyed by
    (trajectory_id, source_dir, step). Duplicate keys collapse — SAFE
    pairs with different perturbations at the same compaction all share
    the same (trajectory, summary), so keeping one is correct.
    """
    by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    with pairs_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if int(row.get("label", 0)) != 1:
                continue
            key = (row["trajectory_id"], row["source_dir"], int(row["step"]))
            if key not in by_key:
                by_key[key] = row
    return by_key


def _extract_compactions(training_json_path: Path) -> Tuple[List[CompactionSnapshot], str]:
    """Return (ordered compaction snapshots, problem_statement).

    Snapshots come out in step order with a 0-based `compaction_index`
    so downstream code can look up the preceding summary without
    re-walking. Steps with `was_compacted=True` but empty `summary` are
    sticky-post-compaction rows and are skipped (they duplicate the
    preceding event).
    """
    data = json.loads(training_json_path.read_text())
    problem = data.get("problem", "") or ""
    snapshots: List[CompactionSnapshot] = []
    ci = 0
    for step_row in data.get("steps", []):
        mem = step_row.get("memory") or {}
        summary = (mem.get("summary") or "").strip()
        if not summary:
            continue
        try:
            step_i = int(step_row.get("step", -1))
        except (TypeError, ValueError):
            continue
        snapshots.append(CompactionSnapshot(
            step=step_i,
            compaction_index=ci,
            summary=summary,
            original_tokens=int(mem.get("original_tokens") or 0),
            filtered_tokens=int(mem.get("filtered_tokens") or 0),
            original_msgs=int(mem.get("original_msgs") or 0),
            filtered_msgs=int(mem.get("filtered_msgs") or 0),
            forgotten=int(mem.get("forgotten") or 0),
            compression_ratio=float(mem.get("compression_ratio") or 1.0),
        ))
        ci += 1
    return snapshots, problem


def _load_steps_for_triples(training_json_path: Path) -> List[Dict[str, Any]]:
    """Read the per-step (thought/action/observation) list used to build
    the user body of the training prompt.
    """
    data = json.loads(training_json_path.read_text())
    steps_out: List[Dict[str, Any]] = []
    for s in data.get("steps", []):
        try:
            step_i = int(s.get("step", -1))
        except (TypeError, ValueError):
            continue
        steps_out.append({
            "step": step_i,
            "thought": s.get("thought") or "",
            "action": s.get("action") or "",
            "observation": s.get("observation") or "",
        })
    return steps_out


def _trajectory_file(
    trajectories_root: Path, source_dir: str, instance_id: str,
) -> Optional[Path]:
    """Resolve `(source_dir, instance_id) -> _training.json path`."""
    traj_root = SOURCE_DIR_TO_TRAJ_ROOT.get(source_dir)
    if traj_root is None:
        return None
    candidate = trajectories_root / traj_root / "trajectories" / f"{instance_id}_training.json"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_pairs(
    trajectories_root: Path,
    pairs_path: Path,
    output_path: Path,
    eval_frac: float = 0.10,
    max_input_chars: int = 48_000,
    seed: int = 17,
) -> Dict[str, Any]:
    """Walk SAFE the training phase pairs, emit `(prompt, completion)` rows.

    `max_input_chars` is a hard safety cap on the rendered user body —
    runtime's NaiveSummarizationMemory tail-truncates at ~12k tokens,
    this is ~4× that, sized to preserve all but genuinely pathological
    rollouts while still fitting comfortably in Qwen3-8B's seq-len=2048
    SFT window after tokenization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_pairs = _load_safe_pairs(pairs_path)
    logger.info("SAFE pair keys (iid, source_dir, step): %d", len(safe_pairs))

    # Group SAFE pairs by trajectory file so we can walk each trajectory
    # once and reuse the previous-summary state across its compaction
    # events (an instance can have multiple SAFE compactions).
    pairs_by_traj: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for key, row in safe_pairs.items():
        iid, source_dir, step = key
        pairs_by_traj[(iid, source_dir)].append({**row, "_step": step})

    rows: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    missing_training_json = 0
    missing_step_match = 0
    oversized_dropped = 0

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

        # Map compaction step -> snapshot for this trajectory.
        by_step: Dict[int, CompactionSnapshot] = {s.step: s for s in snapshots}

        # SAFE pairs for this trajectory, sorted by step so we always see
        # smaller steps first (cheap invariant for reviewing outputs).
        traj_pair_rows.sort(key=lambda r: r["_step"])

        for pair_row in traj_pair_rows:
            target_step = pair_row["_step"]
            snap = by_step.get(target_step)
            if snap is None:
                missing_step_match += 1
                continue

            # Previous-summary lookup for incremental reconstruction.
            prev_snap: Optional[CompactionSnapshot] = (
                snapshots[snap.compaction_index - 1]
                if snap.compaction_index > 0
                else None
            )
            prev_summary = prev_snap.summary if prev_snap else None
            prev_step = prev_snap.step if prev_snap else -1

            # Select the step slice the teacher actually saw at this
            # compaction. On first compaction: all steps strictly before
            # the compaction step. On subsequent: steps strictly between
            # the previous compaction step and this one.
            start = prev_step + 1 if prev_snap else 0
            delta_steps = [s for s in steps_triples if start <= s["step"] < target_step]
            delta_body = _render_history_slice(
                delta_steps,
                problem_statement=problem,
                include_problem=prev_snap is None,
            )

            if prev_summary is None:
                user_body = delta_body
            else:
                user_body = (
                    f"Previous summary: {prev_summary}\n\n"
                    f"New content:\n{delta_body}"
                )

            if len(user_body) > max_input_chars:
                # Tail-truncate: runtime's budget guard keeps the
                # recent context. Preserve that semantics here.
                user_body = "[... earlier content truncated ...]\n" + user_body[-max_input_chars:]
                oversized_dropped += 1  # not dropped, just tracked

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

    # Repo-grouped split, consistent with aug_pairs.assign_splits.
    splits = assign_splits((r["trajectory_id"] for r in rows), eval_frac)
    with output_path.open("w") as out:
        for r in rows:
            r["split"] = splits[r["trajectory_id"].split("__")[0]]
            out.write(json.dumps(r) + "\n")

    split_counts = Counter(r["split"] for r in rows)
    stats = {
        "written": len(rows),
        "by_source_model": dict(source_counts),
        "by_split": dict(split_counts),
        "missing_training_json_instances": missing_training_json,
        "missing_step_match_pairs": missing_step_match,
        "tail_truncated_rows": oversized_dropped,
        "out_path": str(output_path),
    }
    return stats


# ---------------------------------------------------------------------------
# Inspection mode — plan H.1.3 calls for a 20-row manual eyeball
# ---------------------------------------------------------------------------

def inspect_sample(path: Path, n: int = 20, seed: int = 42) -> None:
    """Print `n` random rows with key stats for the data-card review."""
    rng = random.Random(seed)
    rows = [json.loads(line) for line in path.open()]
    rng.shuffle(rows)
    sample = rows[:n]

    ratios = [r["completion_chars"] / max(1, r["num_input_chars"]) for r in rows]
    ratios.sort()
    def _pct(p): return ratios[int(len(ratios) * p)] if ratios else 0.0
    print(f"=== Dataset summary: {path} ===")
    print(f"total rows: {len(rows)}")
    print(f"completion/input char ratio — p10={_pct(0.1):.3f} p50={_pct(0.5):.3f} p90={_pct(0.9):.3f}")
    src = Counter(r.get("source_model", "") for r in rows)
    print(f"by source_model: {dict(src)}")
    comp_idx = Counter(r.get("compaction_index", -1) for r in rows)
    print(f"by compaction_index: first={comp_idx.get(0,0)} subsequent={sum(v for k,v in comp_idx.items() if k and k>0)}")
    splits = Counter(r.get("split", "") for r in rows)
    print(f"by split: {dict(splits)}")
    print()

    for i, r in enumerate(sample):
        print(f"--- sample {i+1}/{len(sample)} ---")
        print(f"  iid={r['trajectory_id']} step={r['step']} ci={r['compaction_index']} "
              f"prev_step={r['prev_compaction_step']} source={r.get('source_model','')} "
              f"input_chars={r['num_input_chars']} completion_chars={r['completion_chars']}")
        print(f"  completion[:300]: {r['completion'][:300]!r}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path,
                        default=Path("data/world_model/trajectories"),
                        help="Root dir containing fork_* trajectory sub-dirs.")
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl"),
                        help="aug_sft_pairs.jsonl to filter by label=1.")
    parser.add_argument("--output", type=Path,
                        default=Path("data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl"),
                        help="Output summarizer-pairs JSONL.")
    parser.add_argument("--eval-frac", type=float, default=0.10,
                        help="Fraction of pairs (by group-shuffle on repo prefix) to hold out.")
    parser.add_argument("--max-input-chars", type=int, default=48_000,
                        help="Tail-truncate the rendered user body if it exceeds this.")
    parser.add_argument("--inspect", type=Path, default=None,
                        help="Inspection mode: print N random rows from this JSONL with stats.")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of rows to show in --inspect mode.")
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
