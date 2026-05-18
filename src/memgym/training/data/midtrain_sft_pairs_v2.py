"""Memory-world-model mid-train SFT pairs V2 — adds SAFE-perturbation augmentation.

V1 (``midtrain_sft_pairs.py``) emits one row per *unique* (iid, source_dir,
step) compaction event with ``label=1`` (SAFE) in ``aug_sft_pairs.jsonl``.
That dedups across the 6 perturbation kinds and yields 582 rows — losing
~1700 SAFE perturbation outcomes that, by RM judgment, were also action-
preserving views of the same compaction.

V2 keeps the v1 baseline AND adds two augmentation streams **without any
LLM calls** — re-applying deterministic perturbation strategies offline
to the already-on-disk trajectory + summary text.

Augmentation buckets:

- **V2-A (output)** — for each SAFE row whose perturbation modifies the
  *summary text* directly, apply that perturbation to the gold summary
  and use the perturbed text as an alternate ``completion``. Prompt is
  the v1 prompt. Perturbations: ``summary_redaction``, ``summary_noise``.

- **V2-B (input)** — for each SAFE row whose perturbation modifies the
  *message tail*, apply that perturbation to the delta-step triples that
  populate ``=== Recent context ===`` in the prompt. Completion is the
  v1 (un-perturbed) gold summary. Perturbations: ``random_drop_0.2``,
  ``truncate_last_10``, ``attr_delete_paths``.

- **Skipped**: ``aggressive_0.5`` — only modifies memory config; without
  a LLM re-run its ``perturb_messages`` is a no-op, so we cannot extract
  an alternate summary offline.

Dedup: if a perturbation produces output identical to the un-perturbed
view (e.g. ``attr_delete_paths`` on a step with no file paths), the row
is dropped — it would just duplicate the baseline.

Schema is a strict superset of v1: adds ``perturbation`` (str, "" for
baseline) and ``aug_kind`` ("baseline" | "output" | "input"). Existing
trainer (``models/sft.py``) ingests it unchanged because it only reads
``prompt``/``completion``/``split``.

    python -m memgym.training.data.midtrain_sft_pairs_v2 \\
        --trajectories data/world_model/trajectories \\
        --pairs data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl \\
        --output data/world_model/training_output/midtrain_sft_v2/midtrain_pairs_v2.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.training.augmentation.perturbations import (
    AttributeDeletion,
    ContextTruncation,
    PerturbationStrategy,
    RandomDrop,
    SummaryNoise,
    SummaryRedaction,
)
from memgym.training.data.aug_pairs import assign_splits
from memgym.training.data.summarizer_pairs import (
    CompactionSnapshot,
    SWE_SYSTEM_PROMPT,
    _extract_compactions,
    _load_steps_for_triples,
    _render_step_triple,
    _trajectory_file,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation registry: maps the perturbation string in aug_sft_pairs.jsonl
# to (kind, factory). Anything not in this map is silently skipped.
#
# We rebuild a fresh strategy instance per row (rather than reusing one
# instance) so RNG-based strategies (RandomDrop, SummaryNoise) are
# deterministic AND independent across compactions: each compaction gets
# the same RNG seed regardless of processing order. Without this, row
# order would change the noise injected into row N.
# ---------------------------------------------------------------------------

OUTPUT_KIND_FACTORIES: Dict[str, Any] = {
    # Modify summary TEXT — perturb gold summary, use as alternate completion.
    "summary_redaction": lambda: SummaryRedaction(),
    "summary_noise": lambda: SummaryNoise(n_snippets=2, seed=42),
}
INPUT_KIND_FACTORIES: Dict[str, Any] = {
    # Modify delta-step list — perturb prompt's recent context, completion unchanged.
    "random_drop_0.2": lambda: RandomDrop(drop_fraction=0.2, seed=42),
    "truncate_last_10": lambda: ContextTruncation(keep_last=10),
    "attr_delete_paths": lambda: AttributeDeletion(delete_type="paths"),
}
# Perturbations we explicitly know about but cannot replay offline.
# Tracked separately so the run-stats include them as "skipped" rather
# than silently drop them — makes it easy to audit "did we lose 405
# aggressive_0.5 rows?" from the stats output alone.
SKIP_PERTURBATIONS: set = {"aggressive_0.5"}


# ---------------------------------------------------------------------------
# Apply perturbations
# ---------------------------------------------------------------------------

def _perturb_summary_text(strategy: PerturbationStrategy, summary: str) -> str:
    """Run a summary-targeting strategy against a single message that wraps
    the gold summary. We synthesize the marker keywords (``USER_CONTEXT``,
    ``TASK_TRACKING``, ``CURRENT_STATE``) so SummaryRedaction/SummaryNoise's
    detector fires; the stored summaries already contain those markers
    (they're emitted by ``LLMSummarizingMemory._summarize_all`` template),
    but adding them defensively keeps the strategy's branch reachable
    even if a teacher used a slightly different format.
    """
    msgs = [{"role": "user", "content": summary}]
    out_msgs = strategy.perturb_messages(msgs, compaction_step_idx=0)
    return out_msgs[0]["content"]


def _perturb_delta_steps(
    strategy: PerturbationStrategy, delta_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply a content-level strategy to delta-step triples.

    We convert each step to a ``{"role":"user","content":<rendered>}`` mes-
    sage so the strategies (which operate on messages-list shape) can be
    reused without modification. After perturbation we keep the rendered
    content as the step's pre-formatted text — downstream rendering joins
    them with ``\\n\\n``, identical to v1.
    """
    msgs = [
        {"role": "user", "content": _render_step_triple(s), "_step": s}
        for s in delta_steps
    ]
    out_msgs = strategy.perturb_messages(msgs, compaction_step_idx=len(msgs))
    return [
        {"_pre_rendered": m["content"], "step": m.get("_step", {}).get("step", -1)}
        for m in out_msgs
    ]


def _render_recent_steps_from_pre(steps_pre: List[Dict[str, Any]]) -> str:
    """Render recent-context section from pre-rendered (possibly perturbed)
    step blocks. ``_pre_rendered`` may already be a step-triple string OR a
    perturbed/regex-rewritten variant; we trust whatever's in it.
    """
    if not steps_pre:
        return "(no new steps)"
    return "\n\n".join(s["_pre_rendered"] for s in steps_pre)


def _build_user_body(
    problem_statement: str,
    prev_summary: Optional[str],
    recent_context_text: str,
) -> str:
    """Same option-B layout as v1: Task / Compressed history / Recent context."""
    sections: List[str] = []
    task = (problem_statement or "").strip()
    if task:
        sections.append(f"=== Task ===\n{task}")
    if prev_summary:
        sections.append(f"=== Compressed history ===\n{prev_summary.strip()}")
    sections.append(f"=== Recent context ===\n{recent_context_text}")
    return "\n\n".join(sections)


def _build_prompt(user_body: str) -> str:
    return (
        f"[System]\n{SWE_SYSTEM_PROMPT}\n\n"
        f"[User]\n{user_body}\n\n"
        f"[Assistant]\n"
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _load_safe_rows_grouped(
    pairs_path: Path,
) -> Dict[Tuple[str, str, int], List[Dict[str, Any]]]:
    """Group SAFE rows by (iid, source_dir, step). Unlike v1 we keep ALL
    SAFE rows per key — each carries a different perturbation tag that
    drives an augmentation arm.
    """
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    with pairs_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if int(row.get("label", 0)) != 1:
                continue
            key = (row["trajectory_id"], row["source_dir"], int(row["step"]))
            grouped[key].append(row)
    return grouped


def build_pairs(
    trajectories_root: Path,
    pairs_path: Path,
    output_path: Path,
    eval_frac: float = 0.10,
    max_input_chars: int = 48_000,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_grouped = _load_safe_rows_grouped(pairs_path)
    logger.info(
        "SAFE compactions: %d unique keys, %d total rows",
        len(safe_grouped),
        sum(len(v) for v in safe_grouped.values()),
    )

    # Bucket compactions by trajectory file.
    by_traj: Dict[Tuple[str, str], List[Tuple[int, List[Dict[str, Any]]]]] = defaultdict(list)
    for (iid, source_dir, step), rows in safe_grouped.items():
        by_traj[(iid, source_dir)].append((step, rows))

    out_rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()

    for (iid, source_dir), step_buckets in by_traj.items():
        traj_path = _trajectory_file(trajectories_root, source_dir, iid)
        if traj_path is None:
            stats["missing_training_json"] += 1
            continue

        snapshots, problem = _extract_compactions(traj_path)
        steps_triples = _load_steps_for_triples(traj_path)
        if not snapshots or not steps_triples:
            stats["missing_training_json"] += 1
            continue

        by_step_snap: Dict[int, CompactionSnapshot] = {s.step: s for s in snapshots}
        step_buckets.sort(key=lambda x: x[0])

        for target_step, safe_rows in step_buckets:
            snap = by_step_snap.get(target_step)
            if snap is None:
                stats["missing_step_match"] += 1
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
            gold_summary = snap.summary.strip()

            # Pre-render delta steps once for the baseline + V2-A rows
            # (V2-B rebuilds them per-perturbation).
            baseline_recent = "\n\n".join(_render_step_triple(s) for s in delta_steps) \
                if delta_steps else "(no new steps)"

            # Pick a representative SAFE row for shared metadata fields
            # (source_model, fork_event_id stem). Multiple SAFE perturba-
            # tions share these; we lock to the alphabetically-first
            # perturbation's row for reproducibility.
            ref_row = sorted(safe_rows, key=lambda r: r.get("perturbation", ""))[0]

            base_meta = dict(
                trajectory_id=iid,
                source_model=ref_row.get("source_model", ""),
                source_dir=source_dir,
                step=target_step,
                compaction_index=snap.compaction_index,
                prev_compaction_step=prev_step,
                problem_statement=problem[:3000],
                compression_ratio=snap.compression_ratio,
                original_tokens=snap.original_tokens,
                filtered_tokens=snap.filtered_tokens,
                forgotten=snap.forgotten,
            )

            # ── Baseline row (= v1) ─────────────────────────────────────
            baseline_user = _build_user_body(problem, prev_summary, baseline_recent)
            tail_truncated = False
            if len(baseline_user) > max_input_chars:
                baseline_user = "[... earlier content truncated ...]\n" + baseline_user[-max_input_chars:]
                tail_truncated = True
            out_rows.append({
                **base_meta,
                "fork_event_id": f"{iid}_{target_step}_baseline",
                "perturbation": "",
                "aug_kind": "baseline",
                "prompt": _build_prompt(baseline_user),
                "completion": gold_summary,
                "num_input_chars": len(baseline_user),
                "completion_chars": len(gold_summary),
            })
            stats["baseline_rows"] += 1
            if tail_truncated:
                stats["tail_truncated"] += 1

            # ── Per-perturbation rows ──────────────────────────────────
            for sr in safe_rows:
                pert = sr.get("perturbation", "")
                if pert in SKIP_PERTURBATIONS:
                    stats[f"skipped_{pert}"] += 1
                    continue

                if pert in OUTPUT_KIND_FACTORIES:
                    strategy = OUTPUT_KIND_FACTORIES[pert]()
                    perturbed_summary = _perturb_summary_text(strategy, gold_summary).strip()
                    if perturbed_summary == gold_summary or not perturbed_summary:
                        stats[f"noop_{pert}"] += 1
                        continue
                    user_body = _build_user_body(problem, prev_summary, baseline_recent)
                    if len(user_body) > max_input_chars:
                        user_body = "[... earlier content truncated ...]\n" + user_body[-max_input_chars:]
                        stats["tail_truncated"] += 1
                    out_rows.append({
                        **base_meta,
                        "fork_event_id": sr.get("fork_event_id", f"{iid}_{target_step}_{pert}"),
                        "perturbation": pert,
                        "aug_kind": "output",
                        "prompt": _build_prompt(user_body),
                        "completion": perturbed_summary,
                        "num_input_chars": len(user_body),
                        "completion_chars": len(perturbed_summary),
                    })
                    stats[f"output_{pert}"] += 1

                elif pert in INPUT_KIND_FACTORIES:
                    strategy = INPUT_KIND_FACTORIES[pert]()
                    perturbed = _perturb_delta_steps(strategy, delta_steps)
                    perturbed_recent = _render_recent_steps_from_pre(perturbed)
                    if perturbed_recent == baseline_recent:
                        stats[f"noop_{pert}"] += 1
                        continue
                    user_body = _build_user_body(problem, prev_summary, perturbed_recent)
                    if len(user_body) > max_input_chars:
                        user_body = "[... earlier content truncated ...]\n" + user_body[-max_input_chars:]
                        stats["tail_truncated"] += 1
                    out_rows.append({
                        **base_meta,
                        "fork_event_id": sr.get("fork_event_id", f"{iid}_{target_step}_{pert}"),
                        "perturbation": pert,
                        "aug_kind": "input",
                        "prompt": _build_prompt(user_body),
                        "completion": gold_summary,
                        "num_input_chars": len(user_body),
                        "completion_chars": len(gold_summary),
                    })
                    stats[f"input_{pert}"] += 1
                else:
                    stats[f"unknown_{pert}"] += 1

    if not out_rows:
        raise RuntimeError("emitted zero rows; check inputs")

    splits = assign_splits((r["trajectory_id"] for r in out_rows), eval_frac)
    with output_path.open("w") as fh:
        for r in out_rows:
            r["split"] = splits[r["trajectory_id"].split("__")[0]]
            fh.write(json.dumps(r) + "\n")

    aug_kind_counts = Counter(r["aug_kind"] for r in out_rows)
    split_counts = Counter(r["split"] for r in out_rows)
    pert_counts = Counter(r["perturbation"] for r in out_rows)
    return {
        "written": len(out_rows),
        "by_aug_kind": dict(aug_kind_counts),
        "by_perturbation": dict(pert_counts),
        "by_split": dict(split_counts),
        "diagnostics": dict(stats),
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
                        default=Path("data/world_model/training_output/midtrain_sft_v2/midtrain_pairs_v2.jsonl"))
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--max-input-chars", type=int, default=48_000)
    args = parser.parse_args()

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
