"""Augmentation-pair adapter: pairs_partial.jsonl → SFT-pairs JSONL.

The recipe builds the prompt directly from the pair's own
`recorded_action` + `predicted_action`
strings — no trajectory/CompactionEvent join needed.

Every pair row has these required keys
(`MemGym/src/memgym/training/augmentation/env_replay.py:125-151`):

    {
        "instance_id": str,
        "step": int,
        "perturbation": str,
        "diverged": bool,
        "label": int,                # 0=HARMFUL, 1=SAFE
        "recorded_action": str,      # ≤500 chars
        "predicted_action": str,     # ≤500 chars
    }

Output columns (one JSONL line per pair):

    {
        "trajectory_id": instance_id,
        "fork_event_id": f"{instance_id}_{step}_{perturbation}",
        "input":   rendered prompt,
        "prompt":  input + RESPONSE_PREFIX (what TRL sees),
        "completion": "SAFE" | "HARMFUL",
        "target":  same as completion (legacy schema column),
        "label":   0 | 1,
        "source_model": "sonnet45" | "gptoss120b" | "haiku45" | ...,
        "split":   "train" | "eval"  (group-shuffled by repo_prefix),
    }

`completion` + `prompt` are the columns TRL ≥1.0 SFTTrainer reads when
`completion_only_loss=True`; `input` + `target` are preserved for any
legacy GRPO/evaluator callers.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

logger = logging.getLogger(__name__)

# Default the training phase label pair. Single-token + balanced under Qwen3.5-9B-Base
# BPE: " Y" -> [783], " N" -> [443]. The answer-hint inside the prompt is
# rebuilt from the label pair so "answer what?" matches "target what?"
# across the label-axis sweep (Phase A). RESPONSE_PREFIX has no trailing
# space — the labels carry their own leading space so they stay
# single-token.
RESPONSE_PREFIX = "\n\nAnswer:"
DEFAULT_LABEL_SAFE = " Y"
DEFAULT_LABEL_HARMFUL = " N"

# Kept as aliases for callers that imported the old names. New code should
# take labels as explicit arguments (parameterized by Phase A).
LABEL_SAFE_STR = DEFAULT_LABEL_SAFE
LABEL_HARMFUL_STR = DEFAULT_LABEL_HARMFUL


def build_prompt_template(label_safe: str, label_harmful: str) -> str:
    """Render the pair-prompt template with a label-aware answer hint.

    The hint mirrors whichever label pair the completion field uses, so
    that Phase A's yes/no and good/bad sweeps see prompts that match
    their targets instead of a fixed "Answer Y / N" instruction.
    """
    return (
        "Trajectory step {step} with perturbation '{perturbation}'.\n\n"
        "Recorded action:\n{recorded}\n\n"
        "Proposed action:\n{predicted}\n\n"
        "Is the proposed action safe (would not regress task resolution)? "
        f"Answer {label_safe.strip()} if safe, {label_harmful.strip()} if harmful."
    )


PROMPT_TEMPLATE = build_prompt_template(DEFAULT_LABEL_SAFE, DEFAULT_LABEL_HARMFUL)

SOURCE_MODEL_BY_DIRNAME = {
    "sonnet_fork_gap554_scaleA_v1": "sonnet45",
    "gptoss_fork_gap554_scaleA_v1": "gptoss120b",
    "haiku45_fork_diverse500_scaleA_v1": "haiku45",
    "sonnet_fork_gap554_scopedB": "sonnet45",
    "sonnet_fork_diverse500_scopedB": "sonnet45",
    "sonnet_fork_gap554_scopedB_v2": "sonnet45",
    # Strategy-OOD held-out dirs (Plan v3 §A0 — memory-gain labeling
    # via `memory_gain_pairs.py`). The `source_model` value encodes the
    # strategy slice so downstream `per_source_threshold_sweep` and
    # `audit_cross_model_variance` bucket on it automatically.
    "heldout_sonnet45_structured_250steps": "sonnet45_structured_250steps",
    "heldout_haiku45_structured_ms100":     "haiku45_structured_ms100",
    # rebuild_strategy_ood_v2 slate (4 sonnet45 strategies, episode-Δr
    # labeling). See plan v3 §A0.
    "heldout_sonnet45_structured_ms200":    "sonnet45_structured_ms200",
    "heldout_sonnet45_obs_masking_w100":    "sonnet45_obs_masking_w100",
    "heldout_sonnet45_pipeline_mask":       "sonnet45_pipeline_mask",
    "heldout_sonnet45_none":                "sonnet45_none",
    # strategy-OOD v2 expansion (5 additional cohorts, Plan v3 §A1 Step 1).
    "heldout_sliding_window_w75":               "sonnet45_sliding_window",
    "heldout_sonnet45_structured_ms400":        "sonnet45_structured_ms400",
    "heldout_sonnet45_structured_ms100_r050":   "sonnet45_structured_r050",
    "heldout_sonnet45_structured_ms100_kf1":    "sonnet45_structured_kf1",
    "heldout_sonnet45_obs_masking_selective":   "sonnet45_obs_masking_selective",
    # EC2 cohorts added opportunistically (larger n, share baseline reeval).
    "heldout_sonnet45_structured_250steps_ec2": "sonnet45_structured_250steps_ec2",
    "heldout_haiku45_structured_ms100_r075":    "haiku45_structured_ms100_r075",
    # v2 rebuilt cohorts — none baseline (memory=none), episode-Δr labeling.
    # These supersede the old llm_summarizing-baseline versions.
    "heldout_sliding_window_w75_v2":              "sonnet45_sliding_window",
    "heldout_sonnet45_structured_ms400_v2":       "sonnet45_structured_ms400",
    "heldout_sonnet45_structured_ms100_r050_v2":  "sonnet45_structured_r050",
    "heldout_sonnet45_structured_ms100_kf1_v2":   "sonnet45_structured_kf1",
    "heldout_sonnet45_obs_masking_selective_v2":  "sonnet45_obs_masking_selective",
    "heldout_sonnet45_obs_masking_w100_v2":       "sonnet45_obs_masking",
    "heldout_sonnet45_pipeline_mask_v2":          "sonnet45_pipeline_mask",
}


def _source_model_for(dir_path: Path) -> str:
    return SOURCE_MODEL_BY_DIRNAME.get(dir_path.name, dir_path.name)


def _iter_pairs(dir_path: Path, skip_bad: bool = True) -> Iterable[dict]:
    """Yield validated pair rows from `<dir_path>/pairs_partial.jsonl`.

    Drops rows with bad JSON, missing required keys, or empty
    `recorded_action` (the GPT-OSS dataset has ~2.8k duplicates and a
    long tail of empty actions that the adapter layer filters).
    """
    pairs_path = dir_path / "pairs_partial.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"missing pairs_partial.jsonl under {dir_path}")
    source = _source_model_for(dir_path)
    bad_json = 0
    bad_schema = 0
    empty_action = 0
    with pairs_path.open() as fh:
        for line_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_bad:
                    bad_json += 1
                    continue
                raise ValueError(f"{pairs_path}:{line_idx} bad JSON: {exc}") from exc
            missing = [k for k in ("instance_id", "step", "perturbation", "label",
                                   "recorded_action", "predicted_action")
                       if k not in row]
            if missing:
                if skip_bad:
                    bad_schema += 1
                    continue
                raise ValueError(f"{pairs_path}:{line_idx} missing keys {missing!r}")
            if not (row.get("recorded_action") or "").strip():
                empty_action += 1
                continue
            yield {**row, "_source_model": source, "_source_dir": dir_path.name}
    if bad_json or bad_schema or empty_action:
        logger.info(
            "skipped from %s: bad_json=%d bad_schema=%d empty_action=%d",
            pairs_path, bad_json, bad_schema, empty_action,
        )


def _to_sft_row(
    raw: dict,
    label_safe: str = DEFAULT_LABEL_SAFE,
    label_harmful: str = DEFAULT_LABEL_HARMFUL,
    template: str = PROMPT_TEMPLATE,
) -> dict:
    label = int(raw["label"])
    target = label_safe if label == 1 else label_harmful
    fork_event_id = f"{raw['instance_id']}_{raw['step']}_{raw['perturbation']}"
    rendered = template.format(
        step=raw["step"],
        perturbation=raw["perturbation"],
        recorded=raw["recorded_action"],
        predicted=raw["predicted_action"],
    )
    return {
        "trajectory_id": raw["instance_id"],
        "fork_event_id": fork_event_id,
        "input": rendered,
        "prompt": rendered + RESPONSE_PREFIX,
        "completion": target,
        "target": target,
        "label": label,
        "source_model": raw["_source_model"],
        "source_dir": raw["_source_dir"],
        "perturbation": raw["perturbation"],
        "step": int(raw["step"]),
    }


def assign_splits(
    trajectory_ids: Iterable[str],
    eval_frac_target: float = 0.10,
) -> dict:
    """Group-shuffle by `repo_prefix = trajectory_id.split("__")[0]`.

    Greedy: rank repos by (pair count ASC, repo name ASC), pick smallest
    first until cumulative pair count >= `eval_frac_target * total`.
    Returns `{repo_prefix: "train"|"eval"}`. The secondary repo-name key
    makes the result independent of `trajectory_ids` iteration order — so
    re-running on the same row set produces byte-identical splits.
    """
    by_repo: Counter = Counter()
    for tid in trajectory_ids:
        by_repo[tid.split("__")[0]] += 1
    total = sum(by_repo.values())
    target = int(total * eval_frac_target)

    ordered = sorted(by_repo.items(), key=lambda kv: (kv[1], kv[0]))
    eval_repos: Set[str] = set()
    cum = 0
    for repo, count in ordered:
        if cum >= target:
            break
        eval_repos.add(repo)
        cum += count
    split = {repo: ("eval" if repo in eval_repos else "train") for repo in by_repo}
    logger.info(
        "split: %d/%d repos to eval (%d/%d pairs = %.1f%%)",
        len(eval_repos), len(by_repo), cum, total, 100 * cum / max(total, 1),
    )
    return split


def convert_aug_dirs(
    dirs: List[Path],
    out_path: Path,
    limit: int = 0,
    eval_frac: float = 0.10,
    label_safe: str = DEFAULT_LABEL_SAFE,
    label_harmful: str = DEFAULT_LABEL_HARMFUL,
) -> dict:
    """Round-robin interleave source dirs, write combined SFT-pairs JSONL.

    `label_safe`/`label_harmful` drive BOTH the `completion` field and
    the answer hint inside the rendered prompt (via
    `build_prompt_template`). Phase A's label-axis sweep flips only
    these two strings.

    Sources are interleaved (not concatenated) so any `limit` cap still
    spans every teacher model. The split is computed once on the fully
    materialized pair stream — train rows are the majority, eval rows
    are the smallest-repo tail.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = build_prompt_template(label_safe, label_harmful)

    # First pass: materialize all rows into memory so we can compute
    # splits before writing. At ~15K rows × ~2 KB each this is ~30 MB.
    iters = [(d, iter(_iter_pairs(d))) for d in dirs]
    rows: List[dict] = []
    counts = {_source_model_for(d): 0 for d in dirs}
    label_counts = {0: 0, 1: 0}
    exhausted = [False] * len(iters)
    while not all(exhausted):
        for idx, (_d, it) in enumerate(iters):
            if exhausted[idx]:
                continue
            try:
                raw = next(it)
            except StopIteration:
                exhausted[idx] = True
                continue
            row = _to_sft_row(raw, label_safe, label_harmful, template)
            rows.append(row)
            counts[row["source_model"]] += 1
            label_counts[row["label"]] += 1
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    splits = assign_splits((r["trajectory_id"] for r in rows), eval_frac)
    with out_path.open("w") as out_fh:
        for r in rows:
            r["split"] = splits[r["trajectory_id"].split("__")[0]]
            out_fh.write(json.dumps(r) + "\n")

    split_counts = Counter(r["split"] for r in rows)
    stats = {
        "written": len(rows),
        "by_source": counts,
        "by_label": label_counts,
        "by_split": dict(split_counts),
        "out_path": str(out_path),
        "label_safe": label_safe,
        "label_harmful": label_harmful,
    }
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", action="append", dest="dirs", required=True,
                        type=Path,
                        help="Augmentation output dir (repeatable)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Combined SFT-pairs JSONL output path")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max rows total (0 = unlimited)")
    parser.add_argument("--eval-frac", type=float, default=0.10,
                        help="Fraction of pairs (by group-shuffle on repo prefix) to hold out")
    parser.add_argument("--label-safe", default=DEFAULT_LABEL_SAFE,
                        help="Completion string for label=1 (must be single-token under the base "
                             "tokenizer; leading space kept with the token, e.g. ' Y', ' yes', ' good')")
    parser.add_argument("--label-harmful", default=DEFAULT_LABEL_HARMFUL,
                        help="Completion string for label=0 (same single-token constraint)")
    args = parser.parse_args()

    stats = convert_aug_dirs(
        args.dirs, args.out, args.limit, args.eval_frac,
        label_safe=args.label_safe, label_harmful=args.label_harmful,
    )
    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
