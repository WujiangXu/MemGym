"""Rebuild strategy_ood_pairs.jsonl using newly-landed reeval reports.

For each sonnet45 OOD strategy with a fresh patch-eval reeval JSON
(structured_ms200, obs_masking, pipeline_mask), call
``memory_gain_pairs.build_pairs_for_strategy`` with explicit
``baseline_reeval`` + ``ood_reeval`` paths so Δr labels are derived
from the patch-eval ``resolved_ids`` (the trajectory's own ``resolved``
field is often stale on EC2-captured replays). Then concatenate all
``pairs_partial.jsonl`` files into a single
``training_output/strategy_ood_v1/strategy_ood_pairs.jsonl``.

NOTE: baseline_none is intentionally NOT in the slate. Its OOD
"strategy" runs no memory at all, so ``extract_compaction_events``
returns zero anchors and the builder yields zero rows by construction.
Including it just produced a misleading 0-row cell — the right
expansion path is to add MORE memory-strategy cohorts (with their own
reeval JSONs), not to keep no-memory cohorts in the slate.

Skips strategies whose required reeval JSON is missing (defer until
the EC2 reeval lands).

Run on EC2 via ``/run-script`` so the trajectory tree + reevals are local:
    python -m memgym.training.scripts.rebuild_strategy_ood_v2 \\
        --repo-root ${REPO_ROOT}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from memgym.training.data.aug_pairs import (
    DEFAULT_LABEL_HARMFUL,
    DEFAULT_LABEL_SAFE,
    SOURCE_MODEL_BY_DIRNAME,
)
from memgym.training.data.memory_gain_pairs import build_pairs_for_strategy
from memgym.training.data.reward_model_pairs import (
    DEFAULT_KEEP_FIRST,
    _build_row,
    _load_trajectory,
)

# Slate of (ood_relpath, ood_reeval_filename, strategy_slug, heldout_dirname,
#          baseline_relpath, baseline_reeval_filename).
#
# Both ``*_relpath`` fields are the run-subdir that actually contains the
# trajectory files (TrajectoryLoader doesn't recurse). The reeval JSON
# lives inside that same subdir.
SLATE = [
    (
        "sonnet45_structured_ms200_r075_kf1/run6d_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-ood-structured-ms200.json",
        "sonnet45_structured_ms200",
        "heldout_sonnet45_structured_ms200",
        "sonnet45_llm_summarizing_ms100_r075_kf3/run7a_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-baseline-llmsumm-n20.json",
    ),
    (
        "sonnet45_obs_masking_w100/run6b_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-ood-obs-masking-w100.json",
        "sonnet45_obs_masking_w100",
        "heldout_sonnet45_obs_masking_w100",
        "sonnet45_llm_summarizing_ms100_r075_kf3/run7a_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-baseline-llmsumm-n20.json",
    ),
    (
        "sonnet45_pipeline_mask_struct_ms100_r075_kf3/run7e_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-ood-pipeline-mask.json",
        "sonnet45_pipeline_mask",
        "heldout_sonnet45_pipeline_mask",
        "sonnet45_llm_summarizing_ms100_r075_kf3/run7a_n20",
        "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-baseline-llmsumm-n20.json",
    ),
    # baseline_none intentionally omitted — no-memory cohorts have zero
    # compaction events to anchor on, so memory_gain_pairs always yields
    # 0 rows. See module docstring.
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: <repo>/training_output/strategy_ood_v1/strategy_ood_pairs.jsonl)",
    )
    args = p.parse_args()

    repo = args.repo_root.resolve()
    traj = repo / "trajectories"
    heldout_root = repo / "data" / "world_model" / "aug_pairs"
    out = args.out or repo / "training_output" / "strategy_ood_v1" / "strategy_ood_pairs.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)

    # (pairs_partial_path, ood_dir) tuples; ood_dir is needed downstream to
    # resolve per-instance training/replay JSONs for long-context inflation.
    built: list[tuple[Path, Path]] = []
    for ood_rel, ood_re_name, slug, heldout_name, base_rel, base_re_name in SLATE:
        ood_dir = traj / ood_rel
        ood_re = ood_dir / ood_re_name
        baseline_dir = traj / base_rel
        base_re = baseline_dir / base_re_name
        out_dir = heldout_root / heldout_name

        if not ood_re.exists():
            print(f"  [skip] {slug}: missing OOD reeval {ood_re}")
            continue
        if not base_re.exists():
            print(f"  [defer] {slug}: baseline reeval not yet ready ({base_re})")
            continue

        print(f"=== building: {slug} ===")
        print(f"  baseline_dir    = {baseline_dir}")
        print(f"  ood_dir         = {ood_dir}")
        print(f"  ood_reeval      = {ood_re}")
        print(f"  baseline_reeval = {base_re}")

        stats = build_pairs_for_strategy(
            baseline_dir=baseline_dir,
            ood_dir=ood_dir,
            strategy_name=slug,
            out_dir=out_dir,
            baseline_reeval=base_re,
            ood_reeval=ood_re,
        )
        print(json.dumps(asdict(stats), indent=2))
        built.append((out_dir / "pairs_partial.jsonl", ood_dir))

    # Concatenate + INFLATE to the long-context aug_sft schema by replaying
    # each pair's OOD trajectory through `reward_model_pairs._build_row`.
    # This mirrors how v2 training data was rendered (full conversation
    # history + active compaction summary + judgment turn), so the
    # eval prompt-length distribution matches training (~22-34K tokens)
    # instead of the ~600-token short-template that the simple
    # PROMPT_TEMPLATE.format() produced.
    #
    # We use the OOD trajectory's view (not the baseline's) because the
    # RM is being asked "is this OOD-strategy memory decision safe?" —
    # the relevant context is the OOD strategy's own state at step k.
    # NB: `_reconstruct_view` derives the view from each step's recorded
    # `step.memory` metadata; for non-summarizing strategies (obs_masking)
    # the trajectory has no summary stored, so it returns the unfiltered
    # raw history slice — close to the agent's view but not byte-identical.
    print(f"\n=== inflating + concatenating into {out} ===")
    n_total = 0
    n_inflate_failed = 0
    with out.open("w") as fh:
        for pf, ood_dir in built:
            if not pf.exists() or pf.stat().st_size == 0:
                continue
            heldout_dir = pf.parent.name
            source_model = SOURCE_MODEL_BY_DIRNAME.get(heldout_dir, heldout_dir)
            n_dir = 0
            cache_by_iid: dict = {}
            with pf.open() as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    label = int(raw["label"])
                    target = DEFAULT_LABEL_SAFE if label == 1 else DEFAULT_LABEL_HARMFUL
                    iid = raw["instance_id"]

                    cache = cache_by_iid.get(iid)
                    if cache is None:
                        training_path = ood_dir / f"{iid}_training.json"
                        replay_path = ood_dir / f"{iid}_replay.json"
                        if not (training_path.exists() and replay_path.exists()):
                            print(f"  [warn] {iid}: missing training/replay under {ood_dir}; skipping")
                            n_inflate_failed += 1
                            continue
                        try:
                            cache = _load_trajectory(training_path, replay_path)
                        except (OSError, json.JSONDecodeError) as exc:
                            print(f"  [warn] {iid}: load error {exc}; skipping")
                            n_inflate_failed += 1
                            continue
                        cache_by_iid[iid] = cache

                    src_row = {
                        "trajectory_id": iid,
                        "instance_id": iid,
                        "step": int(raw["step"]),
                        "perturbation": raw["perturbation"],
                        "label": label,
                        "recorded_action": raw["recorded_action"],
                        "predicted_action": raw["predicted_action"],
                        "diverged": bool(raw.get("diverged", label == 0)),
                        "source_dir": heldout_dir,
                        "source_model": source_model,
                        "fork_event_id": f"{iid}_{raw['step']}_{raw['perturbation']}",
                    }
                    result = _build_row(src_row, cache, DEFAULT_KEEP_FIRST)
                    if result.row is None:
                        print(f"  [warn] {iid} step={raw['step']}: {result.skip_reason}")
                        n_inflate_failed += 1
                        continue

                    row = result.row
                    # Force eval split + carry through OOD bookkeeping.
                    row["split"] = "eval"
                    row["target"] = target
                    row["delta_r"] = raw.get("delta_r")
                    row["ood_strategy"] = raw.get("ood_strategy")
                    fh.write(json.dumps(row) + "\n")
                    n_dir += 1
            n_total += n_dir
            print(f"  + {heldout_dir}: {n_dir} rows  (source_model={source_model})")
    print(f"\ninflated: {n_total} rows  ({n_inflate_failed} failed) → {out}")

    print("\n=== Δr label distribution by strategy ===")
    by_slug: dict[str, Counter] = {}
    with out.open() as fh:
        for line in fh:
            r = json.loads(line)
            s = r.get("ood_strategy", "?")
            by_slug.setdefault(s, Counter())["safe" if r.get("label") == 1 else "harmful"] += 1
            by_slug[s][f"dr_{r.get('delta_r')}"] += 1

    for s, c in sorted(by_slug.items()):
        safe = c.get("safe", 0)
        harm = c.get("harmful", 0)
        dr_breakdown = "  ".join(f"{k}={v}" for k, v in sorted(c.items()) if k.startswith("dr_"))
        print(f"  {s:35} n={safe + harm:4}  SAFE={safe:4}  HARMFUL={harm:4}  {dr_breakdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
