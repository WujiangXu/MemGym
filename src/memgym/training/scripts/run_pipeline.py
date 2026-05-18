"""End-to-end world model training pipeline.

Runs: load trajectories → extract compaction events → (optional) augment
→ build SFT dataset → train → evaluate.

Usage:
    python -m memgym.training.scripts.run_pipeline \
        --strategy-dir ec2_trajectories/train50_sonnet45_llmsummarizing \
        --output-dir training_output/

    # With augmentation
    python -m memgym.training.scripts.run_pipeline \
        --strategy-dir ec2_trajectories/diverse500_sonnet45_llmsumm_ms100 \
        --augment \
        --policy-model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --train \
        --output-dir training_output/

    # With baseline pairing
    python -m memgym.training.scripts.run_pipeline \
        --strategy-dir ec2_trajectories/diverse500_sonnet45_llmsumm_ms100 \
        --baseline-dir ec2_trajectories/sonnet45_baseline_diverse500 \
        --augment --train \
        --output-dir training_output/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end world model training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data sources
    p.add_argument(
        "--strategy-dir", required=True,
        help="Directory with memory-augmented trajectories (_training.json)",
    )
    p.add_argument(
        "--baseline-dir", default=None,
        help="Directory with baseline trajectories (for paired labeling)",
    )

    # Pipeline stages
    p.add_argument("--augment", action="store_true",
                    help="Run counterfactual replay augmentation")
    p.add_argument("--judge", action="store_true",
                    help="Run LLM-as-a-judge (requires API access)")
    p.add_argument("--train", action="store_true",
                    help="Train the world model after dataset construction")
    p.add_argument("--evaluate", action="store_true",
                    help="Evaluate after training")

    # Augmentation options
    p.add_argument(
        "--policy-model",
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="LLM for counterfactual replay",
    )
    p.add_argument(
        "--judge-model",
        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="LLM for judge evaluation",
    )
    p.add_argument("--max-api-calls", type=int, default=500,
                    help="Budget for API calls (augmentation + judge)")
    p.add_argument("--max-events-per-instance", type=int, default=10,
                    help="Max compaction events to augment per instance")

    # Training options
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="Base model for world model training")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)

    # Output
    p.add_argument("--output-dir", default="training_output/",
                    help="Output directory for dataset, checkpoints, metrics")

    # Misc
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # ── Step 1: Load trajectories ──────────────────────────────────────
    from memgym.training.data.loader import TrajectoryLoader

    strategy_dir = Path(args.strategy_dir)
    loader = TrajectoryLoader(strategy_dir.parent)
    experiment_name = strategy_dir.name

    logger.info("Loading trajectories from %s ...", experiment_name)
    trajectories = loader.load_experiment(experiment_name, load_messages=True)
    logger.info("Loaded %d trajectories", len(trajectories))

    if not trajectories:
        logger.error("No trajectories found in %s", strategy_dir)
        sys.exit(1)

    # ── Step 2: Extract compaction events ──────────────────────────────
    from memgym.training.data.compaction import extract_all_compaction_events

    all_events = extract_all_compaction_events(trajectories)
    logger.info("Extracted %d compaction events", len(all_events))

    if not all_events:
        logger.error("No compaction events found. Are _training.json files present?")
        sys.exit(1)

    # ── Step 3: Paired labeling (optional) ─────────────────────────────
    if args.baseline_dir:
        from memgym.training.data.labels import compute_paired_labels

        baseline_dir = Path(args.baseline_dir)
        baseline_trajs = loader.load_experiment(baseline_dir.name)
        labels = compute_paired_labels(
            baseline_trajs, trajectories, lambda_efficiency=0.1,
        )
        logger.info("Computed %d paired labels", len(labels))
        # Save labels
        labels_path = output_dir / "paired_labels.json"
        with open(labels_path, "w") as f:
            json.dump([vars(l) for l in labels], f, indent=2)

    # ── Step 4: Build initial dataset ──────────────────────────────────
    from memgym.training.data.dataset import (
        WorldModelDataset, save_dataset_jsonl,
    )

    dataset = WorldModelDataset()
    added = dataset.add_from_events(all_events)
    logger.info("Added %d episode-label examples", added)

    # ── Step 5: Counterfactual augmentation (optional) ─────────────────
    augmentation_pairs = []
    if args.augment:
        from memgym.training.augmentation.replay import SWEBenchReplayAugmenter

        augmenter = SWEBenchReplayAugmenter(
            policy_model=args.policy_model,
            max_api_calls=args.max_api_calls,
            rate_limit_delay=0.5,
        )

        traj_dict = {t.instance_id: t for t in trajectories}
        results = augmenter.augment_batch(
            traj_dict, all_events,
            max_events_per_instance=args.max_events_per_instance,
        )

        for r in results:
            augmentation_pairs.extend(r.pairs)

        logger.info("Augmentation: %d pairs, %d diverged",
                     len(augmentation_pairs),
                     sum(1 for p in augmentation_pairs if p.diverged))

        events_by_key = {f"{e.instance_id}_{e.step}": e for e in all_events}
        aug_added = dataset.add_from_counterfactual_pairs(
            augmentation_pairs, events_by_key,
        )
        logger.info("Added %d augmentation examples", aug_added)

    # ── Step 6: LLM-as-a-judge (optional) ─────────────────────────────
    if args.judge:
        from memgym.training.augmentation.llm_judge import LLMJudge

        judge = LLMJudge(
            model=args.judge_model,
            max_calls=args.max_api_calls,
        )
        judge_results = judge.judge_batch(all_events, max_events=200)
        logger.info("Judge: %d results", len(judge_results))

        events_by_key = {f"{e.instance_id}_{e.step}": e for e in all_events}
        judge_added = dataset.add_from_judge_results(judge_results, events_by_key)
        logger.info("Added %d judge examples", judge_added)

    # ── Step 7: Save dataset ──────────────────────────────────────────
    dataset_path = output_dir / "dataset.jsonl"
    save_dataset_jsonl(dataset.examples, dataset_path)

    stats = dataset.stats()
    logger.info("Dataset stats: %s", json.dumps(stats, indent=2))
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Step 8: Train / Evaluate ──────────────────────────────────────
    # The old in-process trainer (`WorldModelTrainer`) was dropped in
    # favor of the cheat-doc SFT recipe. If `--train` was passed, tell
    # the user how to run the replacement; don't silently no-op.
    if args.train or args.evaluate:
        logger.warning(
            "--train/--evaluate are no longer handled inside run_pipeline. "
            "Use the SFT recipe instead:\n"
            "  python -m memgym.training.scripts.train_sft ...\n"
            "  python -m memgym.training.scripts.eval_aug_sft "
            "--checkpoint <dir> --pairs <jsonl>"
        )

    elapsed = time.time() - start
    logger.info("Pipeline complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
