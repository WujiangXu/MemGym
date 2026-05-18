"""Augmentation-only CLI for counterfactual replay.

Generates training pairs by replaying compaction events with perturbed
memory contexts and detecting action divergence.

Usage:
    python -m memgym.training.scripts.augment \
        --strategy-dir ec2_trajectories/train50_sonnet45_llmsummarizing \
        --policy-model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --output augmentation_output/

    # Dry run (no API calls, just extract events)
    python -m memgym.training.scripts.augment \
        --strategy-dir ec2_trajectories/train50_sonnet45_llmsummarizing \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual replay augmentation")

    p.add_argument("--strategy-dir", required=True,
                    help="Directory with memory-augmented trajectories")
    p.add_argument("--output", default="augmentation_output/",
                    help="Output directory for augmentation results")

    # API
    p.add_argument("--policy-model",
                    default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--max-api-calls", type=int, default=500)
    p.add_argument("--max-events-per-instance", type=int, default=10)
    p.add_argument("--rate-limit-delay", type=float, default=0.5)

    # Perturbations
    p.add_argument("--perturbations", default="all",
                    help="Comma-separated: aggressive,lenient,random_drop,skip_compression,"
                         "summary_redaction,attr_delete_paths,attr_delete_errors,"
                         "truncate_last_10,truncate_last_20,summary_noise,llm_summary_rewrite")

    # Env-feedback mode
    p.add_argument("--env-feedback", action="store_true",
                    help="Execute perturbed actions in Docker containers (ground-truth labels)")
    p.add_argument("--cleanup-images", action="store_true",
                    help="Delete Docker images after processing each batch (saves disk)")
    p.add_argument("--batch-size", type=int, default=10,
                    help="Number of instances to process before cleaning up images")

    # Judge (optional)
    p.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-a-judge")
    p.add_argument("--judge-model",
                    default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--max-judge-calls", type=int, default=200)

    # Modes
    p.add_argument("--dry-run", action="store_true",
                    help="Extract events only, no API calls")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.verbose:
        logging.getLogger("memgym.training").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        # Keep litellm quiet — its debug output is enormous
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    from memgym.training.data.loader import TrajectoryLoader
    from memgym.training.data.compaction import extract_all_compaction_events

    strategy_dir = Path(args.strategy_dir)
    loader = TrajectoryLoader(strategy_dir.parent)
    experiment_name = strategy_dir.name

    logger.info("Loading trajectories from %s ...", experiment_name)
    trajectories = loader.load_experiment(experiment_name, load_messages=True)
    logger.info("Loaded %d trajectories", len(trajectories))

    all_events_list = extract_all_compaction_events(trajectories)
    # Group by instance_id for batch processing
    all_events: Dict[str, list] = {}
    for e in all_events_list:
        all_events.setdefault(e.instance_id, []).append(e)
    total_events = len(all_events_list)
    logger.info("Extracted %d compaction events from %d instances",
                total_events, len(all_events))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save event summary
    event_summary = {
        iid: len(events) for iid, events in all_events.items()
    }
    with open(output_dir / "event_counts.json", "w") as f:
        json.dump(event_summary, f, indent=2)

    if args.dry_run:
        logger.info("Dry run — skipping API calls. %d events extracted.", total_events)
        return

    # ── Select perturbations ─────────────────────────────────────────
    perturbations = None
    if args.perturbations != "all":
        from memgym.training.augmentation.perturbations import (
            AggressiveCompression, LenientCompression,
            RandomDrop, SkipCompression,
            SummaryRedaction, AttributeDeletion, ContextTruncation,
            SummaryNoise, LLMSummaryRewrite,
        )
        name_map = {
            "aggressive": AggressiveCompression(),
            "lenient": LenientCompression(),
            "random_drop": RandomDrop(),
            "skip_compression": SkipCompression(),
            "summary_redaction": SummaryRedaction(),
            "attr_delete_paths": AttributeDeletion("paths"),
            "attr_delete_errors": AttributeDeletion("errors"),
            "truncate_last_10": ContextTruncation(10),
            "truncate_last_20": ContextTruncation(20),
            "summary_noise": SummaryNoise(),
            "llm_summary_rewrite": LLMSummaryRewrite(),
        }
        perturbations = [
            name_map[n.strip()]
            for n in args.perturbations.split(",")
            if n.strip() in name_map
        ]
        if not perturbations:
            logger.error("No valid perturbations selected from: %s", args.perturbations)
            return

    # ── Replay augmentation ───────────────────────────────────────────
    if args.env_feedback:
        from memgym.training.augmentation.env_replay import EnvFeedbackReplayer

        augmenter = EnvFeedbackReplayer(
            policy_model=args.policy_model,
            perturbations=perturbations,
            max_api_calls=args.max_api_calls,
            rate_limit_delay=args.rate_limit_delay,
            cleanup_images=args.cleanup_images,
        )
        results = augmenter.replay_batch(
            trajectories, all_events,
            max_events_per_instance=args.max_events_per_instance,
            batch_size=args.batch_size,
            checkpoint_dir=output_dir,
        )
    else:
        from memgym.training.augmentation.replay import SWEBenchReplayAugmenter

        augmenter = SWEBenchReplayAugmenter(
            policy_model=args.policy_model,
            perturbations=perturbations,
            max_api_calls=args.max_api_calls,
            rate_limit_delay=args.rate_limit_delay,
        )
        results = augmenter.augment_batch(
            trajectories, all_events,
            max_events_per_instance=args.max_events_per_instance,
        )

    # Save results. When env_feedback is on, `pairs_partial.jsonl` is the
    # append-only source of truth for every pair across all runs on this
    # output dir (populated by env_replay._append_checkpoint). Prefer it so
    # that a cover-remaining run that produces zero new pairs cannot wipe
    # the historical pairs already written by prior runs.
    partial_path = output_dir / "pairs_partial.jsonl"
    all_pairs = []
    if args.env_feedback and partial_path.is_file() and partial_path.stat().st_size > 0:
        with partial_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        for r in results:
            for pair in r.pairs:
                all_pairs.append({
                    "instance_id": pair.instance_id,
                    "step": pair.step,
                    "perturbation": pair.perturbation_name,
                    "diverged": pair.diverged,
                    "label": pair.label,
                    "recorded_action": pair.recorded_action[:500],
                    "predicted_action": pair.predicted_action[:500],
                })

    with open(output_dir / "augmentation_pairs.json", "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Summary
    total_pairs = len(all_pairs)
    diverged = sum(1 for p in all_pairs if p.get("diverged"))
    logger.info(
        "Augmentation complete: %d pairs, %d diverged (%.1f%%), %d API calls",
        total_pairs, diverged,
        100 * diverged / max(1, total_pairs),
        augmenter._api_calls,
    )

    # ── LLM-as-a-judge (optional) ────────────────────────────────────
    if args.judge:
        from memgym.training.augmentation.llm_judge import LLMJudge

        flat_events = [e for events in all_events.values() for e in events]
        judge = LLMJudge(
            model=args.judge_model,
            max_calls=args.max_judge_calls,
        )
        judge_results = judge.judge_batch(flat_events)

        judge_out = []
        for jr in judge_results:
            judge_out.append({
                "instance_id": jr.instance_id,
                "step": jr.step,
                "verdict": jr.verdict,
                "label": jr.label,
                "explanation": jr.explanation[:500],
            })
        with open(output_dir / "judge_results.json", "w") as f:
            json.dump(judge_out, f, indent=2)

        logger.info("Judge: %d results", len(judge_results))


if __name__ == "__main__":
    main()
