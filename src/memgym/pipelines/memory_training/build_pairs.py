#!/usr/bin/env python
"""
Build paired dataset from consolidated trajectories.

Joins baseline + strategy trajectories per instance, attaches reeval labels,
and computes multi-dimensional reward scores.

Usage:
    python -m pipelines.memory_training.build_pairs \
        --trajectories results/trajectories_all \
        --reeval-baseline results/trajectories_all/baseline/reeval_report.json \
        --reeval-strategy results/trajectories_all/llm_summarizing/reeval_report.json \
        --strategy llm_summarizing \
        --output pipelines/memory_training/data/paired.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_reeval_report(path: Path) -> Dict[str, bool]:
    """Load reeval report and return {instance_id: resolved}."""
    if not path.exists():
        print(f"Warning: reeval report not found at {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    resolved_ids = set(data.get("resolved", []))
    # Build mapping for all submitted instances
    all_ids = set(data.get("resolved", [])) | set(data.get("unresolved", []))
    return {iid: iid in resolved_ids for iid in all_ids}


def load_replay(path: Path) -> Optional[Dict]:
    """Load _replay.json and extract key metadata."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    messages = data.get("messages", [])
    steps = data.get("steps", [])

    # Estimate token count from message content
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    estimated_tokens = total_chars // 4  # rough estimate

    return {
        "num_messages": len(messages),
        "num_steps": len(steps),
        "status": data.get("status", "unknown"),
        "patch": data.get("patch", ""),
        "patch_size": len(data.get("patch", "")),
        "memory_strategy": data.get("memory_strategy", "unknown"),
        "memory_config": data.get("memory_config", {}),
        "estimated_tokens": estimated_tokens,
    }


def load_training_json(path: Path) -> Optional[Dict]:
    """Load _training.json and extract memory stats per step."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    steps = data.get("steps", [])
    compression_steps = []
    total_original_tokens = 0
    total_filtered_tokens = 0
    summarization_count = 0

    for step in steps:
        mem = step.get("memory", {})
        total_original_tokens += mem.get("original_tokens", 0)
        total_filtered_tokens += mem.get("filtered_tokens", 0)
        if mem.get("was_compacted", False):
            summarization_count += 1
            compression_steps.append({
                "step": step.get("step"),
                "compression_ratio": mem.get("compression_ratio", 1.0),
                "original_tokens": mem.get("original_tokens", 0),
                "filtered_tokens": mem.get("filtered_tokens", 0),
                "forgotten": mem.get("forgotten", 0),
            })

    return {
        "num_steps": len(steps),
        "total_original_tokens": total_original_tokens,
        "total_filtered_tokens": total_filtered_tokens,
        "summarization_count": summarization_count,
        "compression_steps": compression_steps,
        "avg_compression_ratio": (
            total_original_tokens / total_filtered_tokens
            if total_filtered_tokens > 0
            else 1.0
        ),
    }


def compute_reward(
    baseline_resolved: bool,
    strategy_resolved: bool,
    baseline_tokens: int,
    strategy_tokens: int,
    summarizer_tokens: int = 0,
    lambda_efficiency: float = 0.3,
) -> Dict[str, float]:
    """Compute multi-dimensional reward.

    reward = resolve_score + lambda * token_efficiency_score

    Token efficiency only counts when strategy resolves.
    """
    resolve_score = 1.0 if strategy_resolved else 0.0
    baseline_resolve_score = 1.0 if baseline_resolved else 0.0

    # Net strategy tokens (agent tokens + summarizer overhead)
    strategy_net_tokens = strategy_tokens + summarizer_tokens

    # Token efficiency: how much we saved relative to baseline
    if baseline_tokens > 0 and strategy_resolved:
        token_efficiency = (baseline_tokens - strategy_net_tokens) / baseline_tokens
    else:
        token_efficiency = 0.0

    # Combined reward
    reward = resolve_score + lambda_efficiency * token_efficiency

    return {
        "reward": round(reward, 4),
        "resolve_score": resolve_score,
        "token_efficiency_score": round(token_efficiency, 4),
        "reward_delta": resolve_score - baseline_resolve_score,
        "baseline_tokens": baseline_tokens,
        "strategy_net_tokens": strategy_net_tokens,
        "summarizer_tokens": summarizer_tokens,
    }


def build_pairs(
    trajectories_dir: Path,
    strategy: str,
    reeval_baseline: Dict[str, bool],
    reeval_strategy: Dict[str, bool],
    lambda_efficiency: float = 0.3,
) -> List[Dict]:
    """Build paired dataset for all instances with both baseline + strategy."""
    baseline_traj_dir = trajectories_dir / "baseline" / "trajectories"
    strategy_traj_dir = trajectories_dir / strategy / "trajectories"

    if not baseline_traj_dir.exists():
        print(f"Error: baseline trajectories not found at {baseline_traj_dir}")
        return []
    if not strategy_traj_dir.exists():
        print(f"Error: {strategy} trajectories not found at {strategy_traj_dir}")
        return []

    # Find instances with both baseline and strategy replays
    baseline_replays = {
        f.name.replace("_replay.json", ""): f
        for f in baseline_traj_dir.glob("*_replay.json")
    }
    strategy_replays = {
        f.name.replace("_replay.json", ""): f
        for f in strategy_traj_dir.glob("*_replay.json")
    }

    common_ids = sorted(set(baseline_replays.keys()) & set(strategy_replays.keys()))
    print(f"Found {len(common_ids)} instances with both baseline + {strategy}")

    pairs = []
    for instance_id in common_ids:
        # Load replay metadata
        bl_replay = load_replay(baseline_replays[instance_id])
        st_replay = load_replay(strategy_replays[instance_id])

        if bl_replay is None or st_replay is None:
            continue

        # Load training.json if available
        bl_training_path = baseline_traj_dir / f"{instance_id}_training.json"
        st_training_path = strategy_traj_dir / f"{instance_id}_training.json"
        bl_training = load_training_json(bl_training_path)
        st_training = load_training_json(st_training_path)

        # Get resolution labels
        bl_resolved = reeval_baseline.get(instance_id, False)
        st_resolved = reeval_strategy.get(instance_id, False)

        # Estimate tokens
        bl_tokens = bl_replay["estimated_tokens"]
        st_tokens = st_replay["estimated_tokens"]
        if bl_training:
            bl_tokens = bl_training["total_original_tokens"] or bl_tokens
        if st_training:
            st_tokens = st_training["total_filtered_tokens"] or st_tokens

        # Compute reward
        reward = compute_reward(
            baseline_resolved=bl_resolved,
            strategy_resolved=st_resolved,
            baseline_tokens=bl_tokens,
            strategy_tokens=st_tokens,
            lambda_efficiency=lambda_efficiency,
        )

        # Extract repo name
        repo = instance_id.rsplit("-", 1)[0].replace("__", "/")

        pair = {
            "instance_id": instance_id,
            "repo": repo,
            "strategy": strategy,
            "baseline": {
                "resolved": bl_resolved,
                "status": bl_replay["status"],
                "num_steps": bl_replay["num_steps"],
                "num_messages": bl_replay["num_messages"],
                "estimated_tokens": bl_replay["estimated_tokens"],
                "patch_size": bl_replay["patch_size"],
            },
            "strategy_run": {
                "resolved": st_resolved,
                "status": st_replay["status"],
                "num_steps": st_replay["num_steps"],
                "num_messages": st_replay["num_messages"],
                "estimated_tokens": st_replay["estimated_tokens"],
                "patch_size": st_replay["patch_size"],
                "memory_config": st_replay["memory_config"],
            },
            "reward": reward,
        }

        # Add training-level stats if available
        if st_training:
            pair["strategy_run"]["summarization_count"] = st_training["summarization_count"]
            pair["strategy_run"]["avg_compression_ratio"] = st_training["avg_compression_ratio"]
            pair["strategy_run"]["compression_steps"] = st_training["compression_steps"]

        pairs.append(pair)

    return pairs


def print_stats(pairs: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("PAIRED DATASET STATISTICS")
    print("=" * 60)

    n = len(pairs)
    if n == 0:
        print("No pairs found.")
        return

    bl_resolved = sum(1 for p in pairs if p["baseline"]["resolved"])
    st_resolved = sum(1 for p in pairs if p["strategy_run"]["resolved"])
    both_resolved = sum(
        1 for p in pairs
        if p["baseline"]["resolved"] and p["strategy_run"]["resolved"]
    )
    only_baseline = sum(
        1 for p in pairs
        if p["baseline"]["resolved"] and not p["strategy_run"]["resolved"]
    )
    only_strategy = sum(
        1 for p in pairs
        if not p["baseline"]["resolved"] and p["strategy_run"]["resolved"]
    )
    neither = sum(
        1 for p in pairs
        if not p["baseline"]["resolved"] and not p["strategy_run"]["resolved"]
    )

    print(f"Total pairs: {n}")
    print(f"Baseline resolved: {bl_resolved}/{n} ({bl_resolved/n*100:.1f}%)")
    print(f"Strategy resolved: {st_resolved}/{n} ({st_resolved/n*100:.1f}%)")
    print(f"Both resolved: {both_resolved}")
    print(f"Only baseline: {only_baseline}")
    print(f"Only strategy: {only_strategy}")
    print(f"Neither: {neither}")
    print(f"Union: {both_resolved + only_baseline + only_strategy}")
    print(f"Disagreement instances: {only_baseline + only_strategy} ({(only_baseline + only_strategy)/n*100:.1f}%)")

    # Reward stats
    rewards = [p["reward"]["reward"] for p in pairs]
    deltas = [p["reward"]["reward_delta"] for p in pairs]
    efficiencies = [p["reward"]["token_efficiency_score"] for p in pairs]

    print(f"\nReward stats:")
    print(f"  Avg reward: {sum(rewards)/n:.3f}")
    print(f"  Avg delta (vs baseline): {sum(deltas)/n:.3f}")
    print(f"  Avg token efficiency: {sum(efficiencies)/n:.3f}")

    # Disagreement details
    if only_baseline + only_strategy > 0:
        print(f"\nDisagreement instances:")
        for p in pairs:
            if p["baseline"]["resolved"] != p["strategy_run"]["resolved"]:
                winner = "baseline" if p["baseline"]["resolved"] else p["strategy"]
                print(f"  {p['instance_id']}: {winner} wins (delta={p['reward']['reward_delta']:+.0f})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build paired trajectory dataset")
    parser.add_argument(
        "--trajectories",
        default="results/trajectories_all",
        help="Consolidated trajectories directory",
    )
    parser.add_argument(
        "--strategy",
        default="llm_summarizing",
        help="Strategy to pair with baseline",
    )
    parser.add_argument(
        "--reeval-baseline",
        required=True,
        help="Path to baseline reeval report JSON",
    )
    parser.add_argument(
        "--reeval-strategy",
        required=True,
        help="Path to strategy reeval report JSON",
    )
    parser.add_argument(
        "--lambda-efficiency",
        type=float,
        default=0.3,
        help="Weight for token efficiency in reward (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default="pipelines/memory_training/data/paired.json",
        help="Output path for paired dataset",
    )

    args = parser.parse_args()

    # Load reeval labels
    reeval_baseline = load_reeval_report(Path(args.reeval_baseline))
    reeval_strategy = load_reeval_report(Path(args.reeval_strategy))

    print(f"Reeval labels: {len(reeval_baseline)} baseline, {len(reeval_strategy)} {args.strategy}")

    # Build pairs
    pairs = build_pairs(
        trajectories_dir=Path(args.trajectories),
        strategy=args.strategy,
        reeval_baseline=reeval_baseline,
        reeval_strategy=reeval_strategy,
        lambda_efficiency=args.lambda_efficiency,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {output_path}")

    # Print stats
    print_stats(pairs)


if __name__ == "__main__":
    main()
