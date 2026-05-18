"""Paired labeling and reward computation for world model training.

Compares baseline (no memory) and strategy (memory-augmented) trajectories
to compute per-instance labels and rewards.

Refactored from pipelines/memory_training/build_pairs.py:compute_reward.

Usage:
    from memgym.training.data.loader import TrajectoryLoader
    from memgym.training.data.labels import compute_paired_labels

    loader = TrajectoryLoader("ec2_trajectories")
    baseline = loader.load_experiment("train50_sonnet45_baseline")
    strategy = loader.load_experiment("train50_sonnet45_llmsummarizing")
    bl_labels = loader.load_reeval_labels("path/to/baseline_reeval.json")
    st_labels = loader.load_reeval_labels("path/to/strategy_reeval.json")

    paired = compute_paired_labels(baseline, strategy, bl_labels, st_labels)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from memgym.training.data.loader import LoadedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class PairedLabel:
    """Label for a paired baseline-vs-strategy comparison."""

    instance_id: str
    baseline_resolved: bool
    strategy_resolved: bool

    # Binary label: 1 = memory ok/helped, 0 = memory hurt
    label: int
    # Descriptive category
    label_category: str  # "both_pass", "memory_hurt", "memory_helped", "both_fail"

    # Multi-dimensional reward (from paper Eq. 1)
    reward: float          # R = resolve_score + λ * token_efficiency
    resolve_score: float
    token_efficiency: float
    reward_delta: float    # strategy resolve - baseline resolve

    # Token stats
    baseline_tokens: int
    strategy_tokens: int
    strategy_net_tokens: int  # agent + summarizer tokens


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

    Refactored from build_pairs.py:compute_reward (L104-141).
    """
    resolve_score = 1.0 if strategy_resolved else 0.0
    baseline_resolve_score = 1.0 if baseline_resolved else 0.0

    strategy_net_tokens = strategy_tokens + summarizer_tokens

    if baseline_tokens > 0 and strategy_resolved:
        token_efficiency = (baseline_tokens - strategy_net_tokens) / baseline_tokens
    else:
        token_efficiency = 0.0

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


def _classify_label(baseline_resolved: bool, strategy_resolved: bool) -> tuple[int, str]:
    """Classify the pairing into a label and category.

    Returns (label, category):
      - both_pass:     baseline ok, strategy ok    → label=1 (safe)
      - memory_hurt:   baseline ok, strategy fail  → label=0 (critical negative)
      - memory_helped: baseline fail, strategy ok  → label=1 (memory improved)
      - both_fail:     baseline fail, strategy fail → label=0 (neutral fail)
    """
    if baseline_resolved and strategy_resolved:
        return 1, "both_pass"
    elif baseline_resolved and not strategy_resolved:
        return 0, "memory_hurt"
    elif not baseline_resolved and strategy_resolved:
        return 1, "memory_helped"
    else:
        return 0, "both_fail"


def compute_paired_labels(
    baseline_trajs: Dict[str, LoadedTrajectory],
    strategy_trajs: Dict[str, LoadedTrajectory],
    baseline_labels: Dict[str, bool],
    strategy_labels: Dict[str, bool],
    lambda_efficiency: float = 0.3,
) -> Dict[str, PairedLabel]:
    """Match baseline vs strategy by instance_id and compute labels.

    Args:
        baseline_trajs: Loaded baseline trajectories keyed by instance_id.
        strategy_trajs: Loaded strategy trajectories keyed by instance_id.
        baseline_labels: {instance_id: resolved} from baseline reeval report.
        strategy_labels: {instance_id: resolved} from strategy reeval report.
        lambda_efficiency: Weight for token efficiency in reward.

    Returns:
        {instance_id: PairedLabel} for all instances with both baseline + strategy.
    """
    common_ids = sorted(
        set(baseline_trajs.keys()) & set(strategy_trajs.keys())
    )
    logger.info(
        "Found %d instances with both baseline + strategy (baseline=%d, strategy=%d)",
        len(common_ids), len(baseline_trajs), len(strategy_trajs),
    )

    results: Dict[str, PairedLabel] = {}
    for iid in common_ids:
        bl = baseline_trajs[iid]
        st = strategy_trajs[iid]

        bl_resolved = baseline_labels.get(iid, False)
        st_resolved = strategy_labels.get(iid, False)

        label, category = _classify_label(bl_resolved, st_resolved)

        bl_tokens = bl.total_original_tokens or bl.total_filtered_tokens
        st_tokens = st.total_filtered_tokens or st.total_original_tokens

        reward_info = compute_reward(
            baseline_resolved=bl_resolved,
            strategy_resolved=st_resolved,
            baseline_tokens=bl_tokens,
            strategy_tokens=st_tokens,
            lambda_efficiency=lambda_efficiency,
        )

        results[iid] = PairedLabel(
            instance_id=iid,
            baseline_resolved=bl_resolved,
            strategy_resolved=st_resolved,
            label=label,
            label_category=category,
            reward=reward_info["reward"],
            resolve_score=reward_info["resolve_score"],
            token_efficiency=reward_info["token_efficiency_score"],
            reward_delta=reward_info["reward_delta"],
            baseline_tokens=bl_tokens,
            strategy_tokens=st_tokens,
            strategy_net_tokens=reward_info["strategy_net_tokens"],
        )

    return results


def print_label_stats(labels: Dict[str, PairedLabel]) -> None:
    """Print summary statistics for paired labels."""
    n = len(labels)
    if n == 0:
        print("No paired labels.")
        return

    categories = {}
    for pl in labels.values():
        categories[pl.label_category] = categories.get(pl.label_category, 0) + 1

    bl_resolved = sum(1 for pl in labels.values() if pl.baseline_resolved)
    st_resolved = sum(1 for pl in labels.values() if pl.strategy_resolved)

    print(f"\n{'='*60}")
    print("PAIRED LABEL STATISTICS")
    print(f"{'='*60}")
    print(f"Total pairs: {n}")
    print(f"Baseline resolved: {bl_resolved}/{n} ({bl_resolved/n*100:.1f}%)")
    print(f"Strategy resolved: {st_resolved}/{n} ({st_resolved/n*100:.1f}%)")
    print(f"\nLabel distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} ({count/n*100:.1f}%)")

    rewards = [pl.reward for pl in labels.values()]
    efficiencies = [pl.token_efficiency for pl in labels.values()]
    print(f"\nAvg reward: {sum(rewards)/n:.3f}")
    print(f"Avg token efficiency: {sum(efficiencies)/n:.3f}")

    disagree = categories.get("memory_hurt", 0) + categories.get("memory_helped", 0)
    print(f"Disagreement: {disagree}/{n} ({disagree/n*100:.1f}%)")
    print(f"{'='*60}")
