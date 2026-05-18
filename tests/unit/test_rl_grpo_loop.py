"""Tests for the (see paper) GRPO advantage prep.

Pure-Python tests on `prepare_advantages` only — `snapshot_old_logprobs`
and `grpo_step` need a real torch model, exercised in the EC2 smoke
(plan §H.3'' Way B verification step 3).
"""
import unittest

from memgym.training.rl.env_reward import EpisodeResult, SummaryCall
from memgym.training.rl.grpo_loop import (
    COMPRESSION_FLOOR,
    R_COMPRESSION_PENALTY,
    prepare_advantages,
)


def _mk_episode(
    iid: str,
    reward: float,
    n_summaries: int = 1,
    max_compression_ratio: float = 1.0,
    resolved: bool = False,
    hit_ctx: bool = False,
) -> EpisodeResult:
    summaries = [
        SummaryCall(system="sys", user=f"hist{i}", max_tokens=1024, completion=f"sum{i}")
        for i in range(n_summaries)
    ]
    return EpisodeResult(
        instance_id=iid,
        R_terminal=reward,
        episode_summaries=summaries,
        metadata={"max_compression_ratio": max_compression_ratio},
        episode_resolved=resolved,
        hit_context_limit=hit_ctx,
    )


class TestPrepareAdvantages(unittest.TestCase):
    def test_groups_by_instance_and_normalizes(self):
        # Group A: rewards [1, 0, 0, 0] → mean=0.25, std≈0.433
        # Group B: rewards [1, 1, 1, 0] → mean=0.75, std≈0.433
        outcomes = [
            _mk_episode("A", 1.0),
            _mk_episode("A", 0.0),
            _mk_episode("A", 0.0),
            _mk_episode("A", 0.0),
            _mk_episode("B", 1.0),
            _mk_episode("B", 1.0),
            _mk_episode("B", 1.0),
            _mk_episode("B", 0.0),
        ]
        rollout_idxs = [0, 1, 2, 3, 0, 1, 2, 3]
        advs = prepare_advantages(outcomes, rollout_indices=rollout_idxs,
                                  use_compression_guardrail=False)
        # 8 outcomes × 1 summary each → 8 advantages
        self.assertEqual(len(advs), 8)
        # The high-reward rollout in group A has a much bigger positive
        # advantage than the high-reward rollout in group B (the latter
        # is the majority case — group baseline is high).
        a_high = [a for a in advs if a.instance_id == "A" and a.reward > 0.5][0]
        b_high = [a for a in advs if a.instance_id == "B" and a.reward > 0.5][0]
        self.assertGreater(a_high.advantage, b_high.advantage)

    def test_skips_episodes_without_summaries(self):
        """times_filtered=0 → no summary → no gradient target.

        Plan §H.3'' explicitly excludes these from the update."""
        outcomes = [
            _mk_episode("A", 1.0, n_summaries=0),  # no compaction triggered
            _mk_episode("A", 0.0, n_summaries=2),
        ]
        advs = prepare_advantages(outcomes, rollout_indices=[0, 1],
                                  use_compression_guardrail=False)
        # Only the second rollout's 2 summaries make it in; the first is dropped.
        self.assertEqual(len(advs), 2)
        self.assertEqual(advs[0].instance_id, "A")
        self.assertEqual(advs[0].rollout_idx, 1)

    def test_replicates_advantage_across_summaries_in_episode(self):
        """REINFORCE-style credit assignment: every summary in an
        episode shares the episode's advantage value."""
        outcomes = [
            _mk_episode("A", 1.0, n_summaries=3),
            _mk_episode("A", 0.0, n_summaries=2),
        ]
        advs = prepare_advantages(outcomes, rollout_indices=[0, 1],
                                  use_compression_guardrail=False)
        # First rollout contributes 3 advantages, all equal.
        first_rollout = [a for a in advs if a.rollout_idx == 0]
        self.assertEqual(len(first_rollout), 3)
        self.assertEqual(len({a.advantage for a in first_rollout}), 1)

    def test_compression_guardrail_penalizes_skip_compression(self):
        """Compression ratio < 0.2 incurs the skip-compression penalty."""
        outcomes = [
            _mk_episode("A", 1.0, max_compression_ratio=0.1),  # would-skip
            _mk_episode("A", 1.0, max_compression_ratio=0.5),  # honest compaction
        ]
        with_g = prepare_advantages(outcomes, rollout_indices=[0, 1],
                                    use_compression_guardrail=True)
        without_g = prepare_advantages(outcomes, rollout_indices=[0, 1],
                                       use_compression_guardrail=False)
        # Without guardrail both rewards are 1.0 (zero variance → advantage 0).
        self.assertAlmostEqual(without_g[0].advantage, 0.0, places=4)
        # With guardrail the skip-compression rollout has lower reward
        # → negative advantage; the honest one positive.
        skip_with = [a for a in with_g if a.summary_call.user == "hist0"][0]
        honest_with = [a for a in with_g if a.summary_call.user == "hist0" and a.rollout_idx == 1][0]
        self.assertLess(skip_with.advantage, honest_with.advantage)

    def test_singleton_group_advantage_is_zero(self):
        """A group with n=1 has no relative signal; advantage falls to 0."""
        outcomes = [_mk_episode("A", 1.0)]
        advs = prepare_advantages(outcomes, rollout_indices=[0])
        self.assertEqual(len(advs), 1)
        self.assertAlmostEqual(advs[0].advantage, 0.0, places=4)

    def test_rollout_indices_length_mismatch_raises(self):
        outcomes = [_mk_episode("A", 1.0), _mk_episode("A", 0.0)]
        with self.assertRaises(ValueError):
            prepare_advantages(outcomes, rollout_indices=[0])

    def test_metadata_carries_resolution_flags(self):
        outcomes = [
            _mk_episode("A", 1.0, resolved=True),
            _mk_episode("A", 0.0, hit_ctx=True),
        ]
        advs = prepare_advantages(outcomes, rollout_indices=[0, 1],
                                  use_compression_guardrail=False)
        resolved_adv = [a for a in advs if a.metadata["episode_resolved"]]
        ctx_adv = [a for a in advs if a.metadata["hit_context_limit"]]
        self.assertEqual(len(resolved_adv), 1)
        self.assertEqual(len(ctx_adv), 1)


class TestCompressionConstants(unittest.TestCase):
    """Pin the constants from plan §H.3.2 in case future edits drift them."""

    def test_compression_floor_matches_plan(self):
        self.assertAlmostEqual(COMPRESSION_FLOOR, 0.2)

    def test_compression_penalty_matches_plan(self):
        self.assertAlmostEqual(R_COMPRESSION_PENALTY, -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
