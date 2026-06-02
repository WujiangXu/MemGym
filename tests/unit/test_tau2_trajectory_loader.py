"""Round-trip tests for the tau2 trajectory loader + compaction extractor.

Fixture is a real episode captured via ``Tau2BenchRunner`` (retail task 0,
ms10_kf1_kl2_r0.6, memory_side=both). The invariants pinned here are the
cross-side contract between runner and world-memory trainer: event counts
and msg_index roles must round-trip exactly, or the downstream training
dataset silently mislabels examples.

The captured-episode fixture is committed under
`tests/fixtures/trajectories/tau2_bench_run/`, so these tests run by default.
The module-level `skipif` is a safety net: if that fixture is ever removed (or
while regenerating it), the suite skips cleanly instead of failing, so
`pytest tests/unit` stays green. To regenerate, run a `Tau2BenchRunner`
retail-task-0 episode with `memory_side=both, max_steps=10, kf=1, kl=2, r=0.6`,
strategy=tau2_summarizing, and copy the run dir under
`tests/fixtures/trajectories/tau2_bench_run/memory/retail/0/`.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pytest

from memgym.training.data.compaction import (
    Tau2CompactionEvent,
    extract_all_tau2_compaction_events,
    extract_tau2_compaction_events,
)
from memgym.training.data.loader import (
    Tau2LoadedTrajectory,
    Tau2StepData,
    TrajectoryLoader,
    load_tau2_trajectory,
)

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "trajectories" / "tau2_bench_run"
RETAIL_TASK_DIR = FIXTURE_ROOT / "memory" / "retail" / "0"

pytestmark = pytest.mark.skipif(
    not (RETAIL_TASK_DIR / "result.json").exists(),
    reason=(
        "tau2 captured-episode fixture missing (it is normally committed; "
        "this only trips if it was removed or is being regenerated). "
        "Regenerate with `Tau2BenchRunner` retail task 0; see module docstring."
    ),
)


class TestTau2TrajectoryLoader(unittest.TestCase):
    """Direct loader → single trajectory."""

    def setUp(self) -> None:
        self.traj = load_tau2_trajectory(
            RETAIL_TASK_DIR, task_id="0", phase="memory", domain="retail"
        )
        self.result = json.load(open(RETAIL_TASK_DIR / "result.json"))

    def test_loads(self) -> None:
        self.assertIsNotNone(self.traj)
        self.assertEqual(self.traj.track, "tau2")
        self.assertEqual(self.traj.domain, "retail")
        self.assertEqual(self.traj.task_id, "0")
        self.assertEqual(self.traj.phase, "memory")
        self.assertEqual(self.traj.instance_key, "memory/retail/0")

    def test_top_level_metadata(self) -> None:
        """Strategy fields carry the memory manager's class name, not the adapter suffix."""
        self.assertEqual(self.traj.memory_side, "both")
        self.assertIn("Summarizing", self.traj.memory_strategy_agent)
        self.assertIn("Summarizing", self.traj.memory_strategy_user)

    def test_two_sided_steps(self) -> None:
        """Agent + user steps coexist under a single ``steps`` list."""
        self.assertGreater(len(self.traj.agent_steps), 0)
        self.assertGreaterEqual(len(self.traj.user_steps), 0)
        for s in self.traj.agent_steps:
            self.assertEqual(s.side, "agent")
        for s in self.traj.user_steps:
            self.assertEqual(s.side, "user")

    def test_compaction_count_matches_result(self) -> None:
        """Loader's property matches the runner's episode-level counter."""
        self.assertEqual(
            self.traj.agent_compaction_count, self.result["agent_compaction_count"]
        )
        self.assertEqual(
            self.traj.user_compaction_count, self.result["user_compaction_count"]
        )

    def test_new_compaction_vs_was_compacted(self) -> None:
        """``was_compacted`` is sticky-True after first event; ``new_compaction`` is a single spike."""
        agent_steps = self.traj.agent_steps
        new_comp = sum(1 for s in agent_steps if s.memory.new_compaction)
        was_comp = sum(1 for s in agent_steps if s.memory.was_compacted)
        self.assertLessEqual(new_comp, was_comp)
        self.assertEqual(new_comp, self.traj.agent_compaction_count)

    def test_condensation_history_length_matches(self) -> None:
        """Per-side condensation_history length == per-side new_compaction count."""
        self.assertEqual(
            len(self.traj.condensation_history.get("agent", [])),
            self.traj.agent_compaction_count,
        )
        self.assertEqual(
            len(self.traj.condensation_history.get("user", [])),
            self.traj.user_compaction_count,
        )


class TestTau2CompactionExtraction(unittest.TestCase):
    """Extractor: one event per ``new_compaction=True`` step."""

    def setUp(self) -> None:
        self.traj = load_tau2_trajectory(
            RETAIL_TASK_DIR, task_id="0", phase="memory", domain="retail"
        )
        self.events = extract_tau2_compaction_events(self.traj)
        self.result = json.load(open(RETAIL_TASK_DIR / "result.json"))

    def test_event_count_matches_runner_counters(self) -> None:
        agent_events = [e for e in self.events if e.side == "agent"]
        user_events = [e for e in self.events if e.side == "user"]
        self.assertEqual(len(agent_events), self.result["agent_compaction_count"])
        self.assertEqual(len(user_events), self.result["user_compaction_count"])

    def test_msg_index_role_alignment(self) -> None:
        """Each event's ``msg_index`` lands on the correct role in ``messages``."""
        msgs = self.traj.messages
        for e in self.events:
            self.assertTrue(0 <= e.msg_index < len(msgs))
            expected_role = "assistant" if e.side == "agent" else "user"
            self.assertEqual(msgs[e.msg_index]["role"], expected_role)
            self.assertIsNotNone(e.action_msg)
            self.assertEqual(e.action_msg["role"], expected_role)

    def test_event_identity_fields(self) -> None:
        """``instance_key`` matches the trajectory and carries all components."""
        for e in self.events:
            self.assertEqual(e.domain, "retail")
            self.assertEqual(e.task_id, "0")
            self.assertEqual(e.phase, "memory")
            self.assertEqual(e.instance_key, "memory/retail/0")

    def test_event_memory_payload(self) -> None:
        """Event carries the summary text + compression ratio, non-empty."""
        for e in self.events:
            self.assertGreater(len(e.summary_generated), 0)
            self.assertGreater(e.compression_ratio, 0.0)
            self.assertGreaterEqual(e.context_before_tokens, e.context_after_tokens // 10)

    def test_neighborhood_bounded(self) -> None:
        """pre/post msg windows respect the window parameter + array bounds."""
        events = extract_tau2_compaction_events(self.traj, msg_window=2)
        for e in events:
            self.assertLessEqual(len(e.pre_msgs), 2)
            self.assertLessEqual(len(e.post_msgs), 2)


class TestTrajectoryLoaderTau2Walker(unittest.TestCase):
    """``TrajectoryLoader.load_tau2_experiment`` walks the run dir layout."""

    def test_walker_discovers_memory_phase(self) -> None:
        loader = TrajectoryLoader(FIXTURE_ROOT.parent)
        trajs = loader.load_tau2_experiment(FIXTURE_ROOT.name)
        self.assertIn("memory/retail/0", trajs)
        self.assertEqual(trajs["memory/retail/0"].domain, "retail")

    def test_walker_skips_baseline_by_default(self) -> None:
        """Default phases=['memory']; baseline dirs (no _training.json) are dropped."""
        loader = TrajectoryLoader(FIXTURE_ROOT.parent)
        trajs = loader.load_tau2_experiment(FIXTURE_ROOT.name)
        for key in trajs:
            self.assertTrue(key.startswith("memory/"))

    def test_extract_all_flattens_cross_trajectory(self) -> None:
        loader = TrajectoryLoader(FIXTURE_ROOT.parent)
        trajs = loader.load_tau2_experiment(FIXTURE_ROOT.name)
        all_events = extract_all_tau2_compaction_events(trajs)
        # Exactly equal to the sum of per-trajectory agent+user compaction counters.
        total_expected = sum(
            t.agent_compaction_count + t.user_compaction_count for t in trajs.values()
        )
        self.assertEqual(len(all_events), total_expected)


if __name__ == "__main__":
    unittest.main()
