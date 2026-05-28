"""
E2E production readiness tests for the trajectory pipeline.

Tests the full flow: record → save → reload → replay → verify.

Uses only mocks (no Docker, no LLM) to be fast and CI-friendly.
Each test creates a MemoryAwareSWEAgent with mocked model+env,
runs steps, saves trajectory files, reloads them, replays through
different strategies, and verifies consistency.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

def _msg(role, content):
    return {"role": role, "content": content}

def _make_mock_model(n_steps):
    """Create a mock LLM model that returns n_steps responses then raises LimitsExceeded."""
    from minisweagent.agents.default import LimitsExceeded

    model = MagicMock()
    model.n_calls = 0
    model.cost = 0
    model.config.model_name = "mock-test-model"

    responses = []
    for i in range(1, n_steps + 1):
        responses.append({"content": f"THOUGHT: Working on step {i}\n```bash\necho step_{i}\n```"})

    call_count = [0]

    def mock_query(messages):
        idx = call_count[0]
        call_count[0] += 1
        if idx >= len(responses):
            raise LimitsExceeded()
        return responses[idx]

    model.query = mock_query
    return model

def _make_mock_env():
    """Create a mock environment that returns fixed command outputs."""
    env = MagicMock()

    def mock_execute(command, cwd="", *, timeout=None):
        return {"output": f"Mock output for: {command}", "returncode": 0}

    env.execute = mock_execute
    env.get_template_vars.return_value = {"image": "test", "cwd": "/testbed"}
    return env

def _run_agent_episode(memory, n_steps):
    """Run an agent episode and return (agent, status, message)."""
    from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent

    model = _make_mock_model(n_steps)
    env = _make_mock_env()
    agent = MemoryAwareSWEAgent(model, env, memory)

    status, message = agent.run("Fix the bug in src/main.py. The function calculate_total() returns wrong results.")
    return agent, status, message

def _build_replay_data(agent, instance_id="test__instance-001"):
    """Build replay data from agent state (mirrors swe_env.py logic)."""
    from memgym.gym.swe_bench.env import _build_step_index

    return {
        "version": 1,
        "instance_id": instance_id,
        "dataset": "test",
        "model": "mock-model",
        "memory_strategy": agent.memory_manager.__class__.__name__,
        "memory_config": {},
        "status": "LimitsExceeded",
        "reward": 0.0,
        "patch": "",
        "messages": list(agent.messages),
        "steps": _build_step_index(agent.messages),
    }

class TestE2EPassthrough(unittest.TestCase):
    """E2E test: record → save → reload → replay with PassThroughMemory."""

    def test_e2e_record_save_reload_replay_passthrough(self):
        from memgym.memory.base import PassThroughMemory
        from replay_swe_bench import analyze_replay

        memory = PassThroughMemory()
        agent, status, message = _run_agent_episode(memory, 5)

        # Get training trajectory
        training = agent.get_training_trajectory()
        self.assertEqual(training["num_steps"], 5)

        # Build replay data
        replay_data = _build_replay_data(agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            training_path = Path(tmpdir) / "training.json"
            replay_path = Path(tmpdir) / "replay.json"
            with open(training_path, "w") as f:
                json.dump(training, f, indent=2)
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)

            # Reload
            with open(training_path) as f:
                loaded_training = json.load(f)
            with open(replay_path) as f:
                loaded_replay = json.load(f)

            # Replay with fresh PassThroughMemory
            replay_memory = PassThroughMemory()
            analysis = analyze_replay(loaded_replay, replay_memory)

        # Verify
        self.assertEqual(analysis["num_steps"], loaded_training["num_steps"])
        self.assertEqual(analysis["num_steps"], 5)

        for step in analysis["steps"]:
            self.assertFalse(step["compacted"], f"Step {step['step']}: passthrough should not compact")
            self.assertAlmostEqual(step["compression_ratio"], 1.0, places=2)

class TestE2EObservationMasking(unittest.TestCase):
    """E2E test: record → save → reload → replay with ObservationMaskingMemory."""

    def test_e2e_record_save_reload_replay_observation_masking(self):
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory
        from replay_swe_bench import analyze_replay

        memory = ObservationMaskingMemory(attention_window=3, keep_first=1)
        agent, status, message = _run_agent_episode(memory, 8)

        training = agent.get_training_trajectory()
        replay_data = _build_replay_data(agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "replay.json"
            training_path = Path(tmpdir) / "training.json"
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)
            with open(training_path, "w") as f:
                json.dump(training, f, indent=2)

            with open(replay_path) as f:
                loaded_replay = json.load(f)
            with open(training_path) as f:
                loaded_training = json.load(f)

            replay_memory = ObservationMaskingMemory(attention_window=3, keep_first=1)
            analysis = analyze_replay(loaded_replay, replay_memory)

        self.assertEqual(analysis["num_steps"], 8)

        # Masking should kick in when messages exceed keep_first + attention_window
        compacted_steps = [s for s in analysis["steps"] if s["compacted"]]
        self.assertGreater(len(compacted_steps), 0,
            "Observation masking should compact at least some steps with 8 steps and window=3")

        # Cross-validate training vs analysis step counts
        self.assertEqual(loaded_training["num_steps"], analysis["num_steps"])

class TestE2ENaive(unittest.TestCase):
    """E2E test: record → save → reload → replay with NaiveSummarizationMemory."""

    @patch("memgym.memory.backends.summarizer_backend.completion")
    def test_e2e_record_save_reload_replay_naive(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of previous steps."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        from replay_swe_bench import analyze_replay

        memory = NaiveSummarizationMemory(max_tokens=200, keep_recent=3)
        agent, status, message = _run_agent_episode(memory, 6)

        training = agent.get_training_trajectory()
        replay_data = _build_replay_data(agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "replay.json"
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)

            with open(replay_path) as f:
                loaded_replay = json.load(f)

            replay_memory = NaiveSummarizationMemory(max_tokens=200, keep_recent=3)
            analysis = analyze_replay(loaded_replay, replay_memory)

        self.assertEqual(analysis["num_steps"], 6)

        # Summarization should trigger at some point with low threshold
        compacted_steps = [s for s in analysis["steps"] if s["compacted"]]
        self.assertGreater(len(compacted_steps), 0,
            "Naive summarization should trigger with max_tokens=200")

        # Check compression ratio > 1.0 for compacted steps
        for step in compacted_steps:
            self.assertGreater(step["compression_ratio"], 1.0,
                f"Step {step['step']}: compression_ratio should be > 1.0")

class TestE2ECrossStrategyReplay(unittest.TestCase):
    """Record with one strategy, replay through multiple different strategies."""

    @patch("memgym.memory.backends.summarizer_backend.completion")
    def test_e2e_cross_strategy_replay(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_completion.return_value = mock_response

        from memgym.memory.base import PassThroughMemory
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        from replay_swe_bench import analyze_replay

        # Record with passthrough (no filtering)
        memory = PassThroughMemory()
        agent, _, _ = _run_agent_episode(memory, 6)
        replay_data = _build_replay_data(agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = Path(tmpdir) / "replay.json"
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)

            with open(replay_path) as f:
                loaded = json.load(f)

            # Replay through 3 strategies
            pt_analysis = analyze_replay(loaded, PassThroughMemory())
            om_analysis = analyze_replay(loaded, ObservationMaskingMemory(attention_window=2, keep_first=1))
            naive_analysis = analyze_replay(loaded, NaiveSummarizationMemory(max_tokens=200, keep_recent=2))

        # All should have same number of steps
        self.assertEqual(pt_analysis["num_steps"], 6)
        self.assertEqual(om_analysis["num_steps"], 6)
        self.assertEqual(naive_analysis["num_steps"], 6)

        # Passthrough: 0 compactions
        pt_compactions = sum(1 for s in pt_analysis["steps"] if s["compacted"])
        self.assertEqual(pt_compactions, 0, "Passthrough should have 0 compactions")

        # Observation masking: should have compactions (small window)
        om_compactions = sum(1 for s in om_analysis["steps"] if s["compacted"])
        self.assertGreater(om_compactions, 0, "Obs masking with window=2 should compact")

        # Naive: should have compactions (low threshold)
        naive_compactions = sum(1 for s in naive_analysis["steps"] if s["compacted"])
        self.assertGreater(naive_compactions, 0, "Naive with max_tokens=200 should compact")

class TestE2EReplayFileIntegrity(unittest.TestCase):
    """Verify replay file structure and JSON round-trip integrity."""

    def test_e2e_replay_file_integrity(self):
        from memgym.memory.base import PassThroughMemory

        memory = PassThroughMemory()
        agent, _, _ = _run_agent_episode(memory, 5)
        replay_data = _build_replay_data(agent)

        # Check structure
        self.assertIn("version", replay_data)
        self.assertIn("instance_id", replay_data)
        self.assertIn("messages", replay_data)
        self.assertIn("steps", replay_data)
        self.assertEqual(replay_data["version"], 1)

        # Check messages structure
        for i, msg in enumerate(replay_data["messages"]):
            self.assertIsInstance(msg, dict, f"Message {i} is not a dict")
            self.assertIn("role", msg, f"Message {i} missing 'role'")
            self.assertIn("content", msg, f"Message {i} missing 'content'")
            self.assertIn(msg["role"], ["system", "user", "assistant"],
                f"Message {i} has invalid role: {msg['role']}")

        # Check steps structure
        for step in replay_data["steps"]:
            self.assertIn("step", step)
            self.assertIn("assistant_msg_index", step)
            self.assertIn("observation_msg_index", step)

            # Verify indices are within bounds
            self.assertLess(step["assistant_msg_index"], len(replay_data["messages"]))
            if step["observation_msg_index"] is not None:
                self.assertLess(step["observation_msg_index"], len(replay_data["messages"]))

            # Verify roles match
            self.assertEqual(
                replay_data["messages"][step["assistant_msg_index"]]["role"], "assistant",
                f"Step {step['step']}: assistant_msg_index points to wrong role")
            if step["observation_msg_index"] is not None:
                self.assertEqual(
                    replay_data["messages"][step["observation_msg_index"]]["role"], "user",
                    f"Step {step['step']}: observation_msg_index points to wrong role")

        # JSON round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            with open(path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)
            with open(path) as f:
                reloaded = json.load(f)

            self.assertEqual(len(replay_data["messages"]), len(reloaded["messages"]))
            self.assertEqual(len(replay_data["steps"]), len(reloaded["steps"]))
            for orig, loaded in zip(replay_data["messages"], reloaded["messages"]):
                self.assertEqual(orig["role"], loaded["role"])
                self.assertEqual(orig["content"], loaded["content"])

class TestE2ETrainingTrajectoryCompleteness(unittest.TestCase):
    """Training trajectory must have complete, ordered steps with all fields."""

    def test_e2e_training_trajectory_completeness(self):
        from memgym.memory.base import PassThroughMemory

        memory = PassThroughMemory()
        agent, _, _ = _run_agent_episode(memory, 5)
        training = agent.get_training_trajectory()

        self.assertEqual(training["num_steps"], 5)
        self.assertEqual(len(training["steps"]), 5)

        required_memory_keys = {"original_msgs", "filtered_msgs", "original_tokens",
                                "filtered_tokens", "compression_ratio", "was_compacted"}

        for i, step in enumerate(training["steps"]):
            step_num = step["step"]
            # Steps should be ordered
            self.assertEqual(step_num, i + 1,
                f"Step ordering broken: expected {i+1}, got {step_num}")

            # All fields present
            self.assertIn("thought", step, f"Step {step_num}: missing 'thought'")
            self.assertIn("action", step, f"Step {step_num}: missing 'action'")
            self.assertIn("observation", step, f"Step {step_num}: missing 'observation'")
            self.assertIn("memory", step, f"Step {step_num}: missing 'memory'")

            # Thought and action should be non-empty (our mock always provides them)
            self.assertTrue(step["thought"], f"Step {step_num}: empty thought")
            self.assertTrue(step["action"], f"Step {step_num}: empty action")

            # Memory has required keys
            missing = required_memory_keys - set(step["memory"].keys())
            self.assertEqual(missing, set(),
                f"Step {step_num}: memory missing keys: {missing}")

        # Last step observation must be captured (not empty)
        last_step = training["steps"][-1]
        self.assertTrue(last_step["observation"],
            f"Last step observation is empty — final step flush is broken")

class TestE2EMultipleEpisodesIsolation(unittest.TestCase):
    """Multiple episodes must be isolated — no state leak between runs."""

    def test_e2e_multiple_episodes_isolation(self):
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory

        memory = ObservationMaskingMemory(attention_window=3, keep_first=1)

        # Episode 1: 5 steps
        agent1, _, _ = _run_agent_episode(memory, 5)
        training1 = agent1.get_training_trajectory()
        replay1 = _build_replay_data(agent1, "episode1")

        # Reset memory (as would happen between episodes)
        memory.reset()

        # Episode 2: 3 steps
        agent2, _, _ = _run_agent_episode(memory, 3)
        training2 = agent2.get_training_trajectory()
        replay2 = _build_replay_data(agent2, "episode2")

        # Episode 2 starts fresh
        self.assertEqual(training2["num_steps"], 3)
        self.assertEqual(training2["steps"][0]["step"], 1,
            "Episode 2 step numbering should start at 1")

        # Episode 1 data still intact
        self.assertEqual(training1["num_steps"], 5)
        self.assertEqual(len(replay1["steps"]), 5)

        # Episode 2 should not have episode 1 messages
        self.assertNotEqual(len(replay1["messages"]), len(replay2["messages"]),
            "Episode 1 and 2 should have different message counts (5 vs 3 steps)")

        # Replay episode 1 still works
        from replay_swe_bench import analyze_replay
        replay_memory = ObservationMaskingMemory(attention_window=3, keep_first=1)
        analysis = analyze_replay(replay1, replay_memory)
        self.assertEqual(analysis["num_steps"], 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
