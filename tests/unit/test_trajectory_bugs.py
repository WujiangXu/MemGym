"""
Hard validation tests for memory trajectory collection bugs.

Each test targets a specific confirmed bug and is designed to FAIL
on unfixed code and PASS after fixes.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

def _msg(role, content):
    """Helper to create a message dict."""
    return {"role": role, "content": content}

def _simulate_agent_steps(memory, n_steps):
    """Simulate n agent steps, returning per-step results.

    Mimics MemoryAwareSWEAgent.query() message flow:
    - messages grow: [sys, task, asst1, obs1, asst2, obs2, ...]
    - At each query: history = messages[:-1], current_obs = messages[-1]
    """
    messages = [
        _msg("system", "You are a coding agent."),
        _msg("user", "Fix the bug in src/main.py"),
    ]
    results = []

    for step in range(1, n_steps + 1):
        # Before query: history = messages[:-1], obs = messages[-1]
        history = messages[:-1]
        current_obs = messages[-1]

        result = memory.manage_context(
            original_context=history,
            current_observation=current_obs,
            metadata={"step": step}
        )
        results.append(result)

        # After query: add assistant response + next observation
        messages.append(_msg("assistant", f"THOUGHT: Analyzing step {step}\n```bash\necho step{step}\n```"))
        messages.append(_msg("user", f"<returncode>0</returncode>\n<output>step{step} output " + "x" * 100 + "</output>"))

    return results, messages

class TestNaiveNoDuplication(unittest.TestCase):
    """Bug 1: _observations accumulation causes duplication with original_context."""

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_naive_no_observation_duplication(self, mock_completion):
        """Content returned by manage_context must not contain duplicate messages."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test summary."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        memory = NaiveSummarizationMemory(max_tokens=100000)  # High threshold, no summarization

        results, messages = _simulate_agent_steps(memory, 5)

        for i, result in enumerate(results):
            content = result.content
            # Each message should appear at most once in content
            seen = []
            for msg in content:
                msg_key = (msg.get("role"), msg.get("content"))
                self.assertNotIn(msg_key, seen,
                    f"Step {i+1}: Duplicate message found: role={msg_key[0]}, content={msg_key[1][:50]}...")
                seen.append(msg_key)

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_naive_multi_step_no_token_inflation(self, mock_completion):
        """Token count must grow linearly with messages, not quadratically."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        memory = NaiveSummarizationMemory(max_tokens=100000)  # No summarization

        results, _ = _simulate_agent_steps(memory, 10)

        token_counts = [r.metadata["original_tokens"] for r in results]

        # With fixed-size messages, tokens should grow roughly linearly
        # If duplicating, growth would be quadratic (n + n-1 + n-2 + ...)
        # Check: ratio of last to first should be < 15x (linear), not > 30x (quadratic)
        if token_counts[0] > 0:
            ratio = token_counts[-1] / token_counts[0]
            # Linear growth: step 10 has ~20 msgs vs step 1 has ~2 msgs, so ~10x message count
            # Token ratio may be higher due to varying msg sizes, but should be < 50x
            # Quadratic (duplication bug) would be > 100x
            self.assertLess(ratio, 50,
                f"Token inflation detected: step1={token_counts[0]}, step10={token_counts[-1]}, ratio={ratio:.1f}x")

class TestNaiveCompressionRatio(unittest.TestCase):
    """Bug 2: Missing compression_ratio in naive metadata."""

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_naive_compression_ratio_in_metadata(self, mock_completion):
        """compression_ratio must exist and be > 1.0 when compacted with enough messages."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Short summary."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        memory = NaiveSummarizationMemory(max_tokens=100, keep_recent=2)

        # Build enough content to trigger summarization — many messages so summary is shorter
        history = [
            _msg("system", "System prompt " * 20),
            _msg("user", "Task description " * 20),
            _msg("assistant", "I will analyze " * 20),
            _msg("user", "Output from command " * 20),
            _msg("assistant", "Next I will fix " * 20),
            _msg("user", "More output here " * 20),
            _msg("assistant", "Almost done " * 20),
        ]
        obs = _msg("user", "Final observation " * 20)

        result = memory.manage_context(history, obs)

        self.assertIn("compression_ratio", result.metadata,
            "compression_ratio key missing from naive metadata")

        if result.metadata["was_compacted"]:
            self.assertGreater(result.metadata["compression_ratio"], 1.0,
                f"compression_ratio should be > 1.0 when many messages compressed, got {result.metadata['compression_ratio']}")

class TestNaiveKeepsRecent(unittest.TestCase):
    """Bug 3: Over-aggressive output — only 2 messages after summarization."""

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_naive_keeps_recent_after_summary(self, mock_completion):
        """After summarization, output must have more than 2 messages (summary + tail)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        memory = NaiveSummarizationMemory(max_tokens=50, keep_recent=3)

        # Build enough messages to trigger summarization
        history = [
            _msg("system", "System " * 20),
            _msg("user", "Task " * 10),
            _msg("assistant", "Response " * 10),
            _msg("user", "Obs1 " * 10),
            _msg("assistant", "Response2 " * 10),
        ]
        obs = _msg("user", "Obs2 " * 10)

        result = memory.manage_context(history, obs)

        if result.metadata["was_compacted"]:
            # Must have: 1 summary + keep_recent tail = at least 4 messages
            self.assertGreater(len(result.content), 2,
                f"After summarization, got only {len(result.content)} messages. "
                f"Expected > 2 (summary + recent tail)")

class TestPassThroughDirectMessages(unittest.TestCase):
    """Bug 4: PassThroughMemory missing direct_messages flag."""

    def test_passthrough_has_direct_messages(self):
        """PassThroughMemory metadata must include direct_messages: True."""
        from memgym.memory.base import PassThroughMemory
        memory = PassThroughMemory()

        result = memory.manage_context(
            [_msg("system", "test")],
            _msg("user", "hello")
        )

        self.assertIn("direct_messages", result.metadata,
            "direct_messages key missing from PassThroughMemory metadata")
        self.assertTrue(result.metadata["direct_messages"])

class TestAllStrategiesConsistentMetadata(unittest.TestCase):
    """All strategies must return consistent metadata keys."""

    REQUIRED_KEYS = {"tokens", "original_tokens", "was_compacted",
                     "compression_ratio", "direct_messages", "strategy"}

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_all_strategies_consistent_metadata(self, mock_completion):
        """Every strategy must include all required metadata keys."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_completion.return_value = mock_response

        from memgym.memory.base import PassThroughMemory
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory

        strategies = [
            ("PassThrough", PassThroughMemory()),
            ("ObservationMasking", ObservationMaskingMemory(attention_window=5)),
            ("NaiveSummarization", NaiveSummarizationMemory(max_tokens=100000)),
        ]

        history = [_msg("system", "test"), _msg("user", "task")]
        obs = _msg("assistant", "response")

        for name, strategy in strategies:
            result = strategy.manage_context(history, obs)
            missing = self.REQUIRED_KEYS - set(result.metadata.keys())
            self.assertEqual(missing, set(),
                f"{name} metadata missing keys: {missing}")

class TestFinalStepObservation(unittest.TestCase):
    """Bug 5: Final step observation lookup is fragile."""

    def _make_agent_with_messages(self, messages):
        """Create a MemoryAwareSWEAgent-like object with given messages."""
        from memgym.memory.base import PassThroughMemory

        # We need to mock the agent to test get_training_trajectory()
        # Import the real class
        from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent

        mock_model = MagicMock()
        mock_env = MagicMock()
        memory = PassThroughMemory()

        agent = MemoryAwareSWEAgent(mock_model, mock_env, memory)
        agent.messages = messages
        agent._task_description = "Test task"
        return agent

    def test_final_step_observation_no_returncode(self):
        """Final step must capture observation even without 'returncode' keyword."""
        messages = [
            _msg("system", "System prompt"),
            _msg("user", "Fix the bug"),
            _msg("assistant", "THOUGHT: thinking\n```bash\necho test\n```"),
            _msg("user", "This is the output without returncode keyword"),
        ]

        agent = self._make_agent_with_messages(messages)

        # Simulate that query() was called once (creating a pending step)
        agent._pending_step = {
            "step": 1,
            "thought": "thinking",
            "action": "echo test",
            "observation": "",
            "memory": {"was_compacted": False},
        }

        traj = agent.get_training_trajectory()
        last_step = traj["steps"][-1]

        self.assertNotEqual(last_step["observation"], "",
            "Final step observation should not be empty when last message lacks 'returncode'")
        self.assertIn("output without returncode", last_step["observation"])

    def test_final_step_observation_with_error_msg(self):
        """Final step must capture the ERROR message, not an earlier returncode message."""
        messages = [
            _msg("system", "System prompt"),
            _msg("user", "Fix the bug"),
            _msg("assistant", "THOUGHT: step1\n```bash\nls\n```"),
            _msg("user", "<returncode>0</returncode>\n<output>file1.py</output>"),
            _msg("assistant", "THOUGHT: step2\n```bash\nbad syntax"),
            _msg("user", "Format error: your response must contain a bash code block"),
        ]

        agent = self._make_agent_with_messages(messages)

        # Simulate pending step for step 2
        agent._pending_step = {
            "step": 2,
            "thought": "step2",
            "action": "",
            "observation": "",
            "memory": {"was_compacted": False},
        }

        traj = agent.get_training_trajectory()
        last_step = traj["steps"][-1]

        # Must capture the format error message, not the earlier returncode message
        self.assertIn("Format error", last_step["observation"],
            f"Final step should capture format error message, got: {last_step['observation'][:100]}")

class TestNaiveSummarizationContent(unittest.TestCase):
    """Bug 1 (continued): Content sent to LLM for summarization must not be duplicated."""

    @patch("memgym.memory.strategies.naive_summarization.completion")
    def test_naive_summarization_content_not_duplicated(self, mock_completion):
        """The content sent to LLM for summarization must not duplicate messages."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary."
        mock_completion.return_value = mock_response

        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory
        memory = NaiveSummarizationMemory(max_tokens=50, keep_recent=2)

        # Use unique per-message markers to detect duplication
        # Step 1
        memory.manage_context(
            [_msg("system", "SYS_MARKER_001")],
            _msg("user", "TASK_MARKER_001 " * 20)
        )

        # Step 2 — should trigger summarization
        memory.manage_context(
            [_msg("system", "SYS_MARKER_001"), _msg("user", "TASK_MARKER_001 " * 20),
             _msg("assistant", "RESP_MARKER_001 " * 20)],
            _msg("user", "OBS_MARKER_001 " * 20)
        )

        if mock_completion.called:
            call_args = mock_completion.call_args
            user_content = call_args.kwargs.get("messages", call_args[1].get("messages", []))[-1]["content"]

            # Each unique marker should appear at most twice:
            # once from the message itself, maybe once from "Previous summary" prefix
            sys_count = user_content.count("SYS_MARKER_001")
            self.assertLessEqual(sys_count, 2,
                f"SYS_MARKER_001 appears {sys_count} times in summarization input — duplication detected")

            # RESP_MARKER_001 is in one message — should appear once (not duplicated)
            resp_count = user_content.count("RESP_MARKER_001")
            self.assertLessEqual(resp_count, 20,  # 20 repetitions in ONE message
                f"RESP_MARKER_001 appears {resp_count} times — message duplicated in summarization input")

class TestReplayAnalysisMetadataMatch(unittest.TestCase):
    """Replay analysis must produce same metadata as direct run for deterministic strategies."""

    def test_replay_analysis_metadata_matches_direct(self):
        """Per-step metadata from analyze_replay must match direct manage_context calls."""
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory
        from memgym.gym.swe_bench.env import _build_step_index

        # Build a realistic message sequence
        messages = [
            _msg("system", "You are a coding agent."),
            _msg("user", "Fix the bug in main.py"),
        ]
        for step in range(1, 9):
            messages.append(_msg("assistant", f"THOUGHT: Step {step}\n```bash\necho {step}\n```"))
            messages.append(_msg("user", f"<returncode>0</returncode>\n<output>output{step} " + "x" * 200 + "</output>"))

        steps = _build_step_index(messages)

        # Direct run: call manage_context at each step
        direct_memory = ObservationMaskingMemory(attention_window=3, keep_first=1)
        direct_results = []
        for step_info in steps:
            assistant_idx = step_info["assistant_msg_index"]
            pre_query = messages[:assistant_idx]
            history = pre_query[:-1]
            obs = pre_query[-1]
            result = direct_memory.manage_context(history, obs, {"step": step_info["step"]})
            direct_results.append({
                "step": step_info["step"],
                "filtered_msgs": len(result.content) if isinstance(result.content, list) else 0,
                "was_compacted": result.metadata.get("was_compacted", False),
                "compression_ratio": result.metadata.get("compression_ratio", 1.0),
            })

        # Replay: use analyze_replay
        from replay_swe_bench import analyze_replay
        replay_data = {"messages": messages, "steps": steps, "instance_id": "test", "memory_strategy": "test"}
        replay_memory = ObservationMaskingMemory(attention_window=3, keep_first=1)
        analysis = analyze_replay(replay_data, replay_memory)

        # Compare per-step
        self.assertEqual(len(direct_results), len(analysis["steps"]),
            f"Step count mismatch: direct={len(direct_results)}, replay={len(analysis['steps'])}")

        for d, r in zip(direct_results, analysis["steps"]):
            self.assertEqual(d["step"], r["step"], f"Step number mismatch")
            self.assertEqual(d["filtered_msgs"], r["filtered_msgs"],
                f"Step {d['step']}: filtered_msgs mismatch: direct={d['filtered_msgs']}, replay={r['filtered_msgs']}")
            self.assertEqual(d["was_compacted"], r["compacted"],
                f"Step {d['step']}: was_compacted mismatch: direct={d['was_compacted']}, replay={r['compacted']}")
            self.assertAlmostEqual(d["compression_ratio"], r["compression_ratio"], places=2,
                msg=f"Step {d['step']}: compression_ratio mismatch: direct={d['compression_ratio']}, replay={r['compression_ratio']}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
