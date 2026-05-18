"""Tests for the (see paper) env-reward wrapper.

Unit-only — no SWEMemoryEnv construction, no docker, no vllm. A tiny
FakeEnv mimics the contract: exposes `_memory_model` with a
`summarizer_backend` slot + a `run_episode()` that returns the dict
shape from `SWEMemoryEnv.run_episode()` (see env.py:1001).
"""
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List

from memgym.memory.naive_summarization import NaiveSummarizationMemory
from memgym.memory.summarizer_backend import SummarizerBackend
from memgym.training.rl.env_reward import (
    EpisodeResult,
    RecordingSummarizerBackend,
    SummaryCall,
    run_episode_with_reward,
)


class _StubInner:
    """Minimal `SummarizerBackend` that returns a canned string."""

    def __init__(self, text: str = "STUB"):
        self.text = text
        self.n_calls = 0

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.n_calls += 1
        return self.text


class TestRecordingSummarizerBackend(unittest.TestCase):
    def test_records_all_calls_in_order(self):
        recorder = RecordingSummarizerBackend(_StubInner("OK"))
        recorder.summarize("sys1", "user1", max_tokens=512)
        recorder.summarize("sys2", "user2", max_tokens=1024)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(recorder.calls[0].system, "sys1")
        self.assertEqual(recorder.calls[0].user, "user1")
        self.assertEqual(recorder.calls[0].max_tokens, 512)
        self.assertEqual(recorder.calls[0].completion, "OK")
        self.assertEqual(recorder.calls[1].system, "sys2")
        self.assertEqual(recorder.calls[1].max_tokens, 1024)

    def test_returns_inner_output_verbatim(self):
        recorder = RecordingSummarizerBackend(_StubInner("quack"))
        self.assertEqual(recorder.summarize("s", "u"), "quack")

    def test_satisfies_protocol(self):
        """Recorder itself must duck-type as a SummarizerBackend."""
        self.assertIsInstance(
            RecordingSummarizerBackend(_StubInner()), SummarizerBackend
        )

    def test_empty_inner_return_is_recorded(self):
        """An empty-string inner return (failed summarization) must still
        show up in `.calls` — the RL loop wants to see zero-completion
        rollouts so it can penalize or filter them, not pretend they
        didn't happen."""
        recorder = RecordingSummarizerBackend(_StubInner(""))
        recorder.summarize("s", "u")
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].completion, "")


@dataclass
class _FakeEnv:
    """Enough of SWEMemoryEnv's surface for env_reward to work against."""

    instance_id: str
    _memory_model: Any
    _fake_result: Dict[str, Any]

    def run_episode(self, verbose: bool = False) -> Dict[str, Any]:
        # Simulate a multi-compaction episode — the real memory
        # manager's `_summarize_all` would call the backend each
        # compaction. We do the same in the fake to exercise the
        # recording path end-to-end.
        for i in range(self._fake_result.get("_n_compactions", 0)):
            self._memory_model.summarizer_backend.summarize(
                system=f"prompt-{i}", user=f"history-{i}", max_tokens=1024
            )
        return self._fake_result


class TestRunEpisodeWithReward(unittest.TestCase):
    def _mk_env(self, reward: float, n_compactions: int = 2) -> _FakeEnv:
        mm = NaiveSummarizationMemory(max_tokens=999_999)  # threshold never trips
        return _FakeEnv(
            instance_id="fake__test_001",
            _memory_model=mm,
            _fake_result={
                "reward": reward,
                "status": "Submitted",
                "patch_size": 1234,
                "token_stats": {"total_input": 100, "total_output": 50},
                "wrapper_stats": {
                    "agent": {
                        "avg_compression_ratio": 3.5,
                        "max_compression_ratio": 4.0,
                        "max_original_tokens": 13264,
                        "max_filtered_tokens": 3716,
                    }
                },
                "_n_compactions": n_compactions,
            },
        )

    def test_terminal_reward_passthrough(self):
        env = self._mk_env(reward=1.0)
        out = run_episode_with_reward(env, _StubInner("SUM"))
        self.assertIsInstance(out, EpisodeResult)
        self.assertEqual(out.R_terminal, 1.0)
        self.assertEqual(out.instance_id, "fake__test_001")

    def test_unresolved_episode_is_zero_reward(self):
        env = self._mk_env(reward=0.0)
        out = run_episode_with_reward(env, _StubInner())
        self.assertEqual(out.R_terminal, 0.0)

    def test_captures_every_summary(self):
        env = self._mk_env(reward=0.0, n_compactions=3)
        out = run_episode_with_reward(env, _StubInner("S"))
        self.assertEqual(len(out.episode_summaries), 3)
        self.assertEqual(out.metadata["num_summaries"], 3)
        # Each call pinned both input AND output — this is what GRPO consumes.
        self.assertEqual(out.episode_summaries[0].user, "history-0")
        self.assertEqual(out.episode_summaries[0].completion, "S")

    def test_hot_swap_replaces_memory_manager_backend(self):
        """Contract: after the call, the memory manager's backend is a
        RecordingSummarizerBackend wrapping the one the caller passed.
        The RL loop relies on this — if run_episode_with_reward silently
        left the old backend in place, summaries would go unrecorded."""
        env = self._mk_env(reward=0.0, n_compactions=1)
        stub = _StubInner()
        run_episode_with_reward(env, stub)
        self.assertIsInstance(env._memory_model.summarizer_backend, RecordingSummarizerBackend)
        self.assertIs(env._memory_model.summarizer_backend.inner, stub)

    def test_metadata_surfaces_compression_stats(self):
        env = self._mk_env(reward=1.0)
        out = run_episode_with_reward(env, _StubInner())
        self.assertEqual(out.metadata["avg_compression_ratio"], 3.5)
        self.assertEqual(out.metadata["max_original_tokens"], 13264)
        self.assertEqual(out.metadata["status"], "Submitted")

    def test_env_without_summarizer_backend_raises(self):
        """If the memory manager doesn't expose `summarizer_backend`, the
        wrapper must fail loudly — silently running with the wrong
        summarizer is the worst possible outcome for an RL loop."""

        class _BadManager:
            pass  # No summarizer_backend attr.

        bad_env = _FakeEnv(
            instance_id="x", _memory_model=_BadManager(), _fake_result={"reward": 0.0}
        )
        with self.assertRaises(TypeError) as ctx:
            run_episode_with_reward(bad_env, _StubInner())
        self.assertIn("summarizer_backend", str(ctx.exception))

    def test_env_with_none_memory_manager_raises(self):
        env = _FakeEnv(instance_id="x", _memory_model=None, _fake_result={"reward": 0.0})
        with self.assertRaises(TypeError):
            run_episode_with_reward(env, _StubInner())

    def test_episode_resolved_true_when_reward_positive(self):
        env = self._mk_env(reward=1.0)
        out = run_episode_with_reward(env, _StubInner())
        self.assertTrue(out.episode_resolved)

    def test_episode_resolved_false_when_reward_zero(self):
        env = self._mk_env(reward=0.0)
        out = run_episode_with_reward(env, _StubInner())
        self.assertFalse(out.episode_resolved)

    def test_hit_context_limit_detected_from_error_string(self):
        env = self._mk_env(reward=0.0, n_compactions=0)
        env._fake_result["error"] = "litellm.ContextWindowExceededError: prompt is too long"
        out = run_episode_with_reward(env, _StubInner())
        self.assertTrue(out.hit_context_limit)

    def test_hit_context_limit_false_for_other_errors(self):
        env = self._mk_env(reward=0.0, n_compactions=0)
        env._fake_result["error"] = "ValueError: agent_llm must be set"
        out = run_episode_with_reward(env, _StubInner())
        self.assertFalse(out.hit_context_limit)

    def test_hit_context_limit_false_when_no_error(self):
        env = self._mk_env(reward=1.0)
        out = run_episode_with_reward(env, _StubInner())
        self.assertFalse(out.hit_context_limit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
