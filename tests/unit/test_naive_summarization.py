"""
Tests for NaiveSummarizationMemory.

Tests naive context management with environment-specific prompts (swe, tau2, generic).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory, SUMMARIZATION_PROMPTS
from memgym.memory.backends.summarizer_backend import (
    LitellmSummarizerBackend,
    LocalHFSummarizerBackend,
    SummarizerBackend,
)

def _msg(role, content):
    return {"role": role, "content": content}

class TestNaiveSummarizationMemory(unittest.TestCase):
    """Unit tests with mocked litellm client.

    Patches `completion` at its real import site (summarizer_backend) rather
    than inside naive_summarization — the litellm call lives in the backend
    now, naive_summarization only holds a reference to the backend instance.
    """

    def setUp(self):
        self.mock_completion_patcher = patch("memgym.memory.backends.summarizer_backend.completion")
        self.mock_completion = self.mock_completion_patcher.start()

        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message.content = "Mock summary."
        self.mock_response.choices[0].message.reasoning_content = None
        self.mock_completion.return_value = self.mock_response

    def tearDown(self):
        self.mock_completion_patcher.stop()

    def test_no_summarization_under_threshold(self):
        """No summarization when under token threshold."""
        memory = NaiveSummarizationMemory(max_tokens=100000)
        result = memory.manage_context([], _msg("user", "Short observation."))
        self.assertFalse(result.metadata["was_compacted"])
        self.mock_completion.assert_not_called()

    def test_summarization_triggers_over_threshold(self):
        """Summarization triggers when over token threshold."""
        memory = NaiveSummarizationMemory(max_tokens=20, keep_recent=1)
        result = memory.manage_context(
            [_msg("system", "Long prompt " * 10)],
            _msg("user", "Long observation " * 10)
        )
        self.assertTrue(result.metadata["was_compacted"])
        self.mock_completion.assert_called_once()

    def test_env_specific_prompts(self):
        """Each environment type uses correct prompt."""
        for env_type, keywords in [
            ("generic", ["Key facts"]),
            ("swe", ["coding agent", "Architectural decisions"]),
            ("tau2", ["customer service", "User's original intent"]),
        ]:
            self.mock_completion.reset_mock()
            memory = NaiveSummarizationMemory(max_tokens=10, env_type=env_type, keep_recent=1)
            memory.manage_context(
                [_msg("system", "Long " * 20)],
                _msg("user", "Trigger summarization " * 10)
            )
            if self.mock_completion.called:
                call_args = self.mock_completion.call_args
                messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
                system_msg = messages[0]["content"]
                for kw in keywords:
                    self.assertIn(kw, system_msg, f"{env_type} prompt missing '{kw}'")

    def test_reset(self):
        """Reset clears all state."""
        memory = NaiveSummarizationMemory(max_tokens=10, keep_recent=1)
        memory.manage_context(
            [_msg("system", "Long " * 20)],
            _msg("user", "Long observation " * 20)
        )
        self.assertIsNotNone(memory._summary)
        memory.reset()
        self.assertIsNone(memory._summary)
        self.assertEqual(memory._compaction_count, 0)

    def test_keeps_recent_messages(self):
        """After summarization, output keeps recent tail messages."""
        memory = NaiveSummarizationMemory(max_tokens=20, keep_recent=3)
        msgs = [
            _msg("system", "Long system " * 20),
            _msg("user", "Task " * 10),
            _msg("assistant", "Response " * 10),
            _msg("user", "Obs1 " * 10),
        ]
        obs = _msg("assistant", "Response2 " * 10)
        result = memory.manage_context(msgs, obs)

        if result.metadata["was_compacted"]:
            # Layout: [head (keep_first=1 system prompt), summary_msg, tail (keep_recent=3)]
            # so the summary sits at index 1, not index 0.
            self.assertGreater(len(result.content), 2,
                "Should keep more than just summary + 1 message")
            self.assertIn("[Summary]", result.content[1]["content"])

    def test_compression_ratio_present(self):
        """Metadata includes compression_ratio."""
        memory = NaiveSummarizationMemory(max_tokens=20, keep_recent=1)
        result = memory.manage_context(
            [_msg("system", "Long " * 20)],
            _msg("user", "Long " * 20)
        )
        self.assertIn("compression_ratio", result.metadata)

    def test_registry(self):
        """Memory model is registered correctly."""
        from memgym.memory import get_memory_model, list_memory_models
        self.assertIn("naive", list_memory_models())
        self.assertIn("naive_summarization", list_memory_models())

class TestSummarizerBackendInjection(unittest.TestCase):
    """Contract tests for the pluggable SummarizerBackend interface.

    These pin the behavior the runtime depends on: NaiveSummarizationMemory
    must accept any object satisfying the Protocol, default to litellm, and
    not hard-code a particular backend.
    """

    def test_default_backend_is_litellm(self):
        """Without an explicit backend, construction picks LitellmSummarizerBackend."""
        memory = NaiveSummarizationMemory(summarization_model="gpt-4o-mini")
        self.assertIsInstance(memory.summarizer_backend, LitellmSummarizerBackend)
        self.assertEqual(memory.summarizer_backend.model, "gpt-4o-mini")

    def test_custom_backend_replaces_litellm(self):
        """Passing a backend bypasses LitellmSummarizerBackend entirely."""
        class StubBackend:
            calls = []
            def summarize(self, system, user, max_tokens=1024):
                StubBackend.calls.append((system[:30], user[:30], max_tokens))
                return "STUB SUMMARY"

        stub = StubBackend()
        memory = NaiveSummarizationMemory(
            max_tokens=20,
            keep_recent=1,
            env_type="swe",
            summarizer_backend=stub,
        )
        result = memory.manage_context(
            [_msg("system", "Long " * 20)],
            _msg("user", "Trigger " * 20),
        )
        self.assertTrue(result.metadata["was_compacted"])
        self.assertEqual(len(StubBackend.calls), 1)
        self.assertEqual(memory._summary, "STUB SUMMARY")

    def test_empty_backend_return_keeps_prior_summary(self):
        """Failed backend call (empty return) must not advance _summarized_upto."""
        class FailingBackend:
            def summarize(self, system, user, max_tokens=1024):
                return ""  # simulate API failure / empty content / empty reasoning

        memory = NaiveSummarizationMemory(
            max_tokens=20,
            keep_recent=1,
            summarizer_backend=FailingBackend(),
        )
        memory.manage_context(
            [_msg("system", "Long " * 20)],
            _msg("user", "Trigger " * 20),
        )
        self.assertIsNone(memory._summary, "empty backend return must not create a summary")
        self.assertEqual(memory._summarized_upto, 0,
                         "_summarized_upto must stay at 0 so next attempt retries the same window")

    def test_protocol_runtime_check(self):
        """Any object with `.summarize(...)` satisfies the Protocol (runtime-checkable)."""
        class Duck:
            def summarize(self, system, user, max_tokens=1024):
                return "quack"
        self.assertIsInstance(Duck(), SummarizerBackend)
        self.assertIsInstance(LitellmSummarizerBackend("gpt-4o-mini"), SummarizerBackend)

    def test_local_hf_backend_from_model_satisfies_protocol(self):
        """`from_model()` alternate ctor produces a Protocol-compliant backend.

        Uses a minimal mock so we don't pull transformers or an 8B checkpoint
        into unit tests. The eager-load `__init__` path is exercised in
        integration tests with real weights, not here.
        """

        class _MockTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "PROMPT:" + " | ".join(m["content"] for m in messages)

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                import torch
                # One token per word — crude but deterministic for test.
                tokens = text.split()
                ids = torch.tensor([[hash(t) % 1000 for t in tokens]])
                mask = torch.ones_like(ids)
                return {"input_ids": ids, "attention_mask": mask}

            def decode(self, tokens, skip_special_tokens=True):
                return "  MOCK SUMMARY  "

        class _MockModel:
            device = None

            def eval(self):
                return self

            def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, **kw):
                import torch
                prompt_len = input_ids.shape[1]
                new = torch.zeros((1, 5), dtype=input_ids.dtype)
                return torch.cat([input_ids, new], dim=1)

        backend = LocalHFSummarizerBackend.from_model(_MockModel(), _MockTokenizer())
        self.assertIsInstance(backend, SummarizerBackend)
        result = backend.summarize("system prompt", "history goes here")
        self.assertEqual(result, "MOCK SUMMARY")  # stripped of whitespace

    def test_local_hf_backend_generation_kwargs_are_mutable(self):
        """RL loop must be able to override temperature/top_p per rollout
        without breaking the Protocol signature."""

        class _MockTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def apply_chat_template(self, messages, **kw):
                return "x"

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                import torch
                ids = torch.tensor([[1, 2, 3]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, tokens, skip_special_tokens=True):
                return "s"

        class _MockModel:
            device = None

            def eval(self):
                return self

            def generate(self, input_ids=None, **kw):
                _MockModel.last_kwargs = kw
                import torch
                return torch.cat([input_ids, torch.zeros((1, 1), dtype=input_ids.dtype)], dim=1)

        backend = LocalHFSummarizerBackend.from_model(_MockModel(), _MockTokenizer())
        backend.generation_kwargs["temperature"] = 0.7
        backend.generation_kwargs["do_sample"] = True
        backend.summarize("s", "u")
        self.assertEqual(_MockModel.last_kwargs["temperature"], 0.7)
        self.assertTrue(_MockModel.last_kwargs["do_sample"])


class TestIntegration(unittest.TestCase):
    """Integration tests with real LLM API."""

    @unittest.skipUnless(
        os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_INTEGRATION_TESTS"),
        "Set OPENAI_API_KEY and RUN_INTEGRATION_TESTS=1"
    )
    def test_real_summarization(self):
        """Test real summarization for all environment types."""
        test_cases = [
            ("generic", [
                _msg("user", "User logged in from IP 192.168.1.100 at 10:00 AM."),
                _msg("user", "User browsed electronics category for 15 minutes looking at laptops."),
                _msg("user", "User added MacBook Pro SKU-12345 to cart and applied SAVE10 coupon."),
            ]),
            ("swe", [
                _msg("user", "Reading src/main.py to understand the entry point and module dependencies."),
                _msg("user", "Error: ImportError - missing 'utils' module, need to create src/utils.py."),
                _msg("user", "Implemented calculate_total(items) helper with tax calculation logic."),
            ]),
            ("tau2", [
                _msg("user", "User: I need to cancel my flight booking #ABC123 for March 15th."),
                _msg("user", "Agent: I'll help cancel that. Tool: get_booking(ABC123) -> confirmed."),
                _msg("user", "User: Yes please cancel and refund to original payment method."),
            ]),
        ]

        for env_type, observations in test_cases:
            memory = NaiveSummarizationMemory(max_tokens=30, env_type=env_type, keep_recent=1)
            for i, obs in enumerate(observations):
                history = observations[:i]
                result = memory.manage_context(history, obs)

            print(f"\n=== {env_type.upper()} ===")
            print(f"Summarizations: {result.metadata['summarization_count']}")
            print(f"Summary: {memory._summary}")

            self.assertTrue(result.metadata["has_summary"], f"{env_type} should have summary")

class TestThinkingStripping(unittest.TestCase):
    """H.1.5 — verify <think> blocks are stripped before reaching the summarizer.

    Patches `completion` at the backend's import site to capture the content
    passed to the summarizer. Assertions check that (a) <think>...</think>
    spans and (b) dangling unclosed <think>... prefixes (vLLM #35221) are
    both removed, while messages with no think tokens pass through unchanged.
    """

    def setUp(self):
        self.mock_completion_patcher = patch("memgym.memory.backends.summarizer_backend.completion")
        self.mock_completion = self.mock_completion_patcher.start()
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message.content = "Mock summary."
        self.mock_response.choices[0].message.reasoning_content = None
        self.mock_completion.return_value = self.mock_response

    def tearDown(self):
        self.mock_completion_patcher.stop()

    def _summarizer_user_content(self) -> str:
        """Return the `user` message content actually sent to the summarizer."""
        self.assertTrue(self.mock_completion.called, "summarizer was not invoked")
        kwargs = self.mock_completion.call_args.kwargs
        for m in kwargs["messages"]:
            if m["role"] == "user":
                return m["content"]
        self.fail("no user message passed to summarizer")

    def test_strips_closed_think_block(self):
        """<think>…</think> must be removed before content hits the summarizer."""
        memory = NaiveSummarizationMemory(max_tokens=10, keep_recent=0)
        history = [
            {"role": "assistant", "content": "<think>plan the fix</think>\n\nTHOUGHT: read file"},
            {"role": "user", "content": "tool output here " * 50},
        ]
        memory.manage_context(history[:-1], history[-1])
        sent = self._summarizer_user_content()
        self.assertNotIn("<think>", sent)
        self.assertNotIn("plan the fix", sent)
        self.assertIn("THOUGHT: read file", sent)

    def test_strips_unclosed_think_block(self):
        """vLLM #35221 — dangling <think>... with no closer must be stripped."""
        memory = NaiveSummarizationMemory(max_tokens=10, keep_recent=0)
        history = [
            {"role": "assistant", "content": "<think>truncated mid-thought " * 20},
            {"role": "user", "content": "observation " * 50},
        ]
        memory.manage_context(history[:-1], history[-1])
        sent = self._summarizer_user_content()
        self.assertNotIn("<think>", sent)
        self.assertNotIn("truncated mid-thought", sent)

    def test_nonthinking_content_unchanged(self):
        """Messages with no <think> tokens pass through unchanged."""
        memory = NaiveSummarizationMemory(max_tokens=10, keep_recent=0)
        history = [
            {"role": "assistant", "content": "THOUGHT: no thinking\n```mswea_bash_command\nls\n```"},
            {"role": "user", "content": "file output " * 80},
        ]
        memory.manage_context(history[:-1], history[-1])
        sent = self._summarizer_user_content()
        self.assertIn("THOUGHT: no thinking", sent)
        self.assertIn("```mswea_bash_command", sent)
        self.assertEqual(memory._last_think_chars_stripped, 0)
        self.assertEqual(memory._last_think_strip_ratio, 0.0)

    def test_telemetry_strip_ratio_populated(self):
        """After a thinking-heavy compaction, strip-ratio telemetry is > 0."""
        memory = NaiveSummarizationMemory(max_tokens=10, keep_recent=0)
        payload = "<think>" + ("x " * 500) + "</think>\n\nTHOUGHT: done"
        history = [
            {"role": "assistant", "content": payload},
            {"role": "user", "content": "obs " * 80},
        ]
        memory.manage_context(history[:-1], history[-1])
        stats = memory.get_stats()
        self.assertGreater(stats["last_think_chars_stripped"], 0)
        self.assertGreater(stats["last_think_strip_ratio"], 0.0)
        self.assertLessEqual(stats["last_think_strip_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
