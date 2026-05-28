import unittest
from unittest.mock import MagicMock, patch

from memgym.memory.strategies.adaptive_token_budget import AdaptiveTokenBudgetMemory

def _msg(role, content):
    return {"role": role, "content": content}

class TestAdaptiveTokenBudgetMemory(unittest.TestCase):
    def setUp(self):
        self.mock_completion_patcher = patch("memgym.memory.strategies.adaptive_token_budget.completion")
        self.mock_completion = self.mock_completion_patcher.start()

        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message.content = "Compressed history summary."
        self.mock_completion.return_value = self.mock_response

    def tearDown(self):
        self.mock_completion_patcher.stop()

    def test_compacts_middle_when_token_budget_exceeded(self):
        memory = AdaptiveTokenBudgetMemory(
            max_tokens=20,
            keep_first=1,
            keep_recent=2,
            preserve_first_user=True,
        )
        messages = [
            _msg("system", "System prompt " * 20),
            _msg("user", "Issue statement " * 20),
            _msg("assistant", "Checked parser.py " * 20),
            _msg("user", "Observed traceback " * 20),
        ]
        current = _msg("assistant", "Patched failing test " * 20)

        result = memory.manage_context(messages, current)

        self.assertTrue(result.metadata["was_compacted"])
        self.assertEqual(result.content[0]["role"], "system")
        self.assertEqual(result.content[1]["role"], "user")
        self.assertIn("[Summary]", result.content[2]["content"])
        self.assertEqual(result.metadata["summary_trigger_count"], 1)
        self.assertEqual(result.metadata["num_summaries"], 1)
        self.assertGreater(result.metadata["compression_ratio"], 1.0)
        self.assertLess(result.metadata["raw_tokens_after"], result.metadata["raw_tokens_before"])

    def test_preserve_first_user_can_be_disabled(self):
        memory = AdaptiveTokenBudgetMemory(
            max_tokens=20,
            keep_first=1,
            keep_recent=1,
            preserve_first_user=False,
        )
        messages = [
            _msg("system", "System prompt " * 20),
            _msg("user", "Original issue statement " * 20),
            _msg("assistant", "Intermediate work " * 20),
        ]
        current = _msg("user", "Latest observation " * 20)

        result = memory.manage_context(messages, current)

        contents = [m.get("content", "") for m in result.content if isinstance(m, dict)]
        self.assertTrue(any("[Summary]" in content for content in contents))
        self.assertFalse(any(content == messages[1]["content"] for content in contents))

    def test_stats_track_token_quality_metrics(self):
        memory = AdaptiveTokenBudgetMemory(max_tokens=20, keep_first=1, keep_recent=1)
        messages = [
            _msg("system", "System prompt " * 20),
            _msg("user", "Issue statement " * 20),
            _msg("assistant", "Long middle " * 20),
        ]

        memory.manage_context(messages, _msg("user", "Observation one " * 20))
        memory.manage_context(messages, _msg("assistant", "Observation two " * 20))
        stats = memory.get_stats()

        self.assertEqual(stats["summary_trigger_count"], 2)
        self.assertEqual(stats["num_summaries"], 2)
        self.assertGreater(stats["avg_raw_tokens_before"], 0)
        self.assertGreater(stats["avg_raw_tokens_after"], 0)
        self.assertGreaterEqual(stats["peak_raw_tokens_before"], stats["raw_tokens_before"])
        self.assertIn("avg_compression_ratio", stats)

    def test_registered_name_available(self):
        from memgym.memory import get_memory_model, list_memory_models

        self.assertIn("adaptive_token_budget", list_memory_models())
        memory = get_memory_model("adaptive_token_budget", max_tokens=100)
        self.assertIsInstance(memory, AdaptiveTokenBudgetMemory)

if __name__ == "__main__":
    unittest.main(verbosity=2)
