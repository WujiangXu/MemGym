"""Tests for `_build_memory_model` in eval_memory_ab.py.

Pins the CLI-to-backend wiring so the H.2 SFT smoke-eval path stays
correct across refactors. No SWEMemoryEnv construction — we only
verify the memory manager + backend selection logic.
"""
import argparse
import sys
import types
import unittest
from unittest.mock import MagicMock

from memgym.memory.llm_summarizing import LLMSummarizingMemory
from memgym.memory.naive_summarization import NaiveSummarizationMemory
from memgym.memory.sliding_window_summary import SlidingWindowSummaryMemory
from memgym.memory.structured_summary import StructuredSummaryMemory
from memgym.memory.summarizer_backend import LitellmSummarizerBackend
from memgym.training.scripts.eval_memory_ab import _build_memory_model


def _ns(**overrides) -> argparse.Namespace:
    """Namespace with the eval_memory_ab defaults, overridable per-test."""
    defaults = {
        "memory_strategy": "naive",
        "summarizer_backend": "litellm",
        "summarization_model": None,
        "agent_llm": "openai/qwen36-35b-a3b-fp8",
        "max_tokens": 3000,
        "summarizer_dtype": "bfloat16",
        "summarizer_max_prompt_tokens": 6144,
        "llmlingua2_model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        "llmlingua2_rate": 0.33,
        "llmlingua2_device": "cpu",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildMemoryModel(unittest.TestCase):
    def test_default_litellm_falls_back_to_agent_llm(self):
        """Legacy path: no --summarization-model → use --agent-llm."""
        mm = _build_memory_model(_ns())
        self.assertIsInstance(mm, NaiveSummarizationMemory)
        self.assertIsInstance(mm.summarizer_backend, LitellmSummarizerBackend)
        self.assertEqual(mm.summarizer_backend.model, "openai/qwen36-35b-a3b-fp8")

    def test_litellm_with_explicit_summarization_model(self):
        mm = _build_memory_model(
            _ns(summarization_model="openai/gpt-4o-mini")
        )
        self.assertEqual(mm.summarizer_backend.model, "openai/gpt-4o-mini")

    def test_local_hf_requires_summarization_model(self):
        """local-hf without a model path must fail loudly — silently
        falling back to --agent-llm would load the wrong checkpoint."""
        with self.assertRaises(ValueError) as ctx:
            _build_memory_model(_ns(summarizer_backend="local-hf"))
        self.assertIn("--summarizer-backend local-hf", str(ctx.exception))
        self.assertIn("--summarization-model", str(ctx.exception))

    def test_env_type_always_swe(self):
        """Harness hard-codes env_type=swe so the summarizer uses the
        SWE-specific prompt regardless of backend."""
        mm = _build_memory_model(_ns())
        self.assertEqual(mm.env_type, "swe")

    def test_max_tokens_threshold_plumbed(self):
        mm = _build_memory_model(_ns(max_tokens=5000))
        self.assertEqual(mm.max_tokens, 5000)


class TestMemoryStrategy(unittest.TestCase):
    """Dispatch on `--memory-strategy` — extra baselines."""

    def test_structured_strategy_instantiates(self):
        mm = _build_memory_model(_ns(memory_strategy="structured"))
        self.assertIsInstance(mm, StructuredSummaryMemory)
        # agent_llm fallback applies when --summarization-model is unset.
        self.assertEqual(mm.summarization_model, "openai/qwen36-35b-a3b-fp8")
        # Token-count trigger must be wired (same threshold as Arm B/C naive).
        self.assertEqual(mm.max_token_size, 3000)

    def test_llm_strategy_instantiates(self):
        mm = _build_memory_model(_ns(memory_strategy="llm"))
        self.assertIsInstance(mm, LLMSummarizingMemory)
        self.assertEqual(mm.max_token_size, 3000)

    def test_sliding_strategy_instantiates(self):
        mm = _build_memory_model(_ns(memory_strategy="sliding"))
        self.assertIsInstance(mm, SlidingWindowSummaryMemory)
        self.assertEqual(mm.env_type, "swe")

    def test_non_naive_rejects_local_hf_backend(self):
        """Mixing local-hf backend with a non-naive strategy silently drops
        the SFT checkpoint — must fail loudly instead."""
        with self.assertRaises(ValueError) as ctx:
            _build_memory_model(_ns(
                memory_strategy="structured",
                summarizer_backend="local-hf",
                summarization_model="/path/to/sft-checkpoint",
            ))
        self.assertIn("structured", str(ctx.exception))
        self.assertIn("local-hf", str(ctx.exception))

    def test_non_naive_rejects_llmlingua2_backend(self):
        with self.assertRaises(ValueError) as ctx:
            _build_memory_model(_ns(
                memory_strategy="llm",
                summarizer_backend="llmlingua2",
            ))
        self.assertIn("llmlingua2", str(ctx.exception))

    def test_naive_is_default(self):
        """No `memory_strategy` on the namespace → falls back to naive."""
        args = _ns()
        del args.memory_strategy
        mm = _build_memory_model(args)
        self.assertIsInstance(mm, NaiveSummarizationMemory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
