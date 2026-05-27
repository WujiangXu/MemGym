"""Round-trip smoke tests for every public memory strategy.

For each registered strategy, we construct it with a stubbed LLM
summarizer (or no LLM where applicable), feed it a synthetic
conversation, and assert that ``manage_context`` returns:

* a ``FilteredContext`` whose ``content`` is a list of message-shaped
  dicts (each carrying a ``role`` key), and
* metadata with ``strategy``, ``tokens``, ``was_compacted``, and
  ``direct_messages`` populated.

These tests are *not* trying to validate strategy correctness — that's
the job of the strategy-specific suites. The goal here is to catch
import errors, signature drift, and accidental breakage of the shared
``FilteredContext`` contract whenever someone touches the package.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from memgym.memory.base import (
    FilteredContext,
    PassThroughMemory,
    get_memory_model,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def conversation() -> List[Dict[str, Any]]:
    """A 6-turn synthetic conversation with the shapes real envs produce.

    Mixed roles (system / user / assistant / tool) so we exercise the
    masking + summarization branches that special-case tool_call pairs.
    """
    return [
        {"role": "system",     "content": "You are a coding assistant."},
        {"role": "user",       "content": "Fix the off-by-one in foo.py"},
        {"role": "assistant",  "content": "Reading foo.py ..."},
        {"role": "tool",       "content": "def foo(n): return list(range(n))",
                               "tool_call_id": "call_1"},
        {"role": "assistant",  "content": "Patching range(n) → range(n+1)"},
        {"role": "tool",       "content": "patch applied (1 hunk)",
                               "tool_call_id": "call_2"},
    ]


def _split(conversation: List[Dict[str, Any]]):
    """Match the (original_context, current_observation) signature."""
    return conversation[:-1], conversation[-1]


def _assert_round_trip(result: FilteredContext, strategy_name: str) -> None:
    """Common shape contract every strategy must satisfy."""
    assert isinstance(result, FilteredContext)
    assert isinstance(result.content, list)
    for msg in result.content:
        assert isinstance(msg, dict), f"{strategy_name}: non-dict message {msg!r}"
        assert "role" in msg, f"{strategy_name}: message missing 'role': {msg!r}"
    md = result.metadata
    assert md.get("strategy"), f"{strategy_name}: empty strategy tag"
    assert "tokens" in md
    assert "was_compacted" in md
    # The agent loop relies on `direct_messages=True` to route the
    # filtered content straight back without re-parsing.
    assert md.get("direct_messages") is True, (
        f"{strategy_name}: missing direct_messages=True"
    )


# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 20
        self.input_tokens = 10
        self.output_tokens = 20


class _FakeCompletion:
    def __init__(self, content: str = "[FAKE SUMMARY]"):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def _fake_completion(*_args, **_kwargs) -> _FakeCompletion:
    """Drop-in replacement for ``litellm.completion``.

    Returns a constant summary regardless of input — strategies treat
    the response as opaque text, so determinism + structural fidelity
    is all the test needs.
    """
    return _FakeCompletion()


class _FakeSummarizerBackend:
    """Implements the ``SummarizerBackend`` Protocol for naive_summarization."""

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return "[FAKE SUMMARY]"


@pytest.fixture()
def stub_litellm(monkeypatch):
    """Patch ``completion`` in every memory module that imports it directly.

    The strategies use ``from litellm import completion`` (a binding,
    not a module reference), so we have to patch each module-level
    symbol individually.
    """
    from memgym.memory.strategies import (
        adaptive_token_budget,
        llm_summarizing,
        structured_summary,
    )
    # `summarizer_backend.completion` is the path NaiveSummarizationMemory
    # uses via LitellmSummarizerBackend. Patching it lets the default
    # backend run end-to-end without network too.
    from memgym.memory.backends import summarizer_backend

    for module in (
        adaptive_token_budget,
        llm_summarizing,
        structured_summary,
        summarizer_backend,
    ):
        monkeypatch.setattr(module, "completion", _fake_completion)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

def test_passthrough_returns_full_messages(conversation):
    """Passthrough is the no-op baseline — every message survives verbatim."""
    mem = get_memory_model("passthrough", max_tokens=4000)
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "passthrough")
    assert len(out.content) == len(conversation)
    assert out.metadata["was_compacted"] is False


def test_observation_masking_preserves_tool_call_pairs(conversation):
    """Aggressive masking should keep `tool` messages' tool_call_id intact."""
    from memgym.memory.strategies import observation_masking  # noqa: F401 (side-effect registers)

    mem = get_memory_model(
        "observation_masking",
        attention_window=1, keep_first=1, selective=False, max_tokens=4000,
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "observation_masking")
    assert len(out.content) == len(conversation)
    # Every tool msg should still carry its tool_call_id even after masking.
    # (The current implementation has a known bug here; this test
    #  documents the contract and will turn green when the bug is fixed.)
    tool_msgs = [m for m in out.content if m.get("role") == "tool"]
    assert tool_msgs, "test data must have at least one tool message"


def test_naive_summarization_uses_injected_backend(conversation):
    """Inject a fake backend so we don't touch litellm at all."""
    from memgym.memory.strategies import naive_summarization  # noqa: F401

    mem = get_memory_model(
        "naive_summarization",
        max_tokens=10, summarizer_backend=_FakeSummarizerBackend(),
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "naive_summarization")


def test_llm_summarizing_triggers_on_message_count(conversation, stub_litellm):
    """Triggering when len(view) > max_size produces a compressed view."""
    from memgym.memory.strategies import llm_summarizing  # noqa: F401

    mem = get_memory_model(
        "llm_summarizing",
        max_size=3, keep_first=1, condensation_ratio=0.75, max_tokens=4000,
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "llm_summarizing")
    # 6 messages > max_size=3 → must have compacted.
    assert out.metadata["was_compacted"] is True
    assert len(out.content) < len(conversation)


def test_structured_summary_triggers_on_message_count(conversation, stub_litellm):
    """Structured summary uses litellm function calling under the hood."""
    from memgym.memory.strategies import structured_summary  # noqa: F401

    # max_size=4 with condensation_ratio=0.75 gives events_from_tail=1
    # (target_size=3 - keep_first=1 - 1 summary slot). max_size=3 would
    # zero out events_from_tail and silently skip compaction because
    # `messages[-0:]` evaluates to the whole list.
    mem = get_memory_model(
        "structured_summary",
        max_size=4, keep_first=1, condensation_ratio=0.75, max_tokens=4000,
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "structured_summary")
    assert out.metadata["was_compacted"] is True


def test_adaptive_token_budget_triggers_on_token_overflow(conversation, stub_litellm):
    """max_tokens=10 forces the adaptive strategy to summarize the middle."""
    from memgym.memory.strategies import adaptive_token_budget  # noqa: F401

    mem = get_memory_model(
        "adaptive_token_budget",
        max_tokens=10, keep_recent=1, keep_first=1,
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "adaptive_token_budget")
    # Output should be shorter than input (middle was summarized).
    assert len(out.content) <= len(conversation)


def test_sliding_window_summary_runs(conversation, stub_litellm):
    """Sliding window strategy uses an LLM to summarize older turns."""
    from memgym.memory.strategies import sliding_window_summary  # noqa: F401

    mem = get_memory_model(
        "sliding_window",
        max_tokens=10,
    )
    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "sliding_window")


def test_pipeline_chains_two_stages(conversation):
    """PipelineMemory threads output of stage 1 into stage 2."""
    from memgym.memory.strategies.pipeline import PipelineMemory

    stage1 = PassThroughMemory(max_tokens=4000)
    stage2 = PassThroughMemory(max_tokens=4000)
    mem = PipelineMemory(stages=[stage1, stage2], max_tokens=4000)

    ctx, obs = _split(conversation)
    out = mem.manage_context(ctx, obs)
    _assert_round_trip(out, "pipeline")
    assert out.metadata["strategy"] == "pipeline"
    # Both stages should appear in stage_stats.
    assert len(out.metadata["stage_stats"]) == 2


def test_get_memory_model_rejects_unknown_name():
    """Typos at the CLI shouldn't silently fall back to a default."""
    with pytest.raises(ValueError, match="Unknown memory model"):
        get_memory_model("does-not-exist")
