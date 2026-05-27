"""Regression tests for known open or recently-fixed contract bugs.

Each test pins a specific behavioural contract the codebase is *supposed*
to honour. Tests are marked ``xfail(strict=False)`` where we know the
fix has not landed yet — that way:

* if the bug is still present, the test reports XFAIL and CI stays
  green;
* if someone fixes the bug, the test reports XPASS and a maintainer
  can drop the ``xfail`` marker.

Each test docstring documents:
1. the offending file/lines,
2. the user-visible symptom,
3. the contract the fix should restore.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Bug 1 — observation_masking drops tool_call_id when masking
# ---------------------------------------------------------------------------

def test_observation_masking_preserves_tool_call_id():
    """Masked ``role='tool'`` messages must keep ``tool_call_id``.

    File: src/memgym/memory/observation_masking.py
    Symptom: Bedrock/OpenAI tool-call validation rejects the request
        because the assistant produced a tool_calls entry whose
        matching tool message was rewritten to ``role='user'`` (and
        thus stripped of its tool_call_id) during masking.
    Contract: After masking, every ``tool`` message in the output
        carries the same ``tool_call_id`` it had on input and keeps
        ``role='tool'``.
    """
    from memgym.memory.observation_masking import ObservationMaskingMemory

    mem = ObservationMaskingMemory(
        attention_window=1, keep_first=1, selective=False, max_tokens=4000,
    )
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_x", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "long output", "tool_call_id": "call_x"},
        {"role": "assistant", "content": "done"},
    ]
    out = mem.manage_context(convo[:-1], convo[-1])
    tool_msgs = [m for m in out.content if m.get("role") == "tool"]
    assert tool_msgs, "lost the tool message entirely"
    for m in tool_msgs:
        assert m["role"] == "tool", "tool message was rewritten to a different role"
        assert m.get("tool_call_id") == "call_x", "tool_call_id was dropped during masking"


# ---------------------------------------------------------------------------
# Bug 2 — naive_summarization must keep system prompt across compactions
# ---------------------------------------------------------------------------

def test_naive_summarization_preserves_system_prompt():
    """``NaiveSummarizationMemory(keep_first=1)`` must keep the system
    prompt intact across compactions.

    File: src/memgym/memory/naive_summarization.py
    Symptom: With no keep_first, the system prompt was packed into the
        ``forgotten`` block, summarised, and lost — the agent silently
        switched to no-system-prompt behaviour.
    Contract: With keep_first >= 1, ``out.content[0]`` is the original
        system message verbatim, even when summarization fires.
    """
    from memgym.memory.naive_summarization import NaiveSummarizationMemory

    class _StubBackend:
        def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:
            return "[SUMMARY OF EARLIER]"

    mem = NaiveSummarizationMemory(
        max_tokens=10, keep_first=1, summarizer_backend=_StubBackend(),
    )
    system_msg = {"role": "system", "content": "You are a careful assistant."}
    convo = [system_msg] + [
        {"role": "user", "content": f"turn {i}"} for i in range(8)
    ]
    out = mem.manage_context(convo[:-1], convo[-1])
    assert out.content[0] == system_msg, (
        "system prompt was rewritten or removed during summarization"
    )


# ---------------------------------------------------------------------------
# Bug 3 — sliding_window_summary must emit summary as role='user'
# ---------------------------------------------------------------------------

def test_sliding_window_summary_emits_user_role():
    """The injected summary message must be ``role='user'``.

    File: src/memgym/memory/sliding_window_summary.py:214
    Symptom: A second system-role message in mid-conversation confuses
        chat-template renderers (especially Qwen + tau2's pydantic
        message validator), which expect at most one ``system`` turn at
        the head.
    Contract: After the strategy compacts, any summary message it
        inserts uses ``role='user'`` (never ``'system'``).
    """
    from memgym.memory.sliding_window_summary import (
        SUMMARY_MARKER, SlidingWindowSummaryMemory,
    )
    from memgym.memory import sliding_window_summary as sws_mod

    def _fake_completion(*_a, **_kw):
        class _M: content = "[FAKE]"
        class _C: message = _M()
        class _R:
            choices = [_C()]
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
        return _R()

    # Avoid network — patch the module-level binding.
    sws_mod.completion = _fake_completion

    mem = SlidingWindowSummaryMemory(max_tokens=10)
    convo = [{"role": "user", "content": f"turn {i}"} for i in range(15)]
    out = mem.manage_context(convo[:-1], convo[-1])
    summary_msgs = [
        m for m in out.content
        if isinstance(m, dict) and SUMMARY_MARKER in str(m.get("content", ""))
    ]
    if summary_msgs:    # only assert when a summary actually fired
        for m in summary_msgs:
            assert m["role"] == "user", (
                f"summary message has role={m['role']!r}, expected 'user'"
            )


# ---------------------------------------------------------------------------
# Bug 4 — swe_env config fallback must use warnings.warn, not print
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="env.py:677 still uses print() — fix should switch to warnings.warn so the silent fallback is visible to pytest's -W flag",
)
def test_swe_env_config_fallback_uses_warnings_module(monkeypatch):
    """Fallback to empty env config must emit a Python warning.

    File: src/memgym/gym/swe_bench/env.py:677
    Symptom: When mini-swe-agent ships without ``benchmarks/swebench.yaml``,
        the env silently falls back to ``{}``. Eval ran with text-based
        prompts incompatible with LitellmModel, costing 56% of instances
        a "No tool calls found" loop. ``print()`` does not flow through
        pytest's warning capture or production log aggregation.
    Contract: The fallback path calls ``warnings.warn(...)`` so callers
        can surface the misconfiguration.
    """
    # Build a stub SWEMemoryEnv with just enough state to invoke the
    # helper. The helper reads ``minisweagent.config.builtin_config_dir``;
    # point it at an empty dir so the search misses and we hit the
    # fallback branch.
    import sys
    import types
    fake_minisweagent = types.ModuleType("minisweagent")
    fake_config_mod = types.ModuleType("minisweagent.config")
    fake_config_mod.builtin_config_dir = Path("/nonexistent/minisweagent/config")
    fake_minisweagent.config = fake_config_mod
    monkeypatch.setitem(sys.modules, "minisweagent", fake_minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.config", fake_config_mod)

    from memgym.gym.swe_bench.env import SWEMemoryEnv
    env = SWEMemoryEnv.__new__(SWEMemoryEnv)    # bypass __init__

    with pytest.warns(Warning):
        cfg = env._get_env_config()
    assert cfg == {}


# ---------------------------------------------------------------------------
# Bug 5 — tracker.context_length should reflect tokens (or be renamed)
# ---------------------------------------------------------------------------

def test_tracker_context_length_field_is_documented_as_chars():
    """``tracker.context_length`` is char count by intent — make sure it
    stays consistent across both call sites.

    File: src/memgym/gym/swe_bench/tracker.py:49,80
    Symptom: An earlier internal version computed
        ``compression_ratio = len(str(...))`` on one side and tokens on
        the other, leading to ratios of 4× over reality. The shipped
        version keeps both call sites consistently as ``len(str())``.
    Contract: Both call sites compute the same shape so a downstream
        comparison is apples-to-apples.
    """
    tracker_path = Path(__file__).resolve().parents[2] / (
        "src/memgym/gym/swe_bench/tracker.py"
    )
    text = tracker_path.read_text()
    # Both context_length call sites should use the same shape — either
    # both len(str(...)) or both an actual tokenizer call.
    occurrences = text.count("len(str(context)) if context else 0")
    assert occurrences >= 2, (
        f"tracker.py context_length shape changed at one call site but "
        f"not the other (found {occurrences} canonical occurrences). "
        f"Both sites must agree."
    )
