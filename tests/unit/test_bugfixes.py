"""
Targeted tests for the 5 bug fixes in the SWE-bench + Memory integration.

All tests are pure-Python (no Docker, no API keys, no LLM calls).
Run with: python3 -m pytest test/test_bugfixes.py -v
"""

import sys
import os
import copy

import pytest

# Add src to path

# ============================================================================
# BUG 1: Observation masking must NOT break tool message pairing
# ============================================================================

class TestObservationMaskingToolPairing:
    """
    BUG: Old code converted masked tool messages to role='user' and deleted
    tool_call_id, breaking the API requirement that every assistant tool_calls
    entry has a matching tool response with the same tool_call_id.

    FIX: Keep role='tool' and tool_call_id intact, only mask content.
    """

    def _make_tool_conversation(self, n_exchanges=10):
        """Build a realistic mini-swe-agent conversation with tool_calls."""
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
        ]
        for i in range(n_exchanges):
            tool_call_id = f"call_{i:04d}"
            # Assistant message with tool_calls
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "echo step {i}"}}'}
                }]
            })
            # Matching tool response
            msgs.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Output of step {i}: " + "x" * 200,
            })
        return msgs

    def test_masked_tool_messages_keep_role_and_id(self):
        """After masking, tool messages must retain role='tool' and tool_call_id."""
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory

        mem = ObservationMaskingMemory(attention_window=4, keep_first=1)
        msgs = self._make_tool_conversation(n_exchanges=10)

        # Split into history + current_obs
        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        for msg in output:
            if not isinstance(msg, dict):
                continue
            if msg.get("content") == "<MASKED>" and "tool_call_id" not in msg:
                # This was a masked user message (not tool), that's fine
                continue
            if msg.get("role") == "tool":
                # Tool messages must ALWAYS have tool_call_id
                assert "tool_call_id" in msg, (
                    f"Tool message lost tool_call_id after masking: {msg}"
                )

    def test_tool_call_pairing_preserved(self):
        """Every assistant tool_calls[].id must have a matching tool response."""
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory

        mem = ObservationMaskingMemory(attention_window=4, keep_first=1)
        msgs = self._make_tool_conversation(n_exchanges=10)

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        # Collect all tool_call IDs from assistant messages
        assistant_call_ids = set()
        for msg in output:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    assistant_call_ids.add(tc["id"])

        # Collect all tool_call_ids from tool responses
        tool_response_ids = set()
        for msg in output:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tool_response_ids.add(msg["tool_call_id"])

        # Every assistant tool_call must have a matching tool response
        orphaned = assistant_call_ids - tool_response_ids
        assert not orphaned, (
            f"Orphaned tool_call IDs (no matching tool response): {orphaned}"
        )

    def test_no_role_conversion_to_user(self):
        """Masked tool messages must NOT be converted to role='user'."""
        from memgym.memory.strategies.observation_masking import ObservationMaskingMemory

        mem = ObservationMaskingMemory(attention_window=2, keep_first=1)
        msgs = self._make_tool_conversation(n_exchanges=6)

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        # Count roles — there should be tool messages in the output
        roles = [m.get("role") for m in output if isinstance(m, dict)]
        assert "tool" in roles, (
            f"No tool messages in output — were they converted to user? Roles: {roles}"
        )

        # Check masked tool messages specifically
        for msg in output:
            if isinstance(msg, dict) and msg.get("content") == "<MASKED>":
                if "tool_call_id" in msg:
                    assert msg["role"] == "tool", (
                        f"Masked tool message had role changed to '{msg['role']}' "
                        f"instead of keeping 'tool'"
                    )

# ============================================================================
# BUG 2: Compression ratio must use token count, not char count
# ============================================================================

class TestCompressionRatioTokens:
    """
    BUG: Compression stats used len(str(content)) (character count) instead
    of actual token count via memory_manager.count_tokens().

    FIX: Use self.memory_manager.count_tokens() for accurate ratios.
    """

    def test_wrapper_calls_count_tokens_not_len(self):
        """Verify the source code uses count_tokens(), not len(str(...))."""
        import inspect

        pytest.importorskip("minisweagent")
        from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent

        source = inspect.getsource(MemoryAwareSWEAgent.query)

        # Old buggy pattern should NOT be present
        assert 'len(str(m.get("content"' not in source, (
            "BUG STILL PRESENT: query() still uses len(str()) for compression ratio"
        )

        # New correct pattern should be present
        assert "self.memory_manager.count_tokens" in source, (
            "FIX MISSING: query() should use self.memory_manager.count_tokens()"
        )

# ============================================================================
# BUG 3: NaiveSummarization must protect system prompt (keep_first)
# ============================================================================

class TestNaiveSummarizationKeepFirst:
    """
    BUG: When summarizing, returned [summary_msg] + tail — the system
    prompt (messages[0]) was lost.

    FIX: Add keep_first parameter, output = head + [summary_msg] + tail.
    """

    def test_keep_first_parameter_exists(self):
        """NaiveSummarizationMemory must accept keep_first parameter."""
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory

        mem = NaiveSummarizationMemory(keep_first=2, max_tokens=100)
        assert mem.keep_first == 2

    def test_system_prompt_preserved_after_summarization(self):
        """System prompt must survive summarization (without calling LLM)."""
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory

        mem = NaiveSummarizationMemory(
            max_tokens=100,
            keep_first=1,
            keep_recent=2,
        )

        # Manually inject a summary to skip actual LLM call
        mem._summary = "Previous work: explored files, found bug in parser."

        system_msg = {"role": "system", "content": "You are a coding assistant."}
        msgs = [system_msg] + [
            {"role": "user", "content": f"observation {i}: " + "x" * 50}
            for i in range(10)
        ]

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        # First message must be the system prompt
        assert output[0]["role"] == "system", (
            f"System prompt lost! First message role is '{output[0]['role']}' "
            f"instead of 'system'"
        )
        assert output[0]["content"] == "You are a coding assistant.", (
            f"System prompt content changed: {output[0]['content']}"
        )

        # Second message should be the summary
        assert "[Summary]" in output[1]["content"], (
            f"Summary not in position 1: {output[1]}"
        )

    def test_keep_first_2_preserves_two_messages(self):
        """keep_first=2 should preserve both system prompt and task message."""
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory

        mem = NaiveSummarizationMemory(
            max_tokens=100,
            keep_first=2,
            keep_recent=2,
        )
        mem._summary = "Summary of work done."

        system_msg = {"role": "system", "content": "System prompt"}
        task_msg = {"role": "user", "content": "Fix the bug in parser.py"}
        msgs = [system_msg, task_msg] + [
            {"role": "user", "content": f"step {i}: " + "x" * 50}
            for i in range(10)
        ]

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        # First two messages preserved
        assert output[0]["content"] == "System prompt"
        assert output[1]["content"] == "Fix the bug in parser.py"
        # Then summary
        assert "[Summary]" in output[2]["content"]

    def test_keep_first_in_stats(self):
        """Stats should report keep_first value."""
        from memgym.memory.strategies.naive_summarization import NaiveSummarizationMemory

        mem = NaiveSummarizationMemory(keep_first=3)
        stats = mem.get_stats()
        assert "keep_first" in stats
        assert stats["keep_first"] == 3

# ============================================================================
# BUG 4: SlidingWindowSummary must use role='user', not role='system'
# ============================================================================

class TestSlidingWindowSummaryRole:
    """
    BUG: Summary message inserted as role='system' but system messages
    should only be at position 0. All other strategies use role='user'.

    FIX: Change to role='user'.
    """

    def test_summary_role_is_user(self):
        """Summary message must use role='user', not role='system'."""
        from memgym.memory.strategies.sliding_window_summary import SlidingWindowSummaryMemory

        mem = SlidingWindowSummaryMemory(
            window_size=4,
            keep_first=1,
        )

        # Manually set summary to skip LLM call
        mem._summary = "Previous work summary."
        mem._summary_frontier = 5  # Pretend we've already absorbed 5 messages

        system_msg = {"role": "system", "content": "System prompt"}
        msgs = [system_msg] + [
            {"role": "user", "content": f"step {i}: data"}
            for i in range(10)
        ]

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        # Find the summary message
        summary_msgs = [m for m in output if isinstance(m, dict)
                        and "[Summary]" in str(m.get("content", ""))]
        assert len(summary_msgs) == 1, f"Expected 1 summary, got {len(summary_msgs)}"

        summary = summary_msgs[0]
        assert summary["role"] == "user", (
            f"Summary role is '{summary['role']}', expected 'user'. "
            f"System messages should only be at position 0."
        )

    def test_only_one_system_message(self):
        """After summarization, there should be exactly one system message (index 0)."""
        from memgym.memory.strategies.sliding_window_summary import SlidingWindowSummaryMemory

        mem = SlidingWindowSummaryMemory(window_size=3, keep_first=1)
        mem._summary = "Work done so far."
        mem._summary_frontier = 3

        system_msg = {"role": "system", "content": "System prompt"}
        msgs = [system_msg] + [
            {"role": "user", "content": f"obs {i}"}
            for i in range(8)
        ]

        result = mem.manage_context(msgs[:-1], msgs[-1])
        output = result.content

        system_msgs = [m for m in output if isinstance(m, dict)
                       and m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message, got {len(system_msgs)}: {system_msgs}"
        )

# ============================================================================
# BUG 5: Config fallback must warn when using default.yaml
# ============================================================================

class TestConfigFallbackWarning:
    """
    BUG: When swebench.yaml not found, silently falls back to default.yaml
    which has incompatible prompts. No logging or warning.

    FIX: Print WARNING when falling back.
    """

    def test_get_mini_swe_config_warns_on_fallback(self):
        """Verify the source code includes a warning on fallback."""
        import inspect
        # We can't import swe_env directly (depends on heavy packages),
        # so read the source file instead
        swe_env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memgym", "gym", "swe_bench", "env.py"
        )
        with open(swe_env_path) as f:
            source = f.read()

        # The fallback to default.yaml should have a WARNING
        # Find the _get_mini_swe_config method
        idx = source.find("def _get_mini_swe_config")
        assert idx != -1, "_get_mini_swe_config method not found"
        method_source = source[idx:idx + 1500]

        assert "warnings.warn" in method_source, (
            "_get_mini_swe_config silently falls back to default.yaml "
            "(expected warnings.warn(...))"
        )
        assert "default.yaml" in method_source, (
            "_get_mini_swe_config does not mention default.yaml in warning"
        )

    def test_get_env_config_warns_when_not_found(self):
        """_get_env_config should warn when swebench.yaml not found.

        Source-grep check that a warning IS issued on the fallback path.
        The companion behavioral test
        `test_known_bugs.py::test_swe_env_config_fallback_uses_warnings_module`
        asserts the *correct* mechanism (`warnings.warn`, surfacing through
        pytest's warning capture); this one just guarantees no silent
        regression to `return {}` with nothing logged.
        """
        swe_env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memgym", "gym", "swe_bench", "env.py"
        )
        with open(swe_env_path) as f:
            source = f.read()

        idx = source.find("def _get_env_config")
        assert idx != -1
        method_end = source.find("\n    def ", idx + 1)
        method_source = source[idx:method_end]

        assert "warnings.warn" in method_source, (
            "_get_env_config silently falls back when swebench.yaml not found "
            "(expected warnings.warn(...))"
        )

    def test_get_model_config_warns_when_not_found(self):
        """_get_model_config should warn when swebench.yaml not found."""
        swe_env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "memgym", "gym", "swe_bench", "env.py"
        )
        with open(swe_env_path) as f:
            source = f.read()

        idx = source.find("def _get_model_config")
        assert idx != -1
        method_end = source.find("\n    def ", idx + 1)
        method_source = source[idx:method_end]

        assert "warnings.warn" in method_source, (
            "_get_model_config silently falls back when swebench.yaml not found "
            "(expected warnings.warn(...))"
        )

# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
