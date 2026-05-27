"""Tests for the shared HippoRAG core and its two adapters.

V2 gate from the plan in `when-we-run-the-pure-locket.md`.

Locks down the contract the smoke runs revealed and we then patched:
1. `openai/<model>` litellm prefix is stripped before HippoRAG sees it
   (HippoRAG uses `openai.OpenAI` directly, not litellm).
2. An explicit `api_key=...` overrides a stale `OPENAI_API_KEY` env var
   (we deliberately moved away from `setdefault`).
3. `add_doc` buffers and does NOT call HippoRAG; `retrieve()` triggers
   exactly one `index()` call regardless of how many docs were added.
4. HippoRAG's `graph_search_with_fact_entities` AssertionError must NOT
   abort the eval — `retrieve()` returns `[]` instead.
5. The coding adapter and the IR adapter both share the same core
   instance contract (same model name handling, same env behavior).

These tests stub the upstream `HippoRAG` class with a fake so we never
make real LLM calls — pure logic checks.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_fake_hipporag(monkey: dict, retrieve_side_effect=None,
                            retrieved_docs=None):
    """Install a fake `hipporag` module so HippoRAGSystem._create_system
    returns a controllable mock without touching the real library.

    Returns the mock class; callers can inspect its instances.
    """
    fake_module = types.ModuleType("hipporag")

    fake_instance = MagicMock(name="HippoRAGInstance")
    if retrieve_side_effect is not None:
        fake_instance.retrieve.side_effect = retrieve_side_effect
    else:
        sol = MagicMock()
        sol.docs = retrieved_docs or []
        fake_instance.retrieve.return_value = [sol]

    def _fake_ctor(**kwargs):
        fake_instance.last_init_kwargs = kwargs
        return fake_instance

    fake_module.HippoRAG = MagicMock(side_effect=_fake_ctor)
    monkey["hipporag"] = sys.modules.get("hipporag")
    sys.modules["hipporag"] = fake_module
    return fake_module.HippoRAG, fake_instance


def _restore_fake_hipporag(monkey: dict):
    prior = monkey.get("hipporag")
    if prior is None:
        sys.modules.pop("hipporag", None)
    else:
        sys.modules["hipporag"] = prior


class TestHippoRAGSystemInit(unittest.TestCase):
    """The constructor's contract: prefix strip + env-var override."""

    def setUp(self) -> None:
        self._prior_key = os.environ.get("OPENAI_API_KEY")
        self._prior_base = os.environ.get("OPENAI_API_BASE")
        self._prior_url = os.environ.get("OPENAI_BASE_URL")

    def tearDown(self) -> None:
        for k, v in (
            ("OPENAI_API_KEY", self._prior_key),
            ("OPENAI_API_BASE", self._prior_base),
            ("OPENAI_BASE_URL", self._prior_url),
        ):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_strips_openai_prefix(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        sys_ = HippoRAGSystem(llm_model="openai/gpt-4o-mini")
        self.assertEqual(sys_._llm_model, "gpt-4o-mini")

    def test_leaves_unprefixed_model_alone(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        sys_ = HippoRAGSystem(llm_model="gpt-4o-mini")
        self.assertEqual(sys_._llm_model, "gpt-4o-mini")

    def test_does_not_strip_other_prefixes(self):
        """Only the litellm-style `openai/` prefix is special; bedrock/,
        anthropic/, etc. shouldn't be touched (they would be a config
        error here anyway, but we want to fail loudly downstream)."""
        from memgym.memory.external.hipporag import HippoRAGSystem
        sys_ = HippoRAGSystem(llm_model="bedrock/claude-haiku")
        self.assertEqual(sys_._llm_model, "bedrock/claude-haiku")

    def test_explicit_api_key_overrides_stale_env(self):
        """The earlier `setdefault` bug let a stale env key shadow an
        explicit ctor arg. After the fix, the ctor arg must win."""
        os.environ["OPENAI_API_KEY"] = "sk-stale-rotten-key"
        from memgym.memory.external.hipporag import HippoRAGSystem
        HippoRAGSystem(llm_model="gpt-4o-mini", api_key="sk-fresh-key")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-fresh-key")

    def test_no_api_key_arg_leaves_env_alone(self):
        """If the caller doesn't pass api_key, we shouldn't clobber a
        legitimately-pre-existing env var (the smoke harness relies on
        this)."""
        os.environ["OPENAI_API_KEY"] = "sk-from-shell"
        from memgym.memory.external.hipporag import HippoRAGSystem
        HippoRAGSystem(llm_model="gpt-4o-mini")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-shell")

    def test_explicit_api_base_overrides_env(self):
        os.environ["OPENAI_API_BASE"] = "https://stale.example/v1"
        from memgym.memory.external.hipporag import HippoRAGSystem
        HippoRAGSystem(
            llm_model="gpt-4o-mini",
            api_base="https://fresh.example/v1",
        )
        self.assertEqual(
            os.environ["OPENAI_API_BASE"], "https://fresh.example/v1",
        )
        self.assertEqual(
            os.environ["OPENAI_BASE_URL"], "https://fresh.example/v1",
        )


class TestHippoRAGSystemBuffering(unittest.TestCase):
    """`add_doc` buffers; `commit`/`retrieve` does the work."""

    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_hipporag(self._monkey)

    def test_add_doc_does_not_create_system(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        ctor, _ = _install_fake_hipporag(self._monkey)
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "hello world")
        s.add_doc("doc-2", "another doc")
        ctor.assert_not_called()
        self.assertEqual(len(s._buffered_docs), 2)
        self.assertFalse(s._committed)

    def test_empty_text_is_skipped(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        _install_fake_hipporag(self._monkey)
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "")
        s.add_doc("doc-2", "   \n  ")
        self.assertEqual(s._buffered_docs, [])

    def test_retrieve_with_no_docs_short_circuits(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        ctor, _ = _install_fake_hipporag(self._monkey)
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        result = s.retrieve("any question")
        self.assertEqual(result, [])
        ctor.assert_not_called()

    def test_retrieve_lazy_commits_exactly_once(self):
        """3 add_doc calls + 2 retrieves should produce exactly 1 index()
        call. This is the core of the Phase-5 cost story."""
        from memgym.memory.external.hipporag import HippoRAGSystem
        _, fake_instance = _install_fake_hipporag(
            self._monkey,
            retrieved_docs=["doc-A", "doc-B"],
        )
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        s.add_doc("doc-2", "beta")
        s.add_doc("doc-3", "gamma")
        passages_1 = s.retrieve("question 1")
        passages_2 = s.retrieve("question 2")
        self.assertEqual(passages_1, ["doc-A", "doc-B"])
        self.assertEqual(passages_2, ["doc-A", "doc-B"])
        self.assertEqual(fake_instance.index.call_count, 1)

    def test_add_doc_after_commit_re_indexes_on_next_retrieve(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        _, fake_instance = _install_fake_hipporag(
            self._monkey, retrieved_docs=["x"],
        )
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        s.retrieve("q1")  # commit #1
        s.add_doc("doc-2", "beta")
        s.retrieve("q2")  # commit #2 (new doc invalidated commit)
        self.assertEqual(fake_instance.index.call_count, 2)


class TestHippoRAGSystemDefensiveRetrieve(unittest.TestCase):
    """The brittle-upstream guard: HippoRAG.retrieve raising != crash."""

    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_hipporag(self._monkey)

    def _system_with_retrieve_error(self, exc: BaseException):
        from memgym.memory.external.hipporag import HippoRAGSystem
        _install_fake_hipporag(
            self._monkey, retrieve_side_effect=exc,
        )
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        return s

    def test_assertion_error_returns_empty(self):
        """The exact failure mode caught in the smoke run."""
        s = self._system_with_retrieve_error(
            AssertionError("No phrases found in graph"),
        )
        self.assertEqual(s.retrieve("oddball question"), [])

    def test_runtime_error_returns_empty(self):
        s = self._system_with_retrieve_error(
            RuntimeError("CUDA OOM during PPR"),
        )
        self.assertEqual(s.retrieve("q"), [])

    def test_value_error_returns_empty(self):
        s = self._system_with_retrieve_error(
            ValueError("bad embedding shape"),
        )
        self.assertEqual(s.retrieve("q"), [])

    def test_zero_division_returns_empty(self):
        """Reproduces the very first failure mode (auth-fail → 0 phrases
        → ZeroDivisionError in HippoRAG's own avg-stats code)."""
        s = self._system_with_retrieve_error(ZeroDivisionError())
        self.assertEqual(s.retrieve("q"), [])

    def test_unexpected_error_propagates(self):
        """We deliberately don't swallow `Exception` — surprising upstream
        bugs (TypeError on an API change, KeyError on a missing config)
        should still surface so we notice during V4."""
        s = self._system_with_retrieve_error(TypeError("API changed"))
        with self.assertRaises(TypeError):
            s.retrieve("q")


class TestHippoRAGSystemReset(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_hipporag(self._monkey)

    def test_reset_drops_buffer_and_system(self):
        from memgym.memory.external.hipporag import HippoRAGSystem
        _install_fake_hipporag(self._monkey, retrieved_docs=["x"])
        s = HippoRAGSystem(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        s.retrieve("q")
        s.reset()
        self.assertEqual(s._buffered_docs, [])
        self.assertFalse(s._committed)
        self.assertIsNone(s._system)


class TestHippoRAGCodingAdapter(unittest.TestCase):
    """`HippoRAGMethod` — coding_synthetic CodingMemoryMethod contract."""

    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_hipporag(self._monkey)

    def test_ingest_truncates_to_max_chars(self):
        from memgym.pipelines.coding_synthetic.memory_methods.hipporag_method import (
            HippoRAGMethod,
        )
        _install_fake_hipporag(self._monkey)
        m = HippoRAGMethod(max_ingest_chars=100)
        big = "x" * 5000
        m.ingest("doc-1", big, "task")
        self.assertEqual(len(m._system._buffered_docs), 1)
        # buffered_docs prefix + truncated payload
        self.assertLessEqual(len(m._system._buffered_docs[0]), 100 + 50)

    def test_retrieve_returns_no_memories_string_when_empty(self):
        """When the underlying core returns [], the adapter must return
        a non-empty placeholder string — the answerer prompt template
        crashes on empty notes."""
        from memgym.pipelines.coding_synthetic.memory_methods.hipporag_method import (
            HippoRAGMethod,
        )
        _install_fake_hipporag(self._monkey, retrieved_docs=[])
        m = HippoRAGMethod()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("no relevant memories", result)

    def test_retrieve_joins_passages_with_separator(self):
        from memgym.pipelines.coding_synthetic.memory_methods.hipporag_method import (
            HippoRAGMethod,
        )
        _install_fake_hipporag(
            self._monkey, retrieved_docs=["passage A", "passage B"],
        )
        m = HippoRAGMethod()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("passage A", result)
        self.assertIn("passage B", result)
        self.assertIn("\n---\n", result)


class TestHippoRAGIRAdapter(unittest.TestCase):
    """`IRHippoRAGMemory` — IR BaseMemoryManager contract."""

    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_hipporag(self._monkey)

    def test_registered_in_ir_factory(self):
        """Module-level `register_memory_model(...)` ran on import."""
        # Trigger registration if it hasn't happened yet
        import memgym.memory.ir.ir_hipporag  # noqa: F401
        from memgym.memory.base import get_memory_model
        # get_memory_model raises on unknown names — the assertion is that
        # we don't raise here.
        m = get_memory_model("ir_hipporag", question="test")
        self.assertEqual(m._turn_id, 0)

    def test_manage_context_buffers_observations(self):
        from memgym.memory.ir.ir_hipporag import IRHippoRAGMemory
        _install_fake_hipporag(self._monkey)
        m = IRHippoRAGMemory(question="q")
        m.manage_context([], "first observation")
        m.manage_context([], "second observation")
        self.assertEqual(m._turn_id, 2)
        self.assertEqual(len(m._all_observations), 2)
        self.assertEqual(len(m._system._buffered_docs), 2)

    def test_empty_observation_is_ignored(self):
        from memgym.memory.ir.ir_hipporag import IRHippoRAGMemory
        _install_fake_hipporag(self._monkey)
        m = IRHippoRAGMemory(question="q")
        m.manage_context([], None)
        m.manage_context([], "")
        self.assertEqual(m._turn_id, 0)

    def test_retrieve_for_question_returns_placeholder_when_empty(self):
        from memgym.memory.ir.ir_hipporag import IRHippoRAGMemory
        _install_fake_hipporag(self._monkey, retrieved_docs=[])
        m = IRHippoRAGMemory(question="q")
        m.manage_context([], "obs")
        result = m.retrieve_for_question("q")
        self.assertIn("no relevant memories", result)


if __name__ == "__main__":
    unittest.main()
