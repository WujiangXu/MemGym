"""Tests for the shared SimpleMem core and its two adapters.

Companion to `test_hipporag_backend.py` — same V2 gate from
`when-we-run-the-pure-locket.md`, applied to the second new baseline.

Covers:
1. `openai/<model>` prefix is stripped (SimpleMem's LLM client uses
   `openai.OpenAI` directly, see `simplemem/utils/llm_client.py:8`).
2. `add_doc` lazily creates the upstream system and buffers; `commit()`
   maps to `SimpleMemSystem.finalize()`; `retrieve()` reads through the
   `hybrid_retriever`.
3. `reset()` drops the system reference so a fresh instance can be
   constructed (per-instance temp DBs are how the training phase isolates state).
4. Both pipeline adapters (`SimpleMemMethod`, `IRSimpleMemMemory`) wire
   into the shared core correctly.

Upstream `simplemem` is stubbed so tests never touch the OpenAI API or
the on-disk LanceDB store.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_fake_simplemem(monkey: dict, retrieved_entries=None):
    """Stub the `simplemem` and `simplemem.config` modules.

    `SimpleMemBackend._create_system()` imports both — we need to provide
    a `SimpleMemSystem` factory and a `get_config`/`set_config` pair.
    """
    fake_module = types.ModuleType("simplemem")
    fake_config = types.ModuleType("simplemem.config")

    fake_instance = MagicMock(name="SimpleMemSystemInstance")
    retriever = MagicMock(name="HybridRetriever")
    entries = []
    for text in retrieved_entries or []:
        e = MagicMock()
        e.content = text
        entries.append(e)
    retriever.retrieve.return_value = entries
    fake_instance.hybrid_retriever = retriever

    def _fake_ctor(**kwargs):
        fake_instance.last_init_kwargs = kwargs
        return fake_instance

    fake_module.SimpleMemSystem = MagicMock(side_effect=_fake_ctor)

    # `get_config()` returns a mutable namespace so the core can write
    # `embedding_model` / `embedding_dimension` onto it.
    cfg = types.SimpleNamespace(
        embedding_model="placeholder", embedding_dimension=0,
    )
    fake_config.get_config = MagicMock(return_value=cfg)
    fake_config.set_config = MagicMock()

    monkey["simplemem"] = sys.modules.get("simplemem")
    monkey["simplemem.config"] = sys.modules.get("simplemem.config")
    sys.modules["simplemem"] = fake_module
    sys.modules["simplemem.config"] = fake_config
    return fake_module.SimpleMemSystem, fake_instance


def _restore_fake_simplemem(monkey: dict):
    for name in ("simplemem", "simplemem.config"):
        prior = monkey.get(name)
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


class TestSimpleMemBackendInit(unittest.TestCase):
    def test_strips_openai_prefix(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        b = SimpleMemBackend(llm_model="openai/gpt-4o-mini")
        self.assertEqual(b._llm_model, "gpt-4o-mini")

    def test_leaves_unprefixed_model_alone(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        self.assertEqual(b._llm_model, "gpt-4o-mini")

    def test_does_not_strip_non_openai_prefix(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        b = SimpleMemBackend(llm_model="bedrock/claude-haiku")
        self.assertEqual(b._llm_model, "bedrock/claude-haiku")

    def test_no_system_created_at_init(self):
        """The expensive SimpleMem init (model loads, DB connect) must
        defer until the first add_doc — the training phase spawns hundreds of
        instances, most of which never index anything."""
        from memgym.memory.simplemem_core import SimpleMemBackend
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        self.assertIsNone(b._system)


class TestSimpleMemBackendBuffering(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_simplemem(self._monkey)

    def test_add_doc_creates_system_lazily(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        ctor, fake_instance = _install_fake_simplemem(self._monkey)
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        ctor.assert_not_called()
        b.add_doc("doc-1", "alpha")
        ctor.assert_called_once()
        fake_instance.add_dialogue.assert_called_once()

    def test_add_doc_passes_synthetic_timestamp(self):
        """SimpleMem requires monotonically-increasing timestamps; the
        backend fakes them. If two calls collide on a single timestamp
        SimpleMem may overwrite or merge entries unexpectedly."""
        from memgym.memory.simplemem_core import SimpleMemBackend
        _, fake_instance = _install_fake_simplemem(self._monkey)
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        b.add_doc("doc-1", "alpha")
        b.add_doc("doc-2", "beta")
        ts1 = fake_instance.add_dialogue.call_args_list[0].kwargs["timestamp"]
        ts2 = fake_instance.add_dialogue.call_args_list[1].kwargs["timestamp"]
        self.assertNotEqual(ts1, ts2)
        self.assertLess(ts1, ts2)

    def test_empty_text_is_skipped(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        ctor, _ = _install_fake_simplemem(self._monkey)
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        b.add_doc("doc-1", "")
        b.add_doc("doc-2", "   ")
        ctor.assert_not_called()

    def test_retrieve_with_no_docs_short_circuits(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        ctor, _ = _install_fake_simplemem(self._monkey)
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        result = b.retrieve("any question")
        self.assertEqual(result, [])
        ctor.assert_not_called()

    def test_retrieve_lazy_commits_exactly_once(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        _, fake_instance = _install_fake_simplemem(
            self._monkey, retrieved_entries=["entry A", "entry B"],
        )
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        b.add_doc("doc-1", "alpha")
        b.add_doc("doc-2", "beta")
        passages_1 = b.retrieve("q1")
        passages_2 = b.retrieve("q2")
        self.assertEqual(passages_1, ["entry A", "entry B"])
        self.assertEqual(passages_2, ["entry A", "entry B"])
        self.assertEqual(fake_instance.finalize.call_count, 1)

    def test_add_doc_after_commit_re_finalizes(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        _, fake_instance = _install_fake_simplemem(
            self._monkey, retrieved_entries=["e"],
        )
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        b.add_doc("doc-1", "alpha")
        b.retrieve("q1")
        b.add_doc("doc-2", "beta")
        b.retrieve("q2")
        self.assertEqual(fake_instance.finalize.call_count, 2)

    def test_retrieve_honors_top_k(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        _install_fake_simplemem(
            self._monkey,
            retrieved_entries=["a", "b", "c", "d", "e"],
        )
        b = SimpleMemBackend(llm_model="gpt-4o-mini", retrieve_k=3)
        b.add_doc("doc-1", "alpha")
        result = b.retrieve("q")
        self.assertEqual(len(result), 3)


class TestSimpleMemBackendReset(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_simplemem(self._monkey)

    def test_reset_drops_system_and_resets_dialogue_id(self):
        from memgym.memory.simplemem_core import SimpleMemBackend
        _install_fake_simplemem(self._monkey, retrieved_entries=["x"])
        b = SimpleMemBackend(llm_model="gpt-4o-mini")
        b.add_doc("doc-1", "alpha")
        b.retrieve("q")
        b.reset()
        self.assertEqual(b._dialogue_id, 0)
        self.assertFalse(b._committed)
        self.assertIsNone(b._system)


class TestSimpleMemCodingAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_simplemem(self._monkey)

    def test_ingest_truncates_to_max_chars(self):
        from memgym.pipelines.coding_synthetic.memory_methods.simplemem_method import (
            SimpleMemMethod,
        )
        _install_fake_simplemem(self._monkey)
        m = SimpleMemMethod(max_ingest_chars=100)
        big = "x" * 5000
        m.ingest("doc-1", big, "task")
        # The buffered dialogue's content was truncated before reaching
        # the upstream system.
        call = m._system._system.add_dialogue.call_args
        passed_text = call.kwargs["content"]
        self.assertLessEqual(len(passed_text), 100 + 50)

    def test_retrieve_returns_no_memories_string_when_empty(self):
        from memgym.pipelines.coding_synthetic.memory_methods.simplemem_method import (
            SimpleMemMethod,
        )
        _install_fake_simplemem(self._monkey, retrieved_entries=[])
        m = SimpleMemMethod()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("no relevant memories", result)

    def test_retrieve_joins_passages_with_separator(self):
        from memgym.pipelines.coding_synthetic.memory_methods.simplemem_method import (
            SimpleMemMethod,
        )
        _install_fake_simplemem(
            self._monkey, retrieved_entries=["entry A", "entry B"],
        )
        m = SimpleMemMethod()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("entry A", result)
        self.assertIn("entry B", result)
        self.assertIn("\n---\n", result)


class TestSimpleMemIRAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_simplemem(self._monkey)

    def test_registered_in_ir_factory(self):
        import memgym.memory.ir.ir_simplemem  # noqa: F401
        from memgym.memory.base import get_memory_model
        m = get_memory_model("ir_simplemem", question="test")
        self.assertEqual(m._turn_id, 0)

    def test_manage_context_buffers_observations(self):
        from memgym.memory.ir.ir_simplemem import IRSimpleMemMemory
        _install_fake_simplemem(self._monkey)
        m = IRSimpleMemMemory(question="q")
        m.manage_context([], "first observation")
        m.manage_context([], "second observation")
        self.assertEqual(m._turn_id, 2)
        self.assertEqual(len(m._all_observations), 2)

    def test_empty_observation_is_ignored(self):
        from memgym.memory.ir.ir_simplemem import IRSimpleMemMemory
        _install_fake_simplemem(self._monkey)
        m = IRSimpleMemMemory(question="q")
        m.manage_context([], None)
        m.manage_context([], "")
        self.assertEqual(m._turn_id, 0)

    def test_retrieve_for_question_returns_placeholder_when_empty(self):
        from memgym.memory.ir.ir_simplemem import IRSimpleMemMemory
        _install_fake_simplemem(self._monkey, retrieved_entries=[])
        m = IRSimpleMemMemory(question="q")
        m.manage_context([], "obs")
        result = m.retrieve_for_question("q")
        self.assertIn("no relevant memories", result)


if __name__ == "__main__":
    unittest.main()
