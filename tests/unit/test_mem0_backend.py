"""Tests for the shared Mem0 core and its two adapters.

V2 offline gate — `mem0` is stubbed via `sys.modules`, no network or GPU.
Mirrors the pattern in `test_simplemem_backend.py`.

Covers:
1. `openai/<model>` prefix is stripped (Mem0's LiteLLM-config schema
   expects bare model names for the OpenAI/Bedrock providers).
2. `add_doc` lazily creates the upstream `Memory` and calls `add()` with
   `user_id` set; empty / whitespace-only text is a no-op.
3. `retrieve` shapes both list-style and `{"results": [...]}` SDK returns
   into a flat list of strings.
4. `reset` rotates `user_id` instead of tearing down the embedder
   (cheapest reset for Phase-5's many-instance reuse).
5. Both pipeline adapters wire into the shared core correctly.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_fake_mem0(monkey: dict, search_return=None):
    fake_module = types.ModuleType("mem0")
    fake_memory_instance = MagicMock(name="Mem0MemoryInstance")
    fake_memory_instance.search.return_value = (
        search_return if search_return is not None else {"results": []}
    )
    fake_memory_class = MagicMock()
    fake_memory_class.from_config = MagicMock(return_value=fake_memory_instance)
    fake_module.Memory = fake_memory_class

    monkey["mem0"] = sys.modules.get("mem0")
    sys.modules["mem0"] = fake_module
    return fake_memory_class, fake_memory_instance


def _restore_fake_mem0(monkey: dict):
    prior = monkey.get("mem0")
    if prior is None:
        sys.modules.pop("mem0", None)
    else:
        sys.modules["mem0"] = prior


class TestMem0SystemInit(unittest.TestCase):
    def test_strips_openai_prefix(self):
        from memgym.memory.external.mem0 import Mem0System
        s = Mem0System(llm_model="openai/gpt-4o-mini")
        self.assertEqual(s._llm_model, "gpt-4o-mini")

    def test_leaves_unprefixed_model_alone(self):
        from memgym.memory.external.mem0 import Mem0System
        s = Mem0System(llm_model="gpt-4o-mini")
        self.assertEqual(s._llm_model, "gpt-4o-mini")

    def test_does_not_strip_non_openai_prefix(self):
        from memgym.memory.external.mem0 import Mem0System
        s = Mem0System(llm_model="bedrock/claude-haiku")
        self.assertEqual(s._llm_model, "bedrock/claude-haiku")

    def test_no_system_created_at_init(self):
        from memgym.memory.external.mem0 import Mem0System
        s = Mem0System(llm_model="gpt-4o-mini")
        self.assertIsNone(s._memory)

    def test_user_id_is_unique(self):
        from memgym.memory.external.mem0 import Mem0System
        s1 = Mem0System(llm_model="gpt-4o-mini")
        s2 = Mem0System(llm_model="gpt-4o-mini")
        self.assertNotEqual(s1._user_id, s2._user_id)


class TestMem0SystemAddRetrieve(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_mem0(self._monkey)

    def test_add_doc_creates_memory_lazily(self):
        from memgym.memory.external.mem0 import Mem0System
        cls, inst = _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini")
        cls.from_config.assert_not_called()
        s.add_doc("doc-1", "alpha")
        cls.from_config.assert_called_once()
        inst.add.assert_called_once()

    def test_add_doc_passes_user_id_and_metadata(self):
        from memgym.memory.external.mem0 import Mem0System
        _, inst = _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        call = inst.add.call_args
        self.assertEqual(call.kwargs["user_id"], s._user_id)
        self.assertEqual(call.kwargs["metadata"], {"doc_id": "doc-1"})
        self.assertIn("doc-1", call.kwargs["messages"])
        self.assertIn("alpha", call.kwargs["messages"])

    def test_empty_text_is_skipped(self):
        from memgym.memory.external.mem0 import Mem0System
        cls, _ = _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "")
        s.add_doc("doc-2", "   ")
        cls.from_config.assert_not_called()

    def test_retrieve_with_no_docs_short_circuits(self):
        from memgym.memory.external.mem0 import Mem0System
        cls, _ = _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini")
        result = s.retrieve("any question")
        self.assertEqual(result, [])
        cls.from_config.assert_not_called()

    def test_retrieve_handles_dict_with_results_key(self):
        from memgym.memory.external.mem0 import Mem0System
        _, _ = _install_fake_mem0(
            self._monkey,
            search_return={"results": [
                {"memory": "fact A"}, {"memory": "fact B"},
            ]},
        )
        s = Mem0System(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        passages = s.retrieve("q")
        self.assertEqual(passages, ["fact A", "fact B"])

    def test_retrieve_handles_bare_list(self):
        from memgym.memory.external.mem0 import Mem0System
        _install_fake_mem0(
            self._monkey,
            search_return=[{"memory": "fact A"}, {"text": "fact B"}],
        )
        s = Mem0System(llm_model="gpt-4o-mini")
        s.add_doc("doc-1", "alpha")
        passages = s.retrieve("q")
        self.assertEqual(passages, ["fact A", "fact B"])

    def test_retrieve_passes_limit(self):
        from memgym.memory.external.mem0 import Mem0System
        _, inst = _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini", retrieve_k=3)
        s.add_doc("doc-1", "alpha")
        s.retrieve("q")
        call = inst.search.call_args
        self.assertEqual(call.kwargs["limit"], 3)
        s.retrieve("q", k=7)
        call = inst.search.call_args
        self.assertEqual(call.kwargs["limit"], 7)


class TestMem0SystemReset(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_mem0(self._monkey)

    def test_reset_rotates_user_id(self):
        from memgym.memory.external.mem0 import Mem0System
        _install_fake_mem0(self._monkey)
        s = Mem0System(llm_model="gpt-4o-mini")
        old_user = s._user_id
        s.add_doc("doc-1", "alpha")
        s.reset()
        self.assertNotEqual(s._user_id, old_user)


class TestMem0CodingAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_mem0(self._monkey)

    def test_ingest_truncates_to_max_chars(self):
        _install_fake_mem0(self._monkey)
        from memgym.pipelines.coding_synthetic.memory_methods.mem0_method import (
            Mem0Method,
        )
        m = Mem0Method(max_ingest_chars=100)
        m.ingest("doc-1", "x" * 5000, "task")
        passed = m._system._memory.add.call_args.kwargs["messages"]
        # Header `[doc-1]\n` adds ~9 chars on top of the 100-char body.
        self.assertLessEqual(len(passed), 100 + 50)

    def test_retrieve_returns_no_memories_string_when_empty(self):
        _install_fake_mem0(self._monkey, search_return={"results": []})
        from memgym.pipelines.coding_synthetic.memory_methods.mem0_method import (
            Mem0Method,
        )
        m = Mem0Method()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("no relevant memories", result)

    def test_retrieve_joins_passages_with_separator(self):
        _install_fake_mem0(
            self._monkey,
            search_return={"results": [
                {"memory": "fact A"}, {"memory": "fact B"},
            ]},
        )
        from memgym.pipelines.coding_synthetic.memory_methods.mem0_method import (
            Mem0Method,
        )
        m = Mem0Method()
        m.ingest("doc-1", "alpha", "task")
        result = m.retrieve("q", "task")
        self.assertIn("fact A", result)
        self.assertIn("fact B", result)
        self.assertIn("\n---\n", result)


class TestMem0IRAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake_mem0(self._monkey)

    def test_registered_in_ir_factory(self):
        _install_fake_mem0(self._monkey)
        import memgym.memory.ir.ir_mem0  # noqa: F401
        from memgym.memory.base import get_memory_model
        m = get_memory_model("ir_mem0", question="test")
        self.assertEqual(m._turn_id, 0)

    def test_manage_context_buffers_observations(self):
        _install_fake_mem0(self._monkey)
        from memgym.memory.ir.ir_mem0 import IRMem0Memory
        m = IRMem0Memory(question="q")
        m.manage_context([], "first observation")
        m.manage_context([], "second observation")
        self.assertEqual(m._turn_id, 2)
        self.assertEqual(len(m._all_observations), 2)

    def test_empty_observation_is_ignored(self):
        _install_fake_mem0(self._monkey)
        from memgym.memory.ir.ir_mem0 import IRMem0Memory
        m = IRMem0Memory(question="q")
        m.manage_context([], None)
        m.manage_context([], "")
        self.assertEqual(m._turn_id, 0)

    def test_retrieve_for_question_returns_placeholder_when_empty(self):
        _install_fake_mem0(self._monkey, search_return={"results": []})
        from memgym.memory.ir.ir_mem0 import IRMem0Memory
        m = IRMem0Memory(question="q")
        m.manage_context([], "obs")
        result = m.retrieve_for_question("q")
        self.assertIn("no relevant memories", result)


if __name__ == "__main__":
    unittest.main()
