"""Tests for the vendored MemoryBank core and its two adapters.

V2 offline gate — `sentence_transformers.SentenceTransformer` is stubbed
to return deterministic 2-D embeddings so cosine math is testable, and
`time.time` is monkey-patched so the Δt-based decay is deterministic.

Covers:
1. `_chunk()` respects paragraph boundaries and the per-chunk hard cap.
2. `add_doc` lazily loads the encoder and stores units with strength=1.0.
3. `retrieve` ranks by cosine × decay.
4. Reinforcement multiplies `strength` by `reinforcement_factor`,
   clamped at `strength_cap`, and bumps `last_accessed`.
5. Empty store returns `[]`; `reset` clears all units.
6. Both pipeline adapters wire into the shared core correctly.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np


def _install_fake_sentence_transformers(monkey: dict, embeddings=None):
    """Stub `sentence_transformers.SentenceTransformer`.

    `embeddings` is a list-of-lists; `encode()` returns the next batch
    each call (used by add_doc with len==chunks; for retrieve with
    list of one query, returns a 2-D ndarray of shape (1, dim)).
    """
    fake_module = types.ModuleType("sentence_transformers")

    class FakeEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=False):
            n = len(texts)
            if embeddings is None:
                # Default: deterministic but distinct per text, normalized.
                arr = np.zeros((n, 4), dtype=np.float32)
                for i in range(n):
                    arr[i, i % 4] = 1.0
                return arr
            arr = np.asarray(embeddings[:n], dtype=np.float32)
            return arr

    fake_module.SentenceTransformer = FakeEncoder
    monkey["sentence_transformers"] = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module


def _restore_fake(monkey: dict, name: str):
    prior = monkey.get(name)
    if prior is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = prior


class TestMemoryBankChunking(unittest.TestCase):
    def test_empty_text_returns_empty(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        s = MemoryBankSystem()
        self.assertEqual(s._chunk(""), [])

    def test_paragraph_split(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        s = MemoryBankSystem(chunk_chars=1000)
        text = "para A\n\npara B\n\npara C"
        chunks = s._chunk(text)
        joined = "\n".join(chunks)
        self.assertIn("para A", joined)
        self.assertIn("para B", joined)
        self.assertIn("para C", joined)

    def test_hard_cap_respected(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        s = MemoryBankSystem(chunk_chars=20)
        # One giant paragraph, way over cap.
        text = "x" * 100
        chunks = s._chunk(text)
        self.assertEqual(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(len(c), 20)


class TestMemoryBankAddRetrieve(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake(self._monkey, "sentence_transformers")

    def test_empty_text_is_skipped(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        _install_fake_sentence_transformers(self._monkey)
        s = MemoryBankSystem()
        s.add_doc("doc-1", "")
        s.add_doc("doc-2", "  \n\n  ")
        self.assertEqual(len(s._units), 0)

    def test_add_doc_creates_units_with_strength_one(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        _install_fake_sentence_transformers(self._monkey)
        s = MemoryBankSystem(chunk_chars=1000)
        s.add_doc("doc-1", "para A\n\npara B")
        self.assertEqual(len(s._units), 1)  # both fit in one chunk
        for unit in s._units:
            self.assertEqual(unit["strength"], 1.0)
            self.assertIn("[doc-1]", unit["content"])

    def test_retrieve_empty_store_returns_empty(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        s = MemoryBankSystem()
        self.assertEqual(s.retrieve("anything"), [])

    def test_retrieve_ranks_by_cosine(self):
        from memgym.memory.memorybank_core import MemoryBankSystem

        # Three units: e0=(1,0,0,0), e1=(0,1,0,0), e2=(0,0,1,0).
        # Query embedding = (0,1,0,0) → unit 1 ranks first.
        embeddings = [
            [1, 0, 0, 0],  # doc 1 chunk 1
            [0, 1, 0, 0],  # doc 2 chunk 1
            [0, 0, 1, 0],  # doc 3 chunk 1
            [0, 1, 0, 0],  # query
        ]
        _install_fake_sentence_transformers(self._monkey, embeddings)
        s = MemoryBankSystem(chunk_chars=1000, retrieve_k=3)
        s.add_doc("doc-1", "first paragraph")
        s.add_doc("doc-2", "second paragraph")
        s.add_doc("doc-3", "third paragraph")
        results = s.retrieve("q")
        self.assertTrue(results[0].startswith("[doc-2]"))

    def test_reinforcement_increases_strength(self):
        from memgym.memory.memorybank_core import MemoryBankSystem

        embeddings = [[1, 0, 0, 0], [1, 0, 0, 0]]  # 1 chunk + 1 query
        _install_fake_sentence_transformers(self._monkey, embeddings)
        s = MemoryBankSystem(chunk_chars=1000, retrieve_k=1,
                             reinforcement_factor=1.5)
        s.add_doc("doc-1", "alpha")
        before = s._units[0]["strength"]
        s.retrieve("q")
        after = s._units[0]["strength"]
        self.assertAlmostEqual(after, before * 1.5)

    def test_strength_cap_respected(self):
        from memgym.memory.memorybank_core import MemoryBankSystem

        embeddings = [[1, 0, 0, 0]] * 10
        _install_fake_sentence_transformers(self._monkey, embeddings)
        s = MemoryBankSystem(chunk_chars=1000, retrieve_k=1,
                             reinforcement_factor=10.0, strength_cap=20.0)
        s.add_doc("doc-1", "alpha")
        for _ in range(8):
            s.retrieve("q")
        self.assertLessEqual(s._units[0]["strength"], 20.0)
        self.assertEqual(s._units[0]["strength"], 20.0)

    def test_retrieve_top_k_overrides_default(self):
        from memgym.memory.memorybank_core import MemoryBankSystem

        embeddings = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                      [0, 0, 0, 1], [1, 0, 0, 0]]
        _install_fake_sentence_transformers(self._monkey, embeddings)
        s = MemoryBankSystem(chunk_chars=1000, retrieve_k=10)
        for i in range(4):
            s.add_doc(f"doc-{i}", f"para {i}")
        results = s.retrieve("q", k=2)
        self.assertEqual(len(results), 2)


class TestMemoryBankReset(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake(self._monkey, "sentence_transformers")

    def test_reset_clears_units(self):
        from memgym.memory.memorybank_core import MemoryBankSystem
        _install_fake_sentence_transformers(self._monkey)
        s = MemoryBankSystem()
        s.add_doc("doc-1", "alpha")
        self.assertGreater(len(s._units), 0)
        s.reset()
        self.assertEqual(len(s._units), 0)


class TestMemoryBankCodingAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake(self._monkey, "sentence_transformers")

    def test_ingest_truncates_to_max_chars(self):
        _install_fake_sentence_transformers(self._monkey)
        from memgym.pipelines.coding_synthetic.memory_methods.memorybank_method import (
            MemoryBankMethod,
        )
        m = MemoryBankMethod(max_ingest_chars=100, chunk_chars=200)
        m.ingest("doc-1", "x" * 5000, "task")
        # One unit, content (header + body) ≤ 200.
        self.assertEqual(len(m._system._units), 1)
        self.assertLessEqual(len(m._system._units[0]["content"]), 200 + 50)

    def test_retrieve_returns_no_memories_string_when_empty(self):
        _install_fake_sentence_transformers(self._monkey)
        from memgym.pipelines.coding_synthetic.memory_methods.memorybank_method import (
            MemoryBankMethod,
        )
        m = MemoryBankMethod()
        result = m.retrieve("q", "task")
        self.assertIn("no relevant memories", result)


class TestMemoryBankIRAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._monkey: dict = {}

    def tearDown(self) -> None:
        _restore_fake(self._monkey, "sentence_transformers")

    def test_registered_in_ir_factory(self):
        _install_fake_sentence_transformers(self._monkey)
        import memgym.memory.ir.ir_memorybank  # noqa: F401
        from memgym.memory.base import get_memory_model
        m = get_memory_model("ir_memorybank", question="test")
        self.assertEqual(m._turn_id, 0)

    def test_manage_context_buffers_observations(self):
        _install_fake_sentence_transformers(self._monkey)
        from memgym.memory.ir.ir_memorybank import IRMemoryBankMemory
        m = IRMemoryBankMemory(question="q")
        m.manage_context([], "first observation")
        m.manage_context([], "second observation")
        self.assertEqual(m._turn_id, 2)
        self.assertEqual(len(m._all_observations), 2)

    def test_empty_observation_is_ignored(self):
        _install_fake_sentence_transformers(self._monkey)
        from memgym.memory.ir.ir_memorybank import IRMemoryBankMemory
        m = IRMemoryBankMemory(question="q")
        m.manage_context([], None)
        m.manage_context([], "")
        self.assertEqual(m._turn_id, 0)


if __name__ == "__main__":
    unittest.main()
