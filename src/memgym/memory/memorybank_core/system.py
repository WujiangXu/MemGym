"""Vendored MemoryBank algorithm.

The upstream repository (zhongwanjun/MemoryBank-SiliconFriend) bundles a
chatbot scaffold and is Python-3.10 pinned, so we vendor only the
~30-line algorithmic contribution from the AAAI-2024 paper:

  score = cosine(q, mem) * exp(-Δt_hours / (decay_scale * strength))

On each retrieval, top-k units are reinforced (strength *= reinforcement_factor)
and their access timestamp is bumped, so repeated relevant memories rank
higher and decay slower — the Ebbinghaus forgetting-curve analogue.

Note on the training phase synthetic time: a single eval instance completes in seconds,
so within-instance Δt is essentially zero and the decay term degenerates
toward 1. The discriminating signal is therefore the **reinforcement**
effect (a relevant unit retrieved on hop-1 is boosted for hop-2/hop-3).
For cross-instance temporal effects you'd need a persistent store + real
wall-clock spacing, which the synthetic pipeline does not exercise.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional


class MemoryBankSystem:
    """Vector store with Ebbinghaus decay + retrieval reinforcement."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        chunk_chars: int = 1500,
        decay_hours_scale: float = 24.0,
        reinforcement_factor: float = 1.5,
        strength_cap: float = 100.0,
    ) -> None:
        self._embedding_model = embedding_model
        self._retrieve_k = retrieve_k
        self._chunk_chars = chunk_chars
        self._decay_hours_scale = decay_hours_scale
        self._reinforcement = reinforcement_factor
        self._strength_cap = strength_cap
        self._units: List[Dict[str, Any]] = []
        self._encoder = None  # lazy

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._embedding_model)

    def _chunk(self, text: str) -> List[str]:
        """Split on paragraph boundaries, then hard-cap each chunk by chars."""
        if not text:
            return []
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras:
            paras = [text.strip()]
        chunks: List[str] = []
        buf = ""
        for para in paras:
            if not buf:
                buf = para
            elif len(buf) + 1 + len(para) <= self._chunk_chars:
                buf = f"{buf}\n{para}"
            else:
                chunks.append(buf)
                buf = para
        if buf:
            chunks.append(buf)
        # Hard-cap any oversize chunk (e.g. one giant paragraph).
        capped: List[str] = []
        for c in chunks:
            if len(c) <= self._chunk_chars:
                capped.append(c)
            else:
                for i in range(0, len(c), self._chunk_chars):
                    capped.append(c[i : i + self._chunk_chars])
        return capped

    def add_doc(self, doc_id: str, text: str) -> None:
        if not text or not text.strip():
            return
        self._ensure_encoder()
        chunks = self._chunk(text)
        if not chunks:
            return
        import numpy as np

        embeddings = self._encoder.encode(chunks, normalize_embeddings=True)
        now = time.time()
        for chunk, emb in zip(chunks, embeddings):
            self._units.append(
                {
                    "content": f"[{doc_id}] {chunk}",
                    "embedding": np.asarray(emb, dtype=np.float32),
                    "last_accessed": now,
                    "strength": 1.0,
                }
            )

    def retrieve(self, question: str, k: Optional[int] = None) -> List[str]:
        if not self._units:
            return []
        self._ensure_encoder()
        import numpy as np

        q_emb = self._encoder.encode([question], normalize_embeddings=True)[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        now = time.time()
        scored: List[tuple] = []
        for unit in self._units:
            sim = float(np.dot(q_emb, unit["embedding"]))
            dt_hours = max((now - unit["last_accessed"]) / 3600.0, 0.0)
            decay = math.exp(
                -dt_hours / (self._decay_hours_scale * max(unit["strength"], 1e-3))
            )
            scored.append((sim * decay, unit))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: (k or self._retrieve_k)]
        for _, unit in top:
            unit["last_accessed"] = now
            unit["strength"] = min(
                unit["strength"] * self._reinforcement, self._strength_cap
            )
        return [unit["content"] for _, unit in top]

    def reset(self) -> None:
        self._units.clear()
