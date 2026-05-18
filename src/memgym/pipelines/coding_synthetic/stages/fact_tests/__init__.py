"""Runtime fact verification — lightweight per-fact pytest gate.

Unlike SWE-bench's gold-patch apply-and-test, which is heavy and lives
in `gym/swe_bench/`, this module validates the generated grounding
facts themselves. For each `discoverable=True` fact, it synthesizes a
small pytest test that an LLM believes would prove the fact, then
runs it inside the pre-built SWE-smith Docker image. Two failure
modes are distinguished:

  - **synthesis error** (import / syntax / collection)
      → inconclusive; we don't penalize the fact
  - **assertion failure**
      → the fact is probably wrong about the code

The gate is advisory: the training phase's P7 flags instances where >50% of
tested facts fail assertions as needing review. Nothing is auto-dropped.
"""
from __future__ import annotations

from .schemas import FactTestResult

__all__ = ["FactTestResult"]
