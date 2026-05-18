"""Pydantic schemas for runtime fact verification output."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class FactTestResult(BaseModel):
    """Per-fact runtime verification outcome.

    Two distinct failure modes:
      - `test_synth_error` (import / syntax / collection issue) — the test
        itself is broken; treat as inconclusive, do not penalize fact.
      - `passed=False` with no synth error — the test ran and the fact
        was contradicted by the code.
    """
    fact_id: str = Field(description="Matches GroundingFact.id")
    passed: bool = Field(description="True iff pytest exit code == 0")
    runtime_ms: int = Field(
        default=0,
        description="Wall time for the pytest invocation (ms)",
    )
    test_name: str = Field(
        default="",
        description="Human-readable name of the synthesized test",
    )
    stderr: str = Field(
        default="",
        description="Captured stderr from pytest (truncated to 2000 chars)",
    )
    test_synth_error: Optional[str] = Field(
        default=None,
        description="Non-None iff the test failed to import/collect; "
                    "indicates the test itself is broken, not the fact",
    )

    class Config:
        extra = "allow"
