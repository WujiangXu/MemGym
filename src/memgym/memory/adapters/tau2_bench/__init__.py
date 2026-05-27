"""Tau2-bench dialogue memory strategies.

These adapters mirror the architecture of the WebArena variants but use
a dialogue-tailored 8-field schema (USER_GOAL / USER_CONSTRAINTS /
DECISIONS_MADE / TOOL_CALLS_EXECUTED / INFORMATION_GATHERED /
OUTSTANDING_COMMITMENTS / OPEN_QUESTIONS / NEXT_ACTION). Both adapters
support an explicit ``keep_last`` parameter (the bottom anchor) on top
of the existing ``keep_first`` (top anchor), so the agent always sees
the most recent N turns verbatim alongside the compressed middle.

Both adapters are role-aware: when constructed with ``role="user"``
they swap ``USER_GOAL`` for ``USER_PERSONA`` so the user simulator
remembers what character it's playing rather than what task it's
pursuing — the central piece of the symmetric memory wrapping design.

Importing this module registers the strategies under the names
``tau2_summarizing`` and ``tau2_structured`` in the memory registry.
"""

from .tau2_summarizing import (
    Tau2SummarizingMemory,
    TAU2_SUMMARIZATION_PROMPT_AGENT,
    TAU2_SUMMARIZATION_PROMPT_USER,
)
from .tau2_structured import (
    Tau2StructuredSummary,
    TAU2_STATE_SUMMARY_FIELDS_AGENT,
    TAU2_STATE_SUMMARY_FIELDS_USER,
)

__all__ = [
    "Tau2SummarizingMemory",
    "Tau2StructuredSummary",
    "TAU2_SUMMARIZATION_PROMPT_AGENT",
    "TAU2_SUMMARIZATION_PROMPT_USER",
    "TAU2_STATE_SUMMARY_FIELDS_AGENT",
    "TAU2_STATE_SUMMARY_FIELDS_USER",
]
