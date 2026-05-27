"""WebArena-specific memory strategies for web GUI agent trajectories.

These adapters mirror the architecture of the proven coding-track strategies
(`memory/llm_summarizing.py` and `memory/structured_summary.py`) — same
CondensationRecord persistence, same build_view() rebuild, same trigger
semantics — but use prompts and a structured-summary schema tailored to
web GUI tasks (URLs, forms, DOM elements, extracted data) instead of
code-edit state.

Importing this module registers the strategies under the names
`webarena_summarizing` and `webarena_structured` in the memory registry.
"""

from .webarena_summarizing import (
    WebArenaSummarizingMemory,
    WEBARENA_SUMMARIZATION_PROMPT,
)
from .webarena_structured_summary import (
    WebArenaStructuredSummary,
    WEBARENA_STATE_SUMMARY_FIELDS,
    WEBARENA_STRUCTURED_SUMMARY_PROMPT,
)

__all__ = [
    "WebArenaSummarizingMemory",
    "WebArenaStructuredSummary",
    "WEBARENA_SUMMARIZATION_PROMPT",
    "WEBARENA_STATE_SUMMARY_FIELDS",
    "WEBARENA_STRUCTURED_SUMMARY_PROMPT",
]
