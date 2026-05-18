"""CUA-specific memory strategies for OSWorld GUI agent trajectories.

These strategies are adapted from the base memory managers but use
CUA-specific prompts that track GUI state: screen observations, actions
taken (clicks, typing, navigation), visual outcomes, and progress toward
the user's desktop task.

Registry:
    cua_llm_summarizing  -- LLM-based summarization for GUI agent context
    cua_structured_summary -- Structured summary with GUI-specific fields
    cua_naive            -- Simple token-threshold summarization for GUI context
"""

# Import submodules to trigger their registration
from .cua_llm_summarizing import CUALLMSummarizingMemory  # noqa: F401, E402
from .cua_structured_summary import CUAStructuredSummaryMemory  # noqa: F401, E402
from .cua_naive_summarization import CUANaiveSummarizationMemory  # noqa: F401, E402
