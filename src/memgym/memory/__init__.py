"""
Memory Managers for MemGym.

This module provides memory managers (middleware) that sit between
environment and agent, controlling what context the agent sees.

Usage:
    >>> from memory import PassThroughMemory, LLMSummarizingMemory
    >>> memory = LLMSummarizingMemory(max_size=100)
    >>> filtered = memory.manage_context(history, current_obs)
    >>> action = agent.act(filtered)

    # Or use registry:
    >>> from memory import get_memory_model
    >>> memory = get_memory_model("llm_summarizing", max_size=100)
"""

from .base import (
    BaseMemoryManager,
    PassThroughMemory,
    FilteredContext,
    # Registry
    get_memory_model,
    register_memory_model,
    list_memory_models,
    # Legacy compatibility
    MemoryAction,
    Context,
    BaseMemoryModel,
    PassThroughMemoryModel,
)

# Generic strategies — import triggers registration by name
from .strategies.summarization import SummarizationMemory, SummarizationMemoryModel
from .strategies.naive_summarization import (
    NaiveSummarizationMemory,
    NaiveSummarizationMemoryModel,
    SUMMARIZATION_PROMPTS,
)

# OpenHands-ported condensers (import triggers registration)
from .strategies.observation_masking import ObservationMaskingMemory
from .strategies.llm_summarizing import LLMSummarizingMemory
from .strategies.structured_summary import StructuredSummaryMemory
from .strategies.pipeline import PipelineMemory

# Sliding window + incremental summary (import triggers registration)
from .strategies.sliding_window_summary import SlidingWindowSummaryMemory
from .strategies.adaptive_token_budget import (
    AdaptiveTokenBudgetMemory,
    AdaptiveTokenBudgetMemoryModel,
)

# Multimodal adapter for CUA agents (import triggers registration)
from .strategies.multimodal_adapter import MultimodalMemoryAdapter

# CUA-specific memory strategies (import triggers registration)
from .adapters.cua import CUALLMSummarizingMemory, CUAStructuredSummaryMemory, CUANaiveSummarizationMemory

# IR-specific memory strategies (import triggers registration)
from .ir import IRPassThrough, IRSummarizingMemory, IRStructuredSummary

# WebArena-specific memory strategies (import triggers registration)
from .adapters.webarena import WebArenaSummarizingMemory, WebArenaStructuredSummary

# Tau2-bench dialogue memory strategies (import triggers registration)
from .adapters.tau2_bench import Tau2SummarizingMemory, Tau2StructuredSummary

# A-mem is optional (requires extra dependencies: sentence-transformers, sklearn, etc.)
try:
    from .external.amem import (
        AMemMemoryModel,
        AgenticMemorySystem,
        MemoryNote,
        EmbeddingRetriever,
        SWEAMemModel,
    )
    _AMEM_AVAILABLE = True
except ImportError:
    AMemMemoryModel = None
    AgenticMemorySystem = None
    MemoryNote = None
    EmbeddingRetriever = None
    SWEAMemModel = None
    _AMEM_AVAILABLE = False

__all__ = [
    # Core interface
    "BaseMemoryManager",
    "PassThroughMemory",
    "SummarizationMemory",
    "NaiveSummarizationMemory",
    "FilteredContext",
    # OpenHands-ported condensers
    "ObservationMaskingMemory",
    "LLMSummarizingMemory",
    "StructuredSummaryMemory",
    "PipelineMemory",
    # Sliding window
    "SlidingWindowSummaryMemory",
    "AdaptiveTokenBudgetMemory",
    # Multimodal
    "MultimodalMemoryAdapter",
    # CUA-specific
    "CUALLMSummarizingMemory",
    "CUAStructuredSummaryMemory",
    "CUANaiveSummarizationMemory",
    # IR-specific
    "IRPassThrough",
    "IRSummarizingMemory",
    "IRStructuredSummary",
    # WebArena-specific
    "WebArenaSummarizingMemory",
    "WebArenaStructuredSummary",
    # Tau2-bench-specific
    "Tau2SummarizingMemory",
    "Tau2StructuredSummary",
    # Registry
    "get_memory_model",
    "register_memory_model",
    "list_memory_models",
    # Environment-specific prompts
    "SUMMARIZATION_PROMPTS",
    # Legacy compatibility
    "MemoryAction",
    "Context",
    "BaseMemoryModel",
    "PassThroughMemoryModel",
    "SummarizationMemoryModel",
    "NaiveSummarizationMemoryModel",
    "AdaptiveTokenBudgetMemoryModel",
]

# Add A-mem components to exports if available
if _AMEM_AVAILABLE:
    __all__.extend([
        "AMemMemoryModel",
        "AgenticMemorySystem",
        "MemoryNote",
        "EmbeddingRetriever",
        "SWEAMemModel",
    ])
