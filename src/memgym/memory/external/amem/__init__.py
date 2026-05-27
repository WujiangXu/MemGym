"""
A-mem (Agentic Memory) for MemGym.

A memory system with LLM-based evolution, semantic retrieval,
and Zettelkasten-style knowledge linking.

Paper: https://arxiv.org/pdf/2502.12110

Components:
    - MemoryNote: Basic memory unit with metadata
    - LLMController: LiteLLM-based controller for metadata generation
    - EmbeddingRetriever: SentenceTransformer-based semantic search
    - AgenticMemorySystem: Main system with evolution and threshold-based compression
    - AMemMemoryModel: MemGym interface (base model)
    - SWEAMemModel: SWE-bench code-adapted model

Note: Tau2AMemModel was removed when the legacy tau2 wiring was deleted. The new
gym/tau2_bench track ships its own dialogue-tailored adapters under
memgym.memory.adapters.tau2_bench (tau2_summarizing + tau2_structured). An A-mem variant
can be added later under that namespace if needed.

Features:
    - Full observation storage with LLM-generated metadata
    - Threshold-based context compression (pass-through below, retrieval above)
    - LLM-generated retrieval queries
    - Hybrid context: retrieved memories + recent turns
    - Environment-specific adaptations
"""

# Use relative imports (works when imported as package)
# Fall back to direct imports (works when running standalone tests)
try:
    from .note import MemoryNote
    from .llm_controller import LLMController
    from .retriever import EmbeddingRetriever
    from .system import AgenticMemorySystem
    from .model import AMemMemoryModel
    from .swe_model import SWEAMemModel
except ImportError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    _AMEM_PATH = Path(__file__).parent
    if str(_AMEM_PATH) not in sys.path:
        sys.path.insert(0, str(_AMEM_PATH))

    from note import MemoryNote
    from llm_controller import LLMController
    from retriever import EmbeddingRetriever
    from system import AgenticMemorySystem
    from model import AMemMemoryModel
    from swe_model import SWEAMemModel

__all__ = [
    # Core components
    "MemoryNote",
    "LLMController",
    "EmbeddingRetriever",
    "AgenticMemorySystem",
    # Base model
    "AMemMemoryModel",
    # Environment-specific models
    "SWEAMemModel",
]
