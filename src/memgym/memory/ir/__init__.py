"""IR-specific memory strategies for MemGym-IR evaluation.

These strategies are adapted from the SWE-bench memory managers but track
IR-specific fields: entities, bridge facts, evidence, candidate answers.

Registry:
    ir_passthrough  — No summarization, pass all turns through
    ir_summarizing  — LLM-based summarization for IR research context
    ir_structured   — Structured summary with IR-specific fields
"""

from ..base import (
    BaseMemoryManager,
    FilteredContext,
    register_memory_model,
)
from typing import Any, Dict, List, Optional


class IRPassThrough(BaseMemoryManager):
    """Pass-through memory for IR — keeps all turn notes without summarization.

    Serves as the "unlimited memory" baseline for IR evaluation.
    """

    def __init__(self, max_tokens: int = 100000, **kwargs):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._notes = []

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        if current_observation:
            self._notes.append(current_observation)
        tokens = self.count_tokens(self._notes)
        return FilteredContext(
            content=self._notes.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_passthrough",
            },
        )

    def reset(self) -> None:
        self._notes = []

    def get_notes_text(self) -> str:
        """Return all accumulated notes as a single string."""
        return "\n".join(str(n) for n in self._notes)


register_memory_model("ir_passthrough", IRPassThrough)

# Import submodules to trigger their registration
from .ir_summarizing import IRSummarizingMemory  # noqa: F401, E402
from .ir_structured_summary import IRStructuredSummary  # noqa: F401, E402
from .ir_naive_rag import IRNaiveRAGMemory  # noqa: F401, E402
from .ir_bm25 import IRBM25Memory  # noqa: F401, E402

# A-MEM is an optional dependency (sentence-transformers, scikit-learn).
# Soft-fail so the rest of the registry still loads when it's missing.
try:
    from .ir_amem import IRAMemMemory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_amem unavailable (pip install -r requirements-amem.txt to enable): %s", _e
    )

# LightMem is an optional dependency (pip install lightmem, in a Python<3.12 venv).
# Soft-fail so the rest of the registry still loads when it's missing.
try:
    from .ir_lightmem import IRLightMemMemory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_lightmem unavailable (pip install lightmem in a Python<3.12 venv to enable): %s", _e
    )

# HippoRAG / SimpleMem are heavy optional deps (gated behind requirements-amem.txt).
# Soft-fail so the rest of the registry still loads when they are missing.
try:
    from .ir_hipporag import IRHippoRAGMemory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_hipporag unavailable (pip install hipporag to enable): %s", _e
    )

try:
    from .ir_simplemem import IRSimpleMemMemory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_simplemem unavailable (pip install simplemem to enable): %s", _e
    )

try:
    from .ir_mem0 import IRMem0Memory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_mem0 unavailable (pip install mem0ai to enable): %s", _e
    )

try:
    from .ir_memorybank import IRMemoryBankMemory  # noqa: F401, E402
except ImportError as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ir_memorybank unavailable (requires numpy + sentence-transformers): %s", _e
    )
