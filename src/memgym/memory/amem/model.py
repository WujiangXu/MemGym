"""
AMemMemoryModel: MemGym interface for A-mem.

This module provides the MemGym-compatible interface for the A-mem
(Agentic Memory) system.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import base classes directly to avoid circular import through memory/__init__.py
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

from base import BaseMemoryModel, Context, FilteredContext, MemoryAction, register_memory_model

# Support both package and direct imports
try:
    from .system import AgenticMemorySystem
except ImportError:
    from system import AgenticMemorySystem


class AMemMemoryModel(BaseMemoryModel):
    """
    A-mem (Agentic Memory) wrapper for MemGym's memory interface.

    Uses LLM-based memory evolution with semantic retrieval.
    Memories are stored with auto-generated keywords, tags, and context.
    Related memories are linked using Zettelkasten principles.

    Supports threshold-based context compression:
    - Below threshold: pass-through (full history)
    - Above threshold: LLM generates query -> retrieve -> combine with recent turns

    Config options:
        llm_model: LLM model for memory evolution (default: gpt-4o-mini)
        embedding_model: Embedding model for retrieval (default: all-MiniLM-L6-v2)
        retrieve_k: Number of memories to retrieve (default: 5)
        api_base: LiteLLM API base URL (default: None)
        api_key: API key (default: None, uses env var)
        evo_threshold: Memory consolidation threshold (default: 100)
        enable_evolution: Whether to run memory evolution (default: True)
        max_context_tokens: Token threshold for compression (default: 8000)
        recent_turns: Number of recent turns to keep (default: 3)
        verbose: Print debug information (default: False)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize A-mem memory model.

        Args:
            config: Configuration dict with options above
        """
        config = config or {}
        max_tokens = config.get("max_context_tokens", 8000)
        super().__init__(max_tokens=max_tokens, config=config)

        # Extract config
        self._llm_model = self.config.get("llm_model", "gpt-4o-mini")
        self._embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self._retrieve_k = self.config.get("retrieve_k", 5)
        self._api_base = self.config.get("api_base", None)
        self._api_key = self.config.get("api_key", None)
        self._evo_threshold = self.config.get("evo_threshold", 100)
        self._enable_evolution = self.config.get("enable_evolution", True)
        self._max_context_tokens = self.config.get("max_context_tokens", 8000)
        self._recent_turns = self.config.get("recent_turns", 3)
        self._verbose = self.config.get("verbose", False)

        # Initialize system
        self._memory_system = self._create_system()

        # Track state
        self._observation_count = 0
        self._last_observation = ""
        self._current_tokens = 0  # Track estimated token count

    def _create_system(self) -> AgenticMemorySystem:
        """Create the underlying A-mem system."""
        return AgenticMemorySystem(
            model_name=self._embedding_model,
            llm_model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
            evo_threshold=self._evo_threshold,
            enable_evolution=self._enable_evolution,
            retrieve_k=self._retrieve_k,
            max_context_tokens=self._max_context_tokens,
            recent_turns=self._recent_turns,
            verbose=self._verbose
        )

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """Implement abstract manage_context via process_observation + get_context."""
        self.process_observation(current_observation, metadata)
        ctx = self.get_context()
        content = ctx.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        tokens = self._memory_system.estimate_tokens(str(content))
        original_tokens = self._current_tokens
        return FilteredContext(
            content=content,
            metadata={
                "tokens": tokens,
                "original_tokens": original_tokens,
                "was_compacted": ctx.metadata.get("compressed", False),
                "strategy": "amem",
                **ctx.metadata,
            }
        )

    def set_task_description(self, task: str) -> None:
        """Set the current task description for query generation."""
        self._memory_system.set_task_description(task)

    def process_observation(
        self,
        observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryAction:
        """
        Process observation by adding to A-mem.

        A-mem will:
        1. Generate keywords, tags, context via LLM
        2. Find related memories
        3. Potentially evolve/link memories

        Args:
            observation: New observation (will be converted to string)
            metadata: Optional metadata (timestamp, role, etc.)

        Returns:
            MemoryAction describing what was done
        """
        content = str(observation)
        timestamp = metadata.get("timestamp") if metadata else None

        # Add to A-mem
        note_id = self._memory_system.add_note(content, time=timestamp)

        self._observation_count += 1
        self._last_observation = content

        # Update token estimate
        self._current_tokens += self._memory_system.estimate_tokens(content)

        self._last_action = MemoryAction(
            action_type="store_and_evolve",
            metadata={
                "note_id": note_id,
                "total_memories": len(self._memory_system.memories),
                "observation_count": self._observation_count,
                "current_tokens": self._current_tokens,
                **(metadata or {})
            }
        )
        return self._last_action

    def get_context(self, current_tokens: int = None) -> Context:
        """
        Get context using threshold-based compression.

        Below threshold: returns full history (pass-through)
        Above threshold: LLM generates query -> retrieve -> combine with recent

        Args:
            current_tokens: Current token count (if None, uses internal estimate)

        Returns:
            Context with memories and metadata
        """
        if not self._memory_system.memories:
            self._last_context = Context(
                content="",
                metadata={"strategy": "amem", "num_memories": 0, "compressed": False}
            )
            return self._last_context

        # Use provided token count or internal estimate
        tokens = current_tokens if current_tokens is not None else self._current_tokens

        # Get context with threshold-based compression
        content, metadata = self._memory_system.get_context_with_threshold(tokens)

        metadata.update({
            "num_memories": len(self._memory_system.memories),
            "retrieve_k": self._retrieve_k,
            "has_summary": metadata.get("compressed", False),
            "recent_turns": self._recent_turns
        })

        self._last_context = Context(content=content, metadata=metadata)
        return self._last_context

    def should_use_filtered_context(self) -> bool:
        """
        Determine if filtered context should be used.

        Returns True when:
        1. Context has been compressed (above threshold)
        2. Or there are memories to retrieve from

        This allows pass-through mode when below threshold.
        """
        if self._last_context and self._last_context.metadata.get("compressed"):
            return True
        # Below threshold: don't filter (use full history)
        return self._current_tokens >= self._max_context_tokens

    def reset(self) -> None:
        """Reset memory system for new episode."""
        self._memory_system = self._create_system()
        self._observation_count = 0
        self._last_observation = ""
        self._current_tokens = 0
        self._last_action = None
        self._last_context = None

    def get_stats(self) -> Dict[str, Any]:
        """Get A-mem statistics."""
        base_stats = super().get_stats()
        system_stats = self._memory_system.get_stats()
        base_stats.update({
            "num_memories": system_stats["num_memories"],
            "observation_count": self._observation_count,
            "evolution_count": system_stats["evolution_count"],
            "current_tokens": self._current_tokens,
            "max_context_tokens": self._max_context_tokens,
            "llm_model": self._llm_model,
            "embedding_model": self._embedding_model,
            "retrieve_k": self._retrieve_k,
            "recent_turns": self._recent_turns,
            "enable_evolution": self._enable_evolution
        })
        return base_stats

    @property
    def system(self) -> AgenticMemorySystem:
        """Access the underlying A-mem system for advanced operations."""
        return self._memory_system


# Register with MemGym
register_memory_model("amem", AMemMemoryModel)
register_memory_model("a-mem", AMemMemoryModel)
register_memory_model("agentic-memory", AMemMemoryModel)
