"""
LLM-based summarization memory manager for MemGym.

Uses LLM to summarize historical context when it exceeds token thresholds,
enabling long-term memory testing.
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..base import BaseMemoryManager, FilteredContext, register_memory_model


class SummarizationMemory(BaseMemoryManager):
    """
    Memory manager with LLM-based context summarization.

    When context exceeds token threshold, uses LLM to create
    a concise summary while preserving recent observations.

    Features:
    - Automatic summarization when exceeding max_tokens
    - Preserves recent observations for immediate context
    - Built-in token tracking
    - Configurable LLM model and threshold

    Example:
        >>> memory = SummarizationMemory(max_tokens=2000)
        >>> filtered = memory.manage_context(
        ...     original_context=history,
        ...     current_observation=obs
        ... )
        >>> # filtered.content contains summary + recent if compacted
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
        summarization_model: str = "gpt-4o-mini",
        keep_recent: int = 3
    ):
        """
        Initialize summarization memory manager.

        Args:
            max_tokens: Maximum token count before summarization
            tokenizer_name: Tokenizer for token counting
            config: Optional configuration dictionary
            summarization_model: OpenAI model for summarization
            keep_recent: Number of recent observations to keep after summarization
        """
        super().__init__(max_tokens, tokenizer_name, config)

        self.summarization_model = summarization_model
        self.keep_recent = keep_recent

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = OpenAI(api_key=api_key)

        # Summarization state
        self._summary: Optional[str] = None
        self._observations: List[Any] = []

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Manage context with automatic summarization.

        Internally:
        1. Counts tokens in full context
        2. If exceeds max_tokens, summarizes older content
        3. Returns filtered context (summary + recent + current)

        Args:
            original_context: Complete conversation history
            current_observation: Latest observation
            metadata: Optional step/task info

        Returns:
            FilteredContext with summarized content if needed
        """
        # Add current observation to tracking
        self._observations.append(current_observation)

        # Build full content for token counting
        full_content = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(full_content)

        # Check if compaction needed
        was_compacted = False
        if original_tokens > self.max_tokens:
            self._compact(original_context)
            was_compacted = True

        # Build filtered content
        if self._summary:
            # Summary + recent observations + current
            filtered_content = [
                {"role": "system", "content": f"=== Summary of Previous Context ===\n{self._summary}"},
                *self._observations[-self.keep_recent:]
            ]
        else:
            # No summarization yet, return full content
            filtered_content = full_content

        # Count filtered tokens
        filtered_tokens = self.count_tokens(filtered_content)
        self._update_token_tracking(filtered_tokens)

        compression_ratio = original_tokens / max(1, filtered_tokens) if was_compacted else 1.0

        return FilteredContext(
            content=filtered_content,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": was_compacted,
                "compression_ratio": compression_ratio,
                "strategy": "summarization",
                "has_summary": self._summary is not None,
                "summarization_count": self._compaction_count,
                "recent_observations": len(self._observations[-self.keep_recent:]),
                "step": self._step_count
            }
        )

    def _compact(self, context: List[Any]) -> None:
        """
        Perform summarization compaction.

        Args:
            context: Context to summarize
        """
        if not context and not self._observations:
            return

        # Build content to summarize (older observations, not recent)
        to_summarize = []
        if self._summary:
            to_summarize.append(f"Previous summary:\n{self._summary}")

        # Add older observations (excluding recent ones we want to keep)
        older_obs = self._observations[:-self.keep_recent] if len(self._observations) > self.keep_recent else []
        for i, obs in enumerate(older_obs):
            to_summarize.append(f"Observation {i}: {obs}")

        if not to_summarize:
            return

        full_context = "\n\n".join(to_summarize)

        try:
            response = self._client.chat.completions.create(
                model=self.summarization_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following context concisely, preserving key information and important details."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize:\n\n{full_context}"
                    }
                ],
                max_completion_tokens=500
            )

            self._summary = response.choices[0].message.content.strip()
            self._compaction_count += 1

            # Keep only recent observations
            if len(self._observations) > self.keep_recent:
                self._observations = self._observations[-self.keep_recent:]

        except Exception as e:
            print(f"Warning: Summarization failed: {e}")
            # Fallback: just truncate observations
            if len(self._observations) > self.keep_recent:
                self._observations = self._observations[-self.keep_recent:]

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._summary = None
        self._observations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        # Legacy support
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = super().get_stats()
        # Get current token count
        current_tokens = self._token_count
        base_stats.update({
            "has_summary": self._summary is not None,
            "summarization_model": self.summarization_model,
            "keep_recent": self.keep_recent,
            "current_observations": len(self._observations),
            "current_tokens": current_tokens,
            "recent_observations": len(self._observations[-self.keep_recent:]) if self._observations else 0
        })
        return base_stats


# Legacy alias
SummarizationMemoryModel = SummarizationMemory

# Register in memory model registry
register_memory_model("summarization", SummarizationMemory)
register_memory_model("llm_summarization", SummarizationMemory)
