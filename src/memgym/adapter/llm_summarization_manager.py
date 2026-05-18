"""
LLM-based context summarization for MemGym.

This module provides context management using LLM summarization
when historical context exceeds token threshold.
"""

from typing import Any, Dict, List, Optional
import os
from openai import OpenAI

from memgym.utility.token_tracker import TokenTracker


class LLMSummarizationContextManager:
    """
    Context manager that uses LLM to summarize when exceeding token threshold.

    When historical context exceeds max_tokens, calls LLM to generate
    a concise summary, then replaces full history with the summary.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000,
        summarization_model: str = "gpt-5-nano",
        keep_recent: int = 3
    ):
        """
        Initialize LLM summarization context manager.

        Args:
            config: Optional configuration dictionary
            max_tokens: Maximum token count before summarization
            summarization_model: OpenAI model for summarization
            keep_recent: Number of recent observations to keep after summarization
        """
        self.config = config or {}
        self.max_tokens = max_tokens
        self.summarization_model = summarization_model
        self.keep_recent = keep_recent

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

        # Token tracker
        self.token_tracker = TokenTracker(tokenizer_name="gpt-4")

        # Storage
        self._observations: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None
        self._summarization_count = 0
        self._total_observations = 0

    def add_observation(self, observation: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add observation and trigger summarization if needed."""
        obs_str = str(observation)
        token_count = self.token_tracker.count_tokens(obs_str)

        self._observations.append({
            "content": observation,
            "metadata": metadata or {},
            "tokens": token_count
        })
        self._total_observations += 1

        # Check if we need to summarize
        current_tokens = self._get_current_token_count()
        if current_tokens > self.max_tokens:
            self._summarize()

    def get_context(self) -> str:
        """Get current context (summary + recent observations)."""
        parts = []

        if self._summary:
            parts.append(f"=== Summary ===\n{self._summary}\n")

        if self._observations:
            parts.append("=== Recent ===")
            for i, obs in enumerate(self._observations):
                parts.append(f"[{i}] {obs['content']}")

        return "\n".join(parts) if parts else ""

    def reset(self) -> None:
        """Reset context manager state."""
        self._observations.clear()
        self._summary = None
        self._summarization_count = 0
        self._total_observations = 0
        self.token_tracker = TokenTracker(tokenizer_name="gpt-4")

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about context state."""
        return {
            "manager_type": self.__class__.__name__,
            "max_tokens": self.max_tokens,
            "current_tokens": self._get_current_token_count(),
            "summarization_count": self._summarization_count,
            "total_observations": self._total_observations,
            "recent_observations": len(self._observations),
            "has_summary": self._summary is not None,
            "summarization_model": self.summarization_model
        }

    def _get_current_token_count(self) -> int:
        """Calculate current total token count."""
        total = 0
        if self._summary:
            total += self.token_tracker.count_tokens(self._summary)
        for obs in self._observations:
            total += obs["tokens"]
        return total

    def _summarize(self) -> None:
        """Summarize historical context using LLM."""
        if not self._observations:
            return

        # Build prompt
        obs_text = [f"Observation {i}: {obs['content']}" for i, obs in enumerate(self._observations)]

        context_parts = []
        if self._summary:
            context_parts.append(f"Previous summary:\n{self._summary}\n")
        context_parts.append("New observations:\n" + "\n".join(obs_text))
        full_context = "\n".join(context_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.summarization_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following context concisely, preserving key information."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize:\n\n{full_context}"
                    }
                ],
                max_completion_tokens=500
            )

            self._summary = response.choices[0].message.content.strip()

            # Keep only recent observations
            if len(self._observations) > self.keep_recent:
                self._observations = self._observations[-self.keep_recent:]

            self._summarization_count += 1

        except Exception as e:
            print(f"Warning: Summarization failed: {e}")
            if len(self._observations) > self.keep_recent:
                self._observations = self._observations[-self.keep_recent:]
