"""
Token tracking utility for MemGym.

This module provides functionality to track token usage in conversation histories,
enabling analysis of context window size and informing decisions about when to
apply context management strategies.
"""

from typing import Dict, List, Tuple, Optional
import tiktoken


class TokenTracker:
    """
    Track token usage in conversation histories.

    This class monitors the number of tokens in observations throughout
    an episode, providing statistics that can inform decisions about
    context management strategies.

    Features:
    - Count tokens using tiktoken (OpenAI's tokenizer)
    - Track token history across episode steps
    - Provide statistics (current, max, average, etc.)
    - Support different tokenizer models
    """

    def __init__(self, tokenizer_name: str = "gpt-4"):
        """
        Initialize token tracker.

        Args:
            tokenizer_name: Name of the tokenizer model to use
                          (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3")
        """
        self.tokenizer_name = tokenizer_name
        try:
            self._encoding = tiktoken.encoding_for_model(tokenizer_name)
        except KeyError:
            # Fallback to cl100k_base if model not found
            self._encoding = tiktoken.get_encoding("cl100k_base")

        self._history: List[Tuple[int, int]] = []  # List of (step, token_count)
        self._current_step = 0

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def update(self, observation: str, step: Optional[int] = None) -> int:
        """
        Update tracker with new observation and return token count.

        Args:
            observation: The current observation text
            step: Optional step number (auto-increments if not provided)

        Returns:
            Number of tokens in the observation
        """
        if step is None:
            step = self._current_step
            self._current_step += 1

        token_count = self.count_tokens(observation)
        self._history.append((step, token_count))

        return token_count

    def get_current_tokens(self) -> int:
        """Get token count for the most recent observation."""
        if not self._history:
            return 0
        return self._history[-1][1]

    def get_max_tokens(self) -> int:
        """Get maximum token count across all observations."""
        if not self._history:
            return 0
        return max(count for _, count in self._history)

    def get_avg_tokens(self) -> float:
        """Get average token count across all observations."""
        if not self._history:
            return 0.0
        return sum(count for _, count in self._history) / len(self._history)

    def get_stats(self) -> Dict[str, any]:
        """
        Get comprehensive token statistics.

        Returns:
            Dictionary containing:
            - current_tokens: Token count for latest observation
            - max_tokens: Maximum token count seen
            - avg_tokens: Average token count
            - total_observations: Number of observations tracked
            - token_history: Full history of (step, token_count) tuples
            - tokenizer: Name of tokenizer used
        """
        return {
            "current_tokens": self.get_current_tokens(),
            "max_tokens": self.get_max_tokens(),
            "avg_tokens": self.get_avg_tokens(),
            "total_observations": len(self._history),
            "token_history": self._history.copy(),
            "tokenizer": self.tokenizer_name
        }

    def exceeds_threshold(self, threshold: int) -> bool:
        """
        Check if current token count exceeds a threshold.

        Args:
            threshold: Token threshold to check against

        Returns:
            True if current tokens exceed threshold
        """
        return self.get_current_tokens() > threshold

    def get_token_growth_rate(self) -> float:
        """
        Calculate rate of token growth per step.

        Returns:
            Average token increase per step, or 0 if insufficient data
        """
        if len(self._history) < 2:
            return 0.0

        first_step, first_count = self._history[0]
        last_step, last_count = self._history[-1]

        step_diff = last_step - first_step
        if step_diff == 0:
            return 0.0

        return (last_count - first_count) / step_diff

    def reset(self) -> None:
        """Reset the tracker, clearing all history."""
        self._history.clear()
        self._current_step = 0

    def __repr__(self) -> str:
        """String representation of tracker state."""
        return (
            f"TokenTracker(tokenizer={self.tokenizer_name}, "
            f"observations={len(self._history)}, "
            f"current_tokens={self.get_current_tokens()}, "
            f"max_tokens={self.get_max_tokens()})"
        )
