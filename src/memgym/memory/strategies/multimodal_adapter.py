"""
Multimodal memory adapter for CUA agents.

Wraps any text-based memory manager and adds screenshot handling.
Screenshots are kept only for the most recent `screenshot_window` steps;
older screenshots are replaced with '[screenshot omitted]' or optionally
an LLM-generated text description.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMemoryManager, FilteredContext, register_memory_model


class MultimodalMemoryAdapter(BaseMemoryManager):
    """
    Adapter that wraps a text-based memory manager for multimodal CUA contexts.

    The CUA agent history consists of (text, screenshot_b64) pairs per step.
    This adapter:
    1. Extracts text portions and delegates to the inner text memory manager
    2. Manages screenshots separately: keeps only the most recent ones
    3. Returns a combined FilteredContext with managed text + managed screenshots

    Usage:
        >>> from memory import LLMSummarizingMemory
        >>> text_mem = LLMSummarizingMemory(max_size=20)
        >>> mm_mem = MultimodalMemoryAdapter(text_memory=text_mem, screenshot_window=3)
        >>> filtered = mm_mem.manage_context(history, current_obs)
    """

    def __init__(
        self,
        text_memory: Optional[BaseMemoryManager] = None,
        screenshot_window: int = 2,
        describe_dropped: bool = False,
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            text_memory: Inner text-based memory manager. If None, uses PassThrough.
            screenshot_window: Number of recent screenshots to keep.
            describe_dropped: If True, generate LLM text descriptions for dropped
                screenshots. If False (default), use '[screenshot omitted]'.
            max_tokens: Token limit (passed to base and inner memory).
            tokenizer_name: Tokenizer for token counting.
            config: Additional config.
        """
        super().__init__(max_tokens, tokenizer_name, config)

        if text_memory is None:
            from ..base import PassThroughMemory
            text_memory = PassThroughMemory(max_tokens=max_tokens)

        self.text_memory = text_memory
        self.screenshot_window = screenshot_window
        self.describe_dropped = describe_dropped

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """
        Manage multimodal context (text + screenshots).

        Args:
            original_context: List of dicts with 'text' and 'screenshot' keys,
                representing the agent's history of (observation+action, screenshot) pairs.
            current_observation: Current step's observation (dict with 'text', 'screenshot').
            metadata: Optional metadata (step number, etc.).

        Returns:
            FilteredContext where content is a dict with:
                - 'history': list of {'text': str, 'screenshot': str|None} dicts
                - 'was_compacted': bool
                - Text is managed by the inner memory; screenshots are windowed.
        """
        # Extract text-only history for the inner memory manager
        text_history = []
        for item in original_context:
            if isinstance(item, dict):
                text_history.append(item.get("text", str(item)))
            else:
                text_history.append(str(item))

        current_text = ""
        if isinstance(current_observation, dict):
            current_text = current_observation.get("text", str(current_observation))
        else:
            current_text = str(current_observation)

        # Delegate text management to inner memory
        text_filtered = self.text_memory.manage_context(
            original_context=text_history,
            current_observation=current_text,
            metadata=metadata,
        )

        # Get the filtered text content
        filtered_texts = text_filtered.content
        if not isinstance(filtered_texts, list):
            filtered_texts = [filtered_texts]

        # Determine total steps (original + current)
        all_items = list(original_context) + [current_observation]
        total_steps = len(all_items)

        # Build screenshot list: keep only the last `screenshot_window` screenshots
        screenshots = []
        for i, item in enumerate(all_items):
            if isinstance(item, dict) and "screenshot" in item:
                screenshots.append(item["screenshot"])
            else:
                screenshots.append(None)

        # Apply screenshot windowing
        managed_screenshots = []
        screenshots_sent = 0
        screenshots_dropped = 0
        for i, ss in enumerate(screenshots):
            if i >= total_steps - self.screenshot_window:
                # Keep recent screenshots
                managed_screenshots.append(ss)
                if ss is not None:
                    screenshots_sent += 1
            else:
                # Drop older screenshots
                if ss is not None:
                    screenshots_dropped += 1
                    if self.describe_dropped:
                        managed_screenshots.append("[screenshot: see observation text above]")
                    else:
                        managed_screenshots.append(None)
                else:
                    managed_screenshots.append(None)

        # Combine filtered text + managed screenshots
        # The output length matches the filtered text length
        # If text memory compressed/truncated, we align screenshots accordingly
        managed_history = []
        for i, text in enumerate(filtered_texts):
            ss = managed_screenshots[i] if i < len(managed_screenshots) else None
            managed_history.append({"text": text, "screenshot": ss})

        # Update tracking
        text_tokens = text_filtered.metadata.get("tokens", 0)
        self._update_token_tracking(text_tokens)

        was_compacted = text_filtered.metadata.get("was_compacted", False)
        if screenshots_dropped > 0:
            was_compacted = True

        return FilteredContext(
            content={
                "history": managed_history,
                "text_filtered": text_filtered,
            },
            metadata={
                "tokens": text_tokens,
                "original_tokens": text_filtered.metadata.get("original_tokens", text_tokens),
                "was_compacted": was_compacted,
                "compression_ratio": text_filtered.metadata.get("compression_ratio", 1.0),
                "strategy": f"multimodal({self.text_memory.__class__.__name__})",
                "screenshots_sent": screenshots_sent,
                "screenshots_dropped": screenshots_dropped,
                "screenshot_window": self.screenshot_window,
                "step": self._step_count,
            },
        )

    def reset(self) -> None:
        """Reset both this adapter and the inner text memory."""
        self.text_memory.reset()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from adapter and inner memory."""
        stats = super().get_stats()
        stats["inner_memory_stats"] = self.text_memory.get_stats()
        stats["screenshot_window"] = self.screenshot_window
        stats["describe_dropped"] = self.describe_dropped
        return stats


# Register in the memory model registry
register_memory_model("multimodal", MultimodalMemoryAdapter)
