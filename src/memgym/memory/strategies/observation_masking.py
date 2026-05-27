"""
Observation masking memory manager for MemGym.

Exact port of OpenHands ObservationMaskingCondenser.
Replaces the content of old observations (user messages) outside of a recent
attention window with a '<MASKED>' placeholder.

This is a cheap, no-LLM strategy that significantly reduces token usage
by masking old command outputs while preserving recent context.

Source: openhands/memory/condenser/impl/observation_masking_condenser.py
"""

import copy
import re
from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model

# Patterns that indicate error signals worth preserving
_ERROR_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)"
    r"|Error:|error:"
    r"|FAILED|FAIL"
    r"|AssertionError"
    r"|Exception:"
    r"|raise \w+Error"
    r"|pytest.*ERRORS?"
    r"|assert .+ ==",
    re.IGNORECASE,
)


class ObservationMaskingMemory(BaseMemoryManager):
    """
    A memory manager that masks the content of observations outside of a recent attention window.

    Based on OpenHands ObservationMaskingCondenser, with selective masking:
    observations containing error signals (tracebacks, assertion failures, test
    failures) are preserved even outside the attention window.

    Args:
        attention_window: Number of recent messages to leave unmasked.
            Default: 100 (same as OpenHands default).
        keep_first: Number of initial messages to always preserve (system prompt, task).
            Default: 1 (same as OpenHands default).
        selective: If True, preserve observations containing error signals even
            outside the attention window. Default: True.
    """

    def __init__(
        self,
        attention_window: int = 100,
        keep_first: int = 1,
        selective: bool = True,
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(max_tokens, tokenizer_name, config)
        self.attention_window = attention_window
        self.keep_first = keep_first
        self.selective = selective

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Replace the content of observations outside of the attention window with '<MASKED>'.

        Exact port of OpenHands ObservationMaskingCondenser.condense():
            for i, event in enumerate(view):
                if isinstance(event, Observation) and i < len(view) - self.attention_window:
                    results.append(AgentCondensationObservation('<MASKED>'))
                else:
                    results.append(event)

        In MemGym:
        - Observation = message with role == "user"
        - AgentCondensationObservation('<MASKED>') = {"role": "user", "content": "<MASKED>"}
        """
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)

        # Apply masking (same logic as OpenHands)
        # Never mask: first keep_first messages OR last attention_window messages
        results = []
        any_masked = False
        for i, msg in enumerate(messages):
            # Skip masking if in protected prefix or suffix
            in_prefix = i < self.keep_first
            in_suffix = i >= len(messages) - self.attention_window

            if (isinstance(msg, dict)
                    and msg.get("role") in ("user", "tool")
                    and not in_prefix
                    and not in_suffix):
                # Selective masking: preserve observations with error signals
                content = str(msg.get("content", ""))
                if self.selective and _ERROR_PATTERNS.search(content):
                    results.append(msg)
                    continue

                # Mask this observation
                masked_msg = copy.copy(msg)
                masked_msg["content"] = "<MASKED>"
                # Keep role="tool" and tool_call_id intact — API requires
                # every assistant tool_calls entry to have a matching tool
                # response with the same tool_call_id.
                results.append(masked_msg)
                any_masked = True
            else:
                results.append(msg)

        filtered_tokens = self.count_tokens(results)
        self._update_token_tracking(filtered_tokens)
        if any_masked:
            self._compaction_count += 1

        return FilteredContext(
            content=results,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": any_masked,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "observation_masking",
                "attention_window": self.attention_window,
                "step": self._step_count
            }
        )

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()


# Register in memory model registry
register_memory_model("observation_masking", ObservationMaskingMemory)
