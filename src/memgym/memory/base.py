"""
Base memory manager abstractions for MemGym.

The memory manager is middleware between environment and agent.
It receives full context, decides what to filter/compact, and returns
filtered context for the agent.

Key design:
- Single `manage_context()` method handles everything (token counting, compaction, filtering)
- Token tracking is built-in (core to compaction decisions)
- Easy to swap implementations for benchmarking different memory systems
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

import tiktoken


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilteredContext:
    """
    Result of context management.

    Contains filtered content for agent and metadata about filtering decisions.
    """
    content: Any  # Filtered content for agent (messages, text, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata typically includes:
    #   - tokens: int (current token count)
    #   - original_tokens: int (before filtering)
    #   - was_compacted: bool
    #   - compression_ratio: float
    #   - strategy: str

    def __repr__(self) -> str:
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        tokens = self.metadata.get("tokens", "?")
        return f"FilteredContext(tokens={tokens}, content={content_preview})"


# =============================================================================
# Base Memory Manager
# =============================================================================

class BaseMemoryManager(ABC):
    """
    Abstract base class for memory managers (middleware layer).

    Memory manager sits between environment and agent:
    - Receives full context (conversation history + current observation)
    - Decides what to filter/compact based on token limits
    - Returns filtered context for agent

    Key features:
    - Single `manage_context()` method handles everything internally
    - Built-in token tracking (core to compaction decisions)
    - Easy to swap implementations for testing different memory systems

    Example:
        >>> memory = PassThroughMemory(max_tokens=4000)
        >>> filtered = memory.manage_context(
        ...     original_context=history,
        ...     current_observation=obs
        ... )
        >>> action = agent.act(filtered)
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize memory manager.

        Args:
            max_tokens: Maximum token limit before compaction
            tokenizer_name: Tokenizer model name for token counting
            config: Optional additional configuration
        """
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        self.config = config or {}

        # Initialize tokenizer
        try:
            self._encoding = tiktoken.encoding_for_model(tokenizer_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

        # Token tracking (built-in)
        self._token_count = 0
        self._token_history: List[int] = []

        # Compaction tracking
        self._compaction_count = 0
        self._step_count = 0

        # Episode-scoped state populated by set_episode_state() after env.reset().
        # Carries fields like task instruction or app name that summarization
        # adapters need to keep summaries task-relevant rather than generic.
        self._episode_state: Dict[str, Any] = {}

    @abstractmethod
    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Single entry point for context management.

        Internally handles:
        1. Count tokens in full context
        2. Decide if compaction needed (based on max_tokens)
        3. Compact if needed (summarize, prune, etc.)
        4. Return filtered context with stats

        Args:
            original_context: Complete conversation history
            current_observation: Latest observation from environment
            metadata: Optional step/task info

        Returns:
            FilteredContext with processed content and token/compaction stats
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset memory state for new episode."""
        pass

    def set_episode_state(self, **state: Any) -> None:
        """Receive episode-scoped state from the runner.

        Called once per episode, immediately after env.reset() returns its
        info dict. Default behavior stores everything in self._episode_state
        so subclasses can read fields without overriding this hook.

        Task-aware summarization adapters (e.g., WebArenaSummarizingMemory)
        override this to capture the task instruction so summaries can stay
        relevant to the goal rather than dumping generic page descriptions.
        """
        self._episode_state = dict(state)

    # =========================================================================
    # Token Tracking (built-in)
    # =========================================================================

    def count_tokens(self, content: Any) -> int:
        """Count tokens in content."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(self._encoding.encode(content))
        elif isinstance(content, list):
            return sum(self.count_tokens(item) for item in content)
        elif isinstance(content, dict):
            # Handle message dicts like {"role": "user", "content": "..."}
            if "content" in content:
                return self.count_tokens(content["content"])
            return self.count_tokens(str(content))
        else:
            return len(self._encoding.encode(str(content)))

    @property
    def token_count(self) -> int:
        """Current token count."""
        return self._token_count

    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining before limit."""
        return max(0, self.max_tokens - self._token_count)

    def _update_token_tracking(self, tokens: int) -> None:
        """Update token tracking state."""
        self._token_count = tokens
        self._token_history.append(tokens)
        self._step_count += 1

    # =========================================================================
    # Stats for Benchmarking
    # =========================================================================

    # =========================================================================
    # Legacy API (backward compatibility)
    # =========================================================================

    def process_observation(self, observation: Any, metadata: Optional[Dict[str, Any]] = None) -> "MemoryAction":
        """Legacy method: Process a new observation. Use manage_context() instead."""
        if not hasattr(self, '_legacy_observations'):
            self._legacy_observations = []
        tokens_before = self._token_count
        self._legacy_observations.append(observation)
        filtered = self.manage_context(
            original_context=self._legacy_observations[:-1],
            current_observation=observation,
            metadata=metadata
        )
        tokens_after = filtered.metadata.get("tokens", 0)
        return MemoryAction(
            action_type="store" if not filtered.metadata.get("was_compacted") else "summarize",
            metadata={
                "total_observations": len(self._legacy_observations),
                "tokens": tokens_after,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "exceeded_threshold": tokens_before > self.max_tokens,
                "summarization_count": self._compaction_count,
                **(metadata or {})
            }
        )

    def get_context(self) -> "Context":
        """Legacy method: Get context for reasoning model. Use manage_context() instead."""
        observations = getattr(self, '_legacy_observations', [])
        if not observations:
            return Context(content="", metadata={"num_observations": 0, "strategy": "passthrough"})
        filtered = self.manage_context(
            original_context=observations[:-1] if len(observations) > 1 else [],
            current_observation=observations[-1] if observations else "",
            metadata=None
        )
        content = filtered.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return Context(
            content=content,
            metadata={
                "num_observations": len(observations),
                "strategy": filtered.metadata.get("strategy", "unknown"),
                **filtered.metadata
            }
        )

    def should_use_filtered_context(self) -> bool:
        """Legacy method: Check if filtered context should be used."""
        return self._compaction_count > 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics for logging/benchmarking.

        Returns comprehensive stats useful for comparing memory systems.
        """
        # Legacy observation count
        legacy_obs = getattr(self, '_legacy_observations', [])
        num_observations = len(legacy_obs)

        return {
            "model_type": self.__class__.__name__,
            "max_tokens": self.max_tokens,
            "token_count": self._token_count,
            "tokens_remaining": self.tokens_remaining,
            "compaction_count": self._compaction_count,
            "summarization_count": self._compaction_count,  # Legacy alias
            "step_count": self._step_count,
            "num_observations": num_observations,  # Legacy key
            "total_observations": num_observations,  # Legacy key
            "max_tokens_seen": max(self._token_history) if self._token_history else 0,
            "avg_tokens": sum(self._token_history) / len(self._token_history) if self._token_history else 0,
            "tokenizer": self.tokenizer_name,
            "config": self.config
        }


# =============================================================================
# PassThrough Memory (Baseline)
# =============================================================================

class PassThroughMemory(BaseMemoryManager):
    """
    Simple pass-through memory manager (baseline).

    Returns full context without any filtering or compaction.
    Useful as baseline to compare against more complex memory strategies.

    Note: Does NOT enforce max_tokens - just tracks them.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(max_tokens, tokenizer_name, config)
        self._observations: List[Any] = []

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Return full context without filtering.

        Just passes through all content, tracking tokens but not enforcing limits.
        """
        # Build full content
        full_content = list(original_context) + [current_observation]

        # Count tokens
        tokens = self.count_tokens(full_content)
        self._update_token_tracking(tokens)

        return FilteredContext(
            content=full_content,
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "compression_ratio": 1.0,
                "direct_messages": True,
                "strategy": "passthrough",
                "step": self._step_count
            }
        )

    def reset(self) -> None:
        """Reset memory state."""
        self._observations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        # Legacy support
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()


# =============================================================================
# Memory Manager Registry
# =============================================================================

_MEMORY_REGISTRY: Dict[str, Type[BaseMemoryManager]] = {
    "passthrough": PassThroughMemory,
    "pass_through": PassThroughMemory,
    "baseline": PassThroughMemory,
}


def register_memory_model(name: str, model_class: Type[BaseMemoryManager]) -> None:
    """Register a memory manager class."""
    _MEMORY_REGISTRY[name.lower()] = model_class


def get_memory_model(name: str, **kwargs) -> BaseMemoryManager:
    """
    Get a memory manager instance by name.

    Args:
        name: Model name ("passthrough", "summarization", etc.)
        **kwargs: Arguments passed to model constructor

    Returns:
        Memory manager instance
    """
    name_lower = name.lower()
    if name_lower not in _MEMORY_REGISTRY:
        available = list(_MEMORY_REGISTRY.keys())
        raise ValueError(f"Unknown memory model: {name}. Available: {available}")
    return _MEMORY_REGISTRY[name_lower](**kwargs)


def list_memory_models() -> List[str]:
    """List all registered memory model names."""
    return list(_MEMORY_REGISTRY.keys())


# =============================================================================
# Legacy API (backward compatibility for Tau2/SWE envs, adapters, A-mem)
# =============================================================================

@dataclass
class MemoryAction:
    """Legacy: Action taken by memory model."""
    action_type: str
    metadata: Dict[str, Any]


@dataclass
class Context:
    """Legacy: Context for reasoning model."""
    content: Any
    metadata: Dict[str, Any]


# =============================================================================
# Tool Call Pair Repair (defense-in-depth for memory condensers)
# =============================================================================

def repair_tool_call_pairs(messages: list, pending_tool_ids: set | None = None) -> list:
    """
    Remove orphaned tool_result messages and fix broken tool_call pairs.

    After memory condensation, tool_call (assistant) / tool_result (tool) pairs
    can be broken when one half is in the forgotten range. Bedrock and other APIs
    require every tool_result to reference a tool_call in the immediately preceding
    assistant message, and every tool_call to have a subsequent tool_result.

    This function:
    1. Drops tool messages whose tool_call_id doesn't match any tool_call
       in the preceding assistant message
    2. Strips orphaned tool_calls entries from assistant messages that lost
       their tool_result responses

    Args:
        messages: Message dicts to repair.
        pending_tool_ids: Tool call IDs whose results are NOT yet in the
            message list but will be appended by the caller (e.g. the
            current observation that tau2 will append after this returns).
            These IDs are treated as "matched" in pass 2 so the
            corresponding tool_calls are preserved.

    Safe to call on any message list (no-op if pairs are already valid).
    """
    if not messages:
        return messages

    result = []
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else None

        if role == "tool":
            # Check if there's a matching tool_call in the preceding assistant message
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                # Tool message without tool_call_id can't be matched to any
                # tool_call — drop it to avoid Bedrock/API validation errors.
                continue
            if result:
                # Find the most recent assistant message before this
                matched = False
                for prev in reversed(result):
                    prev_role = prev.get("role") if isinstance(prev, dict) else None
                    if prev_role == "assistant":
                        tool_calls = prev.get("tool_calls") or []
                        if any(tc.get("id") == tool_call_id for tc in tool_calls):
                            matched = True
                        break  # Only check the most recent assistant message
                    elif prev_role == "tool":
                        continue  # Skip other tool messages in same batch
                    else:
                        break  # Hit a non-assistant, non-tool message
                if not matched:
                    # Orphaned tool_result — skip it
                    continue
            else:
                # No preceding messages at all — tool result is orphaned
                continue
            result.append(msg)
        else:
            result.append(msg)

    # Second pass: strip orphaned tool_calls from assistant messages
    # (tool_calls whose tool_results were dropped)
    tool_result_ids = set(pending_tool_ids or ())
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid:
                tool_result_ids.add(tid)

    cleaned = []
    for msg in result:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Filter to only tool_calls that have matching results
            original_calls = msg["tool_calls"]
            valid_calls = [tc for tc in original_calls if tc.get("id") in tool_result_ids]
            if valid_calls != original_calls:
                msg = dict(msg)  # Shallow copy to avoid mutating original
                if valid_calls:
                    msg["tool_calls"] = valid_calls
                else:
                    # All tool_calls orphaned — remove the field entirely
                    msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    # If content is empty/None, add placeholder so message isn't empty
                    if not msg.get("content"):
                        msg["content"] = "(continued)"
        cleaned.append(msg)

    return cleaned


# Legacy aliases
BaseMemoryModel = BaseMemoryManager
PassThroughMemoryModel = PassThroughMemory
