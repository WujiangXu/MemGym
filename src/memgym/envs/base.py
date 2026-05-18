"""
Base environment abstractions for MemGym.

Provides BaseMemoryEnvironment and environment registry for easy loading.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field

import gymnasium as gym


# =============================================================================
# Memory State Classes
# =============================================================================

@dataclass
class MemorySnapshot:
    """A snapshot of memory state at a specific point in time."""
    step: int
    timestamp: float
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemoryState(ABC):
    """Abstract base class for memory state management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._snapshots: List[MemorySnapshot] = []
        self._current_step = 0

    @abstractmethod
    def update(self, observation: Any, action: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update memory state with new observation and action."""
        pass

    @abstractmethod
    def retrieve(self, query: Any, k: int = 5) -> List[Any]:
        """Retrieve relevant memories based on a query."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current memory state as a dictionary."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the memory state to initial conditions."""
        pass

    def snapshot(self, timestamp: float) -> MemorySnapshot:
        """Create a snapshot of the current memory state."""
        snapshot = MemorySnapshot(
            step=self._current_step,
            timestamp=timestamp,
            content=self.get_state()
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all memory snapshots."""
        return self._snapshots.copy()


class SimpleMemoryState(BaseMemoryState):
    """Simple implementation of memory state using a list buffer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_capacity = self.config.get("max_capacity", 1000)
        self._buffer: List[Dict[str, Any]] = []

    def update(self, observation: Any, action: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update memory buffer with new observation and action."""
        item = {
            "step": self._current_step,
            "observation": observation,
            "action": action,
            "metadata": metadata or {}
        }
        self._buffer.append(item)
        if len(self._buffer) > self.max_capacity:
            self._buffer.pop(0)
        self._current_step += 1

    def retrieve(self, query: Any, k: int = 5) -> List[Any]:
        """Retrieve the k most recent items."""
        return self._buffer[-k:] if self._buffer else []

    def get_state(self) -> Dict[str, Any]:
        """Get current memory state."""
        return {
            "buffer": self._buffer.copy(),
            "current_step": self._current_step,
            "capacity": self.max_capacity
        }

    def reset(self) -> None:
        """Reset memory state."""
        self._buffer.clear()
        self._snapshots.clear()
        self._current_step = 0


# =============================================================================
# Base Environment
# =============================================================================

class BaseMemoryEnvironment(gym.Env, ABC):
    """
    Abstract base class for memory-testing environments.

    All concrete memory environments should inherit from this class.
    Provides memory tracking, trajectory recording, and standard gym interface.

    Example:
        >>> env = get_env("tau2", domain="airline", task_id="book_flight_1")
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        memory_state: Optional[BaseMemoryState] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.config = config or {}
        self._memory_state = memory_state or SimpleMemoryState(config)
        self._episode_step = 0
        self._total_steps = 0

    @property
    def memory_state(self) -> BaseMemoryState:
        """Get the current memory state."""
        return self._memory_state

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and memory state."""
        super().reset(seed=seed)
        self._episode_step = 0

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self._episode_step += 1
        self._total_steps += 1

    @abstractmethod
    def _calculate_reward(self, observation: Any, action: Any, info: Dict[str, Any]) -> float:
        """Calculate the reward for the current step."""
        pass

    @abstractmethod
    def _evaluate_memory(self) -> Dict[str, Any]:
        """Evaluate memory-specific performance."""
        pass

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get current memory performance metrics."""
        return self._evaluate_memory()

    def reset_memory(self) -> None:
        """Reset only the memory state, not the entire environment."""
        self._memory_state.reset()

    @property
    def episode_step(self) -> int:
        """Get the current step number in this episode."""
        return self._episode_step

    @property
    def total_steps(self) -> int:
        """Get the total number of steps across all episodes."""
        return self._total_steps

    # =========================================================================
    # Trajectory methods (to be implemented by subclasses)
    # =========================================================================

    def get_trajectory(self) -> Dict[str, Any]:
        """Get recorded trajectory. Override in subclass."""
        return {}

    def save_trajectory(self, filepath: str) -> None:
        """Save trajectory to file. Override in subclass."""
        pass


# =============================================================================
# Environment Registry
# =============================================================================

_ENV_REGISTRY: Dict[str, Type[BaseMemoryEnvironment]] = {}


def register_env(name: str, env_class: Type[BaseMemoryEnvironment]) -> None:
    """Register an environment class."""
    _ENV_REGISTRY[name.lower()] = env_class


def get_env(name: str, **kwargs) -> BaseMemoryEnvironment:
    """
    Get an environment instance by name.

    Args:
        name: Environment name ("tau2", "swe", etc.)
        **kwargs: Arguments passed to environment constructor

    Returns:
        Environment instance

    Example:
        >>> env = get_env("tau2", domain="airline", task_id="book_flight_1")
        >>> env = get_env("swe", dataset="princeton-nlp/SWE-bench_Lite")
    """
    name_lower = name.lower()
    if name_lower not in _ENV_REGISTRY:
        available = list(_ENV_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name}. Available: {available}")
    return _ENV_REGISTRY[name_lower](**kwargs)


def list_envs() -> List[str]:
    """List all registered environment names."""
    return list(_ENV_REGISTRY.keys())
