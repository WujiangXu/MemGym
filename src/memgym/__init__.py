"""
MemGym: A modular framework for testing memory abilities of AI agents.

MemGym provides a Memory-Reasoning separation architecture for evaluating
agent memory across different benchmarks (SWE-bench, WebArena, tau2-bench, etc.).

Quick Start:
    >>> from memgym import get_env, get_agent, get_memory_model
    >>>
    >>> # Create environment
    >>> env = get_env("swe", dataset="princeton-nlp/SWE-bench_Lite")
    >>> # Or: env = get_env("webarena", task_id="...")
    >>>
    >>> # Run episode
    >>> obs, info = env.reset(seed=42)
    >>> obs, reward, done, _, info = env.step(action)
    >>>
    >>> # Get trajectory
    >>> trajectory = env.get_trajectory()
"""

# =============================================================================
# Environments
# =============================================================================
from memgym.envs import (
    # Base classes
    BaseMemoryEnvironment,
    BaseMemoryState,
    SimpleMemoryState,
    # Environments
    SWEMemoryEnv,
    SWETask,
    # Registry
    get_env,
    register_env,
    list_envs,
)

# =============================================================================
# Memory Managers
# =============================================================================
from memgym.memory import (
    # New interface
    BaseMemoryManager,
    PassThroughMemory,
    SummarizationMemory,
    FilteredContext,
    # Legacy aliases
    BaseMemoryModel,
    PassThroughMemoryModel,
    SummarizationMemoryModel,
    MemoryAction,
    Context,
    # Registry
    get_memory_model,
    register_memory_model,
    list_memory_models,
)

# =============================================================================
# Agents (formerly Reasoning Models)
# =============================================================================
from memgym.agents import (
    # New interface
    BaseAgent,
    AgentAction,
    SWEAgent,
    SWEAgentTracker,
    get_agent,
    register_agent,
    list_agents,
    # Legacy aliases
    BaseReasoningModel,
    ReasoningAction,
    SWEReasoningModel,
    get_reasoning_model,
    register_reasoning_model,
    list_reasoning_models,
)

# =============================================================================
# Runners (Experiment Utilities)
# =============================================================================
# New runners module
from memgym.runners import (
    BaseRunner,
    SimpleRunner,
    SWERunner,
)

# Legacy runner module (for backward compatibility)
from memgym.runner import (
    EpisodeRunner,
    TrajectoryCollector,
)

# =============================================================================
# Adapters & Utilities
# =============================================================================
from memgym.adapter import TrajectoryRecorder
from memgym.utility import TokenTracker

# =============================================================================
# Types
# =============================================================================
from .types import (
    TopicType,
    MemoryTestType,
    MetricType,
    MemoryMetrics,
    ObservationType,
)

__version__ = "0.2.0"

__all__ = [
    # Environments
    "BaseMemoryEnvironment",
    "BaseMemoryState",
    "SimpleMemoryState",
    "SWEMemoryEnv",
    "SWETask",
    "get_env",
    "register_env",
    "list_envs",
    # Memory (new interface)
    "BaseMemoryManager",
    "PassThroughMemory",
    "SummarizationMemory",
    "FilteredContext",
    "get_memory_model",
    "register_memory_model",
    "list_memory_models",
    # Memory (legacy)
    "BaseMemoryModel",
    "PassThroughMemoryModel",
    "SummarizationMemoryModel",
    "MemoryAction",
    "Context",
    # Agents (new interface)
    "BaseAgent",
    "AgentAction",
    "SWEAgent",
    "SWEAgentTracker",
    "get_agent",
    "register_agent",
    "list_agents",
    # Agents (legacy)
    "BaseReasoningModel",
    "ReasoningAction",
    "SWEReasoningModel",
    "get_reasoning_model",
    "register_reasoning_model",
    "list_reasoning_models",
    # Runners (new)
    "BaseRunner",
    "SimpleRunner",
    "SWERunner",
    # Runners (legacy)
    "EpisodeRunner",
    "TrajectoryCollector",
    # Utilities
    "TrajectoryRecorder",
    "TokenTracker",
    # Types
    "TopicType",
    "MemoryTestType",
    "MetricType",
    "MemoryMetrics",
    "ObservationType",
]
