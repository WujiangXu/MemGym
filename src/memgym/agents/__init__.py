"""
Agents for MemGym.

This module provides agents that receive context from memory managers
and produce task-specific actions.

Usage:
    >>> from agents import get_agent, SWEAgent
    >>> agent = get_agent("swe")
    >>> action = agent.act(filtered_context)

    # For SWE-bench with mini-swe-agent:
    >>> from agents import MemoryAwareSWEAgent
    >>> agent = MemoryAwareSWEAgent(model, env, memory_manager)
"""

from .base import (
    # New interface
    BaseAgent,
    AgentAction,
    get_agent,
    register_agent,
    list_agents,
    # Legacy aliases
    BaseReasoningModel,
    ReasoningAction,
    get_reasoning_model,
    register_reasoning_model,
    list_reasoning_models,
)

# Tau2-bench agent now lives at memgym.gym.tau2_bench.agent (lazy import).
try:
    from memgym.gym.tau2_bench.agent import Tau2BenchAgent
    _TAU2_AVAILABLE = True
except ImportError:
    Tau2BenchAgent = None
    _TAU2_AVAILABLE = False


from memgym.gym.swe_bench.tracker import SWEAgent, SWEAgentTracker, SWEReasoningModel

# Mini-swe-agent wrapper (requires mini-swe-agent package)
try:
    from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent
    _MINI_SWE_AVAILABLE = True
except ImportError:
    MemoryAwareSWEAgent = None
    _MINI_SWE_AVAILABLE = False

__all__ = [
    # New interface
    "BaseAgent",
    "AgentAction",
    "get_agent",
    "register_agent",
    "list_agents",
    # Agents
    "SWEAgent",
    "SWEAgentTracker",
    # Legacy aliases
    "BaseReasoningModel",
    "ReasoningAction",
    "get_reasoning_model",
    "register_reasoning_model",
    "list_reasoning_models",
    "SWEReasoningModel",
]

# Add mini-swe-agent wrapper if available
if _MINI_SWE_AVAILABLE:
    __all__.append("MemoryAwareSWEAgent")

# Add tau2-bench agent if available
if _TAU2_AVAILABLE:
    __all__.append("Tau2BenchAgent")
