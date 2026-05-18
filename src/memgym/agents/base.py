"""
Base agent abstraction for MemGym.

Agents receive context (from memory manager) and produce task-specific actions.
This module provides the base class and registry for different agent types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentAction:
    """
    Action decided by the agent.

    Represents the task-specific action that the agent decided to take
    based on the provided context.
    """
    action: Any  # The actual action (format depends on task)
    metadata: Dict[str, Any]  # Info about the decision process

    def __repr__(self) -> str:
        action_preview = str(self.action)[:50] + "..." if len(str(self.action)) > 50 else str(self.action)
        return f"AgentAction(action={action_preview}, metadata={self.metadata})"


# Legacy alias
ReasoningAction = AgentAction


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for agents.

    Agents take context (from memory manager) and produce task-specific actions.
    Different downstream tasks (tau2-bench, SWE-bench, web navigation, etc.)
    will have their own implementations.

    Key features:
    - Takes context as input (from memory manager)
    - Produces task-specific actions
    - Behavior is tracked and exposed for training
    - Can be any model (rule-based, LLM, trained policy, etc.)

    Example:
        >>> agent = get_agent("tau2")
        >>> action = agent.act(filtered_context)
        >>> stats = agent.get_stats()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._last_action: Optional[AgentAction] = None
        self._action_history: List[AgentAction] = []
        # Episode-scoped state populated by set_episode_state() after env.reset().
        # Carries fields like task instruction or app name that are constant for
        # the whole episode and should not be re-derived per step.
        self._episode_state: Dict[str, Any] = {}

    @abstractmethod
    def act(self, context: Any) -> AgentAction:
        """
        Decide what action to take given context.

        Args:
            context: FilteredContext from memory manager

        Returns:
            AgentAction containing the decided action and metadata
        """
        pass

    def set_action_from_external(self, action: Any, context: Any = None) -> AgentAction:
        """
        Record an externally-provided action (from gym step or LLM).

        This is called when actions come from outside (e.g., gym interface
        or real LLM agent). It records the action for trajectory tracking.

        Args:
            action: The action that was taken
            context: The context that was provided

        Returns:
            AgentAction with the action and metadata
        """
        agent_action = AgentAction(
            action=action,
            metadata={
                "context_length": len(str(context)) if context else 0,
                "source": "external"
            }
        )
        self._record_action(agent_action)
        return agent_action

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state."""
        pass

    def set_episode_state(self, **state: Any) -> None:
        """Receive episode-scoped state from the runner.

        Called once per episode, immediately after env.reset() returns its
        info dict. Default behavior stores everything in self._episode_state
        so subclasses can read fields without overriding this hook.

        Subclasses (e.g., WebArenaAgent) override this when they need to
        eagerly extract specific fields like the task instruction.
        """
        self._episode_state = dict(state)

    @property
    def last_action(self) -> Optional[AgentAction]:
        """Get the last action taken."""
        return self._last_action

    @property
    def action_history(self) -> List[AgentAction]:
        """Get history of all actions."""
        return self._action_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about agent behavior."""
        return {
            "agent_type": self.__class__.__name__,
            "config": self.config,
            "total_actions": len(self._action_history)
        }

    def _record_action(self, action: AgentAction) -> None:
        """Record an action for tracking."""
        self._last_action = action
        self._action_history.append(action)


# Legacy alias
BaseReasoningModel = BaseAgent


# =============================================================================
# Agent Registry
# =============================================================================

_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register an agent class.

    Args:
        name: Name to register under (e.g., "tau2", "swe")
        agent_class: The agent class to register
    """
    _AGENT_REGISTRY[name.lower()] = agent_class


# Legacy alias
register_reasoning_model = register_agent


def get_agent(name: str, **kwargs) -> BaseAgent:
    """
    Get an agent instance by name.

    Args:
        name: Agent name ("tau2", "swe", etc.)
        **kwargs: Arguments passed to agent constructor

    Returns:
        Agent instance

    Example:
        >>> agent = get_agent("tau2")
        >>> agent = get_agent("swe", config={"verbose": True})
    """
    name_lower = name.lower()
    if name_lower not in _AGENT_REGISTRY:
        available = list(_AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return _AGENT_REGISTRY[name_lower](**kwargs)


# Legacy alias
get_reasoning_model = get_agent


def list_agents() -> List[str]:
    """List all registered agent names."""
    return list(_AGENT_REGISTRY.keys())


# Legacy alias
list_reasoning_models = list_agents
