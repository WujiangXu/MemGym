"""
Tau2BenchAgent — light tracking wrapper around tau2-bench's LLMAgent.

Why this is just a wrapper, not the real agent
==============================================
Unlike WebArena (where ``WebArenaAgent.act()`` is called from a runner step
loop), tau2-bench's ``Orchestrator`` owns the entire turn taking. The "real"
agent is tau2's ``LLMAgent`` / ``LLMSoloAgent``, constructed inside
``Tau2BenchMemoryEnv.run_tau2_episode()`` and possibly wrapped by a
``MemoryManagerAdapter``.

So why have this class at all? Three reasons:

1. ``BaseRunner`` and the per-track CLI registry expect a ``BaseAgent``
   instance — even when the runner never actually calls ``act()``. Without a
   tracking wrapper, the tau2 track would have to special-case every agent
   factory in ``evaluate.py`` and ``launch.py``.
2. After ``run_tau2_episode()`` returns, the runner needs a single place to
   read post-hoc stats (turn count, agent llm name, memory adapter stats).
   This wrapper holds those stats so the runner can call ``get_stats()`` like
   it does for every other track.
3. ``act()`` exists as a guard rail: if anything ever tries to drive the
   tau2 agent step-by-step from outside the orchestrator (e.g., a copy-paste
   bug from the WebArena runner), it raises immediately with a clear message
   instead of silently doing nothing.

All ``tau2.*`` imports are deferred — this module loads cleanly even if
``third_party/tau2-bench`` hasn't been installed yet.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from memgym.agents.base import AgentAction, BaseAgent, register_agent

logger = logging.getLogger("memgym.tau2_bench.agent")


class Tau2BenchAgent(BaseAgent):
    """Tracking wrapper for tau2-bench episodes.

    Args:
        agent_llm: LiteLLM model id of the underlying tau2 agent (e.g.
            ``bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0``).
            Stored only for stats / trajectory metadata — the tau2 env
            constructs the actual ``LLMAgent`` from the same value.
        config: Arbitrary config dict (passed to BaseAgent).
    """

    def __init__(
        self,
        agent_llm: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config)
        self.agent_llm = agent_llm

        # Stats refreshed by the runner after run_tau2_episode() returns.
        # Keys mirror what MemoryManagerAdapter.get_stats() exposes so the
        # trajectory recorder has a single flat schema to serialize.
        self._episode_stats: Dict[str, Any] = {
            "num_messages": 0,
            "num_agent_turns": 0,
            "num_user_turns": 0,
            "num_tool_calls": 0,
            "compaction_count": 0,
            "user_compaction_count": 0,
        }

    # ---- BaseAgent contract ----

    def act(self, context: Any) -> AgentAction:
        """Not used in the tau2 track — orchestrator owns the loop.

        Raising here (rather than returning a no-op) catches the bug at
        the call site instead of letting an empty action propagate into
        a tau2 episode where it would be ignored anyway.
        """
        raise NotImplementedError(
            "Tau2BenchAgent.act() is never called. tau2-bench's Orchestrator "
            "owns the agent-user-env loop. Use Tau2BenchMemoryEnv."
            "run_tau2_episode() — see Tau2BenchRunner for the standard usage."
        )

    def reset(self) -> None:
        """Clear per-episode tracking state."""
        self._action_history = []
        self._last_action = None
        self._episode_stats = {
            "num_messages": 0,
            "num_agent_turns": 0,
            "num_user_turns": 0,
            "num_tool_calls": 0,
            "compaction_count": 0,
            "user_compaction_count": 0,
        }

    def set_episode_state(self, **state: Any) -> None:
        """Capture domain / task / solo_mode for the current episode.

        Tau2 doesn't have a free-form task instruction the way WebArena does
        — the "instruction" is implied by the loaded task object. We still
        store ``instruction`` if the runner passes one (for symmetry with
        memory managers that key off it) and we always pull out the
        domain / task_id / solo_mode triple so ``get_stats()`` can include
        them in trajectory metadata without a second source of truth.
        """
        super().set_episode_state(**state)
        self._instruction = str(state.get("instruction") or "").strip()
        self._domain = str(state.get("domain") or "").strip()
        self._task_id = str(state.get("task_id") or "").strip()
        self._solo_mode = bool(state.get("solo_mode", False))

    # ---- Tau2-specific stat ingestion ----

    def record_episode_outcome(
        self,
        simulation_run: Any,
        agent_memory_stats: Optional[Dict[str, Any]] = None,
        user_memory_stats: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
    ) -> None:
        """Pull post-episode counters out of the tau2 simulation run.

        Called by Tau2BenchRunner after ``env.run_tau2_episode()`` returns.
        We count agent / user / tool messages by walking ``simulation_run.
        messages`` and inspecting each message's ``role`` field. The
        compaction counts come from the memory adapter's stats dict
        (None when memory wrapping is disabled).
        """
        num_messages = 0
        num_agent_turns = 0
        num_user_turns = 0
        num_tool_calls = 0

        messages = getattr(simulation_run, "messages", None) or []
        for msg in messages:
            num_messages += 1
            role = getattr(msg, "role", None) or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            if role == "assistant":
                num_agent_turns += 1
            elif role == "user":
                num_user_turns += 1
            elif role == "tool":
                num_tool_calls += 1

        self._episode_stats = {
            "num_messages": num_messages,
            "num_agent_turns": num_agent_turns,
            "num_user_turns": num_user_turns,
            "num_tool_calls": num_tool_calls,
            "compaction_count": int(
                (agent_memory_stats or {}).get("compaction_count", 0)
            ),
            "user_compaction_count": int(
                (user_memory_stats or {}).get("compaction_count", 0)
            ),
            "reward": float(reward) if reward is not None else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return the merged BaseAgent + tau2 episode stats."""
        base_stats = super().get_stats()
        base_stats.update({
            "agent_llm": self.agent_llm,
            "domain": getattr(self, "_domain", ""),
            "task_id": getattr(self, "_task_id", ""),
            "solo_mode": getattr(self, "_solo_mode", False),
            **self._episode_stats,
        })
        return base_stats


# Register so `get_agent("tau2-bench")` works alongside the env registration.
register_agent("tau2-bench", Tau2BenchAgent)
register_agent("tau2", Tau2BenchAgent)
