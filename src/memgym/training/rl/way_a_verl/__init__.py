"""VeRL adapter package for Way A (agent-RL on SWE-Gym).

Way A trains the agent itself end-to-end: the policy IS the SWE-Bench
agent, AND the same policy summarizes its own context when the conversation
gets long. VeRL handles the rollout-trainer co-location that our hand-rolled
``train_rl_online.py`` does not support (vLLM weight-sync after each
GRPO step). Way B continues to use the hand-rolled stack — see
``memgym.training.rl.grpo_loop`` and
``memgym.training.scripts.train_rl_online``.

Modules:
    tool — ``MemGymSWETool(BaseTool)`` wrapping ``SWEMemoryEnv`` lifecycle.
    agent_loop — registers the ``backticks`` tool parser for mini-swe-agent.
    memgym_agent_loop — registers the ``memgym_tool`` agent loop:
        ``ToolAgentLoop`` + self-summarize. The trainable Qwen3-8B is BOTH
        the reasoning policy AND the summarizer; summary tokens carry RL
        gradient (``response_mask=1``). Mirrors the trajectory-collection
        condenser in ``src/memgym/memory/llm_summarizing.py``.
    reward — ``compute_reward(...)`` producing the scalar VeRL needs.
    train — CLI wrapper invoking VeRL's PPO trainer with our config.

Importing this package side-effect-registers both the ``backticks`` parser
and the ``memgym_tool`` agent loop in verl's global registries, so
``train.py`` only needs to ``import memgym.training.rl.way_a_verl`` before
launching ``ppo_trainer.main``.
"""

# Side-effect registration: importing these modules populates verl's
# ``ToolParser._registry`` and ``_agent_loop_registry``.
from . import agent_loop  # noqa: F401  (registers "backticks" tool parser)

try:
    from . import memgym_agent_loop  # noqa: F401  (registers "memgym_tool" agent loop)
except Exception:  # noqa: BLE001
    # verl 0.7.1 may not be installed in non-Way-A environments (e.g. Way B
    # venv that imports ``rl/`` for shared helpers). Defer the registration
    # error to actual use rather than blocking package import.
    import logging as _logging

    _logging.getLogger(__name__).debug(
        "memgym_agent_loop registration skipped (verl 0.7.1 not importable)",
        exc_info=True,
    )
