"""
Tau2BenchRunner — episode runner for tau2-bench dialogue tasks.

Mirrors WebArenaRunner's reset → set_episode_state → run → save-trajectory
shape, but **does not contain a step loop**. Tau2-bench's ``Orchestrator``
owns the entire agent ↔ user ↔ environment turn taking, so the runner's
job is to:

1. Reset env / agent / both memory managers / recorder
2. Hand episode-scoped state (domain, task_id, instruction, solo_mode) to
   each component via ``set_episode_state``
3. Call ``env.run_tau2_episode(memory_agent=..., memory_user=..., ...)``
   exactly once
4. Replay the returned ``simulation_run.messages`` into the trajectory
   recorder so the trajectory file matches the format the WebArena and
   SWE-bench tracks produce
5. Pull stats out of both memory managers and the tracking agent, save
   ``trajectory.json`` + ``result.json`` under ``<output_dir>/<domain>/
   <task_id>/`` (mirroring WebArena's per-task layout)

The runner constructs **two** memory managers — ``memory_agent`` and
``memory_user`` — to support symmetric wrapping. ``memory_user`` is
``None`` when ``solo_mode=True`` (no UserSimulator exists). ``BaseRunner``
only knows about one memory manager, so we pass the agent-side instance
through to the parent and store ``memory_user`` as a Tau2BenchRunner-only
attribute. The reported ``memory_stats`` dict has both ``agent`` and
``user`` sub-dicts so per-side ablation lands in ``summary.json``.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from memgym.agents.base import BaseAgent
from memgym.memory import BaseMemoryManager, PassThroughMemory

from .base import BaseRunner

logger = logging.getLogger("memgym.tau2_bench.runner")


class Tau2BenchRunner(BaseRunner):
    """Episode runner for tau2-bench dialogue tasks.

    Args:
        env: Tau2BenchMemoryEnv instance.
        agent: ``Tau2BenchAgent`` (tracking wrapper, not the real LLMAgent).
        memory_agent: BaseMemoryManager for the agent side. Defaults to
            ``PassThroughMemory()`` so the runner is usable without memory.
        memory_user: BaseMemoryManager for the user simulator side. Pass
            ``None`` to disable user-side wrapping (required when
            ``env.solo_mode=True`` since there is no user simulator).
        output_dir: Directory to save trajectories + per-episode artifacts.
        verbose: Print per-episode progress.
    """

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        memory_agent: Optional[BaseMemoryManager] = None,
        memory_user: Optional[BaseMemoryManager] = None,
        output_dir: str = "./results/tau2_bench",
        verbose: bool = False,
    ):
        super().__init__(
            env=env,
            agent=agent,
            memory=memory_agent or PassThroughMemory(),
            output_dir=output_dir,
            verbose=verbose,
        )
        # BaseRunner stores `memory` (the agent-side instance). The user-side
        # instance is tau2-runner-specific so we store it on the subclass.
        self.memory_agent = self.memory
        self.memory_user = memory_user

    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one tau2-bench episode end-to-end.

        Unlike SWE/WebArena there is no per-step loop here — tau2's
        Orchestrator runs the whole conversation inside one synchronous
        ``env.run_tau2_episode(...)`` call. After it returns we replay the
        message history into the trajectory recorder so downstream tooling
        (e.g. ``trajectory_to_replay.py``) sees the same shape as other tracks.
        """
        domain = self.env.domain
        task_id = self.env.task_id
        episode_dir = self.output_dir / domain / str(task_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # ---- Reset everything ----
        self.agent.reset()
        self.memory_agent.reset()
        if self.memory_user is not None:
            self.memory_user.reset()
        self.recorder.reset()

        # ---- Hand episode-scoped state to each component ----
        episode_state = {
            "domain": domain,
            "task_id": task_id,
            "solo_mode": self.env.solo_mode,
            "agent_llm": self.env.agent_llm,
        }
        if task_config:
            # Optional per-task overrides — runner accepts them through the
            # gymnasium-style options dict for parity with the other tracks.
            episode_state.update({k: v for k, v in task_config.items() if v is not None})

        self.agent.set_episode_state(**episode_state)
        self.memory_agent.set_episode_state(**episode_state)
        if self.memory_user is not None:
            # User-side memory wants role="user" so its prompt template
            # picks USER_PERSONA over USER_GOAL.
            self.memory_user.set_episode_state(
                **{k: v for k, v in episode_state.items() if k != "role"},
                role="user",
            )

        self.recorder.start_episode(metadata={
            "domain": domain,
            "task_id": task_id,
            "solo_mode": self.env.solo_mode,
            "agent_llm": self.env.agent_llm,
            "memory_agent_strategy": self.memory_agent.__class__.__name__,
            "memory_user_strategy": (
                self.memory_user.__class__.__name__ if self.memory_user else "none"
            ),
        })

        # ---- Single episode-level call ----
        if self.verbose:
            logger.info("episode start: domain=%s task=%s solo=%s", domain, task_id, self.env.solo_mode)

        try:
            episode_result = self.env.run_tau2_episode(
                memory_agent=self.memory_agent if not _is_passthrough(self.memory_agent) else None,
                memory_user=self.memory_user,
                seed=seed,
                verbose=self.verbose,
            )
        except Exception as exc:
            # Hard failure: log + return a result row so the worker pool
            # records the crash rather than dropping the task silently.
            logger.exception("tau2 episode failed: %s", exc)
            elapsed = time.time() - start_time
            error_result = {
                "domain": domain,
                "task_id": task_id,
                "solo_mode": self.env.solo_mode,
                "reward": 0.0,
                "success": False,
                "error": str(exc),
                "wall_time_seconds": round(elapsed, 2),
                "memory_stats": {"agent": {}, "user": {}},
                "agent_compaction_count": 0,
                "user_compaction_count": 0,
            }
            with (episode_dir / "result.json").open("w") as f:
                json.dump(error_result, f, indent=2, default=str)
            return error_result

        simulation_run = episode_result.get("simulation_run")
        evaluation = episode_result.get("evaluation")
        reward = float(episode_result.get("reward", 0.0))
        agent_memory_stats = episode_result.get("agent_memory_stats") or {}
        user_memory_stats = episode_result.get("user_memory_stats") or {}

        # ---- Replay simulation messages into the trajectory recorder ----
        # We dump each tau2 Message as a "step" in the recorder so the
        # resulting trajectory.json has the same per-step structure other
        # tracks emit. Tau2 messages are pydantic models — model_dump()
        # gives us the dict shape recorder expects.
        messages = getattr(simulation_run, "messages", None) or []
        for i, msg in enumerate(messages):
            msg_dict = _message_to_dict(msg)
            role = msg_dict.get("role", "?")
            terminated = (i == len(messages) - 1)
            self.recorder.record_step(
                observation=msg_dict if role != "assistant" else None,
                action=msg_dict if role == "assistant" else None,
                reward=reward if terminated else 0.0,
                terminated=terminated,
                truncated=False,
                info={"role": role, "msg_index": i},
            )

        self.recorder.end_episode()

        # ---- Pull tracking-agent stats (counts assistant/user/tool turns) ----
        try:
            self.agent.record_episode_outcome(
                simulation_run=simulation_run,
                agent_memory_stats=agent_memory_stats,
                user_memory_stats=user_memory_stats,
                reward=reward,
            )
        except AttributeError:
            # Custom agent without record_episode_outcome — skip silently.
            pass
        agent_stats = self.agent.get_stats()

        # ---- Build result dict ----
        elapsed = time.time() - start_time
        agent_compaction = int(agent_memory_stats.get("compaction_count", 0))
        user_compaction = int(user_memory_stats.get("compaction_count", 0))
        # Wrapper-side metrics live alongside the BaseMemoryManager metrics
        # under namespaced keys (see env._extract_wrapper_stats).
        agent_coercion_failures = int(agent_memory_stats.get("wrapper_coercion_failures", 0))
        user_coercion_failures = int(user_memory_stats.get("wrapper_coercion_failures", 0))
        result = {
            "domain": domain,
            "task_id": task_id,
            "solo_mode": self.env.solo_mode,
            "agent_llm": self.env.agent_llm,
            "reward": reward,
            "success": reward >= 1.0,
            "num_messages": len(messages),
            "evaluation": evaluation,
            "memory_stats": {
                "agent": agent_memory_stats,
                "user": user_memory_stats,
            },
            "agent_compaction_count": agent_compaction,
            "user_compaction_count": user_compaction,
            "agent_coercion_failures": agent_coercion_failures,
            "user_coercion_failures": user_coercion_failures,
            "agent_stats": agent_stats,
            "wall_time_seconds": round(elapsed, 2),
        }

        # ---- Save trajectory + result JSONs ----
        traj_path = episode_dir / "trajectory.json"
        trajectory = self.recorder.to_dict()
        trajectory["metadata"] = result
        with traj_path.open("w") as f:
            json.dump(trajectory, f, indent=2, default=str)
        with (episode_dir / "result.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)

        # ---- Save _training.json + _replay.json (world-memory-model pipeline) ----
        # These two files feed `src/memgym/training/data/loader.py`'s tau2 path:
        #   _training.json   → per-step memory metadata + condensation history
        #   _replay.json     → full message list + per-side step-to-msg index
        # Written alongside trajectory.json/result.json under <domain>/<task_id>/.
        self._write_training_artifacts(
            episode_dir=episode_dir,
            domain=domain,
            task_id=task_id,
            reward=reward,
            messages=messages,
            agent_memory_stats=agent_memory_stats,
            user_memory_stats=user_memory_stats,
        )

        if self.verbose:
            logger.info(
                "episode done: reward=%.3f messages=%d agent_compactions=%d user_compactions=%d",
                reward, len(messages), agent_compaction, user_compaction,
            )

        return result

    # -------------------------------------------------------------------------
    # Training artefact writer
    # -------------------------------------------------------------------------
    def _write_training_artifacts(
        self,
        *,
        episode_dir: Any,
        domain: str,
        task_id: str,
        reward: float,
        messages: List[Any],
        agent_memory_stats: Dict[str, Any],
        user_memory_stats: Dict[str, Any],
    ) -> None:
        """Build and save ``{task_id}_training.json`` + ``{task_id}_replay.json``.

        These two files are the contract between the collection side (here)
        and the world-memory-model training side (``src/memgym/training/
        data/loader.py``'s tau2 path). Schema matches plan §1.

        Artifacts are skipped gracefully if there are no wrapper calls on
        either side — e.g. when both sides use PassThroughMemory.
        """
        agent_step_log = list(agent_memory_stats.get("wrapper_step_log") or [])
        user_step_log = list(user_memory_stats.get("wrapper_step_log") or [])

        if not agent_step_log and not user_step_log:
            return

        msg_dicts = [_message_to_dict(m) for m in messages]
        agent_msg_indices_all = [i for i, m in enumerate(msg_dicts) if m.get("role") == "assistant"]
        user_msg_indices_all = [i for i, m in enumerate(msg_dicts) if m.get("role") == "user"]

        # Trailing alignment: tau2's LLMAgent emits an inline greeting during
        # orchestrator init (never through generate_next_message), so the first
        # ~1 assistant messages aren't backed by a wrapper call. We align the
        # tail — the LAST N assistant messages map to the N wrapper calls —
        # and log the leading delta so mismatches surface in the run.
        agent_msg_indices, agent_delta = _align_tail(agent_step_log, agent_msg_indices_all)
        user_msg_indices, user_delta = _align_tail(user_step_log, user_msg_indices_all)
        if agent_delta:
            logger.warning(
                "tau2 msg/step leading-skip: domain=%s task=%s agent_steps=%d agent_msgs=%d skipped=%d",
                domain, task_id, len(agent_step_log), len(agent_msg_indices_all), agent_delta,
            )
        if user_delta:
            logger.warning(
                "tau2 msg/step leading-skip: domain=%s task=%s user_steps=%d user_msgs=%d skipped=%d",
                domain, task_id, len(user_step_log), len(user_msg_indices_all), user_delta,
            )

        agent_steps = _pair_steps_to_msgs(agent_step_log, agent_msg_indices)
        user_steps = _pair_steps_to_msgs(user_step_log, user_msg_indices)

        # Condensation histories — only emitted by adapters that record them
        # (tau2_summarizing + tau2_structured; PassThrough + other strategies
        # are skipped gracefully).
        condensation_agent = _safe_condensation_history(self.memory_agent)
        condensation_user = _safe_condensation_history(self.memory_user)

        # Flat per-step view with side tags — easier for the loader to walk.
        all_steps: List[Dict[str, Any]] = []
        for rec, ms_idx in zip(agent_step_log, agent_msg_indices):
            all_steps.append(_tag_step(rec, side="agent", msg_index=ms_idx))
        for rec, ms_idx in zip(user_step_log, user_msg_indices):
            all_steps.append(_tag_step(rec, side="user", msg_index=ms_idx))

        memory_side = _infer_memory_side(
            agent=self.memory_agent, user=self.memory_user
        )

        training_payload = {
            "domain": domain,
            "task_id": task_id,
            "memory_strategy_agent": self.memory_agent.__class__.__name__,
            "memory_strategy_user": (
                self.memory_user.__class__.__name__ if self.memory_user else None
            ),
            "memory_side": memory_side,
            "episode_reward": reward,
            "episode_outcome": "resolved" if reward >= 1.0 else "unresolved",
            "condensation_history": {
                "agent": condensation_agent,
                "user": condensation_user,
            },
            "steps": all_steps,
        }
        replay_payload = {
            "domain": domain,
            "task_id": task_id,
            "messages": msg_dicts,
            "agent_steps": agent_steps,
            "user_steps": user_steps,
        }

        training_path = episode_dir / f"{task_id}_training.json"
        replay_path = episode_dir / f"{task_id}_replay.json"
        with training_path.open("w") as f:
            json.dump(training_payload, f, indent=2, default=str)
        with replay_path.open("w") as f:
            json.dump(replay_payload, f, indent=2, default=str)


# =============================================================================
# Helpers
# =============================================================================

def _is_passthrough(mm: BaseMemoryManager) -> bool:
    """Don't bother wrapping the agent if the memory manager is a no-op pass-through."""
    return mm.__class__.__name__ == "PassThroughMemory"


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Best-effort conversion of a tau2 Message to a dict for trajectory recording."""
    if isinstance(msg, dict):
        return msg
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return {
        "role": getattr(msg, "role", "?"),
        "content": str(getattr(msg, "content", msg)),
    }


def _align_tail(
    step_log: List[Dict[str, Any]], msg_indices_all: List[int]
) -> Tuple[List[int], int]:
    """Trailing-align a wrapper step log against candidate msg positions.

    Tau2's ``LLMAgent`` / ``UserSimulator`` can emit an inline greeting
    during orchestrator construction (e.g. retail's "Hi! How can I help
    you today?") that never goes through ``generate_next_message``. That
    leaves the first ~1 assistant/user messages unbacked by a wrapper call,
    so naive zip-from-head pairing assigns the wrong ``msg_index`` to
    every step and inflates the step/msg mismatch warning.

    We anchor to the tail instead: the LAST ``len(step_log)`` positions
    in ``msg_indices_all`` correspond 1:1 to wrapper calls, and the
    leading ``delta`` positions are init-emitted and dropped. The caller
    is expected to log a non-zero ``delta`` so any drift from this
    assumption stays visible.
    """
    delta = max(0, len(msg_indices_all) - len(step_log))
    return msg_indices_all[delta:], delta


def _pair_steps_to_msgs(
    step_log: List[Dict[str, Any]], msg_indices: List[int]
) -> List[Dict[str, Any]]:
    """Zip a wrapper's per-call step log with the msg positions it produced.

    Each wrapper call returns exactly one assistant (or user) message, so
    the i-th step log entry maps to the i-th assistant/user position in
    ``simulation_run.messages``. Zipping stops at the shorter of the two
    so a length mismatch produces a truncated-but-valid mapping rather
    than a crash.
    """
    paired: List[Dict[str, Any]] = []
    for rec, ms_idx in zip(step_log, msg_indices):
        paired.append({
            "step": rec.get("step"),
            "turn_idx": rec.get("turn_idx"),
            "msg_index": ms_idx,
        })
    return paired


def _safe_condensation_history(mm: Optional[BaseMemoryManager]) -> List[Dict[str, Any]]:
    """Pull condensation history from a memory manager if the adapter supports it.

    tau2_summarizing + tau2_structured implement ``get_condensation_history()``;
    PassThrough and other strategies don't — returning [] keeps the
    downstream training loader's schema stable.
    """
    if mm is None:
        return []
    fn = getattr(mm, "get_condensation_history", None)
    if not callable(fn):
        return []
    try:
        return list(fn()) or []
    except Exception as exc:
        logger.warning("get_condensation_history failed on %s: %s", type(mm).__name__, exc)
        return []


def _tag_step(rec: Dict[str, Any], *, side: str, msg_index: int) -> Dict[str, Any]:
    """Normalize a step-log record into the training schema's step entry."""
    return {
        "step": rec.get("step"),
        "turn_idx": rec.get("turn_idx"),
        "msg_index": msg_index,
        "side": side,
        "memory": {
            "was_compacted": rec.get("was_compacted", False),
            "new_compaction": rec.get("new_compaction", False),
            "memory_failed": rec.get("memory_failed", False),
            "coercion_failed": rec.get("coercion_failed", False),
            "strategy": rec.get("strategy"),
            "original_msgs": rec.get("original_msgs"),
            "filtered_msgs": rec.get("filtered_msgs"),
            "original_tokens": rec.get("original_tokens"),
            "filtered_tokens": rec.get("filtered_tokens"),
            "compression_ratio": rec.get("compression_ratio", 1.0),
            "summary": rec.get("summary"),
            "forgotten": rec.get("forgotten"),
            "summarizer_prompt_tokens": rec.get("summarizer_prompt_tokens", 0),
            "summarizer_completion_tokens": rec.get("summarizer_completion_tokens", 0),
            "summarization_time_seconds": rec.get("summarization_time_seconds", 0.0),
        },
    }


def _infer_memory_side(
    *, agent: Optional[BaseMemoryManager], user: Optional[BaseMemoryManager]
) -> str:
    """Reconstruct the ``memory_side`` setting from which managers are live."""
    agent_live = agent is not None and not _is_passthrough(agent)
    user_live = user is not None and not _is_passthrough(user)
    if agent_live and user_live:
        return "both"
    if agent_live:
        return "agent"
    if user_live:
        return "user"
    return "none"
