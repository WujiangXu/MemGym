"""
Tau2BenchMemoryEnv — orchestrator-delegating dialogue env for MemGym.

Unlike SWE-bench and WebArena, tau2-bench's `Orchestrator` owns the entire
agent ↔ user ↔ environment loop. We cannot easily drive it step-by-step from
a runner, so this env exposes a single episode-level entry point —
``run_tau2_episode(memory_agent, memory_user, recorder, ...)`` — instead of
the gym ``reset/step`` pair. The Tau2BenchRunner (under
``memgym.runners.tau2_bench_runner``) calls it once per episode and replays
the resulting message stream into the trajectory recorder.

The class still subclasses ``BaseMemoryEnvironment`` so it shows up in
``memgym.envs.list_envs()``, but the inherited ``reset/step`` methods raise
``NotImplementedError`` with a clear pointer at ``run_tau2_episode``. This is
the intentional deviation from the WebArena pattern documented in the plan.

All ``tau2.*`` imports are deferred to method bodies so that
``import memgym.gym.tau2_bench`` works even when ``third_party/tau2-bench``
hasn't been installed yet (gym/tau2_bench/install.py handles that).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.envs.base import BaseMemoryEnvironment, register_env

logger = logging.getLogger("memgym.tau2_bench.env")


# =============================================================================
# Domain → solo_mode auto-detection table.
# =============================================================================
#
# Tau2-bench's mock and telecom domains are designed to run with a single
# agent (no user simulator). airline and retail require the dual-agent loop
# because their tasks are scripted around an active customer persona.
# This is the single source of truth for the auto-detection — `evaluate.py`
# imports `DOMAIN_SOLO_MODE_SUPPORT` from here.
DOMAIN_SOLO_MODE_SUPPORT: Dict[str, bool] = {
    "mock": True,
    "telecom": True,
    "airline": False,
    "retail": False,
}


# =============================================================================
# Helpers
# =============================================================================

def _ensure_tau2_on_path() -> None:
    """Add ``third_party/tau2-bench/src`` to ``sys.path`` if it's not installed.

    Preferred installation path is `pip install -e third_party/tau2-bench`,
    in which case this fallback is a no-op (the package is importable normally).
    The fallback exists so a hand-checked-out tau2-bench repo also works.
    """
    try:
        import tau2  # noqa: F401
        return
    except ImportError:
        pass

    # Fallback: walk up from this file to find a `third_party/tau2-bench/src`
    # directory and add it to sys.path. Limited search depth so we don't
    # scan the whole filesystem if the repo isn't checked out.
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / "third_party" / "tau2-bench" / "src"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


def _extract_wrapper_stats(entity: Any) -> Dict[str, Any]:
    """Pull wrapper-side counters from a memory_adapter wrapper instance.

    The wrapper sets ``_stats`` directly in ``__dict__`` via
    ``object.__setattr__`` (init bypass), so we read it the same way to
    avoid touching tau2's __getattr__ proxy. Returns ``{}`` if the entity
    isn't a wrapper (e.g. memory was disabled and the raw tau2 agent is
    in use).
    """
    try:
        d = object.__getattribute__(entity, "__dict__")
    except AttributeError:
        return {}
    stats = d.get("_stats")
    if not isinstance(stats, dict):
        return {}
    # Namespace the wrapper-side counters so they don't collide with the
    # BaseMemoryManager keys (compaction_count is reported by both — the
    # manager wins after the dict-merge in run_tau2_episode).
    return {
        "wrapper_total_calls": stats.get("total_calls", 0),
        "wrapper_compaction_count": stats.get("compaction_count", 0),
        "wrapper_coercion_failures": stats.get("coercion_failures", 0),
        "wrapper_last_compression_ratio": stats.get("last_compression_ratio", 1.0),
        # Per-call memory records assembled during _intercept_generate_next_message.
        # Runner reads this to write {task_id}_training.json.
        "wrapper_step_log": list(stats.get("step_log", [])),
    }


_NL_PATCHED = False


def _patch_nl_json_extraction(nl_eval_module: Any) -> None:
    """Monkey-patch ``NLAssertionsEvaluator.evaluate_nl_assertions`` to
    tolerate non-JSON LLM output (Haiku wraps JSON in markdown/preamble).

    Idempotent — only patches once per process.
    """
    global _NL_PATCHED
    if _NL_PATCHED:
        return
    _NL_PATCHED = True

    import json as _json
    import re as _re

    _orig = nl_eval_module.NLAssertionsEvaluator.evaluate_nl_assertions

    @classmethod  # type: ignore[misc]
    def _safe_evaluate(cls, trajectory, nl_assertions):
        """Wrapper that extracts JSON from preamble text if json.loads fails."""
        _ensure_tau2_on_path()
        from tau2.data_model.simulation import NLAssertionCheck  # type: ignore

        from tau2.utils.llm_utils import generate  # type: ignore
        from tau2.data_model.message import SystemMessage, UserMessage  # type: ignore

        if not nl_assertions:
            return []

        trajectory_str = "\n".join(
            [f"{m.role}: {m.content}" for m in trajectory]
        )
        system_prompt = nl_eval_module.NLAssertionsEvaluator.evaluate_nl_assertions.__func__  # access the original
        # Just call the original — if it works, great
        try:
            return _orig.__func__(cls, trajectory, nl_assertions)
        except (_json.JSONDecodeError, Exception) as exc:
            if not isinstance(exc, _json.JSONDecodeError):
                raise
            # Haiku returned non-JSON. Retry with explicit JSON instruction.
            logger.warning("NL judge returned non-JSON; retrying with explicit JSON instruction")

        # Build messages matching the tau2 evaluator prompt
        system_prompt_text = """
        TASK
        - You will be given a list of expected outcomes and a conversation.
        - Grade each expected outcome individually.
        - RESPOND WITH ONLY VALID JSON, no other text.

        FORMAT
        {
            "results": [
                {
                    "expectedOutcome": "<the expected outcome>",
                    "reasoning": "<reasoning>",
                    "metExpectation": true or false
                }
            ]
        }
        """
        user_prompt = f"conversation:\n{trajectory_str}\n\nexpectedOutcomes:\n{nl_assertions}"
        messages = [
            SystemMessage(role="system", content=system_prompt_text),
            UserMessage(role="user", content=user_prompt),
        ]
        resp = generate(
            model=nl_eval_module.DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            call_name="nl_assertions_eval_retry",
            **nl_eval_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        content = resp.content
        # Try to extract JSON from the response
        match = _re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                data = _json.loads(match.group())
                return [
                    NLAssertionCheck(
                        nl_assertion=r["expectedOutcome"],
                        met=r["metExpectation"],
                        justification=r.get("reasoning", ""),
                    )
                    for r in data.get("results", [])
                ]
            except (_json.JSONDecodeError, KeyError):
                pass

        # Final fallback: mark all unresolved
        logger.warning("NL judge retry also failed; marking all assertions as not-met")
        return [
            NLAssertionCheck(
                nl_assertion=a, met=False,
                justification="NL judge returned unparseable response",
            )
            for a in nl_assertions
        ]

    nl_eval_module.NLAssertionsEvaluator.evaluate_nl_assertions = _safe_evaluate


def load_domain_tasks(domain: str, task_split: str = "base") -> List[Any]:
    """Return the task list for a tau2 domain via ``tau2.registry``.

    Args:
        domain: One of {"mock", "telecom", "airline", "retail"}.
        task_split: Split name — "base" (all tasks, default), "train", or
            "test".  Passed to ``registry.get_tasks_loader(domain)(split)``.
    """
    _ensure_tau2_on_path()
    from tau2.registry import registry  # type: ignore
    return list(registry.get_tasks_loader(domain)(task_split))


# =============================================================================
# Tau2BenchMemoryEnv
# =============================================================================

class Tau2BenchMemoryEnv(BaseMemoryEnvironment):
    """Orchestrator-delegating tau2-bench env.

    Args:
        domain: One of {"mock", "telecom", "airline", "retail"}.
        task_id: Tau2 task identifier (matches `task.id` from
            `registry.get_tasks_loader(domain)`).
        agent_llm: LiteLLM model id for the agent (e.g.
            ``bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0``).
        user_llm: LiteLLM model id for the user simulator. Defaults to
            ``agent_llm`` so single-LLM ablations are the default.
        agent_llm_args: Extra kwargs forwarded to tau2's LLMAgent constructor.
        user_llm_args: Extra kwargs for the UserSimulator.
        solo_mode: True → single-agent (no user simulator). Auto-detected
            from ``DOMAIN_SOLO_MODE_SUPPORT`` if None.
        max_steps: Hard episode cap (passed to tau2's Orchestrator).
        config: Arbitrary config dict (passed to BaseMemoryEnvironment).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        domain: str,
        task_id: str,
        agent_llm: str,
        user_llm: Optional[str] = None,
        agent_llm_args: Optional[Dict[str, Any]] = None,
        user_llm_args: Optional[Dict[str, Any]] = None,
        solo_mode: Optional[bool] = None,
        max_steps: int = 100,
        nl_judge_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(memory_state=None, config=config)
        self.domain = domain
        self.task_id = str(task_id)
        self.agent_llm = agent_llm
        self.user_llm = user_llm or agent_llm
        self.agent_llm_args = dict(agent_llm_args or {})
        self.user_llm_args = dict(user_llm_args or {})
        self.max_steps = max_steps
        self.nl_judge_model = nl_judge_model or agent_llm

        # Auto-detect solo_mode if not specified, with a soft override warning
        # if the caller asks for solo on a domain that doesn't support it.
        if solo_mode is None:
            self.solo_mode = DOMAIN_SOLO_MODE_SUPPORT.get(domain, False)
        else:
            domain_supports_solo = DOMAIN_SOLO_MODE_SUPPORT.get(domain, False)
            if solo_mode and not domain_supports_solo:
                logger.warning(
                    "domain %r does not support solo_mode; falling back to dual-agent",
                    domain,
                )
                self.solo_mode = False
            else:
                self.solo_mode = solo_mode

    # =========================================================================
    # Inherited gym API — intentionally NotImplemented for tau2.
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(
            "Tau2BenchMemoryEnv does not expose reset/step because tau2's "
            "Orchestrator owns the agent-user-env loop. Call "
            "run_tau2_episode(memory_agent=..., memory_user=..., recorder=...) "
            "instead — see Tau2BenchRunner for the standard usage."
        )

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError(
            "Tau2BenchMemoryEnv does not expose reset/step. Use "
            "run_tau2_episode() — see Tau2BenchRunner."
        )

    def _calculate_reward(self, observation: Any, action: Any, info: Dict[str, Any]) -> float:
        # Reward comes from tau2's evaluate_simulation, not per-step.
        return 0.0

    def _evaluate_memory(self) -> Dict[str, Any]:
        # Memory metrics are owned by the BaseMemoryManager instances passed
        # into run_tau2_episode; the env itself does not track memory state.
        return {}

    # =========================================================================
    # Tau2 episode-level entry point
    # =========================================================================

    def _build_environment(self) -> Any:
        _ensure_tau2_on_path()
        from tau2.registry import registry  # type: ignore
        return registry.get_env_constructor(self.domain)(solo_mode=self.solo_mode)

    def _resolve_task(self) -> Any:
        for task in load_domain_tasks(self.domain):
            if str(task.id) == self.task_id:
                return task
        raise ValueError(
            f"No task with id={self.task_id!r} in domain={self.domain!r}. "
            f"List available tasks via load_domain_tasks({self.domain!r})."
        )

    def _build_base_agent(self, environment: Any, task: Any) -> Any:
        """Construct the unwrapped tau2 agent (LLMSoloAgent or LLMAgent)."""
        _ensure_tau2_on_path()
        from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent  # type: ignore

        tools = environment.get_tools()
        if self.solo_mode:
            user_tools = environment.get_user_tools() if environment.user_tools else []
            tools = tools + user_tools
            return LLMSoloAgent(
                tools=tools,
                domain_policy=environment.get_policy(),
                task=task,
                llm=self.agent_llm,
                llm_args=self.agent_llm_args,
            )
        return LLMAgent(
            tools=tools,
            domain_policy=environment.get_policy(),
            llm=self.agent_llm,
            llm_args=self.agent_llm_args,
        )

    def _build_base_user(self, environment: Any, task: Any) -> Any:
        """Construct the unwrapped tau2 user (UserSimulator or DummyUser)."""
        _ensure_tau2_on_path()
        from tau2.user.user_simulator import UserSimulator, DummyUser  # type: ignore

        if self.solo_mode:
            return DummyUser()

        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None

        return UserSimulator(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=self.user_llm,
            llm_args=self.user_llm_args,
        )

    def run_tau2_episode(
        self,
        memory_agent: Optional[Any] = None,
        memory_user: Optional[Any] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run a single tau2 episode end-to-end.

        Args:
            memory_agent: Optional ``BaseMemoryManager`` instance whose
                ``manage_context`` will be invoked on every agent turn via
                ``MemoryManagerAdapter``.
            memory_user: Optional ``BaseMemoryManager`` for the user simulator
                side. Ignored when ``solo_mode=True`` (no UserSimulator exists).
            seed: Optional seed forwarded to the agent / user via ``set_seed``.
            verbose: Print one-line progress summaries.

        Returns:
            A dict with: ``simulation_run`` (tau2's ``SimulationRun``),
            ``evaluation`` (tau2's ``EvaluationResult`` as dict), ``reward``
            (float), ``agent_memory_stats`` (dict), ``user_memory_stats``
            (dict). The runner replays ``simulation_run.messages`` into its
            recorder and serializes the rest into ``result.json``.
        """
        _ensure_tau2_on_path()
        from tau2.orchestrator.orchestrator import Orchestrator  # type: ignore
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # type: ignore

        environment = self._build_environment()
        task = self._resolve_task()
        base_agent = self._build_base_agent(environment, task)
        base_user = self._build_base_user(environment, task)

        # Wrap with the symmetric memory adapter pair when memory managers are
        # supplied. Imported lazily because memory_adapter itself imports
        # tau2's agent classes for the isinstance trick.
        agent = base_agent
        user = base_user
        if memory_agent is not None:
            from .memory_adapter import wrap_agent_with_memory
            agent = wrap_agent_with_memory(base_agent, memory_agent, role="agent")
        if memory_user is not None and not self.solo_mode:
            from .memory_adapter import wrap_user_with_memory
            user = wrap_user_with_memory(base_user, memory_user, role="user")

        if seed is not None:
            for entity in (agent, user):
                if hasattr(entity, "set_seed"):
                    entity.set_seed(seed)

        if verbose:
            logger.info(
                "tau2 episode start: domain=%s task=%s solo=%s agent_llm=%s",
                self.domain, self.task_id, self.solo_mode, self.agent_llm,
            )

        orchestrator = Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.max_steps,
            solo_mode=self.solo_mode,
        )
        simulation_run = orchestrator.run()

        evaluation_result = None
        reward = 0.0
        if simulation_run is not None:
            # Monkey-patch the NL-assertion judge model so tasks whose
            # reward_basis includes NL_ASSERTION use our Bedrock model
            # instead of tau2's default gpt-4.1.
            # IMPORTANT: must patch the evaluator module's local copy, not
            # tau2.config — `from X import Y` copies the value at import
            # time so patching the source module has no effect.
            import tau2.evaluator.evaluator_nl_assertions as _nl_eval  # type: ignore
            _nl_eval.DEFAULT_LLM_NL_ASSERTIONS = self.nl_judge_model
            _patch_nl_json_extraction(_nl_eval)

            evaluation_result = evaluate_simulation(
                simulation=simulation_run,
                task=task,
                evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                solo_mode=self.solo_mode,
                domain=self.domain,
            )
            reward = float(evaluation_result.reward)

        # Pull memory stats out of the BaseMemoryManager (compaction_count,
        # token tracking) AND merge in the wrapper-side counters
        # (coercion_failures, total_calls) which live on the wrapper instance's
        # __dict__ rather than the manager. ``_extract_wrapper_stats`` returns
        # an empty dict if the entity wasn't wrapped, so this is a no-op when
        # memory is disabled.
        agent_memory_stats = (
            {**memory_agent.get_stats(), **_extract_wrapper_stats(agent)}
            if memory_agent is not None else {}
        )
        user_memory_stats = (
            {**memory_user.get_stats(), **_extract_wrapper_stats(user)}
            if memory_user is not None and not self.solo_mode else {}
        )

        if verbose:
            num_messages = len(simulation_run.messages) if simulation_run else 0
            logger.info(
                "tau2 episode done: messages=%d reward=%.3f", num_messages, reward,
            )

        return {
            "simulation_run": simulation_run,
            "evaluation": (
                evaluation_result.model_dump()
                if evaluation_result is not None and hasattr(evaluation_result, "model_dump")
                else (evaluation_result.__dict__ if evaluation_result is not None else None)
            ),
            "reward": reward,
            "agent_memory_stats": agent_memory_stats,
            "user_memory_stats": user_memory_stats,
        }

    def close(self) -> None:
        # tau2's Environment / Orchestrator manage their own teardown inside
        # `orchestrator.run()`. Nothing to release here.
        pass


# Register the env so it shows up in `memgym.envs.list_envs()`. Both names
# are kept so existing CLIs ("tau2", "tau2-bench", "dialogue") still resolve.
register_env("tau2-bench", Tau2BenchMemoryEnv)
register_env("tau2", Tau2BenchMemoryEnv)
register_env("dialogue", Tau2BenchMemoryEnv)
