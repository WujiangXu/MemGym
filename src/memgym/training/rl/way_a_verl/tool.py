"""verl 0.7.1 ``BaseTool`` adapter — one Docker-backed SWE instance per tool call.

verl 0.7.1's ``ToolAgentLoop`` invokes ``tool.create()`` / ``tool.execute()`` /
``tool.release()`` **per assistant turn** (see ``ToolAgentLoop._call_tool``). For an
SWE-Bench episode that's 75 turns × the same instance — we can't tear down the
container between bash steps. So we cache one ``DockerEnvironment`` per dataset
``instance_id`` at the class level: ``create`` is idempotent on cache hit,
``release`` is a no-op, and an atexit hook closes everything at worker shutdown.

Wire-up assumptions (set in ``train.py``):

1. The parquet row carries ``tools_kwargs={"MemGymSWETool": {"create_kwargs":
   {"instance_id": ..., "image_name": ..., "ground_truth": ..., "repo": ...}}}``.
   verl forwards that dict into ``create_kwargs`` per turn.
2. The tool config YAML lists this class via fully-qualified path
   (``memgym.training.rl.way_a_verl.tool.MemGymSWETool``) with
   ``config.type: native`` so ``initialize_tools_from_config`` picks the
   ``ToolType.NATIVE`` branch.
3. The agent emits actions in mini-swe-agent's backticks format
   (``THOUGHT: ...\\n```bash\\n<cmd>\\n```\\n``) — parsed by
   ``BackticksToolParser`` registered alongside this tool in ``agent_loop.py``.

The reward path: every non-submit bash step returns ``(text, 0.0, info)``.
Only when the model emits a ``submit`` (canonical mini-swe-agent end-of-task
signal) do we collect the patch via ``git diff`` inside the container and
shell out to SWE-bench's harness via ``_evaluate_patch``. That single step
returns the terminal reward; verl's ``custom_reward_function`` pulls it via
``extra_fields["tool_rewards"][-1]`` — see ``reward.py``.
"""
from __future__ import annotations

import atexit
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# mini-swe-agent's submit marker. Anything containing the literal token is
# treated as end-of-task; the actual command can vary
# (``submit``, ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, ...).
_SUBMIT_MARKER = re.compile(
    r"\b(?:submit|COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT)\b",
    re.IGNORECASE,
)


def _make_tool_class():
    """Lazy import — verl is an optional dependency under ``[rl-way-a]``."""
    from verl.tools.base_tool import BaseTool  # type: ignore
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse  # type: ignore

    class MemGymSWETool(BaseTool):
        """One ``MemGymSWETool`` instance is created per worker process; the
        env cache spans the whole rollout pool.

        Concurrency: ``create`` may be called concurrently for the same
        ``instance_id`` from sibling rollouts of the same dataset row. We
        guard the cache with a ``threading.Lock`` and let only the first
        winner pay the docker-pull cost; later callers reuse the env. The
        same env is shared across same-instance rollouts because each rollout
        runs sequentially within a single ``run()`` coroutine on a Ray actor —
        verl does not parallel-step within one trajectory.
        """

        # Class-level so every Ray worker on this process sees one cache.
        # ``_env_cache`` uses an OrderedDict so we can implement LRU eviction
        # (audit H9): on cache miss past ``_max_envs``, evict the oldest by
        # last-touch order. Each ``execute()`` calls ``move_to_end`` on the
        # touched key so the LRU view stays accurate.
        _env_cache: "OrderedDict[str, Any]" = OrderedDict()
        _env_meta: Dict[str, Dict[str, Any]] = {}
        # Audit H6: ``meta["step"]`` was per-instance (shared across rollouts)
        # so 8 concurrent rollouts of the same instance shared one counter
        # and ``step >= max_steps`` fired after collectively 75 turns instead
        # of 75 per rollout. Track step per-rollout via the verl-supplied
        # ``request_id`` (each rollout has a unique id). Key shape:
        # ``f"{instance_id}:{request_id}"``. Lazy-init at first ``execute``.
        _rollout_step: Dict[str, int] = {}
        _cache_lock: threading.Lock = threading.Lock()
        _atexit_registered: bool = False
        # Default LRU cap; overridable via tool config ``max_envs``.
        _max_envs: int = 16
        # Audit H6 followup: bound the per-rollout step dict so a crash
        # path that skips ``release()`` doesn't leak entries forever.
        _rollout_step_cap: int = 1024

        def __init__(self, config: Dict[str, Any],
                     tool_schema: Optional["OpenAIFunctionToolSchema"] = None):
            super().__init__(config=config, tool_schema=tool_schema)
            self._dataset = config.get("dataset", "swe-gym")
            self._docker_image_source = config.get("docker_image_source", "swe-gym")
            self._max_steps = int(config.get("max_steps", 75))
            self._cwd = config.get("cwd", "/testbed")
            self._timeout = int(config.get("step_timeout", 60))
            self._verbose = bool(config.get("verbose", False))
            # Tool-config override for the LRU cap. Mutates class-level
            # state — last constructor wins, but in practice all
            # constructors come from the same tool_config.yaml.
            cap = config.get("max_envs")
            if cap is not None:
                type(self)._max_envs = max(1, int(cap))

            # Register atexit cleanup once per process.
            if not type(self)._atexit_registered:
                atexit.register(type(self)._cleanup_all)
                type(self)._atexit_registered = True

        # ------------------------------------------------------------------
        # BaseTool API
        # ------------------------------------------------------------------

        async def create(self, instance_id: Optional[str] = None,
                         **kwargs: Any) -> Tuple[str, "ToolResponse"]:
            """Idempotent — cache hit on ``dataset_iid``, build on miss."""
            create_kwargs = kwargs.get("create_kwargs", {}) or {}
            dataset_iid = (
                create_kwargs.get("instance_id")
                or instance_id
                or uuid4().hex
            )
            image_name = create_kwargs.get("image_name") or self._image_for(dataset_iid)
            ground_truth = create_kwargs.get("ground_truth", "") or ""
            repo = create_kwargs.get("repo", "")

            with type(self)._cache_lock:
                if dataset_iid in type(self)._env_cache:
                    type(self)._env_cache.move_to_end(dataset_iid)
                    type(self)._env_meta[dataset_iid]["last_used"] = time.time()
                    return dataset_iid, ToolResponse()

            # Build outside the lock to avoid serializing all docker pulls.
            env = self._build_env(image_name)
            evicted: list = []
            with type(self)._cache_lock:
                # Lost the race? Discard our env, reuse the cached one.
                if dataset_iid in type(self)._env_cache:
                    self._cleanup_one(env)
                    type(self)._env_cache.move_to_end(dataset_iid)
                    return dataset_iid, ToolResponse()
                # Audit H9: enforce LRU cap before insert.
                while len(type(self)._env_cache) >= type(self)._max_envs:
                    old_iid, old_env = type(self)._env_cache.popitem(last=False)
                    type(self)._env_meta.pop(old_iid, None)
                    evicted.append(old_env)
                type(self)._env_cache[dataset_iid] = env
                type(self)._env_meta[dataset_iid] = {
                    "ground_truth": ground_truth,
                    "image_name": image_name,
                    "repo": repo,
                    "last_used": time.time(),
                }
            for old_env in evicted:
                self._cleanup_one(old_env)
            return dataset_iid, ToolResponse()

        async def execute(self, instance_id: str, parameters: Dict[str, Any],
                          **kwargs: Any) -> Tuple["ToolResponse", float, Dict[str, Any]]:
            """One bash step. Returns the verl 3-tuple ``(resp, reward, info)``."""
            action = (parameters or {}).get("action") or (parameters or {}).get("bash") or ""
            action = str(action).strip()

            with type(self)._cache_lock:
                env = type(self)._env_cache.get(instance_id)
                meta = type(self)._env_meta.get(instance_id, {})
                if env is not None:
                    # Touch LRU on every step so eviction order reflects actual use.
                    type(self)._env_cache.move_to_end(instance_id)
                    meta["last_used"] = time.time()

            if env is None:
                return (
                    ToolResponse(text=f"[tool error] no live env for {instance_id}"),
                    0.0,
                    {"resolved": False, "error": "no_env", "step": 0},
                )

            # Audit H6: per-rollout step counter keyed on (instance_id, request_id)
            # so concurrent rollouts of the same instance don't share the budget.
            request_id = str(kwargs.get("request_id") or "default")
            step_key = f"{instance_id}:{request_id}"
            with type(self)._cache_lock:
                step = type(self)._rollout_step.get(step_key, 0) + 1
                # Bound the dict so a crash path that skips ``release`` can't leak
                # entries forever. FIFO eviction is fine — the bound is loose.
                if (
                    step_key not in type(self)._rollout_step
                    and len(type(self)._rollout_step) >= type(self)._rollout_step_cap
                ):
                    oldest = next(iter(type(self)._rollout_step))
                    type(self)._rollout_step.pop(oldest, None)
                type(self)._rollout_step[step_key] = step

            if _SUBMIT_MARKER.search(action) and "git diff" not in action:
                # Submission path: pull the patch from the container and
                # let the SWE-Bench harness score it. The reward is the
                # terminal reward for this episode.
                patch_text, patch_err = self._collect_patch(env)
                resolved, reward, info_extra = self._evaluate(
                    instance_id, patch_text, meta,
                )
                info = {
                    "resolved": bool(resolved),
                    "step": step,
                    "done": True,
                    "patch_bytes": len(patch_text or ""),
                    **info_extra,
                }
                if patch_err:
                    info["patch_collect_error"] = patch_err
                obs = (
                    f"Submission scored. resolved={resolved}, reward={reward:.3f}.\n"
                    f"Patch ({len(patch_text or '')} bytes) was sent to the harness."
                )
                return ToolResponse(text=obs), float(reward), info

            try:
                result = env.execute(action, cwd=self._cwd, timeout=self._timeout) \
                    if hasattr(env, "execute") else {"output": "", "returncode": 1}
            except Exception as exc:  # noqa: BLE001 — env failure surfaces as observation
                return (
                    ToolResponse(text=f"[exec error] {exc}"),
                    0.0,
                    {"resolved": False, "step": step, "error": "exec_failed"},
                )

            output = result.get("output", "") if isinstance(result, dict) else str(result)
            rc = result.get("returncode", 0) if isinstance(result, dict) else 0
            done = step >= self._max_steps
            return (
                ToolResponse(text=str(output)),
                0.0,
                {"resolved": False, "step": step, "returncode": rc, "done": done},
            )

        async def release(self, instance_id: str, **kwargs: Any) -> None:
            """No env teardown — real container cleanup happens in
            ``_cleanup_all`` at process exit (verl calls ``release`` per
            assistant turn, not per rollout, so killing the container here
            would tear it down mid-episode).

            We **do** drop the per-rollout step counter here so that across
            many rollouts the ``_rollout_step`` dict stays bounded. ``release``
            is still called once per turn, but popping the key is cheap and
            the next ``execute`` will lazy-re-init it (counter resets — which
            is wrong if we relied on this for budgeting). To preserve budget
            semantics we only drop on what verl signals as a final release;
            absent a clear signal, leave the entry for ``_rollout_step_cap``
            FIFO eviction to handle.
            """
            return None

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _image_for(self, dataset_iid: str) -> str:
            """SWE-Gym / SWE-Smith image naming convention.

            Falls back to ``swebench/sweb.eval.x86_64.<iid>:latest`` if the
            dataset row didn't carry ``image_name``. Friends who use a
            mirrored registry override via ``create_kwargs``.
            """
            if self._docker_image_source == "swe-gym":
                # SWE-Gym uses lowercased instance ids with double-underscore.
                slug = dataset_iid.replace("/", "_").lower()
                return f"xingyaoww/sweb.eval.x86_64.{slug}:latest"
            slug = dataset_iid.replace("/", "_").replace(".", "_").lower()
            return f"swebench/sweb.eval.x86_64.{slug}:latest"

        def _build_env(self, image_name: str) -> Any:
            """Build a mini-swe-agent ``DockerEnvironment`` (pre-pulls image)."""
            from memgym.gym.swe_bench.container import DockerBackend  # type: ignore

            backend = DockerBackend()
            return backend.create_env(
                image=image_name,
                cwd=self._cwd,
                env_vars={},
                timeout=self._timeout,
                verbose=self._verbose,
            )

        def _collect_patch(self, env: Any) -> Tuple[str, Optional[str]]:
            # Audit H8: ``git add -A`` would stage every untracked file under
            # /testbed including ``__pycache__/``, build artifacts, .pyc — the
            # SWE-Bench harness rejects diffs that touch unrelated files.
            # ``git diff HEAD`` returns the diff of all *tracked* modifications,
            # which is exactly what the harness scoring expects. Do NOT add
            # ``-A`` back without a follow-up to filter the patch.
            try:
                result = env.execute("git -C /testbed diff HEAD",
                                     cwd=self._cwd, timeout=30)
                if isinstance(result, dict):
                    return result.get("output", "") or "", None
                return str(result), None
            except Exception as exc:  # noqa: BLE001
                return "", f"{type(exc).__name__}: {exc}"

        def _evaluate(self, dataset_iid: str, patch_text: str,
                      meta: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
            """Score the patch via SWE-Bench harness. Returns (resolved, reward, info).

            Audit H7: ``SWEMemoryEnv`` constructs a *second* Docker container
            for the harness shape while our live container is still in
            ``_env_cache``. Until we lift the harness call out (proper fix is
            to call ``swe_bench/evaluate`` directly), at minimum guarantee the
            second container is torn down — success or exception — via
            try/finally with an explicit ``cleanup`` call.
            """
            if not patch_text.strip():
                return False, 0.0, {"reason": "empty_patch"}
            env_runner = None
            try:
                from memgym.gym.swe_bench.env import SWEMemoryEnv  # type: ignore

                env_runner = SWEMemoryEnv(
                    dataset=self._dataset,
                    split="train",
                    instance_id=dataset_iid,
                    agent_llm="openai/local-policy",  # only for harness shape
                    memory_model=None,
                    use_context_management=False,
                    use_docker=True,
                    environment_class="docker",
                    docker_image_source=self._docker_image_source,
                    max_steps=1,
                    agent_type="mini-swe",
                    skip_eval=False,
                    memory_gate=None,
                )
                # Use only the harness path; do not run the agent.
                if hasattr(env_runner, "_evaluate_patch"):
                    res = env_runner._evaluate_patch(patch_text)
                    if isinstance(res, tuple) and len(res) == 2:
                        resolved, info = res
                    elif isinstance(res, dict):
                        resolved = bool(res.get("resolved", False))
                        info = res
                    else:
                        resolved, info = bool(res), {}
                else:
                    resolved, info = False, {"reason": "no_evaluate_patch"}
            except Exception as exc:  # noqa: BLE001
                logger.exception("evaluate failed for %s", dataset_iid)
                return False, 0.0, {"reason": "harness_error", "error": str(exc)}
            finally:
                if env_runner is not None:
                    try:
                        type(self)._cleanup_one(env_runner)
                    except Exception as cleanup_exc:  # noqa: BLE001
                        logger.warning(
                            "second-env cleanup failed for %s: %s",
                            dataset_iid, cleanup_exc,
                        )
            reward = 1.0 if resolved else 0.0
            return resolved, reward, {"harness": info}

        @classmethod
        def _cleanup_one(cls, env: Any) -> None:
            try:
                close = getattr(env, "cleanup", None) or getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                logger.warning("env cleanup failed: %s", exc)

        @classmethod
        def _cleanup_all(cls) -> None:
            with cls._cache_lock:
                envs = list(cls._env_cache.values())
                cls._env_cache.clear()
                cls._env_meta.clear()
            for env in envs:
                cls._cleanup_one(env)

    return MemGymSWETool


# Module-level export — verl's ``initialize_tools_from_config`` resolves
# ``class_name: memgym.training.rl.way_a_verl.tool.MemGymSWETool`` by
# importing this module and reading ``MemGymSWETool`` off it. Build the
# class on import so the symbol exists at module load time.
MemGymSWETool = _make_tool_class()


def get_tool_class():
    """Public factory — preserved for callers that imported the old name."""
    return MemGymSWETool


__all__ = ["MemGymSWETool", "get_tool_class"]
