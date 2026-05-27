"""
SWE-bench Memory Environment for MemGym.

Provides a memory-testing wrapper for SWE-bench coding tasks.
"""

import subprocess
import json
import tempfile
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


from memgym.envs.base import BaseMemoryEnvironment, register_env
from memgym.memory import PassThroughMemoryModel, SummarizationMemoryModel, BaseMemoryModel
from memgym.agents import SWEReasoningModel, BaseReasoningModel
from memgym.utils.trajectory_recorder import TrajectoryRecorder
from memgym.utils.token_tracker import TokenTracker


class SwerexEnvironmentAdapter:
    """
    Adapter that wraps SWE-ReX DockerDeployment to match mini-swe-agent's environment interface.

    mini-swe-agent DockerEnvironment interface:
    - execute(command, cwd="", *, timeout=None) -> dict with "output" and "returncode"
    - get_template_vars() -> dict
    - cleanup()
    """

    def __init__(self, image: str, timeout: int = 60, cwd: str = "/testbed",
                 env_vars: Optional[Dict[str, str]] = None, verbose: bool = False):
        self.image = image
        self.timeout = timeout
        self.cwd = cwd
        self.env = env_vars or {}
        self.verbose = verbose
        self._deployment = None
        self._runtime = None

    def _ensure_started(self):
        """Lazily start the deployment."""
        if self._deployment is not None:
            return

        try:
            from swerex.deployment.docker import DockerDeployment
        except ImportError:
            raise ImportError("swerex not installed. Run: pip install swe-rex>=1.4.0")

        self._deployment = DockerDeployment(image=self.image)
        self._deployment.start()
        self._runtime = self._deployment.runtime

    def get_template_vars(self) -> Dict[str, Any]:
        """Return template variables (compatible with mini-swe-agent)."""
        return {
            "image": self.image,
            "cwd": self.cwd,
            "env": self.env,
            "timeout": self.timeout,
        }

    def execute(self, action, cwd: str = "", *, timeout: int = 0) -> Dict[str, Any]:
        """Execute a command and return dict with 'output' and 'returncode'.

        Args:
            action: Command string or dict with 'command' key (v2.2.4+ format)
        """
        self._ensure_started()

        # v2.2.4+ passes dict {"command": "..."}, v1.x passes string
        command = action.get("command", "") if isinstance(action, dict) else action

        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd

        # Prepend env vars and cd to working directory
        env_prefix = " ".join(f"{k}={v}" for k, v in self.env.items())
        full_command = f"cd {effective_cwd} && {env_prefix} {command}" if env_prefix else f"cd {effective_cwd} && {command}"

        try:
            result = self._runtime.execute(full_command, timeout=effective_timeout)
            output = result.stdout
            if result.stderr:
                output = output + "\n" + result.stderr if output else result.stderr
            return {"output": output, "returncode": result.exit_code}
        except Exception as e:
            return {"output": str(e), "returncode": 1}

    def cleanup(self):
        """Stop the deployment."""
        if self._deployment is not None:
            try:
                self._deployment.stop()
            except Exception:
                pass
            self._deployment = None
            self._runtime = None

    def close(self):
        """Alias for cleanup."""
        self.cleanup()

    def __del__(self):
        """Cleanup on garbage collection."""
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


@dataclass
class SWETask:
    """Represents a SWE-bench task instance.

    Fields base_commit, hints_text, test_patch, version, and environment_setup_commit
    are optional (default "") to support datasets like SWE-smith that lack them.
    image_name is used by SWE-smith (per-instance Docker image).
    """
    instance_id: str
    repo: str
    problem_statement: str
    patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    base_commit: str = ""
    hints_text: str = ""
    created_at: str = ""
    test_patch: str = ""
    version: str = ""
    environment_setup_commit: str = ""
    image_name: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SWETask":
        """Create SWETask from dictionary (HuggingFace format)."""
        # Parse FAIL_TO_PASS and PASS_TO_PASS (stored as JSON strings in the dataset)
        fail_to_pass = data.get("FAIL_TO_PASS", "[]")
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        pass_to_pass = data.get("PASS_TO_PASS", "[]")
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)

        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            problem_statement=data.get("problem_statement", ""),
            patch=data.get("patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            base_commit=data.get("base_commit", ""),
            hints_text=data.get("hints_text", ""),
            created_at=data.get("created_at", ""),
            test_patch=data.get("test_patch", ""),
            version=data.get("version", ""),
            environment_setup_commit=data.get("environment_setup_commit", ""),
            image_name=data.get("image_name", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to HuggingFace-style dictionary for swebench harness."""
        d = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "created_at": self.created_at,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "version": self.version,
            "environment_setup_commit": self.environment_setup_commit,
            "FAIL_TO_PASS": json.dumps(self.fail_to_pass),
            "PASS_TO_PASS": json.dumps(self.pass_to_pass),
        }
        if self.image_name:
            d["image_name"] = self.image_name
        return d


def _build_step_index(messages):
    """Build step-to-message-index mapping from agent.messages.

    Walks the message list after the initial system+user pair.
    Each (assistant, user) pair is one agent step.
    Extra user messages (from NonTerminatingExceptions) are skipped.
    """
    steps = []
    step_num = 0
    i = 2  # Skip system (0) and initial task (1)
    while i < len(messages):
        if messages[i].get("role") == "assistant":
            step_num += 1
            entry = {"step": step_num, "assistant_msg_index": i}
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                entry["observation_msg_index"] = i + 1
                i += 2
            else:
                entry["observation_msg_index"] = None
                i += 1
            steps.append(entry)
        else:
            i += 1  # Extra user message (format error, timeout)
    return steps


class SWEMemoryEnv(BaseMemoryEnvironment):
    """
    Memory-testing wrapper for SWE-bench tasks.

    Features:
    - Token tracking: Monitor context window size
    - Trajectory recording: Record full interaction history
    - Optional context management: LLM-based summarization
    - Docker evaluation: Use Epoch AI images for testing
    - Dataset access: Load tasks from HuggingFace

    Example:
        >>> env = get_env("swe", dataset="princeton-nlp/SWE-bench_Lite")
        >>> obs, info = env.reset(seed=42)  # Random task
        >>> action = {"patch": "diff --git a/..."}
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    DATASETS = {
        "swe-bench": "princeton-nlp/SWE-bench",
        "swe-bench-lite": "princeton-nlp/SWE-bench_Lite",
        "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
    }

    # Docker image sources
    DOCKER_IMAGE_SOURCES = {
        "epoch": "ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}",
        "swebench": "docker.io/swebench/sweb.eval.x86_64.{instance_id}:latest",
        "swe-gym": "docker.io/xingyaoww/sweb.eval.x86_64.{instance_id}:latest",
    }

    def __init__(
        self,
        instance_id: Optional[str] = None,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        memory_model: Optional[BaseMemoryModel] = None,
        reasoning_model: Optional[BaseReasoningModel] = None,
        use_context_management: bool = False,
        max_context_tokens: int = 8000,
        tokenizer_name: str = "gpt-4",
        max_steps: int = 10,
        use_docker: bool = True,
        docker_arch: str = "x86_64",
        environment_class: str = "docker",
        docker_image_dir: str = "${HOME}/swe-bench/dockers",
        docker_image_source: str = "swebench",
        agent_llm: Optional[str] = None,
        agent_llm_args: Optional[Dict] = None,
        agent_type: str = "mini-swe",  # "mini-swe" or "codeact"
        skip_eval: bool = False,
        bwrap_rootfs: Optional[str] = None,
        bwrap_executable: Optional[str] = None,
        bwrap_wrapper_args: Optional[List[str]] = None,
        memory_gate: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config=config)
        self._memory_gate = memory_gate

        self._fixed_instance_id = instance_id
        self.instance_id = instance_id
        self.dataset = dataset
        self.split = split
        self._use_context_management = use_context_management
        self._max_context_tokens = max_context_tokens
        self._max_steps = max_steps
        self._tokenizer_name = tokenizer_name
        self._use_docker = use_docker
        self._docker_arch = docker_arch
        self._environment_class = environment_class
        self._docker_image_dir = docker_image_dir
        self._docker_image_source = docker_image_source
        self._agent_llm = agent_llm
        self._agent_llm_args = agent_llm_args or {}
        self._agent_type = agent_type
        self._skip_eval = skip_eval
        self._bwrap_rootfs = bwrap_rootfs
        self._bwrap_executable = bwrap_executable
        self._bwrap_wrapper_args = list(bwrap_wrapper_args or [])

        # Initialize memory model
        if memory_model is not None:
            self._memory_model = memory_model
        elif use_context_management:
            self._memory_model = SummarizationMemoryModel(max_tokens=max_context_tokens)
        else:
            self._memory_model = PassThroughMemoryModel()

        # Initialize reasoning model
        self._reasoning_model = reasoning_model or SWEReasoningModel()

        # Initialize trackers
        self._token_tracker = TokenTracker(tokenizer_name=tokenizer_name)
        self._trajectory_recorder = TrajectoryRecorder()

        # State
        self._task: Optional[SWETask] = None
        self._tasks_cache: Optional[List[Dict]] = None
        self._current_step = 0
        self._episode_observations: List[str] = []
        self._last_reward = 0.0
        self._done = False
        self._docker_env = None  # Store Docker env for cleanup

    def _load_dataset(self) -> List[Dict]:
        """Load tasks from HuggingFace dataset."""
        if self._tasks_cache is not None:
            return self._tasks_cache
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.dataset, split=self.split)
            self._tasks_cache = list(dataset)
            return self._tasks_cache
        except ImportError:
            raise ImportError("datasets package required: pip install datasets")

    def _get_task(self, instance_id: str) -> SWETask:
        """Get a specific task by instance ID."""
        tasks = self._load_dataset()
        for task_data in tasks:
            if task_data.get("instance_id") == instance_id:
                return SWETask.from_dict(task_data)
        raise ValueError(f"Task not found: {instance_id}")

    def _get_task_list(self) -> List[str]:
        """Get list of all available task instance IDs."""
        tasks = self._load_dataset()
        return [t.get("instance_id", "") for t in tasks]

    @property
    def is_random_mode(self) -> bool:
        """Check if environment samples random tasks on reset."""
        return self._fixed_instance_id is None

    @property
    def num_tasks(self) -> int:
        """Get number of available tasks in the dataset split."""
        return len(self._get_task_list())

    def get_all_task_ids(self) -> List[str]:
        """Get all available task instance IDs."""
        return self._get_task_list()

    @property
    def memory_model(self) -> BaseMemoryModel:
        return self._memory_model

    @property
    def reasoning_model(self) -> BaseReasoningModel:
        return self._reasoning_model

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode."""
        import random

        if seed is not None:
            random.seed(seed)

        # Determine task
        if options and "instance_id" in options:
            instance_id = options["instance_id"]
        elif self._fixed_instance_id is not None:
            instance_id = self._fixed_instance_id
        else:
            task_ids = self._get_task_list()
            instance_id = random.choice(task_ids)

        self._task = self._get_task(instance_id)
        self.instance_id = instance_id

        # Reset state
        self._current_step = 0
        self._episode_observations = []
        self._last_reward = 0.0
        self._done = False

        # Reset models
        self._memory_model.reset()
        self._reasoning_model.reset()
        self._token_tracker.reset()

        # Start trajectory
        self._trajectory_recorder.start_episode(metadata={
            "instance_id": instance_id,
            "repo": self._task.repo,
            "dataset": self.dataset,
            "split": self.split
        })

        observation = self._build_observation()
        memory_action = self._memory_model.process_observation(
            observation["problem_statement"],
            metadata={"step": 0}
        )

        obs_text = json.dumps(observation)
        token_count = self._token_tracker.update(obs_text, step=0)

        info = {
            "instance_id": instance_id,
            "repo": self._task.repo,
            "memory_info": {
                "tokens": token_count,
                "memory_action": str(memory_action),
                "memory_stats": self._memory_model.get_stats()
            }
        }

        return observation, info

    def _build_observation(self) -> Dict[str, Any]:
        """Build the observation dictionary."""
        if self._task is None:
            return {}
        return {
            "instance_id": self._task.instance_id,
            "repo": self._task.repo,
            "problem_statement": self._task.problem_statement,
            "hints_text": self._task.hints_text,
            "base_commit": self._task.base_commit,
            "version": self._task.version,
            "step": self._current_step
        }

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self._current_step += 1
        self._episode_step += 1
        self._total_steps += 1

        patch = action.get("patch", "")
        action_str = f"Patch submitted:\n{patch[:500]}..." if len(patch) > 500 else f"Patch submitted:\n{patch}"

        memory_action = self._memory_model.process_observation(
            action_str,
            metadata={"step": self._current_step}
        )

        context = self._memory_model.get_context()
        token_count = self._token_tracker.update(action_str, step=self._current_step)

        # Evaluate
        reward = 0.0
        evaluation_result = None
        if self._use_docker and patch and not self._skip_eval:
            evaluation_result = self._evaluate_patch(patch)
            reward = 1.0 if evaluation_result.get("success", False) else 0.0

        terminated = reward > 0 or self._current_step >= self._max_steps
        truncated = self._current_step >= self._max_steps and reward == 0
        self._done = terminated or truncated
        self._last_reward = reward

        observation = self._build_observation()
        if evaluation_result:
            observation["evaluation_feedback"] = evaluation_result.get("message", "")

        reasoning_action = self._reasoning_model.set_action_from_external(action_str, context)

        self._trajectory_recorder.record_step(
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"evaluation": evaluation_result},
            metadata={"tokens": token_count, "episode_step": self._current_step},
            memory_action=memory_action,
            context=context,
            reasoning_action=reasoning_action
        )

        if self._done:
            self._trajectory_recorder.end_episode()

        info = {
            "instance_id": self.instance_id,
            "evaluation": evaluation_result,
            "memory_info": {
                "tokens": token_count,
                "memory_action": str(memory_action),
                "memory_stats": self._memory_model.get_stats(),
                "reasoning_stats": self._reasoning_model.get_stats()
            }
        }

        return observation, reward, terminated, truncated, info

    def _get_docker_image_name(self) -> str:
        """Get the Docker image name based on configuration."""
        # SWE-smith: use per-instance image_name field directly
        if self._docker_image_source == "swe-smith" and self._task and self._task.image_name:
            return self._task.image_name

        pattern = self.DOCKER_IMAGE_SOURCES.get(self._docker_image_source,
                                                  self.DOCKER_IMAGE_SOURCES["swebench"])
        # Each registry uses a different __ replacement convention:
        #   swebench: _1776_  (official SWE-bench convention)
        #   swe-gym:  _s_     (xingyaoww convention)
        #   epoch:    __      (keeps as-is)
        if self._docker_image_source == "swe-gym":
            instance_id_docker = self.instance_id.replace("__", "_s_")
        elif self._docker_image_source == "epoch":
            instance_id_docker = self.instance_id
        else:
            instance_id_docker = self.instance_id.replace("__", "_1776_")
        return pattern.format(arch=self._docker_arch, instance_id=instance_id_docker).lower()

    def _evaluate_patch(self, patch: str, env=None) -> Dict[str, Any]:
        """Evaluate a patch using the official SWE-bench harness.

        Uses swebench.harness.run_evaluation to run tests in a Docker container
        and swebench.harness.grading to parse results — the same pipeline used
        by sb-cli and the SWE-bench leaderboard.

        Args:
            patch: The unified diff patch to evaluate
            env: Unused (kept for API compatibility). The official harness
                 manages its own containers.
        """
        if not self._use_docker:
            return {"success": False, "message": "Docker evaluation disabled"}

        if not self._is_valid_patch(patch):
            return {"success": False, "message": "Invalid patch format (not a valid unified diff)"}

        return self._evaluate_with_harness(patch)

    def _evaluate_with_harness(self, patch: str) -> Dict[str, Any]:
        """Evaluate a patch using the official SWE-bench evaluation harness.

        This delegates to swebench.harness which:
        1. Builds an eval script from the repo's test spec (correct test runner,
           correct directives extracted from test_patch, correct env activation).
        2. Applies the model patch and test patch inside a Docker container.
        3. Runs the project-specific test command (e.g. runtests.py for Django,
           pytest for others) on the relevant test modules.
        4. Parses the test output with repo-specific log parsers.
        5. Grades results by comparing individual test outcomes against the
           FAIL_TO_PASS and PASS_TO_PASS lists.

        This avoids all the pitfalls of our previous custom eval:
        - No need to convert test IDs or build test commands ourselves
        - No docstring-style test name issues
        - No conda activation / locale issues
        - No exit-code-only checking (uses log parsing instead)
        """
        try:
            import docker as docker_mod
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError as e:
            return {"success": False, "message": f"swebench harness not installed: {e}"}

        if not self._task:
            return {"success": False, "message": "No task loaded"}

        # Patch swebench with SWE-Gym repo specs if needed
        if self._docker_image_source == "swe-gym":
            from memgym.utils.swegym_specs import patch_swebench_specs
            patch_swebench_specs()

        try:
            # Build the instance dict that swebench expects
            instance = self._task.to_dict()

            # Create the test spec (encapsulates eval script, test commands, etc.)
            test_spec = make_test_spec(instance)

            # Build the prediction dict
            pred = {
                "instance_id": self._task.instance_id,
                "model_patch": patch,
                "model_name_or_path": self._agent_llm or "unknown",
            }

            # Use namespace="swebench" to pull pre-built images from DockerHub
            client = docker_mod.from_env()
            run_id = f"memgym_{self._task.instance_id}"

            result = run_instance(
                test_spec=test_spec,
                pred=pred,
                rm_image=False,
                force_rebuild=False,
                client=client,
                run_id=run_id,
                timeout=300,
            )

            resolved = result.get("resolved", False)
            return {
                "success": resolved,
                "message": f"SWE-bench harness: {'RESOLVED' if resolved else 'UNRESOLVED'}",
                "returncode": 0 if resolved else 1,
                "harness_result": result,
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "message": f"SWE-bench harness error: {e}\n{traceback.format_exc()}",
                "returncode": 1,
            }

    @staticmethod
    def _is_valid_patch(patch: str) -> bool:
        """Validate patch is a proper unified diff."""
        if not patch or not patch.strip():
            return False
        lines = patch.split('\n')
        has_header = any(l.startswith('diff --git') or l.startswith('---') for l in lines)
        has_hunk = any(l.startswith('@@') for l in lines)
        has_changes = any(l.startswith('+') or l.startswith('-') for l in lines)
        return has_header and has_hunk and has_changes

    def _calculate_reward(self, observation: Any, action: Any, info: Dict[str, Any]) -> float:
        evaluation = info.get("evaluation", {})
        return 1.0 if evaluation and evaluation.get("success", False) else 0.0

    def _evaluate_memory(self) -> Dict[str, Any]:
        return {
            "context_management_enabled": self._use_context_management,
            "max_context_tokens": self._max_context_tokens,
            "token_stats": self._token_tracker.get_stats(),
            "memory_model_stats": self._memory_model.get_stats()
        }

    def get_trajectory(self) -> Dict[str, Any]:
        return self._trajectory_recorder.to_dict()

    def save_trajectory(self, filepath: str) -> None:
        """Save both detailed and concise trajectories."""
        # Save detailed trajectory (original format)
        self._trajectory_recorder.to_json(filepath, indent=2)

        # Save concise trajectory (messages only for training)
        from pathlib import Path
        path = Path(filepath)
        concise_path = path.parent / f"{path.stem}_concise{path.suffix}"
        self._trajectory_recorder.to_json_concise(str(concise_path), indent=2)

    def _get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration from SWE-bench config."""
        import yaml
        from minisweagent.config import builtin_config_dir

        # Try benchmarks/ first (mini-swe-agent v2.2+), then extra/ (legacy)
        for subdir in ["benchmarks", "extra"]:
            config_path = builtin_config_dir / subdir / "swebench.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                return config_data.get("environment", {})
        print("WARNING: swebench.yaml not found in benchmarks/ or extra/. "
              "Using empty environment config. Install mini-swe-agent v2.2+ "
              "or check your minisweagent installation.")
        return {}

    def _create_environment(self, verbose: bool = False):
        """Create the appropriate environment based on environment_class.

        Uses the container backend abstraction from container.py to support
        Docker, Singularity, Bubblewrap, SWE-ReX, and local environments.
        """
        from memgym.gym.swe_bench.container import get_backend

        image_name = self._get_docker_image_name()
        env_config = self._get_env_config()

        # Standard SWE-bench environment settings
        cwd = env_config.get("cwd", "/testbed")
        env_vars = env_config.get("env", {
            "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
            "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"
        })
        timeout = env_config.get("timeout", 60)

        backend_kwargs = {}
        if self._environment_class == "bubblewrap":
            backend_kwargs["rootfs_dir"] = getattr(self, "_bwrap_rootfs", None)
            backend_kwargs["executable"] = getattr(self, "_bwrap_executable", None)
            backend_kwargs["wrapper_args"] = getattr(self, "_bwrap_wrapper_args", [])

        backend = get_backend(self._environment_class, **backend_kwargs)
        return backend.create_env(
            image=image_name,
            cwd=cwd,
            env_vars=env_vars,
            timeout=timeout,
            verbose=verbose,
            instance_id=getattr(self, "instance_id", "unknown"),
            rootfs=getattr(self, "_bwrap_rootfs", None),
            executable=getattr(self, "_bwrap_executable", None),
            wrapper_args=getattr(self, "_bwrap_wrapper_args", []),
        )

    def run_episode(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run a full episode using mini-swe-agent or CodeAct agent with memory filtering.

        Args:
            verbose: Print detailed progress

        Returns:
            Dictionary with episode results including reward, patch, and memory stats
        """
        if self._agent_llm is None:
            raise ValueError("agent_llm must be set to use run_episode()")

        # Reset environment first to initialize task
        if self._task is None:
            self.reset()

        # Dispatch to CodeAct agent if requested
        if self._agent_type == "codeact":
            return self._run_codeact_episode(verbose=verbose)

        # Reset memory and reasoning models
        self._memory_model.reset()
        self._reasoning_model.reset()
        self._token_tracker.reset()

        # Get observation
        obs = self._build_observation()

        # Import mini-swe-agent components
        try:
            from minisweagent.models import get_model
            from minisweagent.agents.default import DefaultAgent, AgentConfig
            from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent
        except ImportError as e:
            raise ImportError(
                f"mini-swe-agent not installed. Run: ./install.sh --swe\nError: {e}"
            )

        # Create mini-swe-agent environment based on environment_class
        env = self._create_environment(verbose=verbose)
        self._docker_env = env  # Store for cleanup in close()

        # Enable modify_params for Bedrock models to fix consecutive user/tool block merging
        if self._agent_llm and self._agent_llm.startswith("bedrock/"):
            import litellm
            litellm.modify_params = True
            os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

        # Self-hosted OpenAI-compatible endpoints (e.g. vLLM for the
        # memory A/B harness) have no litellm price entry, so mini-swe-agent's cost
        # tracker raises on every response. Default to ignore_errors when
        # OPENAI_API_BASE points at a non-Anthropic, non-OpenAI server.
        api_base = os.environ.get("OPENAI_API_BASE", "")
        if api_base and ("127.0.0.1" in api_base or "localhost" in api_base):
            os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

        # Create LLM model — merge swebench.yaml model config with CLI overrides.
        # `model_kwargs` is a nested dict (e.g. drop_params, temperature); a plain
        # top-level .update() would wipe the yaml's kwargs whenever CLI passes
        # any kwarg. Deep-merge at the model_kwargs level instead.
        model_config = self._get_model_config()
        cli_overrides = dict(self._agent_llm_args or {})
        cli_model_kwargs = cli_overrides.pop("model_kwargs", None)
        model_config.update(cli_overrides)
        if cli_model_kwargs:
            merged_kwargs = dict(model_config.get("model_kwargs") or {})
            merged_kwargs.update(cli_model_kwargs)
            model_config["model_kwargs"] = merged_kwargs

        # Fix Bedrock compatibility: Bedrock's Converse API does not support
        # parallel_tool_calls, and dict-style tool_choice (e.g. {"type": "auto"})
        # gets incorrectly mapped by litellm to toolChoice.tool with empty name.
        # Solution: strip both params — litellm defaults to auto tool_choice.
        if self._agent_llm and self._agent_llm.startswith("bedrock/"):
            mkw = model_config.get("model_kwargs", {})
            mkw.pop("parallel_tool_calls", None)
            mkw.pop("tool_choice", None)
            model_config["model_kwargs"] = mkw

        # Use text-based model for Llama/Nemotron (Bedrock tool calling is broken)
        # and for self-hosted Qwen3-thinking (tool_choice="auto" not accepted
        # by our vLLM launch). Route through our reasoning-aware subclass so
        # vLLM's --reasoning-parser qwen3 (which routes the whole turn to
        # `reasoning_content` and leaves `content=""`) doesn't starve the
        # action parser. See `reasoning_litellm_model.py` for B2/B6 details.
        if self._is_textbased_model():
            from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel  # noqa: F401 — import kept for any downstream isinstance checks
            model_config["model_class"] = (
                "memgym.gym.swe_bench.reasoning_litellm_model.ReasoningAwareLitellmTextbasedModel"
            )
            if verbose:
                print(f"  Using text-based mode (reasoning-aware: B2 content-fallback + B6 <think>-strip)")

        model = get_model(
            input_model_name=self._agent_llm,
            config=model_config
        )

        # Create agent (with or without memory wrapper)
        if self._use_context_management:
            agent = MemoryAwareSWEAgent(
                model=model,
                env=env,
                memory_manager=self._memory_model,
                verbose=verbose,
                memory_gate=self._memory_gate,
                config_class=AgentConfig,
                **self._get_mini_swe_config()
            )
        else:
            agent = DefaultAgent(
                model=model,
                env=env,
                config_class=AgentConfig,
                **self._get_mini_swe_config()
            )

        # Start trajectory recording
        self._trajectory_recorder.start_episode(metadata={
            "instance_id": self.instance_id,
            "repo": obs["repo"],
            "agent_llm": self._agent_llm,
            "context_management": self._use_context_management,
            "agent_type": "mini-swe-agent"
        })

        if verbose:
            print(f"\nStarting mini-swe-agent episode:")
            print(f"  Instance: {self.instance_id}")
            print(f"  Repo: {obs['repo']}")
            print(f"  Agent LLM: {self._agent_llm}")
            print(f"  Memory: {'ENABLED' if self._use_context_management else 'DISABLED'}")
            print()

        # Build task description
        task = f"{obs['problem_statement']}\n\nHints: {obs.get('hints_text', 'None')}"

        # Run agent (mini-swe-agent loop)
        # Note: Don't pass 'version' - conflicts with platform.uname() version
        try:
            result = agent.run(
                task=task,
                repo=obs['repo'],
                base_commit=obs.get('base_commit', '')
            )

            # v2.2.4+ returns dict {"exit_status": ..., "submission": ...}
            if isinstance(result, dict):
                status = result.get("exit_status", "Unknown")
                message = result.get("submission", "")
            else:
                # v1.x returned tuple (status, message)
                status, message = result

            if verbose:
                print(f"\nAgent finished: {status}")
                print(f"Final output length: {len(message)} chars")

            # Extract patch from final output
            patch = message

            def _is_valid_patch(p):
                """Check if string looks like a valid unified diff."""
                if not p or not p.strip():
                    return False
                stripped = p.strip()
                return stripped.startswith("diff --git") or stripped.startswith("---")

            # If patch doesn't look like a valid diff (e.g., v2.2.4 returns "submission"),
            # or agent hit limits, extract diff from the still-running container
            if not _is_valid_patch(patch) or status == "LimitsExceeded":
                try:
                    # Only diff .py files that existed at HEAD (the original repo state).
                    # Excludes any new files the agent created and staged with git add
                    # (debug scripts, .md summaries, .backup files, test helpers, etc.)
                    # that cause "Patch Apply Failed" during swebench evaluation.
                    extract_cmd = (
                        "cd /testbed && git diff HEAD -- "
                        "$(git ls-tree -r HEAD --name-only | grep '\\.py$' | tr '\\n' ' ')"
                    )
                    diff_output = env.execute({"command": extract_cmd})
                    extracted_patch = diff_output.get("output", "").strip()
                    if extracted_patch:
                        patch = extracted_patch
                        if verbose:
                            print(f"  Extracted patch from container: {len(patch)} chars")
                    else:
                        # Fallback: include staged changes (may include new files)
                        fallback_cmd = "cd /testbed && git add -A -- '*.py' && git diff --cached -- '*.py'"
                        diff_output = env.execute({"command": fallback_cmd})
                        extracted_patch = diff_output.get("output", "").strip()
                        if extracted_patch:
                            patch = extracted_patch
                            if verbose:
                                print(f"  Extracted patch (fallback) from container: {len(patch)} chars")
                except Exception:
                    pass  # Container may already be gone

        except Exception as e:
            if verbose:
                print(f"\nAgent error: {e}")
                import traceback
                traceback.print_exc()
            status = "Error"
            message = str(e)
            patch = f"# Error: {e}"

            # Return early on error to avoid further issues
            return {
                "reward": 0.0,
                "patch": patch,
                "patch_size": len(patch),
                "status": status,
                "terminated": True,
                "evaluation": {"error": str(e)},
                "memory_metrics": self._evaluate_memory(),
                "token_stats": self._token_tracker.get_stats(),
                "trajectory": self.get_trajectory(),
                "wrapper_stats": {},
                "error": str(e)
            }

        # Record all agent steps to trajectory
        for i, msg in enumerate(agent.messages):
            self._trajectory_recorder.record_step(
                observation=f"{msg['role']}: {(msg.get('content') or '')[:200]}...",
                action=msg.get('content') or '',
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
                metadata={"step": i, "role": msg["role"]},
                memory_action=None,
                context=None,
                reasoning_action=None
            )

        # Evaluate patch with Docker (use running container for test deps)
        reward = 0.0
        evaluation_result = None
        if self._use_docker and patch and not self._skip_eval:
            evaluation_result = self._evaluate_patch(patch, env=env)
            reward = 1.0 if evaluation_result.get("success", False) else 0.0

            if verbose:
                eval_status = 'PASS' if reward > 0 else 'FAIL'
                print(f"Docker evaluation: {eval_status}")
                # Show evaluation details for debugging
                eval_msg = evaluation_result.get("message", "")
                if eval_msg and not reward:
                    # Show last 500 chars of failure output
                    print(f"  Eval output (last 500 chars): ...{eval_msg[-500:]}")

            # Record evaluation result in trajectory
            self._trajectory_recorder.record_step(
                observation=f"Docker evaluation: {'PASS' if reward > 0 else 'FAIL'}",
                action="evaluate_patch",
                reward=reward,
                terminated=True,
                truncated=False,
                info={"evaluation": evaluation_result},
                metadata={"step": "evaluation", "patch_valid": self._is_valid_patch(patch)},
                memory_action=None,
                context=None,
                reasoning_action=None
            )

        # End trajectory
        self._trajectory_recorder.end_episode()

        # Collect wrapper statistics and training trajectory
        wrapper_stats = {}
        training_trajectory = None
        if self._use_context_management and hasattr(agent, 'get_wrapper_stats'):
            wrapper_stats["agent"] = agent.get_wrapper_stats()
        if hasattr(agent, 'get_training_trajectory'):
            training_trajectory = agent.get_training_trajectory()
        if hasattr(agent, 'memory_manager') and hasattr(agent.memory_manager, 'get_condensation_history'):
            if training_trajectory:
                training_trajectory["condensation_history"] = agent.memory_manager.get_condensation_history()
            if verbose:
                stats = wrapper_stats["agent"]
                print(f"\nMemory wrapper stats:")
                print(f"  Total queries: {stats['total_queries']}")
                print(f"  Times filtered: {stats['times_filtered']}")
                print(f"  Avg compression: {stats.get('avg_compression_ratio', 1.0):.2f}x")

        result = {
            "reward": reward,
            "patch": patch,
            "patch_size": len(patch),
            "status": status,
            "terminated": True,
            "evaluation": evaluation_result,
            "memory_metrics": self._evaluate_memory(),
            "token_stats": self._token_tracker.get_stats(),
            "trajectory": self.get_trajectory()
        }

        # Add wrapper stats and training trajectory if available
        if wrapper_stats:
            result["wrapper_stats"] = wrapper_stats
        if training_trajectory:
            training_trajectory["instance_id"] = self.instance_id
            training_trajectory["repo"] = obs.get("repo", "")
            training_trajectory["outcome"] = "resolved" if reward > 0 else "unresolved"
            training_trajectory["patch"] = patch[:5000]  # Truncate for training
            result["training_trajectory"] = training_trajectory

        # Build replay data (full untruncated messages for replay mechanism)
        replay_data = None
        if hasattr(agent, 'messages') and agent.messages:
            memory_config = {"max_tokens": getattr(self._memory_model, 'max_tokens', None)}
            for attr in ['max_size', 'keep_first', 'attention_window', 'summarization_model']:
                if hasattr(self._memory_model, attr):
                    memory_config[attr] = getattr(self._memory_model, attr)
            replay_data = {
                "version": 1,
                "instance_id": self.instance_id,
                "dataset": self.dataset,
                "model": self._agent_llm,
                "memory_strategy": self._memory_model.__class__.__name__,
                "memory_config": memory_config,
                "status": status,
                "reward": reward,
                "patch": patch,
                "messages": list(agent.messages),
                "steps": _build_step_index(agent.messages),
            }
            result["replay_data"] = replay_data

        if verbose:
            print(f"\nEpisode completed:")
            print(f"  Reward: {reward}")
            print(f"  Patch size: {len(patch)} chars")

        return result

    def _is_textbased_model(self) -> bool:
        """Check if the agent model requires text-based mode (no function calling).

        Llama models on Bedrock's Converse API return tool calls as plain text
        JSON in the content field instead of structured toolUse blocks. This
        causes mini-swe-agent's tool_call parser to see empty tool_calls,
        raise FormatError, and the agent submits with empty patches.

        Fix: Use LitellmTextbasedModel + swebench_backticks.yaml config which
        extracts commands from ```mswea_bash_command``` code blocks via regex.
        """
        if not self._agent_llm:
            return False
        model_lower = self._agent_llm.lower()
        # Llama, Nemotron, and GPT-OSS on Bedrock don't return structured
        # toolUse blocks — they embed tool calls as plain-text JSON in the
        # content field, which mini-swe-agent's parser can't handle.
        #
        # Self-hosted Qwen reasoning models (e.g. `openai/qwen36-35b-a3b-fp8`
        # served by vLLM for the memory A/B harness) also need text mode:
        # vLLM rejects `tool_choice="auto"` unless launched with
        # `--enable-auto-tool-choice --tool-call-parser`, and the backticks
        # config omits the `parallel_tool_calls` flag that triggers litellm
        # to inject tool_choice.
        textbased_names = [
            "llama", "nemotron", "gpt-oss",
            "qwen36", "qwen3-14b-thinking", "qwen3-8b-thinking",
        ]
        return any(name in model_lower for name in textbased_names)

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from SWE-bench config."""
        import yaml
        from minisweagent.config import builtin_config_dir

        # Use text-based config for models that don't support Bedrock tool calling
        if self._is_textbased_model():
            config_name = "swebench_backticks.yaml"
        else:
            config_name = "swebench.yaml"

        # Try benchmarks/ first (mini-swe-agent v2.2+), then extra/ (legacy)
        for subdir in ["benchmarks", "extra"]:
            config_path = builtin_config_dir / subdir / config_name
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                return config_data.get("model", {})
        # Fallback to swebench.yaml if backticks config doesn't exist
        if config_name != "swebench.yaml":
            for subdir in ["benchmarks", "extra"]:
                config_path = builtin_config_dir / subdir / "swebench.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                    return config_data.get("model", {})
        print("WARNING: swebench.yaml not found in benchmarks/ or extra/. "
              "Using empty model config. This may cause tool-call parsing "
              "issues (text-based prompts vs function-calling).")
        return {}

    def _get_mini_swe_config(self) -> Dict[str, Any]:
        """Get configuration for mini-swe-agent using SWE-bench config."""
        import yaml
        from minisweagent.config import get_config_path, builtin_config_dir

        # Use text-based config for models that don't support Bedrock tool calling
        if self._is_textbased_model():
            config_name = "swebench_backticks.yaml"
        else:
            config_name = "swebench.yaml"

        # Use SWE-bench specific config (handles Docker templates properly)
        # Try benchmarks/ first (mini-swe-agent v2.2+), then extra/ (legacy)
        config_path = None
        for subdir in ["benchmarks", "extra"]:
            candidate = builtin_config_dir / subdir / config_name
            if candidate.exists():
                config_path = candidate
                break
        # Fallback to swebench.yaml if backticks not found
        if config_path is None and config_name != "swebench.yaml":
            for subdir in ["benchmarks", "extra"]:
                candidate = builtin_config_dir / subdir / "swebench.yaml"
                if candidate.exists():
                    config_path = candidate
                    break
        if config_path is None:
            config_path = get_config_path("default.yaml")
            print("WARNING: swebench.yaml not found in benchmarks/ or extra/. "
                  "Falling back to default.yaml which uses text-based prompts "
                  "incompatible with LitellmModel's function-calling format. "
                  "This will cause 'No tool calls found' errors.")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Extract agent config and override limits
        agent_config = config_data.get("agent", {})
        agent_config["step_limit"] = self._max_steps
        agent_config["cost_limit"] = 3.0

        return agent_config

    def _run_codeact_episode(self, verbose: bool = False) -> Dict[str, Any]:
        """Run a full episode using OpenHands CodeAct agent.

        Uses OpenHands' own controller loop with MemGym strategy mapped to
        an OpenHands condenser config. Runtime is created externally so it
        stays alive for proper patch extraction via git diff.
        """
        try:
            from memgym.agents.codeact_wrapper import run_codeact_episode
        except ImportError as e:
            raise ImportError(
                f"OpenHands not installed. Run: ./install.sh --openhands\nError: {e}"
            )

        obs = self._build_observation()
        # Build task with repo context for the CodeAct agent.
        # The agent runs in OpenHands' generic container with /workspace.
        # If the SWE-bench repo is pre-installed at /testbed, tell the agent.
        # Otherwise, include repo/commit info so the agent can clone it.
        repo = obs.get('repo', '')
        base_commit = obs.get('base_commit', '')
        task = (
            f"Please fix the following issue in the {repo} repository.\n"
            f"The repository may already be cloned at /testbed. "
            f"If not, clone it: git clone https://github.com/{repo}.git /workspace/{repo.split('/')[-1]} && "
            f"cd /workspace/{repo.split('/')[-1]}"
        )
        if base_commit:
            task += f" && git checkout {base_commit}"
        task += (
            f"\n\n{obs['problem_statement']}\n\nHints: {obs.get('hints_text', 'None')}"
        )

        # Determine memory strategy name from the memory model class
        strategy_name = self._memory_model.__class__.__name__
        strategy_map = {
            "PassThroughMemory": "none",
            "NaiveSummarizationMemory": "naive",
            "ObservationMaskingMemory": "observation_masking",
            "LLMSummarizingMemory": "llm_summarizing",
            "StructuredSummaryMemory": "structured_summary",
            "SlidingWindowSummaryMemory": "sliding_window",
            "PipelineMemory": "none",  # Fallback; pipeline needs special handling
        }
        memory_strategy = strategy_map.get(strategy_name, "none")

        # Get strategy-specific params from the memory model
        summarization_model = getattr(self._memory_model, 'summarization_model', 'gpt-4o-mini')
        max_size = getattr(self._memory_model, 'max_size', 100)
        keep_first_val = getattr(self._memory_model, 'keep_first', 1)
        attention_window_val = getattr(self._memory_model, 'attention_window', 100)

        # Build Docker image name for SWE-bench
        sandbox_image = self._get_docker_image_name() if self._use_docker else None

        if verbose:
            print(f"\nStarting CodeAct agent episode:")
            print(f"  Instance: {self.instance_id}")
            print(f"  Repo: {obs['repo']}")
            print(f"  Agent LLM: {self._agent_llm}")
            print(f"  Memory: {memory_strategy} (from {strategy_name})")

        result = run_codeact_episode(
            task=task,
            model=self._agent_llm,
            memory_strategy=memory_strategy,
            sandbox_image=sandbox_image,
            max_iterations=self._max_steps,
            max_budget=100.0,  # Don't limit by cost — use step_limit instead
            runtime="docker" if self._use_docker else "local",
            save_trajectory_path=None,
            verbose=verbose,
            summarization_model=summarization_model,
            max_size=max_size,
            keep_first=keep_first_val,
            attention_window=attention_window_val,
        )

        # Evaluate patch with official harness
        patch = result.get("patch", "")
        reward = 0.0
        evaluation_result = None
        if self._use_docker and patch and not self._skip_eval:
            evaluation_result = self._evaluate_patch(patch)
            reward = 1.0 if evaluation_result.get("success", False) else 0.0

        if verbose:
            eval_status = 'PASS' if reward > 0 else 'FAIL'
            print(f"  Evaluation: {eval_status}")

        # Build wrapper_stats in the same shape as mini-swe-agent path
        # so evaluate.py's save_summary() can process both uniformly
        wrapper_stats = {}
        condenser_stats = result.get("condenser_stats", {})
        if condenser_stats:
            wrapper_stats["agent"] = condenser_stats

        # Build training trajectory in same format as mini-swe-agent path
        training_trajectory = result.get("training_trajectory")
        if training_trajectory:
            training_trajectory["instance_id"] = self.instance_id
            training_trajectory["repo"] = obs.get("repo", "")
            training_trajectory["model"] = self._agent_llm
            training_trajectory["outcome"] = "resolved" if reward > 0 else "unresolved"
            training_trajectory["patch"] = patch[:5000]

        output = {
            "reward": reward,
            "patch": patch,
            "patch_size": len(patch),
            "status": result.get("status", "Unknown"),
            "terminated": True,
            "evaluation": evaluation_result,
            "memory_metrics": self._evaluate_memory(),
            "token_stats": self._token_tracker.get_stats(),
            "trajectory": {},
            "agent_type": "codeact",
            "codeact_stats": result,
        }
        if wrapper_stats:
            output["wrapper_stats"] = wrapper_stats
        if training_trajectory:
            output["training_trajectory"] = training_trajectory

        return output

    def close(self) -> None:
        """Cleanup Docker container if running."""
        if self._docker_env is not None:
            try:
                self._docker_env.cleanup()
            except Exception:
                pass
            self._docker_env = None


# Register in environment registry
register_env("swe", SWEMemoryEnv)
register_env("swe-bench", SWEMemoryEnv)
