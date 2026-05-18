"""
SWE-bench runner for MemGym.

Orchestrates SWE-bench coding episodes with memory-aware agents.
"""

from typing import Any, Dict, Optional

from .base import BaseRunner
from memgym.memory import BaseMemoryManager, PassThroughMemory
from memgym.agents import BaseAgent, SWEAgentTracker


class SWERunner(BaseRunner):
    """
    Runner for SWE-bench coding tasks.

    Composes:
    - SWEMemoryEnv (environment)
    - SWEAgentTracker (agent)
    - Memory manager (context filtering)

    Note: This runner currently delegates to env.run_episode() because
    SWE-bench tasks use mini-swe-agent which has its own loop.
    Future versions may extract orchestration entirely.

    Example:
        >>> from envs import SWEMemoryEnv
        >>> from agents import SWEAgentTracker
        >>> from memory import SummarizationMemory
        >>>
        >>> env = SWEMemoryEnv(
        ...     dataset="princeton-nlp/SWE-bench_Lite",
        ...     model="gpt-4"
        ... )
        >>> agent = SWEAgentTracker()
        >>> memory = SummarizationMemory(max_tokens=4000)
        >>>
        >>> runner = SWERunner(env=env, agent=agent, memory=memory)
        >>> result = runner.run_episode(task_config={"instance_id": "django__django-11099"})
    """

    def __init__(
        self,
        env: Any,  # SWEMemoryEnv
        agent: Optional[BaseAgent] = None,
        memory: Optional[BaseMemoryManager] = None,
        output_dir: str = "./trajectories",
        verbose: bool = False
    ):
        """
        Initialize SWE runner.

        Args:
            env: SWEMemoryEnv instance
            agent: Agent for tracking (defaults to SWEAgentTracker)
            memory: Memory manager (defaults to env's memory model)
            output_dir: Directory to save trajectories
            verbose: Print progress information
        """
        # Use env's memory if not provided
        if memory is None and hasattr(env, 'memory_model'):
            memory = env.memory_model

        # Create default agent if not provided
        if agent is None:
            agent = SWEAgentTracker()

        super().__init__(
            env=env,
            agent=agent,
            memory=memory or PassThroughMemory(),
            output_dir=output_dir,
            verbose=verbose
        )

    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single SWE-bench episode.

        Currently delegates to env.run_episode() because mini-swe-agent
        has its own agent loop.

        Args:
            task_config: Task configuration with optional "instance_id"
            seed: Random seed (passed to env.reset)

        Returns:
            Episode results including reward, steps, memory stats
        """
        # Reset agent (env.run_episode handles memory reset)
        self.agent.reset()

        # Reset env with task config
        config = task_config or {}
        if seed is not None:
            config["seed"] = seed

        # If instance_id provided, pass as options
        options = None
        if "instance_id" in config:
            options = {"instance_id": config.pop("instance_id")}
            self.env.reset(seed=seed, options=options)
        else:
            self.env.reset(seed=seed)

        if self.verbose:
            print(f"Running SWE episode")

        # Use env's run_episode
        result = self.env.run_episode(verbose=self.verbose)

        # Record agent actions from trajectory
        trajectory = result.get("trajectory", {})
        for step in trajectory.get("steps", []):
            action = step.get("action")
            if action:
                self.agent.set_action_from_external(
                    action=action,
                    context=step.get("context")
                )

        # Add agent stats
        result["agent_stats"] = self.agent.get_stats()

        return result


class SWEPureRunner(BaseRunner):
    """
    Pure runner for SWE-bench that handles orchestration externally.

    This is the "ideal" pattern where the runner fully controls orchestration.
    Requires a refactored SWEEnv without mini-swe-agent integration.

    Example (future pattern):
        >>> env = PureSWEEnv(repo="django/django", issue="Bug in form validation")
        >>> agent = LLMSWEAgent(llm="gpt-4")
        >>> memory = SummarizationMemory(max_tokens=8000)
        >>>
        >>> runner = SWEPureRunner(env=env, agent=agent, memory=memory)
        >>> result = runner.run_episode()
    """

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        memory: Optional[BaseMemoryManager] = None,
        output_dir: str = "./trajectories",
        max_steps: int = 50,
        verbose: bool = False
    ):
        """
        Initialize pure SWE runner.

        Args:
            env: Pure environment (reset/step only)
            agent: Agent that generates patches/commands
            memory: Memory manager
            output_dir: Directory to save trajectories
            max_steps: Maximum steps per episode
            verbose: Print progress
        """
        super().__init__(
            env=env,
            agent=agent,
            memory=memory or PassThroughMemory(),
            output_dir=output_dir,
            verbose=verbose
        )
        self.max_steps = max_steps

    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run episode with external orchestration.

        Args:
            task_config: Task configuration
            seed: Random seed

        Returns:
            Episode results
        """
        # Reset everything
        reset_kwargs = task_config or {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs, info = self.env.reset(**reset_kwargs)
        self.agent.reset()
        self.memory.reset()
        self.recorder.reset()

        # Episode loop
        history = []
        total_reward = 0
        step = 0
        done = False

        while not done and step < self.max_steps:
            step += 1

            # Memory filters context
            filtered = self.memory.manage_context(
                original_context=history,
                current_observation=obs,
                metadata={"step": step, "info": info}
            )

            # Agent decides action
            action_result = self.agent.act(filtered)
            action = action_result.action

            # Environment executes
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Record
            self.recorder.record_step(
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
                memory_metadata=filtered.metadata
            )

            history.append({
                "observation": obs,
                "action": action,
                "reward": reward
            })

            obs = next_obs
            info = next_info

            if self.verbose:
                tokens = filtered.metadata.get("tokens", "?")
                print(f"  Step {step}: tokens={tokens}, reward={reward}")

        return {
            "reward": total_reward,
            "steps": step,
            "terminated": terminated,
            "truncated": truncated,
            "memory_stats": self.memory.get_stats(),
            "agent_stats": self.agent.get_stats()
        }
