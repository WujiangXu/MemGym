"""
Base runner for MemGym experiments.

Runners compose environment + agent + memory manager and handle episode orchestration.
This keeps environments pure (just reset/step/close) while runners handle the full loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time

from memgym.memory import BaseMemoryManager, PassThroughMemory, FilteredContext
from memgym.agents import BaseAgent
from memgym.utils import TrajectoryRecorder


class BaseRunner(ABC):
    """
    Abstract base class for experiment runners.

    Runners compose:
    - Environment (pure - just reset/step/close)
    - Agent (makes decisions based on filtered context)
    - Memory manager (filters context for agent)

    The runner handles the episode loop, calling memory.manage_context()
    before each agent action.

    Example:
        >>> runner = SWERunner(
        ...     env=env,
        ...     agent=agent,
        ...     memory=memory,
        ...     output_dir="./trajectories"
        ... )
        >>> results = runner.run_episode(task_config)
    """

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        memory: Optional[BaseMemoryManager] = None,
        output_dir: str = "./trajectories",
        verbose: bool = False
    ):
        """
        Initialize runner.

        Args:
            env: Environment instance (pure - reset/step/close only)
            agent: Agent instance
            memory: Memory manager (defaults to PassThroughMemory)
            output_dir: Directory to save trajectories
            verbose: Print progress information
        """
        self.env = env
        self.agent = agent
        self.memory = memory or PassThroughMemory()
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory recorder
        self.recorder = TrajectoryRecorder()

        # Results storage
        self.results: List[Dict[str, Any]] = []

    @abstractmethod
    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode.

        The episode loop should:
        1. Reset env, agent, memory
        2. Loop until done:
           a. memory.manage_context(history, current_obs)
           b. agent.act(filtered_context)
           c. env.step(action)
           d. Record trajectory
        3. Return results

        Args:
            task_config: Task-specific configuration
            seed: Random seed

        Returns:
            Episode results dict
        """
        pass

    def run(
        self,
        num_episodes: int = 1,
        task_configs: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple episodes.

        Args:
            num_episodes: Number of episodes to run
            task_configs: List of task configs (if None, runs same task repeatedly)
            seed: Base random seed

        Returns:
            List of episode results
        """
        configs = task_configs or [{}] * num_episodes

        for i, config in enumerate(configs[:num_episodes]):
            episode_seed = seed + i if seed is not None else None

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Episode {i + 1}/{num_episodes}")
                print(f"{'='*50}")

            result = self.run_episode(task_config=config, seed=episode_seed)
            self.results.append(result)

            if self.verbose:
                print(f"Reward: {result.get('reward', 0)}")
                print(f"Steps: {result.get('steps', 0)}")

        return self.results

    def save_trajectory(self, episode: int, extra_metadata: Optional[Dict] = None) -> str:
        """
        Save trajectory from recorder.

        Args:
            episode: Episode number
            extra_metadata: Additional metadata to save

        Returns:
            Path to saved trajectory file
        """
        trajectory = self.recorder.get_trajectory()

        if extra_metadata:
            trajectory["metadata"] = {**trajectory.get("metadata", {}), **extra_metadata}

        filepath = self.output_dir / f"trajectory_{episode:04d}.json"
        with open(filepath, 'w') as f:
            json.dump(trajectory, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved trajectory to {filepath}")

        return str(filepath)

    def save_results(self, filename: str = "results.json") -> str:
        """
        Save all results to a JSON file.

        Args:
            filename: Name of results file

        Returns:
            Path to saved results file
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump({
                "env_type": self.env.__class__.__name__,
                "agent_type": self.agent.__class__.__name__,
                "memory_type": self.memory.__class__.__name__,
                "results": self.results
            }, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved results to {filepath}")

        return str(filepath)

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.env, 'close'):
            self.env.close()


class SimpleRunner(BaseRunner):
    """
    Simple runner with standard episode loop.

    Works with any environment that has standard reset/step interface.
    """

    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode with standard loop.

        Args:
            task_config: Task-specific configuration (passed to env.reset)
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

        while not done:
            step += 1

            # Memory filters context
            filtered = self.memory.manage_context(
                original_context=history,
                current_observation=obs,
                metadata={"step": step, "info": info}
            )

            # Agent decides action based on filtered context
            action_result = self.agent.act(filtered)
            action = action_result.action

            # Environment executes action
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Record trajectory step
            self.recorder.record_step(
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
                memory_metadata=filtered.metadata
            )

            # Update history
            history.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "info": info
            })

            obs = next_obs
            info = next_info

            if self.verbose:
                print(f"  Step {step}: reward={reward}, tokens={filtered.metadata.get('tokens', '?')}")

        return {
            "reward": total_reward,
            "steps": step,
            "terminated": terminated,
            "truncated": truncated,
            "memory_stats": self.memory.get_stats(),
            "agent_stats": self.agent.get_stats()
        }
