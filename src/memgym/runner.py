"""
Runner abstractions for MemGym experiments.

DEPRECATED: This module is deprecated. Please use `runners` module instead.

Provides abstract classes and utilities for running experiments,
collecting trajectories, and evaluating agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

# Re-export from new runners module for backward compatibility
try:
    from memgym.runners import (
        BaseRunner as NewBaseRunner,
        SimpleRunner,
        SWERunner,
    )
except ImportError:
    pass


class BaseRunner(ABC):
    """
    Abstract base class for experiment runners.

    Subclass this to create custom experiment workflows.

    Example:
        >>> class MyRunner(BaseRunner):
        ...     def run(self):
        ...         env = self.get_env()
        ...         for episode in range(self.num_episodes):
        ...             obs, info = env.reset()
        ...             # Run episode...
        ...             self.save_trajectory(env, episode)
    """

    def __init__(
        self,
        env_name: str,
        env_kwargs: Optional[Dict[str, Any]] = None,
        output_dir: str = "./trajectories",
        num_episodes: int = 1,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize runner.

        Args:
            env_name: Name of environment ("tau2", "swe", etc.)
            env_kwargs: Arguments passed to environment constructor
            output_dir: Directory to save trajectories
            num_episodes: Number of episodes to run
            seed: Random seed for reproducibility
            verbose: Print progress information
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}
        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.seed = seed
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[Dict[str, Any]] = []

    def get_env(self):
        """Create and return an environment instance."""
        from memgym.envs import get_env
        return get_env(self.env_name, **self.env_kwargs)

    def get_reasoning_model(self, model_name: Optional[str] = None):
        """Create and return a reasoning model instance."""
        from memgym.agents import get_reasoning_model
        name = model_name or self.env_name
        return get_reasoning_model(name)

    def get_memory_model(self, model_name: str = "passthrough", **kwargs):
        """Create and return a memory model instance."""
        from memgym.memory import get_memory_model
        return get_memory_model(model_name, **kwargs)

    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the experiment.

        Returns:
            List of episode results
        """
        pass

    def save_trajectory(self, env, episode: int, extra_metadata: Optional[Dict] = None) -> str:
        """
        Save trajectory from environment.

        Args:
            env: Environment with trajectory recorder
            episode: Episode number
            extra_metadata: Additional metadata to save

        Returns:
            Path to saved trajectory file
        """
        trajectory = env.get_trajectory()

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
                "env_name": self.env_name,
                "env_kwargs": self.env_kwargs,
                "num_episodes": self.num_episodes,
                "seed": self.seed,
                "results": self.results
            }, f, indent=2, default=str)

        if self.verbose:
            print(f"Saved results to {filepath}")

        return str(filepath)


class EpisodeRunner(BaseRunner):
    """
    Simple runner that executes episodes and collects trajectories.

    Example:
        >>> runner = EpisodeRunner(
        ...     env_name="tau2",
        ...     env_kwargs={"domain": "airline", "task_id": "book_flight_1"},
        ...     num_episodes=10
        ... )
        >>> results = runner.run()
    """

    def __init__(
        self,
        env_name: str,
        agent_action_fn: Optional[callable] = None,
        **kwargs
    ):
        """
        Initialize episode runner.

        Args:
            env_name: Name of environment
            agent_action_fn: Function that takes (observation, info) and returns action.
                            If None, uses env.run_episode() for LLM agents.
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(env_name, **kwargs)
        self.agent_action_fn = agent_action_fn

    def run(self) -> List[Dict[str, Any]]:
        """Run episodes and collect trajectories."""
        env = self.get_env()

        for episode in range(self.num_episodes):
            seed = self.seed + episode if self.seed is not None else None

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Episode {episode + 1}/{self.num_episodes}")
                print(f"{'='*50}")

            if self.agent_action_fn is None:
                # Use LLM agent mode
                result = env.run_episode(seed=seed, verbose=self.verbose)
            else:
                # Use custom action function
                result = self._run_manual_episode(env, seed)

            self.results.append(result)
            self.save_trajectory(env, episode)

            if self.verbose:
                print(f"Reward: {result.get('reward', 0)}")

        self.save_results()
        env.close()

        return self.results

    def _run_manual_episode(self, env, seed: Optional[int]) -> Dict[str, Any]:
        """Run episode with custom action function."""
        obs, info = env.reset(seed=seed)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = self.agent_action_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        return {
            "reward": total_reward,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "memory_metrics": env.get_memory_metrics(),
            "trajectory": env.get_trajectory()
        }


class TrajectoryCollector(BaseRunner):
    """
    Collects trajectories across multiple tasks for training.

    Example:
        >>> collector = TrajectoryCollector(
        ...     env_name="swe",
        ...     env_kwargs={"dataset": "princeton-nlp/SWE-bench_Lite"},
        ...     num_episodes=100
        ... )
        >>> trajectories = collector.run()
    """

    def __init__(
        self,
        env_name: str,
        task_ids: Optional[List[str]] = None,
        sample_random: bool = True,
        **kwargs
    ):
        """
        Initialize trajectory collector.

        Args:
            env_name: Name of environment
            task_ids: Specific task IDs to collect from (if None, samples randomly)
            sample_random: If True and no task_ids, sample random tasks
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(env_name, **kwargs)
        self.task_ids = task_ids
        self.sample_random = sample_random

    def run(self) -> List[Dict[str, Any]]:
        """Collect trajectories from multiple tasks."""
        env = self.get_env()

        # Get task list if needed
        if self.task_ids is None and hasattr(env, 'get_all_task_ids'):
            available_tasks = env.get_all_task_ids()
            if self.sample_random:
                import random
                if self.seed is not None:
                    random.seed(self.seed)
                self.task_ids = random.sample(
                    available_tasks,
                    min(self.num_episodes, len(available_tasks))
                )
            else:
                self.task_ids = available_tasks[:self.num_episodes]

        for i, task_id in enumerate(self.task_ids or range(self.num_episodes)):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Task {i + 1}/{len(self.task_ids or [])}: {task_id}")
                print(f"{'='*50}")

            try:
                if isinstance(task_id, str):
                    obs, info = env.reset(options={"instance_id": task_id})
                else:
                    seed = self.seed + i if self.seed is not None else None
                    obs, info = env.reset(seed=seed)

                result = env.run_episode(verbose=self.verbose) if hasattr(env, 'run_episode') else {
                    "reward": 0, "steps": 0
                }

                result["task_id"] = task_id
                self.results.append(result)
                self.save_trajectory(env, i, {"task_id": task_id})

            except Exception as e:
                if self.verbose:
                    print(f"Error on task {task_id}: {e}")
                self.results.append({
                    "task_id": task_id,
                    "error": str(e),
                    "reward": 0
                })

        self.save_results()
        env.close()

        return self.results
