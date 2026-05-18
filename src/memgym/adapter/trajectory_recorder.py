"""
Trajectory recording for MemGym.

This module provides functionality to record complete interaction trajectories
between agents and environments, capturing all observations, actions, rewards,
and metadata for later analysis.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json
import time


@dataclass
class Step:
    """A single step in the trajectory."""

    step_id: int
    timestamp: float
    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Memory-Reasoning separation
    memory_action: Optional[Any] = None  # What memory model did
    context: Optional[Any] = None  # Context given to reasoning
    reasoning_action: Optional[Any] = None  # What reasoning model decided

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary format."""
        return asdict(self)


@dataclass
class Episode:
    """A complete episode trajectory."""

    episode_id: int
    steps: List[Step] = field(default_factory=list)
    total_reward: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Step) -> None:
        """Add a step to the episode."""
        self.steps.append(step)
        self.total_reward += step.reward

    def finalize(self) -> None:
        """Mark episode as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get episode duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def num_steps(self) -> int:
        """Get number of steps in episode."""
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary format."""
        return {
            "episode_id": self.episode_id,
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps]
        }


class TrajectoryRecorder:
    """
    Records complete interaction trajectories.

    This class captures all interactions between agents and environments,
    storing them in a structured format for later analysis, visualization,
    or training.

    Features:
    - Episode-based organization
    - Step-level granularity
    - Metadata attachment
    - Export to JSON/dict format
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trajectory recorder.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._episodes: List[Episode] = []
        self._current_episode: Optional[Episode] = None
        self._current_step_id = 0
        self._episode_count = 0

    def start_episode(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start recording a new episode.

        Args:
            metadata: Optional metadata for the episode
        """
        if self._current_episode is not None:
            # Finalize previous episode if not already done
            self.end_episode()

        self._current_episode = Episode(
            episode_id=self._episode_count,
            metadata=metadata or {}
        )
        self._episode_count += 1
        self._current_step_id = 0

    def record_step(
        self,
        observation: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_action: Optional[Any] = None,
        context: Optional[Any] = None,
        reasoning_action: Optional[Any] = None
    ) -> None:
        """
        Record a single step in the current episode.

        Args:
            observation: The observation received
            action: The action taken
            reward: The reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Optional info dict
            metadata: Optional step metadata
            memory_action: Optional memory model action
            context: Optional context provided to reasoning model
            reasoning_action: Optional reasoning model action
        """
        if self._current_episode is None:
            self.start_episode()

        step = Step(
            step_id=self._current_step_id,
            timestamp=time.time(),
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
            metadata=metadata or {},
            memory_action=memory_action,
            context=context,
            reasoning_action=reasoning_action
        )

        self._current_episode.add_step(step)
        self._current_step_id += 1

    def end_episode(self) -> Episode:
        """
        End the current episode and return it.

        Returns:
            The completed episode
        """
        if self._current_episode is None:
            raise ValueError("No episode in progress")

        self._current_episode.finalize()
        self._episodes.append(self._current_episode)
        episode = self._current_episode
        self._current_episode = None
        return episode

    def get_episodes(self) -> List[Episode]:
        """Get all recorded episodes."""
        return self._episodes.copy()

    def get_current_episode(self) -> Optional[Episode]:
        """Get the current episode in progress."""
        return self._current_episode

    @property
    def num_episodes(self) -> int:
        """Get total number of completed episodes."""
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        """Get total number of steps across all episodes."""
        return sum(ep.num_steps for ep in self._episodes)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export all trajectories to dictionary format.

        Returns:
            Dictionary containing all episodes
        """
        return {
            "num_episodes": self.num_episodes,
            "total_steps": self.total_steps,
            "config": self.config,
            "episodes": [ep.to_dict() for ep in self._episodes]
        }

    def to_json(self, filepath: str, **kwargs) -> None:
        """
        Export trajectories to JSON file.

        Args:
            filepath: Path to save JSON file
            **kwargs: Additional arguments for json.dump
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, default=str, **kwargs)

    def to_concise_dict(self) -> Dict[str, Any]:
        """
        Export concise trajectories (messages only) for training.

        Returns:
            Dictionary with messages only (role, content, timestamp)
        """
        concise_episodes = []

        for episode in self._episodes:
            messages = []
            for step in episode.steps:
                # Extract message from observation/action
                msg = self._extract_message(step)
                if msg:
                    messages.append(msg)

            concise_episodes.append({
                "episode_id": episode.episode_id,
                "num_messages": len(messages),
                "metadata": episode.metadata,
                "messages": messages
            })

        return {
            "num_episodes": self.num_episodes,
            "total_messages": sum(len(ep["messages"]) for ep in concise_episodes),
            "episodes": concise_episodes
        }

    def _extract_message(self, step: Step) -> Optional[Dict[str, Any]]:
        """Extract concise message from step."""
        # Parse observation/action strings (format: "role: content")
        obs_str = str(step.observation)

        # Skip if not a message format
        if ": " not in obs_str:
            return None

        parts = obs_str.split(": ", 1)
        if len(parts) != 2:
            return None

        role, content = parts

        return {
            "step": step.step_id,
            "role": role.strip(),
            "content": content.strip(),
            "timestamp": step.timestamp
        }

    def to_json_concise(self, filepath: str, **kwargs) -> None:
        """
        Export concise trajectories (messages only) to JSON file.

        Args:
            filepath: Path to save JSON file
            **kwargs: Additional arguments for json.dump
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_concise_dict(), f, default=str, **kwargs)

    def reset(self) -> None:
        """Clear all recorded trajectories."""
        self._episodes.clear()
        self._current_episode = None
        self._current_step_id = 0
        self._episode_count = 0
