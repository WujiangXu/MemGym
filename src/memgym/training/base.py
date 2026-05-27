"""Base classes for the MemGym training module.

WorldModelEnv wraps a trained world model checkpoint to predict
compression outcomes without running Docker containers.

TrainingDataset provides a convenience wrapper around the data
pipeline (loader → compaction → dataset).

Usage:
    from memgym.training.base import WorldModelEnv

    env = WorldModelEnv.from_checkpoint("checkpoints/world_model/final")
    prediction, confidence = env.predict_compression_outcome(
        problem_statement="Fix bug in ...",
        summary="The agent explored ...",
        compression_ratio=2.5,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompressionQuery:
    """Input for world model prediction."""

    problem_statement: str
    summary: str
    agent_action: str = ""
    compression_ratio: float = 1.0
    context_before_tokens: int = 0
    context_after_tokens: int = 0
    forgotten_count: int = 0
    step: int = 0
    total_steps: int = 0
    memory_strategy: str = ""
    model: str = ""


class WorldModelEnv:
    """Environment backed by a trained world model.

    Predicts whether a compression event will be SAFE or HARMFUL,
    enabling rapid memory strategy evaluation without Docker rollouts.
    """

    def __init__(self, model: Any = None):
        self._model = model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **model_kwargs,
    ) -> "WorldModelEnv":
        """Load from a trained checkpoint.

        Args:
            checkpoint_path: Path to saved LoRA adapter weights.
            **model_kwargs: Passed to ModelConfig.
        """
        from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

        config = ModelConfig(**model_kwargs) if model_kwargs else None
        model = MemoryWorldModel.from_checkpoint(checkpoint_path, config)
        return cls(model=model)

    def predict_compression_outcome(
        self,
        query: CompressionQuery,
    ) -> Tuple[str, float]:
        """Predict whether a compression event is SAFE or HARMFUL.

        Args:
            query: CompressionQuery with task context and compression details.

        Returns:
            (prediction: "SAFE"|"HARMFUL", confidence: float)
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Use WorldModelEnv.from_checkpoint()."
            )

        from memgym.training.data.dataset import SYSTEM_PROMPT, USER_TEMPLATE

        user_content = USER_TEMPLATE.format(
            problem_statement=query.problem_statement[:2000],
            step=query.step,
            total_steps=query.total_steps,
            model=query.model,
            memory_strategy=query.memory_strategy,
            before_tokens=query.context_before_tokens,
            before_msgs=0,
            after_tokens=query.context_after_tokens,
            after_msgs=0,
            ratio=query.compression_ratio,
            forgotten=query.forgotten_count,
            summary=query.summary[:3000],
            agent_action=query.agent_action[:1000],
            pre_steps_text="(not available)",
            resolved="unknown",
            reward=0.0,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return self._model.predict(messages)

    def batch_predict(
        self,
        queries: List[CompressionQuery],
    ) -> List[Tuple[str, float]]:
        """Predict outcomes for multiple compression events."""
        return [self.predict_compression_outcome(q) for q in queries]


class TrainingDataset:
    """Convenience wrapper for building training datasets from trajectories.

    Chains: loader → compaction extraction → dataset construction.

    Usage:
        ds = TrainingDataset.from_trajectory_dir(
            "trajectories/train50_sonnet45_llmsummarizing"
        )
        ds.save("training_output/dataset.jsonl")
    """

    def __init__(self):
        self._examples = []

    @classmethod
    def from_trajectory_dir(
        cls,
        strategy_dir: str | Path,
        baseline_dir: Optional[str | Path] = None,
    ) -> "TrainingDataset":
        """Build dataset from a trajectory directory.

        Args:
            strategy_dir: Directory with memory-augmented trajectories.
            baseline_dir: Optional baseline directory for paired labeling.
        """
        from memgym.training.data.loader import TrajectoryLoader
        from memgym.training.data.compaction import extract_all_compaction_events
        from memgym.training.data.dataset import WorldModelDataset

        loader = TrajectoryLoader(Path(strategy_dir).parent)
        experiment_name = Path(strategy_dir).name
        trajectories = loader.load_experiment(experiment_name)

        all_events = extract_all_compaction_events(trajectories)

        # Flatten events
        flat_events = []
        for events in all_events.values():
            flat_events.extend(events)

        dataset = WorldModelDataset.from_events_and_pairs(flat_events)

        instance = cls()
        instance._examples = dataset.examples
        instance._dataset = dataset
        return instance

    def save(self, output_path: str | Path) -> None:
        """Save to JSONL."""
        from memgym.training.data.dataset import save_dataset_jsonl

        save_dataset_jsonl(self._examples, output_path)

    def __len__(self) -> int:
        return len(self._examples)

    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        if hasattr(self, "_dataset"):
            return self._dataset.stats()
        return {"total": len(self._examples)}
