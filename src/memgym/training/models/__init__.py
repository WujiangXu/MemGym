"""World model architecture, training, and evaluation.

Training lives in `memgym.training.models.sft` (TRL SFTTrainer +
`completion_only_loss=True`). The older `WorldModelTrainer` was removed;
use `memgym.training.scripts.train_sft` as the CLI entrypoint.
"""

from memgym.training.models.evaluator import (
    EvalMetrics,
    PredictionResult,
    WorldModelEvaluator,
    compute_metrics,
)
from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

__all__ = [
    "EvalMetrics",
    "MemoryWorldModel",
    "ModelConfig",
    "PredictionResult",
    "WorldModelEvaluator",
    "compute_metrics",
]
