"""
MemGym Training Module.

Pipeline: trajectory data → compaction events → augmentation → SFT dataset → world model.

Subpackages:
  - training.data: Trajectory loading, labeling, compaction extraction, dataset construction
  - training.augmentation: Counterfactual replay, perturbations, LLM-as-a-judge
  - training.models: World model (Qwen2.5 + LoRA), SFT trainer, evaluator
  - training.scripts: CLI entry points (run_pipeline, train, augment)
  - training.base: WorldModelEnv (wraps trained model for inference)

Usage:
    # End-to-end
    python -m memgym.training.scripts.run_pipeline \\
        --strategy-dir ec2_trajectories/diverse500_sonnet45_llmsumm_ms100 \\
        --augment --train --output-dir training_output/

    # Programmatic
    from memgym.training.base import WorldModelEnv, CompressionQuery
    env = WorldModelEnv.from_checkpoint("checkpoints/world_model/final")
    pred, conf = env.predict_compression_outcome(CompressionQuery(...))
"""

from memgym.training.base import CompressionQuery, TrainingDataset, WorldModelEnv

__all__ = ["CompressionQuery", "TrainingDataset", "WorldModelEnv"]
