"""
MemGym training module — the Memory Reward Model (MemRM) stack.

Pipeline: trajectory data → compaction events → counterfactual-replay
augmentation → paired SFT dataset → MemRM (Qwen3-1.7B QLoRA reward model).

Subpackages:
  - training.data: trajectory loading, labeling, compaction extraction, dataset construction
  - training.augmentation: counterfactual replay, perturbations, LLM-as-a-judge
  - training.models: MemRM architecture (Qwen3 + LoRA), SFT trainer, evaluator
  - training.eval: the ``memgym-eval-rm`` / ``memgym-eval-memory`` console scripts
  - training.scripts: dataset builders + SFT/eval launchers (build_reward_model_dataset, train_sft, augment)
  - training.base: inference wrapper around a trained checkpoint

Usage:
    # Build the paired SFT dataset, then train the reward model
    python -m memgym.training.scripts.augment   --strategy-dir trajectories/<run> ...
    python -m memgym.training.scripts.train_sft --out-dir training_output/

    # Programmatic inference
    from memgym.training.base import WorldModelEnv, CompressionQuery
    env = WorldModelEnv.from_checkpoint("checkpoints/memrm/final")
    pred, conf = env.predict_compression_outcome(CompressionQuery(...))
"""

from memgym.training.base import CompressionQuery, TrainingDataset, WorldModelEnv

__all__ = ["CompressionQuery", "TrainingDataset", "WorldModelEnv"]
