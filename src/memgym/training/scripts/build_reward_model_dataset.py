"""CLI entry point for the reward-model dataset rebuild.

Thin shim over ``memgym.training.data.reward_model_pairs.main`` so the
build is callable with the standard scripts/ layout used by the other
training launchers (e.g. `train_sft.py`).

Usage:
    python -m memgym.training.scripts.build_reward_model_dataset \
        --pairs-in data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl \
        --trajectories-root data/world_model/trajectories \
        --out-jsonl data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl \
        --limit 20    # smoke gate
"""
from __future__ import annotations

import sys

from memgym.training.data.reward_model_pairs import main


if __name__ == "__main__":
    sys.exit(main())
