"""Data augmentation via counterfactual replay and LLM-as-a-judge.

Two replay modes:
  - LLM-only (replay.py): Compare action text, no Docker needed, cheap
  - Env-feedback (env_replay.py): Execute in Docker, compare observations, ground-truth
"""
