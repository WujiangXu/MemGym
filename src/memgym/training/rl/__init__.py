"""(see paper) RL training loop for the 8B memory summarizer.

`env_reward` wraps `SWEMemoryEnv.run_episode()` with a recording
summarizer backend so each rollout yields the exact
(prompt, completion, R_terminal) tuples GRPO needs.

`grpo_summarizer` (to land next) owns the policy-gradient update.
Importing this package is cheap — heavy deps (TRL, vllm) live in the
submodules, not here.
"""

from .env_reward import (
    EpisodeResult,
    RecordingSummarizerBackend,
    SummaryCall,
    run_episode_with_reward,
)

__all__ = [
    "EpisodeResult",
    "RecordingSummarizerBackend",
    "SummaryCall",
    "run_episode_with_reward",
]
