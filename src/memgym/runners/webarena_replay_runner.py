"""
ObservationReplayRunner — offline memory-strategy evaluator for WebArena.

Loads baseline trajectories (saved by WebArenaRunner), replays recorded
observations through a memory strategy, and re-queries the policy model
only at steps where memory actually compresses the context.  Steps where
memory does not trigger use the recorded baseline action for free.

This avoids browser/app-server overhead entirely — the only cost is
LLM API calls at compaction steps + summariser calls.  Typical savings
vs a live run: ≥10× (most episodes have ≤5 compaction steps out of
10-50 total steps).

Usage (CLI):
    python -m memgym.gym.webarena --replay-from results/webarena/full_baseline/haiku45/hard/gmail \\
        --memory_model webarena_structured --max_size 20 \\
        --policy_model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
        --result_dir results/webarena/replay/haiku45/struct_ms20/gmail

Usage (API):
    POST /run-webarena-replay  (see launch.py)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from memgym.adapter.trajectory_recorder import TrajectoryRecorder
from memgym.agents.base import BaseAgent
from memgym.memory import BaseMemoryManager

logger = logging.getLogger("memgym.webarena.replay_runner")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Per-episode result from observation replay."""

    task_id: str
    total_steps: int
    compaction_steps: int = 0        # Steps where memory triggered condensation
    policy_calls: int = 0            # = compaction_steps
    divergence_steps: int = 0        # Compaction steps where predicted != recorded
    baseline_success: bool = False   # Did the original baseline trajectory succeed?
    baseline_tokens: int = 0         # Cumulative tokens in uncompressed context
    compressed_tokens: int = 0       # Cumulative tokens in memory-managed context
    summarizer_tokens: int = 0       # Tokens consumed by summariser model
    wall_time_seconds: float = 0.0

    # Per-compaction-step details
    compaction_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return self.divergence_steps / self.compaction_steps if self.compaction_steps else 0.0

    @property
    def action_match_rate(self) -> float:
        return 1.0 - (self.divergence_steps / self.total_steps) if self.total_steps else 1.0

    @property
    def compression_ratio(self) -> float:
        return self.baseline_tokens / self.compressed_tokens if self.compressed_tokens else 1.0


# ---------------------------------------------------------------------------
# Trajectory loading helpers
# ---------------------------------------------------------------------------

def _load_trajectory(trajectory_dir: Path, task_id: str) -> Optional[Dict]:
    """Load trajectory.json for a given task.  Returns None if missing."""
    traj_path = trajectory_dir / task_id / "trajectory.json"
    if not traj_path.exists():
        logger.warning("no trajectory at %s — skipping", traj_path)
        return None
    with traj_path.open() as f:
        return json.load(f)


def _extract_steps(traj: Dict) -> List[Dict]:
    """Pull the flat step list from a trajectory dict."""
    episodes = traj.get("episodes", [])
    if not episodes:
        return []
    return episodes[0].get("steps", [])


def _obs_to_history_content(obs: Any) -> str:
    """Reconstruct the history content string from a recorded observation.

    Mirrors ``_history_content()`` in webarena_runner.py so the memory
    manager sees the same input format as during a live run.
    """
    if not isinstance(obs, dict):
        return str(obs)
    url = obs.get("url", "")
    title = obs.get("title", "")
    tree = obs.get("accessibility_tree", "")
    if tree:
        return f"URL: {url}\nTitle: {title}\n\n{tree}"
    return f"URL: {url}\nTitle: {title}"


# ---------------------------------------------------------------------------
# ObservationReplayRunner
# ---------------------------------------------------------------------------

class ObservationReplayRunner:
    """Replay baseline trajectories through a memory strategy offline.

    Parameters
    ----------
    agent : BaseAgent
        Policy model agent (only called at compaction steps).
    memory : BaseMemoryManager
        The memory strategy under evaluation.
    trajectory_dir : Path | str
        Directory containing ``<task_id>/trajectory.json`` files from a
        baseline run.
    output_dir : Path | str
        Where to write per-task replay results.
    verbose : bool
        Log per-step progress.
    """

    def __init__(
        self,
        agent: BaseAgent,
        memory: BaseMemoryManager,
        trajectory_dir: str | Path,
        output_dir: str | Path = "./results/webarena/replay",
        verbose: bool = False,
    ):
        self.agent = agent
        self.memory = memory
        self.trajectory_dir = Path(trajectory_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core: replay a single episode
    # ------------------------------------------------------------------

    def replay_episode(self, task_id: str) -> Optional[ReplayResult]:
        """Replay one baseline trajectory through the memory strategy.

        Returns None if the trajectory file is missing.
        """
        traj = _load_trajectory(self.trajectory_dir, task_id)
        if traj is None:
            return None

        steps = _extract_steps(traj)
        if not steps:
            logger.warning("trajectory for %s has no steps — skipping", task_id)
            return None

        meta = traj.get("metadata", {})
        baseline_success = meta.get("success", False)

        start_time = time.time()

        # Reset agent + memory for this episode.
        self.agent.reset()
        self.memory.reset()

        # Propagate episode state so memory adapters that need the task
        # instruction / app name get it (same as WebArenaRunner).
        episode_state = {
            k: v for k, v in meta.items()
            if k in ("instruction", "app_name", "observation_mode") and v
        }
        self.agent.set_episode_state(**episode_state)
        self.memory.set_episode_state(**episode_state)

        # Recorder for saving the replay trajectory
        recorder = TrajectoryRecorder()
        recorder.start_episode(metadata={
            "task_id": task_id,
            "replay_from": str(self.trajectory_dir),
            "memory_strategy": self.memory.__class__.__name__,
            "baseline_success": baseline_success,
        })

        history: List[Dict[str, Any]] = []
        result = ReplayResult(
            task_id=task_id,
            total_steps=len(steps),
            baseline_success=baseline_success,
        )

        for step_idx, step in enumerate(steps, 1):
            obs = step.get("observation", {})
            recorded_action = step.get("action", "")
            reward = step.get("reward", 0.0)
            terminated = step.get("terminated", False)
            truncated = step.get("truncated", False)

            obs_text = _obs_to_history_content(obs)

            # --- 1. Run memory manager on the current history ---
            filtered = self.memory.manage_context(
                original_context=history,
                current_observation=obs_text,
                metadata={"step": step_idx, "info": step.get("info", {})},
            )

            was_compacted = filtered.metadata.get("was_compacted", False)
            tokens_after = filtered.metadata.get("tokens", 0)
            original_tokens = filtered.metadata.get("original_tokens", tokens_after)
            result.baseline_tokens += original_tokens
            result.compressed_tokens += tokens_after
            result.summarizer_tokens += (
                filtered.metadata.get("summarizer_prompt_tokens", 0)
                + filtered.metadata.get("summarizer_completion_tokens", 0)
            )

            predicted_action = recorded_action  # default: use baseline
            policy_called = False

            # --- 2. Only re-query policy at compaction steps ---
            if was_compacted:
                result.compaction_steps += 1
                result.policy_calls += 1
                policy_called = True

                action_result = self.agent.act(filtered)
                predicted_action = str(action_result.action)

                diverged = predicted_action.strip() != recorded_action.strip()
                if diverged:
                    result.divergence_steps += 1

                result.compaction_details.append({
                    "step": step_idx,
                    "recorded_action": recorded_action,
                    "predicted_action": predicted_action,
                    "diverged": diverged,
                    "tokens_before": original_tokens,
                    "tokens_after": tokens_after,
                })

                if self.verbose:
                    logger.info(
                        "step %d: COMPACTED tokens=%d→%d predicted=%r recorded=%r %s",
                        step_idx, original_tokens, tokens_after,
                        predicted_action, recorded_action,
                        "DIVERGED" if diverged else "match",
                    )

            # --- 3. Record step ---
            recorder.record_step(
                observation=obs,
                action=predicted_action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=step.get("info"),
                memory_action=filtered.metadata,
                reasoning_action={
                    "mode": "replay",
                    "policy_called": policy_called,
                    "was_compacted": was_compacted,
                    "recorded_action": recorded_action,
                },
            )

            # --- 4. Always append RECORDED action to history ---
            # (stay on the baseline trajectory regardless of divergence)
            history.append({"role": "user", "content": obs_text})
            history.append({"role": "assistant", "content": str(recorded_action)})

        recorder.end_episode()
        result.wall_time_seconds = round(time.time() - start_time, 2)

        # --- Save results ---
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        with (task_dir / "replay_result.json").open("w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        traj_out = recorder.to_dict()
        traj_out["metadata"] = {
            "task_id": task_id,
            "replay_from": str(self.trajectory_dir),
            "memory_strategy": self.memory.__class__.__name__,
            **asdict(result),
        }
        with (task_dir / "trajectory.json").open("w") as f:
            json.dump(traj_out, f, indent=2, default=str)

        if self.verbose:
            logger.info(
                "replay %s: %d steps, %d compactions, %d divergences, "
                "compress=%.2fx, %.1fs",
                task_id, result.total_steps, result.compaction_steps,
                result.divergence_steps, result.compression_ratio,
                result.wall_time_seconds,
            )

        return result

    # ------------------------------------------------------------------
    # Batch run across multiple tasks
    # ------------------------------------------------------------------

    def run(
        self,
        task_ids: List[str],
        workers: int = 1,
    ) -> Dict[str, ReplayResult]:
        """Replay multiple tasks.  Returns {task_id: ReplayResult}.

        When workers > 1, tasks are distributed across threads.
        (Memory strategies are stateful per-episode but independent across
        tasks, so thread-level parallelism is safe as long as each thread
        gets its own memory + agent instance — handled by the CLI harness.)
        """
        results: Dict[str, ReplayResult] = {}

        if workers <= 1:
            for tid in task_ids:
                r = self.replay_episode(tid)
                if r is not None:
                    results[tid] = r
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # For thread safety, each worker needs its own
            # memory/agent clone.  For now, run single-threaded
            # and let the CLI harness handle multi-process parallelism
            # (same pattern as WebArenaRunner).
            logger.warning(
                "workers=%d requested but replay runner currently uses "
                "single-thread mode.  Use the CLI --workers flag for "
                "process-level parallelism.",
                workers,
            )
            for tid in task_ids:
                r = self.replay_episode(tid)
                if r is not None:
                    results[tid] = r

        # --- Write aggregate summary ---
        if results:
            self._write_summary(results)

        return results

    def _write_summary(self, results: Dict[str, ReplayResult]) -> None:
        total_steps = sum(r.total_steps for r in results.values())
        total_compactions = sum(r.compaction_steps for r in results.values())
        total_divergences = sum(r.divergence_steps for r in results.values())
        total_policy_calls = sum(r.policy_calls for r in results.values())
        total_baseline_tokens = sum(r.baseline_tokens for r in results.values())
        total_compressed_tokens = sum(r.compressed_tokens for r in results.values())
        total_summarizer_tokens = sum(r.summarizer_tokens for r in results.values())
        baseline_successes = sum(1 for r in results.values() if r.baseline_success)

        summary = {
            "tasks": len(results),
            "total_steps": total_steps,
            "total_compaction_steps": total_compactions,
            "total_policy_calls": total_policy_calls,
            "total_divergence_steps": total_divergences,
            "divergence_rate": total_divergences / total_compactions if total_compactions else 0.0,
            "policy_call_savings": 1.0 - (total_policy_calls / total_steps) if total_steps else 0.0,
            "baseline_success_rate": baseline_successes / len(results),
            "compression_ratio": total_baseline_tokens / total_compressed_tokens if total_compressed_tokens else 1.0,
            "total_baseline_tokens": total_baseline_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "total_summarizer_tokens": total_summarizer_tokens,
            "memory_strategy": next(iter(results.values())).__class__.__name__
                if results else "unknown",
            "wall_time_seconds": sum(r.wall_time_seconds for r in results.values()),
        }

        # Fix: use the actual strategy name from the result metadata
        summary["memory_strategy"] = self.memory.__class__.__name__

        with (self.output_dir / "replay_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "replay summary: %d tasks, %d total steps, %d compactions (%d diverged), "
            "%.0f%% policy calls saved, compress=%.2fx",
            summary["tasks"], total_steps, total_compactions, total_divergences,
            summary["policy_call_savings"] * 100, summary["compression_ratio"],
        )
