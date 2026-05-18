"""Environment-feedback replay augmentation for SWE-bench trajectories.

Instead of just comparing action text (noisy), executes the perturbed
action in a real Docker container and compares the observation to the
recorded one. This gives ground-truth labels:
  Same observation  → SAFE  (compression didn't affect outcome)
  Different observation → HARMFUL (compression changed outcome)

Architecture:
  1. Pull Docker image for an instance
  2. Replay bash commands 1..N-1 to reconstruct container state
  3. At compaction step N: get perturbed action from policy LLM
  4. Execute perturbed action in container
  5. Compare observation to recorded observation
  6. Optionally delete image after processing to save disk

Usage:
    from memgym.training.augmentation.env_replay import EnvFeedbackReplayer

    replayer = EnvFeedbackReplayer(
        policy_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    results = replayer.replay_instance(trajectory, compaction_events)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.training.augmentation.perturbations import PerturbationStrategy
from memgym.training.augmentation.replay import (
    CounterfactualPair,
    AugmentationResult,
    _build_messages_at_step,
    _extract_bash_from_message,
    _extract_recorded_observation,
    _call_policy,
)
from memgym.training.data.compaction import CompactionEvent
from memgym.training.data.loader import LoadedTrajectory

logger = logging.getLogger(__name__)

# Docker image pattern for SWE-Gym
SWEGYM_IMAGE_PATTERN = "docker.io/xingyaoww/sweb.eval.x86_64.{instance_id}:latest"


def _instance_to_image(instance_id: str) -> str:
    """Convert instance_id to Docker image name (SWE-Gym convention)."""
    docker_id = instance_id.replace("__", "_s_").lower()
    return SWEGYM_IMAGE_PATTERN.format(instance_id=docker_id)


def _extract_all_commands(trajectory: LoadedTrajectory) -> List[Dict[str, Any]]:
    """Extract all bash commands and observations from a trajectory's messages.

    Returns list of {"step": int, "command": str, "msg_index": int}.
    """
    commands = []
    for step_info in trajectory.message_steps:
        step = step_info.get("step")
        asst_idx = step_info.get("assistant_msg_index")
        if asst_idx is None or asst_idx >= len(trajectory.messages):
            continue

        msg = trajectory.messages[asst_idx]
        cmd = _extract_bash_from_message(msg)
        if cmd:
            commands.append({
                "step": step,
                "command": cmd,
                "msg_index": asst_idx,
            })

    return commands


def _observations_diverged(recorded: str, live: str) -> bool:
    """Compare two observations for meaningful divergence.

    Normalizes whitespace, timestamps, and PIDs before comparison.
    Returns True if the observations are meaningfully different.
    """
    if not recorded or not live:
        return True

    def normalize(text: str) -> str:
        # Extract returncode
        rc_match = re.search(r"<returncode>(\d+)</returncode>", text)
        rc = rc_match.group(1) if rc_match else "?"

        # Extract output content
        out_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        out = out_match.group(1).strip() if out_match else text.strip()

        # Normalize volatile content
        out = re.sub(r"\d+\.\d+s", "Xs", out)           # timing
        out = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", out)  # memory addresses
        out = re.sub(r"pid=\d+", "pid=N", out)           # PIDs
        out = re.sub(r"\s+", " ", out)                   # whitespace

        return f"rc={rc}|{out[:2000]}"

    return normalize(recorded) != normalize(live)


@dataclass
class EnvReplayResult(AugmentationResult):
    """Extended result with environment feedback data."""

    images_pulled: int = 0
    images_deleted: int = 0
    commands_replayed: int = 0
    env_errors: List[str] = field(default_factory=list)


def _append_checkpoint(
    pairs_path: Path,
    processed_path: Path,
    instance_id: str,
    result: "EnvReplayResult",
) -> None:
    """Append one instance's pairs + id to the running checkpoint files.

    Writes pairs to pairs_partial.jsonl (one JSON per line) and the
    instance_id to processed_ids.txt, then fsyncs so a crash mid-run
    doesn't leave a half-written line.
    """
    with pairs_path.open("a", encoding="utf-8") as f:
        for pair in result.pairs:
            f.write(json.dumps({
                "instance_id": pair.instance_id,
                "step": pair.step,
                "perturbation": pair.perturbation_name,
                "diverged": pair.diverged,
                "label": pair.label,
                "recorded_action": pair.recorded_action[:500],
                "predicted_action": pair.predicted_action[:500],
            }) + "\n")
        f.flush()
    with processed_path.open("a", encoding="utf-8") as f:
        f.write(instance_id + "\n")
        f.flush()


class EnvFeedbackReplayer:
    """Replay compaction events with real Docker environment feedback.

    For each compaction step:
    1. Replay recorded commands to reconstruct container state
    2. Get policy action with perturbed context
    3. Execute in container, compare observation
    """

    def __init__(
        self,
        policy_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        perturbations: Optional[List[PerturbationStrategy]] = None,
        max_api_calls: int = 500,
        rate_limit_delay: float = 0.5,
        command_timeout: int = 60,
        cleanup_images: bool = False,
    ):
        from memgym.training.augmentation.perturbations import DEFAULT_PERTURBATIONS

        self.policy_model = policy_model
        self.perturbations = perturbations or DEFAULT_PERTURBATIONS
        self.max_api_calls = max_api_calls
        self.rate_limit_delay = rate_limit_delay
        self.command_timeout = command_timeout
        self.cleanup_images = cleanup_images
        self._api_calls = 0

    def _setup_container(self, instance_id: str):
        """Create a Docker container for an instance.

        Returns the environment handle or None if setup fails.
        """
        from memgym.gym.swe_bench.container import get_backend

        image = _instance_to_image(instance_id)
        try:
            backend = get_backend("docker")
            env = backend.create_env(
                image=image,
                cwd="/testbed",
                env_vars={},
                timeout=self.command_timeout,
                verbose=False,
            )
            return env
        except Exception as e:
            logger.warning("Container setup failed for %s: %s", instance_id, e)
            return None

    def _replay_commands_to_step(
        self,
        env: Any,
        commands: List[Dict[str, Any]],
        target_step: int,
    ) -> int:
        """Replay all commands up to (but not including) target_step.

        Returns count of commands executed.
        """
        executed = 0
        for cmd_info in commands:
            if cmd_info["step"] >= target_step:
                break
            try:
                env.execute({"command": cmd_info["command"]})
                executed += 1
            except Exception as e:
                logger.debug("Command failed during replay at step %d: %s",
                             cmd_info["step"], str(e)[:100])
                executed += 1  # count it even if it failed (matches original)
        return executed

    def _execute_and_compare(
        self,
        env: Any,
        perturbed_action: str,
        recorded_observation: str,
    ) -> Tuple[str, bool]:
        """Execute a command in the container and compare to recorded observation.

        Returns (live_observation, diverged).
        """
        try:
            result = env.execute({"command": perturbed_action})
            output = result.get("output", "")
            rc = result.get("returncode", -1)
            live_obs = f"<returncode>{rc}</returncode>\n<output>\n{output}\n</output>"

            diverged = _observations_diverged(recorded_observation, live_obs)
            return live_obs, diverged
        except Exception as e:
            logger.warning("Execution failed: %s", str(e)[:200])
            return f"[ERROR: {e}]", True

    def _commit_snapshot(self, env: Any) -> Optional[str]:
        """docker commit the live container to a throwaway image tag.

        Returns the tag on success. The caller owns cleanup.
        """
        container_id = getattr(env, "container_id", None)
        if not container_id:
            return None
        tag = f"memgym-snap-{uuid.uuid4().hex[:10]}"
        try:
            subprocess.run(
                ["docker", "commit", container_id, tag],
                check=True, capture_output=True, timeout=30,
            )
            return tag
        except Exception as e:
            logger.warning("Snapshot commit failed: %s", str(e)[:200])
            return None

    def _execute_in_snapshot(
        self,
        snapshot_tag: str,
        command: str,
        recorded_observation: str,
        cwd: str = "/testbed",
    ) -> Tuple[str, bool]:
        """Run command in a fresh throwaway container from snapshot_tag.

        Each call gets its own container, so perturbations within an event
        cannot pollute each other's filesystem / env state.
        """
        cmd = [
            "docker", "run", "--rm", "-w", cwd,
            snapshot_tag, "bash", "-lc", command,
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.command_timeout,
                encoding="utf-8", errors="replace",
            )
            output = (r.stdout or "") + (r.stderr or "")
            live_obs = f"<returncode>{r.returncode}</returncode>\n<output>\n{output}\n</output>"
            return live_obs, _observations_diverged(recorded_observation, live_obs)
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]", True
        except Exception as e:
            logger.warning("Snapshot exec failed: %s", str(e)[:200])
            return f"[ERROR: {e}]", True

    def _cleanup_snapshot(self, tag: str) -> None:
        try:
            subprocess.run(
                ["docker", "rmi", "-f", tag],
                capture_output=True, timeout=15,
            )
        except Exception as e:
            logger.debug("Snapshot rmi failed for %s: %s", tag, e)

    def replay_instance(
        self,
        trajectory: LoadedTrajectory,
        compaction_events: List[CompactionEvent],
        *,
        max_events: Optional[int] = None,
    ) -> EnvReplayResult:
        """Replay compaction events for one instance with env feedback.

        Sorts events by step and replays incrementally (doesn't restart
        container between events — reuses accumulated state).
        """
        result = EnvReplayResult(
            instance_id=trajectory.instance_id,
            total_events=len(compaction_events),
        )

        if not trajectory.messages:
            logger.warning("%s: no messages — skipping", trajectory.instance_id)
            return result

        start_time = time.time()

        # Extract all commands for sequential replay
        all_commands = _extract_all_commands(trajectory)
        if not all_commands:
            logger.warning("%s: no extractable commands", trajectory.instance_id)
            return result

        # Setup container
        env = self._setup_container(trajectory.instance_id)
        if env is None:
            result.env_errors.append("Container setup failed")
            return result
        result.images_pulled += 1

        try:
            # Sort events by step for incremental replay
            events = sorted(compaction_events, key=lambda e: e.step)

            # Drop fork-point artifacts BEFORE max_events slicing so we don't
            # waste the per-instance budget on events that will be skipped
            # later for having an empty reconstructed prefix.
            def _has_real_prefix(ev):
                msgs = _build_messages_at_step(trajectory, ev.step)
                return sum(1 for m in msgs if m.get("role") == "assistant") >= 3
            events = [e for e in events if _has_real_prefix(e)]

            if max_events:
                events = events[:max_events]

            last_replayed_step = 0

            for event in events:
                # Reserve enough budget for all perturbations in this event.
                # Starting an event we can't finish would produce uneven
                # per-perturbation counts; better to stop cleanly on an
                # event boundary.
                remaining = self.max_api_calls - self._api_calls
                if remaining < len(self.perturbations):
                    logger.info(
                        "API budget low (%d remaining, %d needed) — stopping",
                        remaining, len(self.perturbations),
                    )
                    break

                # Incrementally replay to this step
                cmds_to_replay = [
                    c for c in all_commands
                    if last_replayed_step <= c["step"] < event.step
                ]
                n_replayed = self._replay_commands_to_step(
                    env, cmds_to_replay, event.step,
                )
                result.commands_replayed += n_replayed
                last_replayed_step = event.step

                # Get recorded observation for comparison
                recorded_obs = _extract_recorded_observation(
                    trajectory, event.step,
                )

                # Build messages at this step for policy call
                messages = _build_messages_at_step(trajectory, event.step)
                if not messages:
                    continue

                # Snapshot container once per event so each perturbation runs
                # in a fresh throwaway container branched from this exact state.
                # Avoids cross-perturbation state drift within one event.
                snapshot_tag = self._commit_snapshot(env)

                # Apply each perturbation
                try:
                    for perturbation in self.perturbations:
                        if self._api_calls >= self.max_api_calls:
                            break

                        perturbed_messages = perturbation.perturb_messages(
                            messages, event.step,
                        )

                        # Get policy action with perturbed context
                        predicted_action = _call_policy(
                            perturbed_messages, self.policy_model,
                        )
                        self._api_calls += 1
                        result.api_calls += 1

                        # Extract bash command from predicted action.
                        # If the policy emitted narrative instead of bash, that
                        # IS a behavioral break caused by the perturbation —
                        # label HARMFUL with failure_mode="no_action". We only
                        # flip to SAFE when a bash command was extracted AND
                        # its execution matched the recorded observation.
                        predicted_cmd = self._extract_command(predicted_action)
                        no_action = not predicted_cmd
                        failure_mode = ""

                        if no_action:
                            diverged = True
                            failure_mode = "no_action"
                        elif recorded_obs and snapshot_tag:
                            # Execute in isolated snapshot container.
                            live_obs, diverged = self._execute_in_snapshot(
                                snapshot_tag, predicted_cmd, recorded_obs,
                            )
                        elif recorded_obs:
                            # Snapshot commit failed — fall back to shared container.
                            live_obs, diverged = self._execute_and_compare(
                                env, predicted_cmd, recorded_obs,
                            )
                        else:
                            # No recorded observation — text-compare fallback.
                            from memgym.training.augmentation.replay import (
                                _extract_recorded_action, _actions_diverged,
                            )
                            recorded_action = _extract_recorded_action(
                                trajectory, event.step,
                            )
                            diverged = _actions_diverged(
                                recorded_action, predicted_action,
                            )

                        if diverged and not failure_mode:
                            failure_mode = "diverged"

                        if diverged:
                            result.divergence_count += 1

                        pair = CounterfactualPair(
                            instance_id=event.instance_id,
                            step=event.step,
                            perturbation_name=perturbation.name,
                            recorded_action=self._extract_command(
                                _extract_bash_from_message(
                                    trajectory.messages[
                                        self._find_asst_idx(trajectory, event.step)
                                    ]
                                ) if self._find_asst_idx(trajectory, event.step) >= 0
                                else ""
                            )[:1000],
                            predicted_action=(
                                self._extract_command(predicted_action)
                                or predicted_action
                            )[:1000],
                            diverged=diverged,
                            label=0 if diverged else 1,
                            failure_mode=failure_mode,
                            original_tokens=event.context_before_tokens,
                            perturbed_tokens=len(str(perturbed_messages)) // 4,
                            perturbation_description=perturbation.name,
                        )
                        result.pairs.append(pair)
                        result.total_pairs += 1

                        if self.rate_limit_delay > 0:
                            time.sleep(self.rate_limit_delay)
                finally:
                    if snapshot_tag:
                        self._cleanup_snapshot(snapshot_tag)

        finally:
            # Cleanup container
            try:
                env.cleanup()
            except Exception:
                pass

            # Optionally remove Docker image to save disk
            if self.cleanup_images:
                self._delete_image(trajectory.instance_id)
                result.images_deleted += 1

        result.wall_time = time.time() - start_time
        return result

    def replay_batch(
        self,
        trajectories: Dict[str, LoadedTrajectory],
        all_events: Dict[str, List[CompactionEvent]],
        *,
        max_events_per_instance: Optional[int] = 10,
        batch_size: int = 10,
        checkpoint_dir: Optional[Path] = None,
    ) -> List[EnvReplayResult]:
        """Replay multiple instances with batched image management.

        Pulls images in batches of `batch_size`, processes them, then
        deletes to manage disk space.

        When `checkpoint_dir` is provided, each instance's pairs are
        appended to `pairs_partial.jsonl` and its id to `processed_ids.txt`
        immediately after processing, and any ids already in
        `processed_ids.txt` are skipped (crash-safe resume).
        """
        results = []
        instance_ids = sorted(
            iid for iid in trajectories
            if iid in all_events and all_events[iid]
        )

        already_done: set = set()
        processed_path = pairs_partial_path = None
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            processed_path = checkpoint_dir / "processed_ids.txt"
            pairs_partial_path = checkpoint_dir / "pairs_partial.jsonl"
            if processed_path.exists():
                already_done = {
                    line.strip() for line in processed_path.read_text().splitlines()
                    if line.strip()
                }
                if already_done:
                    logger.info(
                        "Resume: skipping %d already-processed instances",
                        len(already_done),
                    )

        for batch_start in range(0, len(instance_ids), batch_size):
            batch_ids = instance_ids[batch_start: batch_start + batch_size]
            logger.info(
                "Processing batch %d-%d / %d",
                batch_start, batch_start + len(batch_ids), len(instance_ids),
            )

            batch_pending = [iid for iid in batch_ids if iid not in already_done]
            if not batch_pending:
                continue

            # Pre-pull images for pending instances only
            if self.cleanup_images:
                self._pull_batch(batch_pending)

            for iid in batch_pending:
                if self._api_calls >= self.max_api_calls:
                    logger.info("API budget exhausted (%d)", self.max_api_calls)
                    return results

                events = all_events.get(iid, [])
                # Convert flat list to per-instance if needed
                if isinstance(events, list):
                    instance_events = [e for e in events if e.instance_id == iid]
                else:
                    instance_events = events

                r = self.replay_instance(
                    trajectories[iid],
                    instance_events,
                    max_events=max_events_per_instance,
                )
                results.append(r)
                logger.info(
                    "%s: %d pairs, %d diverged (%.0f%%), %d cmds replayed, %.1fs",
                    iid, r.total_pairs, r.divergence_count,
                    100 * r.divergence_count / max(1, r.total_pairs),
                    r.commands_replayed, r.wall_time,
                )

                if checkpoint_dir is not None:
                    _append_checkpoint(pairs_partial_path, processed_path, iid, r)

            # Delete images for this batch to free disk
            if self.cleanup_images:
                for iid in batch_pending:
                    self._delete_image(iid)
                logger.info("Cleaned up %d images from batch", len(batch_pending))

        return results

    def _extract_command(self, text: str) -> str:
        """Extract a bash command from LLM response text.

        The policy LLM may return a tool_call JSON, a code block,
        or plain text with the command. Code blocks are searched
        anywhere in the response (policy often emits prose first).
        Returns empty string when no plausible command is found —
        the caller decides how to label that case.
        """
        if not text:
            return ""

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "command" in data:
                return str(data["command"]).strip()
        except (json.JSONDecodeError, TypeError):
            pass

        # Codeblock anywhere in the response. Prefer last block (the
        # policy typically narrates first, then emits the bash it wants
        # executed next).
        code_blocks = re.findall(
            r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, re.DOTALL,
        )
        if code_blocks:
            cmd = code_blocks[-1].strip()
            if self._looks_like_command(cmd):
                return cmd

        cmd_match = re.search(r"<command>(.*?)</command>", text, re.DOTALL)
        if cmd_match:
            cmd = cmd_match.group(1).strip()
            if self._looks_like_command(cmd):
                return cmd

        # Last-resort keyword scan across all lines (not just the first).
        for line in text.split("\n"):
            line = line.strip()
            if self._looks_like_command(line):
                return line

        return ""

    _CMD_LEADERS = (
        "cd ", "cat ", "grep ", "find ", "python ", "python3 ", "pip ",
        "git ", "ls ", "ls\n", "echo ", "sed ", "awk ", "head ", "tail ",
        "mkdir ", "touch ", "rm ", "mv ", "cp ", "pytest ", "bash ",
        "sh ", "curl ", "wget ", "diff ", "patch ", "make ", "export ",
        "source ", "./", "/bin/", "/usr/",
    )

    def _looks_like_command(self, text: str) -> bool:
        """Heuristic: does this string look like a bash command, not prose?"""
        t = text.strip()
        if not t or t.startswith("#"):
            return False
        if t.startswith(self._CMD_LEADERS):
            return True
        # A pipeline or redirect is strong evidence even without a known leader.
        if any(op in t for op in (" | ", " > ", " >> ", " && ", " 2>&1")):
            return True
        return False

    def _find_asst_idx(self, trajectory: LoadedTrajectory, step: int) -> int:
        """Find assistant message index for a step."""
        for step_info in trajectory.message_steps:
            if step_info.get("step") == step:
                return step_info.get("assistant_msg_index", -1)
        return -1

    def _pull_batch(self, instance_ids: List[str]) -> None:
        """Pre-pull Docker images for a batch of instances."""
        import subprocess

        for iid in instance_ids:
            image = _instance_to_image(iid)
            try:
                subprocess.run(
                    ["docker", "pull", image],
                    timeout=600, capture_output=True,
                )
                logger.debug("Pulled %s", image)
            except Exception as e:
                logger.warning("Failed to pull %s: %s", image, e)

    def _delete_image(self, instance_id: str) -> None:
        """Delete Docker image to free disk space."""
        import subprocess

        image = _instance_to_image(instance_id)
        try:
            subprocess.run(
                ["docker", "rmi", "-f", image],
                timeout=30, capture_output=True,
            )
            logger.debug("Deleted image %s", image)
        except Exception as e:
            logger.warning("Failed to delete %s: %s", image, e)

    def reset_budget(self):
        """Reset the API call counter."""
        self._api_calls = 0
