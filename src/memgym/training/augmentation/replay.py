"""Counterfactual single-step replay augmentation for SWE-bench trajectories.

Adapted from runners/webarena_replay_runner.py:ObservationReplayRunner.

At each compaction event in a memory-augmented trajectory:
1. Reconstruct the message history up to that step
2. Apply a perturbation (modified compression config or message dropout)
3. Call the policy LLM with the perturbed context
4. Compare the action to the recorded baseline action
5. Diverged → label as HARMFUL; same → label as SAFE

This creates counterfactual training pairs without re-running Docker containers.

Usage:
    from memgym.training.augmentation.replay import SWEBenchReplayAugmenter
    from memgym.training.augmentation.perturbations import DEFAULT_PERTURBATIONS

    augmenter = SWEBenchReplayAugmenter(
        policy_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        perturbations=DEFAULT_PERTURBATIONS,
    )
    pairs = augmenter.augment_instance(trajectory, compaction_events)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memgym.training.augmentation.perturbations import PerturbationStrategy
from memgym.training.data.compaction import CompactionEvent
from memgym.training.data.loader import LoadedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualPair:
    """A counterfactual comparison at a single compaction step."""

    instance_id: str
    step: int
    perturbation_name: str

    # Original action (from recorded trajectory)
    recorded_action: str

    # Counterfactual action (from perturbed context)
    predicted_action: str

    # Did the action change?
    diverged: bool

    # Derived label: compression affected downstream behavior.
    # 1=SAFE (same bash, matching observation),
    # 0=HARMFUL (observation diverged OR policy failed to produce bash).
    label: int

    # Nature of the HARMFUL case (empty for SAFE):
    #   "diverged"  — executed but produced different observation
    #   "no_action" — policy emitted prose/narrative instead of bash
    failure_mode: str = ""

    # Context for debugging
    original_tokens: int = 0
    perturbed_tokens: int = 0
    perturbation_description: str = ""


@dataclass
class AugmentationResult:
    """Summary of augmentation for one instance."""

    instance_id: str
    total_events: int = 0
    total_pairs: int = 0
    divergence_count: int = 0
    api_calls: int = 0
    wall_time: float = 0.0
    pairs: List[CounterfactualPair] = field(default_factory=list)


def _build_messages_at_step(
    trajectory: LoadedTrajectory,
    target_step: int,
) -> List[Dict[str, Any]]:
    """Reconstruct the message history up to a given step.

    Uses _replay.json message_steps to find the assistant_msg_index
    for the target step, then returns messages[:assistant_msg_index].
    """
    if not trajectory.messages or not trajectory.message_steps:
        return []

    # Find the step entry matching our target
    for step_info in trajectory.message_steps:
        if step_info.get("step") == target_step:
            assistant_idx = step_info.get("assistant_msg_index")
            if assistant_idx is not None and assistant_idx <= len(trajectory.messages):
                return trajectory.messages[:assistant_idx]

    # Fallback: if exact step not found, try matching by position
    # _training.json steps are 1-indexed, message_steps may differ
    for step_info in trajectory.message_steps:
        if step_info.get("step") == target_step - 1:
            obs_idx = step_info.get("observation_msg_index")
            if obs_idx is not None and obs_idx + 1 <= len(trajectory.messages):
                return trajectory.messages[: obs_idx + 1]

    logger.debug(
        "Could not find messages for step %d in %s (has %d message_steps)",
        target_step, trajectory.instance_id, len(trajectory.message_steps),
    )
    return []


def _extract_bash_from_message(msg: Dict[str, Any]) -> str:
    """Extract bash command from an assistant message.

    Handles both formats:
    - tool_calls: [{"function": {"name": "bash", "arguments": '{"command": "..."}' }}]
    - plain content: text with the action
    """
    # Try tool_calls first (standard format for mini-swe-agent).
    # GPT-OSS trajectories store tool_calls as None (not []) on turns
    # with no tool call, so coerce.
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") == "bash":
            try:
                args = json.loads(fn.get("arguments", "{}"))
                cmd = args.get("command", "")
                if cmd:
                    return cmd
            except (json.JSONDecodeError, TypeError):
                pass

    # Fall back to content
    content = msg.get("content", "")
    if content:
        return str(content)

    return ""


def _extract_observation_from_message(msg: Dict[str, Any]) -> str:
    """Extract observation text from a tool response message.

    Format: <returncode>N</returncode>\n<output>...</output>
    """
    return str(msg.get("content", ""))


def _extract_recorded_action(
    trajectory: LoadedTrajectory,
    target_step: int,
) -> str:
    """Get the recorded agent action at a step.

    Priority:
    1. Replay messages tool_calls (has actual bash commands)
    2. Training.json steps (has thought/action, but often empty)
    """
    # From replay.json message_steps — parse tool_calls for bash commands
    for step_info in trajectory.message_steps:
        if step_info.get("step") == target_step:
            assistant_idx = step_info.get("assistant_msg_index")
            if assistant_idx is not None and assistant_idx < len(trajectory.messages):
                msg = trajectory.messages[assistant_idx]
                cmd = _extract_bash_from_message(msg)
                if cmd:
                    return cmd

    # Fallback: training.json steps
    for step in trajectory.steps:
        if step.step == target_step:
            if step.action:
                return step.action
            if step.thought:
                return step.thought

    return ""


def _extract_recorded_observation(
    trajectory: LoadedTrajectory,
    target_step: int,
) -> str:
    """Get the recorded environment observation at a step.

    Reads the tool response message following the assistant action.
    """
    for step_info in trajectory.message_steps:
        if step_info.get("step") == target_step:
            assistant_idx = step_info.get("assistant_msg_index")
            if assistant_idx is not None:
                # Tool response is the message after the assistant message
                obs_idx = assistant_idx + 1
                if obs_idx < len(trajectory.messages):
                    msg = trajectory.messages[obs_idx]
                    if msg.get("role") == "tool":
                        return _extract_observation_from_message(msg)
    return ""


def _flatten_messages_for_chat(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten tool_calls/tool messages into plain text chat format.

    Bedrock Converse API requires toolConfig when messages contain
    tool_use/toolResult blocks. We avoid this by converting:
    - assistant messages with tool_calls → text showing the command
    - tool role messages → user messages with the output
    This also merges consecutive same-role messages (required by some APIs).
    """
    flat: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "tool":
            # Convert tool response to user message
            tool_text = str(content) if content else "(no output)"
            role = "user"
            content = f"[Tool Output]\n{tool_text}"
        elif role == "assistant":
            # If assistant used tool_calls, extract the command as text
            tool_calls = m.get("tool_calls", [])
            if tool_calls and not content:
                parts = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = fn.get("arguments", "")
                    if name == "bash" and isinstance(args, dict):
                        parts.append(args.get("command", str(args)))
                    else:
                        parts.append(f"{name}({args})")
                content = "\n".join(parts)
            else:
                content = str(content) if content else ""
        elif role in ("system", "user"):
            content = str(content) if content else ""
        else:
            continue  # skip unknown roles

        # Merge consecutive same-role messages
        if flat and flat[-1]["role"] == role:
            flat[-1]["content"] += "\n" + content
        else:
            flat.append({"role": role, "content": content})

    return flat


def _call_policy(
    messages: List[Dict[str, Any]],
    model: str,
) -> str:
    """Call the policy LLM with a message history.

    Returns the assistant's response text.
    """
    import litellm

    chat_messages = _flatten_messages_for_chat(messages)

    if not chat_messages:
        return ""

    try:
        response = litellm.completion(
            model=model,
            messages=chat_messages,
            max_tokens=4000,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Policy call failed: %s", e)
        return f"[ERROR: {e}]"


_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.DOTALL)


def _extract_command_from_response(text: str) -> str:
    """Extract a bash command from an LLM text response.

    The policy LLM returns full text with reasoning. We need to extract
    just the bash command for comparison against the recorded action.

    Handles: ```bash ...```, inline `command`, or lines starting with $ / cd / cat / etc.
    """
    # Try fenced code block first
    m = _BASH_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    # Try generic code block
    m = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Look for lines that look like shell commands
    cmd_starters = (
        "cd ", "cat ", "find ", "grep ", "python ", "pip ", "git ",
        "echo ", "mkdir ", "rm ", "cp ", "mv ", "ls ", "sed ", "awk ",
        "touch ", "chmod ", "curl ", "wget ", "tar ", "apt ", "make ",
        "export ", "source ", "./", "bash ", "sh ",
    )
    for line in text.split("\n"):
        stripped = line.strip()
        # Strip leading $ prompt
        if stripped.startswith("$ "):
            return stripped[2:]
        if stripped.lower().startswith(cmd_starters):
            return stripped

    # Fallback: return raw text
    return text.strip()


def _actions_diverged(recorded: str, predicted: str) -> bool:
    """Check if two actions are meaningfully different.

    Extracts bash commands from both strings and compares them.
    The predicted string may be a full LLM response with reasoning.
    """
    r = recorded.strip()
    p = _extract_command_from_response(predicted)

    if not r or not p:
        # Ambiguous: cannot extract a bash command from either side. Returning
        # True here silently defaults every extraction failure to "diverged",
        # which inflates the divergence rate. Return False and let the caller
        # detect/skip via the (empty command, empty command) signature.
        return False

    # Normalize for comparison
    r_norm = r.lower().strip()
    p_norm = p.lower().strip()

    # Exact match
    if r_norm == p_norm:
        return False

    # Check if recorded is contained in predicted (predicted might have extra context)
    if r_norm in p_norm or p_norm in r_norm:
        return False

    # Compare first meaningful line (the core command)
    r_lines = [l.strip() for l in r_norm.split("\n") if l.strip()]
    p_lines = [l.strip() for l in p_norm.split("\n") if l.strip()]
    if r_lines and p_lines and r_lines[0] == p_lines[0]:
        return False

    # Compare command name + first arg (e.g., both run "python -m pytest ...")
    r_parts = r_norm.split()
    p_parts = p_norm.split()
    if len(r_parts) >= 2 and len(p_parts) >= 2:
        if r_parts[0] == p_parts[0] and r_parts[1] == p_parts[1]:
            return False

    return True


class SWEBenchReplayAugmenter:
    """Generate counterfactual training pairs by replaying SWE-bench trajectories.

    Adapted from runners/webarena_replay_runner.py:ObservationReplayRunner.

    For each compaction event, replays the step with perturbed memory context
    and checks if the policy LLM produces a different action.
    """

    def __init__(
        self,
        policy_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        perturbations: Optional[List[PerturbationStrategy]] = None,
        max_api_calls: int = 500,
        rate_limit_delay: float = 0.5,
    ):
        from memgym.training.augmentation.perturbations import DEFAULT_PERTURBATIONS

        self.policy_model = policy_model
        self.perturbations = perturbations or DEFAULT_PERTURBATIONS
        self.max_api_calls = max_api_calls
        self.rate_limit_delay = rate_limit_delay
        self._api_calls = 0

    def augment_instance(
        self,
        trajectory: LoadedTrajectory,
        compaction_events: List[CompactionEvent],
        *,
        max_events: Optional[int] = None,
    ) -> AugmentationResult:
        """Generate counterfactual pairs for one instance.

        Args:
            trajectory: Must have messages loaded (load_messages=True).
            compaction_events: Events from extract_compaction_events.
            max_events: Limit events per instance (for budget control).

        Returns:
            AugmentationResult with all counterfactual pairs.
        """
        result = AugmentationResult(
            instance_id=trajectory.instance_id,
            total_events=len(compaction_events),
        )

        if not trajectory.messages:
            logger.warning(
                "Trajectory %s has no messages — cannot replay",
                trajectory.instance_id,
            )
            return result

        start_time = time.time()
        events_to_process = compaction_events
        if max_events:
            events_to_process = compaction_events[:max_events]

        for event in events_to_process:
            if self._api_calls >= self.max_api_calls:
                logger.info("API call budget exhausted (%d)", self.max_api_calls)
                break

            # Reconstruct messages at this step
            messages = _build_messages_at_step(trajectory, event.step)
            if not messages:
                logger.debug(
                    "No messages at step %d for %s — skipping",
                    event.step, event.instance_id,
                )
                continue

            recorded_action = _extract_recorded_action(trajectory, event.step)

            # Apply each perturbation
            for perturbation in self.perturbations:
                if self._api_calls >= self.max_api_calls:
                    break

                # Perturb messages (for RandomDrop-type perturbations)
                perturbed_messages = perturbation.perturb_messages(
                    messages, event.step,
                )

                # Call policy with perturbed context
                predicted_action = _call_policy(perturbed_messages, self.policy_model)
                self._api_calls += 1
                result.api_calls += 1

                diverged = _actions_diverged(recorded_action, predicted_action)
                extracted_cmd = _extract_command_from_response(predicted_action)
                logger.debug(
                    "step=%d %s: recorded=%s predicted=%s diverged=%s",
                    event.step, perturbation.name,
                    recorded_action[:120], extracted_cmd[:120], diverged,
                )
                if diverged:
                    result.divergence_count += 1

                pair = CounterfactualPair(
                    instance_id=event.instance_id,
                    step=event.step,
                    perturbation_name=perturbation.name,
                    recorded_action=recorded_action[:1000],
                    predicted_action=predicted_action[:1000],
                    diverged=diverged,
                    label=0 if diverged else 1,  # HARMFUL if diverged
                    original_tokens=event.context_before_tokens,
                    perturbed_tokens=len(str(perturbed_messages)) // 4,
                    perturbation_description=perturbation.perturb_config(
                        trajectory.memory_config
                    ).description,
                )
                result.pairs.append(pair)
                result.total_pairs += 1

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

        result.wall_time = time.time() - start_time
        return result

    def augment_batch(
        self,
        trajectories: Dict[str, LoadedTrajectory],
        all_events: Dict[str, List[CompactionEvent]],
        *,
        max_events_per_instance: Optional[int] = 10,
    ) -> List[AugmentationResult]:
        """Augment multiple instances.

        Args:
            trajectories: {instance_id: LoadedTrajectory} with messages loaded.
            all_events: {instance_id: [CompactionEvent]} from extraction.
            max_events_per_instance: Limit events per instance for budget control.
        """
        results = []
        for iid in sorted(trajectories.keys()):
            if iid not in all_events or not all_events[iid]:
                continue
            if self._api_calls >= self.max_api_calls:
                logger.info("API call budget exhausted (%d)", self.max_api_calls)
                break

            result = self.augment_instance(
                trajectories[iid],
                all_events[iid],
                max_events=max_events_per_instance,
            )
            results.append(result)

            logger.info(
                "%s: %d pairs, %d diverged (%.1f%%), %d API calls",
                iid, result.total_pairs, result.divergence_count,
                100 * result.divergence_count / max(1, result.total_pairs),
                result.api_calls,
            )

        return results

    def reset_budget(self):
        """Reset the API call counter."""
        self._api_calls = 0
