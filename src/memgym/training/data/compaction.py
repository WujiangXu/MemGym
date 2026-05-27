"""Extract compression-event neighborhoods from trajectories.

For each trajectory with memory, walks steps and finds was_compacted=True.
Extracts the context around each compaction for world model training.

Refactored from the legacy memory_training pipeline (removed in the
repository reorganization; recoverable from git history).

Usage:
    from memgym.training.data.loader import TrajectoryLoader
    from memgym.training.data.compaction import extract_compaction_events

    loader = TrajectoryLoader("trajectories")
    trajs = loader.load_experiment("train50_sonnet45_llmsummarizing")
    events = extract_compaction_events(trajs["getmoto__moto-4881"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memgym.training.data.loader import (
    LoadedTrajectory,
    StepData,
    Tau2LoadedTrajectory,
    Tau2StepData,
)

logger = logging.getLogger(__name__)

# Max chars for context text fields to keep examples manageable
MAX_CONTEXT_CHARS = 30_000


@dataclass
class CompactionEvent:
    """A single compression event with its surrounding context."""

    instance_id: str
    step: int
    total_steps: int

    # --- Before compression ---
    context_before_tokens: int = 0
    context_before_msgs: int = 0

    # --- The compression action ---
    summary_generated: str = ""
    forgotten_count: int = 0
    compression_ratio: float = 1.0
    view_size: Optional[int] = None

    # --- After compression ---
    context_after_tokens: int = 0
    context_after_msgs: int = 0

    # --- Agent response at this step ---
    agent_action: str = ""
    agent_thought: str = ""

    # --- Neighborhood steps (for richer context) ---
    pre_steps: List[Dict[str, str]] = field(default_factory=list)
    post_steps: List[Dict[str, str]] = field(default_factory=list)

    # --- Episode-level outcome ---
    episode_resolved: Optional[bool] = None
    episode_reward: float = 0.0

    # --- Task context ---
    problem_statement: str = ""
    model: str = ""
    memory_strategy: str = ""


def _step_to_summary(step: StepData) -> Dict[str, str]:
    """Flatten a step into a ReAct-shaped dict for the pre-compaction window.

    mini-swe-agent puts the full reasoning in ``thought`` and leaves ``action``
    empty most of the time, so thought is the primary signal here. We don't
    per-field truncate — the prompt builder sizes its window to seq_len=32k.
    """
    return {
        "step": str(step.step),
        "thought": step.thought or "",
        "action": step.action or "",
        "observation": step.observation or "",
    }


def extract_compaction_events(
    trajectory: LoadedTrajectory,
    *,
    episode_resolved: Optional[bool] = None,
    episode_reward: float = 0.0,
    context_window: int = 30,
) -> List[CompactionEvent]:
    """Extract all compaction events from a trajectory.

    Args:
        trajectory: Loaded trajectory with per-step memory data.
        episode_resolved: Whether the episode resolved (from paired labels).
        episode_reward: Combined reward for the episode.
        context_window: Number of steps before/after compaction to include.
            Default 30 is sized to cover the typical ms=100 forgotten range
            (~50 messages ≈ 25 agent steps) plus slack.

    Returns:
        List of CompactionEvent, one per *new* compaction snapshot. SWE-Gym
        ``_training.json`` only records the sticky ``was_compacted`` flag,
        so we de-duplicate by the (original_msgs, filtered_msgs, summary)
        identity to emit one event per true compaction rather than one per
        post-compaction step.
    """
    if not trajectory.has_step_data:
        logger.debug(
            "Trajectory %s has no step data — skipping compaction extraction",
            trajectory.instance_id,
        )
        return []

    steps = trajectory.steps
    total_steps = len(steps)
    events = []

    for i, step in enumerate(steps):
        if not step.memory.was_compacted:
            continue

        # In SWE-Gym _training.json, ``was_compacted`` is sticky-True after
        # the first compaction. The *summary* field is the true event marker:
        # it is populated only on the step where compaction actually fires,
        # and None on all subsequent sticky steps.
        if not step.memory.summary:
            continue

        # Pre-compression context (window of steps before this one)
        pre_start = max(0, i - context_window)
        pre_steps = [_step_to_summary(steps[j]) for j in range(pre_start, i)]

        # Post-compression context (window of steps after this one)
        post_end = min(total_steps, i + 1 + context_window)
        post_steps = [_step_to_summary(steps[j]) for j in range(i + 1, post_end)]

        event = CompactionEvent(
            instance_id=trajectory.instance_id,
            step=step.step,
            total_steps=total_steps,
            # Before
            context_before_tokens=step.memory.original_tokens,
            context_before_msgs=step.memory.original_msgs or 0,
            # Compression
            summary_generated=step.memory.summary or "",
            forgotten_count=step.memory.forgotten or 0,
            compression_ratio=step.memory.compression_ratio,
            view_size=step.memory.view_size,
            # After
            context_after_tokens=step.memory.filtered_tokens,
            context_after_msgs=step.memory.filtered_msgs or 0,
            # Agent at this step
            agent_action=step.action[:2000],
            agent_thought=step.thought[:2000],
            # Neighborhood
            pre_steps=pre_steps,
            post_steps=post_steps,
            # Episode outcome
            episode_resolved=episode_resolved,
            episode_reward=episode_reward,
            # Task context
            problem_statement=trajectory.problem_statement[:3000],
            model=trajectory.model,
            memory_strategy=trajectory.memory_strategy,
        )
        events.append(event)

    return events


def extract_all_compaction_events(
    trajectories: Dict[str, LoadedTrajectory],
    labels: Optional[Dict[str, Any]] = None,
    context_window: int = 30,
) -> List[CompactionEvent]:
    """Extract compaction events from all trajectories.

    Args:
        trajectories: {instance_id: LoadedTrajectory} from loader.
        labels: Optional {instance_id: PairedLabel} for resolve/reward info.
        context_window: Steps before/after each compaction event.

    Returns:
        List of all CompactionEvent across all trajectories.
    """
    all_events = []
    for iid, traj in sorted(trajectories.items()):
        resolved = None
        reward = 0.0
        if labels and iid in labels:
            pl = labels[iid]
            resolved = pl.strategy_resolved if hasattr(pl, "strategy_resolved") else None
            reward = pl.reward if hasattr(pl, "reward") else 0.0

        events = extract_compaction_events(
            traj,
            episode_resolved=resolved,
            episode_reward=reward,
            context_window=context_window,
        )
        all_events.extend(events)

    logger.info(
        "Extracted %d compaction events from %d trajectories",
        len(all_events), len(trajectories),
    )
    return all_events


def print_compaction_stats(events: List[CompactionEvent]) -> None:
    """Print summary statistics for extracted compaction events."""
    n = len(events)
    if n == 0:
        print("No compaction events.")
        return

    instances = set(e.instance_id for e in events)
    ratios = [e.compression_ratio for e in events]
    forgotten = [e.forgotten_count for e in events]
    summary_lens = [len(e.summary_generated) for e in events]

    print(f"\n{'='*60}")
    print("COMPACTION EVENT STATISTICS")
    print(f"{'='*60}")
    print(f"Total events: {n}")
    print(f"Unique instances: {len(instances)}")
    print(f"Avg events/instance: {n/len(instances):.1f}")
    print(f"\nCompression ratio: min={min(ratios):.2f}, avg={sum(ratios)/n:.2f}, max={max(ratios):.2f}")
    print(f"Forgotten msgs: min={min(forgotten)}, avg={sum(forgotten)/n:.1f}, max={max(forgotten)}")
    print(f"Summary length (chars): min={min(summary_lens)}, avg={sum(summary_lens)/n:.0f}, max={max(summary_lens)}")

    resolved_counts = {"True": 0, "False": 0, "None": 0}
    for e in events:
        resolved_counts[str(e.episode_resolved)] += 1
    print(f"\nEpisode resolved: {resolved_counts}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Tau2CompactionEvent — same shape as CompactionEvent but keyed by
# ``(domain, task_id, side, turn_idx)`` and carrying the full tau2 Message
# at ``action_msg`` rather than flattened ReAct strings. Kept as a distinct
# dataclass (not a subclass) because the ``instance_id + step`` identity
# SWE code assumes doesn't survive the two-sided split.
# ---------------------------------------------------------------------------

# Max msgs in each half of the neighborhood; mirrors SWE's context_window
# default of 3 but operates on raw tau2 Messages instead of ReAct triples.
DEFAULT_MSG_WINDOW = 3


@dataclass
class Tau2CompactionEvent:
    """A single tau2 compaction event, one per ``new_compaction=True`` step."""

    # Identity
    domain: str
    task_id: str
    phase: str           # "baseline" | "memory"
    side: str            # "agent" | "user"
    step: int            # per-side step index
    turn_idx: int        # orchestrator-level turn counter
    msg_index: int       # index into ``trajectory.messages``

    # Progress within this side's dialogue
    total_steps: int = 0

    # Before compression
    context_before_tokens: int = 0
    context_before_msgs: int = 0

    # The compression action
    summary_generated: str = ""
    forgotten_count: int = 0
    compression_ratio: float = 1.0
    view_size: Optional[int] = None

    # After compression
    context_after_tokens: int = 0
    context_after_msgs: int = 0

    # The wrapper-produced message for this step + its orchestrator neighborhood
    action_msg: Optional[Dict[str, Any]] = None
    pre_msgs: List[Dict[str, Any]] = field(default_factory=list)
    post_msgs: List[Dict[str, Any]] = field(default_factory=list)

    # Episode-level
    episode_outcome: str = ""
    episode_reward: float = 0.0

    # Config
    memory_strategy: str = ""     # per-side strategy string
    memory_side: str = ""         # "both" | "agent" | "user"

    @property
    def instance_key(self) -> str:
        parts = [p for p in (self.phase, self.domain, self.task_id) if p]
        return "/".join(parts)


def _slim_msg(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop heavy fields from a tau2 Message to keep event payloads small.

    tau2 Message dumps carry ``raw_data`` (the full LLM response dict) and
    audio-channel fields that bloat JSON with no training value. We keep
    role, content, tool_calls, turn_idx, and cost for downstream labeling.
    """
    keep = ("role", "content", "tool_calls", "turn_idx", "cost", "usage")
    return {k: msg.get(k) for k in keep if k in msg}


def extract_tau2_compaction_events(
    trajectory: Tau2LoadedTrajectory,
    *,
    msg_window: int = DEFAULT_MSG_WINDOW,
) -> List[Tau2CompactionEvent]:
    """Walk a tau2 trajectory and emit one event per ``new_compaction=True`` step.

    Events are emitted for BOTH sides — the caller gets a single flat list
    with ``side`` discriminating. ``was_compacted`` is deliberately NOT the
    trigger; using it would fire once per step after the first compaction
    (it's sticky-True post-event) and wildly inflate event counts.
    """
    events: List[Tau2CompactionEvent] = []
    msgs = trajectory.messages
    agent_total = len(trajectory.agent_steps)
    user_total = len(trajectory.user_steps)

    for s in trajectory.steps:
        if not s.memory.new_compaction:
            continue

        mi = s.msg_index
        pre_start = max(0, mi - msg_window)
        post_end = min(len(msgs), mi + 1 + msg_window)
        action_msg = _slim_msg(msgs[mi]) if 0 <= mi < len(msgs) else None
        pre = [_slim_msg(msgs[j]) for j in range(pre_start, mi)] if mi >= 0 else []
        post = [_slim_msg(msgs[j]) for j in range(mi + 1, post_end)] if mi >= 0 else []

        strategy = (
            trajectory.memory_strategy_agent if s.side == "agent"
            else trajectory.memory_strategy_user
        )

        events.append(Tau2CompactionEvent(
            domain=trajectory.domain,
            task_id=trajectory.task_id,
            phase=trajectory.phase,
            side=s.side,
            step=s.step,
            turn_idx=s.turn_idx,
            msg_index=mi,
            total_steps=agent_total if s.side == "agent" else user_total,
            context_before_tokens=s.memory.original_tokens,
            context_before_msgs=s.memory.original_msgs or 0,
            summary_generated=s.memory.summary or "",
            forgotten_count=s.memory.forgotten or 0,
            compression_ratio=s.memory.compression_ratio,
            view_size=s.memory.view_size,
            context_after_tokens=s.memory.filtered_tokens,
            context_after_msgs=s.memory.filtered_msgs or 0,
            action_msg=action_msg,
            pre_msgs=pre,
            post_msgs=post,
            episode_outcome=trajectory.episode_outcome,
            episode_reward=trajectory.episode_reward,
            memory_strategy=strategy,
            memory_side=trajectory.memory_side,
        ))

    return events


def extract_all_tau2_compaction_events(
    trajectories: Dict[str, Tau2LoadedTrajectory],
    *,
    msg_window: int = DEFAULT_MSG_WINDOW,
) -> List[Tau2CompactionEvent]:
    """Flatten event extraction across a tau2 run's loaded trajectories."""
    all_events: List[Tau2CompactionEvent] = []
    for _, traj in sorted(trajectories.items()):
        all_events.extend(extract_tau2_compaction_events(traj, msg_window=msg_window))
    logger.info(
        "Extracted %d tau2 compaction events from %d trajectories",
        len(all_events), len(trajectories),
    )
    return all_events


def print_tau2_compaction_stats(events: List[Tau2CompactionEvent]) -> None:
    """Compact summary of a tau2 event list, partitioned by side."""
    n = len(events)
    if n == 0:
        print("No tau2 compaction events.")
        return

    agent = [e for e in events if e.side == "agent"]
    user = [e for e in events if e.side == "user"]
    tasks = {(e.phase, e.domain, e.task_id) for e in events}
    domains = {e.domain for e in events}

    print(f"\n{'='*60}")
    print("TAU2 COMPACTION EVENT STATISTICS")
    print(f"{'='*60}")
    print(f"Total events: {n}  (agent={len(agent)} user={len(user)})")
    print(f"Unique tasks: {len(tasks)}  Domains: {sorted(domains)}")
    if agent:
        ratios = [e.compression_ratio for e in agent]
        print(f"Agent compression: min={min(ratios):.2f} avg={sum(ratios)/len(ratios):.2f} max={max(ratios):.2f}")
    if user:
        ratios = [e.compression_ratio for e in user]
        print(f"User  compression: min={min(ratios):.2f} avg={sum(ratios)/len(ratios):.2f} max={max(ratios):.2f}")
    summary_lens = [len(e.summary_generated) for e in events]
    print(f"Summary length (chars): min={min(summary_lens)} avg={sum(summary_lens)//len(summary_lens)} max={max(summary_lens)}")
    print(f"{'='*60}")
