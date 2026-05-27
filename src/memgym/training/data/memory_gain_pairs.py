"""Memory-gain pair builder — OOD eval pairs from existing trajectories.

Builds `pairs_partial.jsonl` files (the same schema `aug_pairs.py` consumes)
without running any forking pipeline. Instead, we exploit a property of
the existing trajectory data: for each instance we have a baseline run
(no memory or weak memory) AND a memory-strategy run, both with a final
episode-resolved flag. The signed delta of those two flags is the
memory-gain label:

    Δr = int(ood.resolved) − int(baseline.resolved)
    Δr ≥ 0  → SAFE      (label = 1)   memory did NOT regress the task
    Δr <  0 → HARMFUL   (label = 0)   memory hurt

We emit one pair per **memory-trigger step** of the OOD trajectory
(i.e. where its strategy actually fired a compaction/summarization),
NOT at every aligned step. Memory-trigger detection is delegated to
`extract_compaction_events` so the gate logic matches what the v2 RM
was trained on.

Input  : two trajectory dirs (baseline + OOD), same instance set
Output : <out_dir>/pairs_partial.jsonl in the schema
         `aug_pairs._iter_pairs` validates against —
         {instance_id, step, perturbation, label, recorded_action,
          predicted_action, diverged} plus bookkeeping
         (delta_r, baseline_reward, ood_reward, ood_strategy).

Wire-up: extend `aug_pairs.SOURCE_MODEL_BY_DIRNAME` with the new
heldout dir name → strategy label, then run `aug_pairs.py` against
the dir as you would a forking-pipeline output.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from memgym.training.data.compaction import extract_compaction_events
from memgym.training.data.loader import LoadedTrajectory, TrajectoryLoader


@dataclass
class _TriggerEvent:
    """Lightweight stand-in for ``CompactionEvent`` when we only have the
    ``was_compacted`` rising edge (server-side ``_training.json`` captures drop
    the ``summary`` field that ``extract_compaction_events`` keys on).
    """
    step: int
    agent_action: str


def _action_from_replay(traj: LoadedTrajectory, step: int) -> str:
    """Reconstruct the agent's bash command from `_replay.json` tool_calls.

    Agent-captured ``_training.json`` files leave ``step.action`` empty
    because the action lives in the assistant message's ``tool_calls``.
    Map ``step.step=k`` → the k-th assistant message (1-indexed; mirrors
    what the agent loop emits). Concatenate every tool_call's
    ``function.arguments.command`` so multi-call steps don't lose
    secondary commands. Returns "" if the message has no usable
    tool_calls (e.g. the agent's final reply).
    """
    if not getattr(traj, "messages", None):
        return ""
    asst = [m for m in traj.messages if m.get("role") == "assistant"]
    if step < 1 or step > len(asst):
        return ""
    m = asst[step - 1]
    parts: List[str] = []
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        cmd = args.get("command") or args.get("text") or ""
        if cmd:
            parts.append(cmd)
    return "\n".join(parts)


def _step_action(traj: LoadedTrajectory, step_idx: int) -> str:
    """Action text for `step.step==step_idx`, with replay fallback."""
    for s in traj.steps:
        if s.step == step_idx:
            return s.action or _action_from_replay(traj, step_idx)
    return ""


def _extract_trigger_steps(traj: LoadedTrajectory) -> List:
    """Memory-trigger steps for the OOD trajectory.

    Two capture lineages coexist in the corpus:
    - Local ``/trajectories/`` style: each compaction step has a non-empty
      ``summary``; ``extract_compaction_events`` is the canonical detector
      and the v2 RM was trained on its output.
    - Server-side ``/trajectories/`` style: ``summary`` is omitted (lives in
      ``_replay.json`` instead), so the same detector returns []. We fall
      back to a ``was_compacted`` rising-edge scan: one event per
      False→True transition. Sticky-True steps after the first fire are
      not separate events.

    Always tries the canonical path first; only falls back when it
    produces nothing AND there are sticky compactions to find.
    """
    events = extract_compaction_events(
        traj,
        episode_resolved=bool(traj.resolved) if traj.resolved is not None else False,
        episode_reward=float(traj.resolved or 0),
    )
    if events:
        return events
    out: List[_TriggerEvent] = []
    prev = False
    for s in traj.steps:
        cur = bool(getattr(s.memory, "was_compacted", False))
        if cur and not prev:
            action = s.action or _action_from_replay(traj, s.step)
            out.append(_TriggerEvent(step=s.step, agent_action=action))
        prev = cur
    return out

logger = logging.getLogger(__name__)


def load_resolved_set(reeval_path: Path) -> Optional[Tuple[set, set]]:
    """Read a reeval JSON, return ``(resolved_ids, completed_ids)``.

    Two reeval JSON shapes coexist in this repo:
    - newer: ``{resolved_instances: [<ids>], unresolved_instances: [...], ...}``
    - older: ``{instance_results: [{instance_id, reward, ...}, ...]}``

    `completed_ids` is the union of resolved + unresolved (everything
    the harness actually ran). `resolved_ids` is the subset that passed
    the test suite. Callers use the difference to distinguish "ran AND
    failed" (label=0) from "never reeval'd" (drop).

    Returns None if the path is missing — callers that pass None as
    the ``*_resolved_set`` will fall back to ``LoadedTrajectory.resolved``.
    """
    if reeval_path is None or not reeval_path.exists():
        return None
    with reeval_path.open() as fh:
        d = json.load(fh)
    # Newer SWE-Bench-harness shape: lists are under `*_ids` keys.
    # Counts (`resolved_instances`, `unresolved_instances`, ...) are ints.
    if "resolved_ids" in d:
        resolved = set(d["resolved_ids"])
        completed = set(d.get("completed_ids", [])) | resolved \
            | set(d.get("unresolved_ids", []))
        return resolved, completed
    if "instance_results" in d:
        resolved = {ir["instance_id"] for ir in d["instance_results"]
                    if float(ir.get("reward", 0)) >= 1.0}
        completed = {ir["instance_id"] for ir in d["instance_results"]}
        return resolved, completed
    if "results" in d and isinstance(d["results"], list):
        # `replay_summary.json` shape — per-instance ``resolved`` bool.
        resolved = {r["instance_id"] for r in d["results"] if r.get("resolved")}
        completed = {r["instance_id"] for r in d["results"]}
        return resolved, completed
    raise ValueError(
        f"Unrecognized reeval shape in {reeval_path}: "
        f"keys={sorted(d.keys())}"
    )


@dataclass
class BuildStats:
    instance_overlap: int
    instances_dropped_no_resolved: int
    instances_kept: int
    n_safe: int
    n_harmful: int
    n_pairs_written: int
    events_dropped_no_baseline_step: int
    events_dropped_empty_action: int


def _label_from_delta(delta_r: int) -> int:
    """SAFE if memory didn't regress (Δr ≥ 0), HARMFUL otherwise.

    Per user decision in the planning conversation: '<0 means harmful,
    >=0 is safe'. Δr=0 is therefore SAFE (memory matched baseline).
    """
    return 1 if delta_r >= 0 else 0


def _resolved_int(
    traj: LoadedTrajectory,
    reeval_sets: Optional[Tuple[set, set]] = None,
) -> Optional[int]:
    """Binary 0/1 resolution flag, with reeval-set override.

    When `reeval_sets` is provided (loaded via `load_resolved_set`):
    - in `resolved_set`            → 1
    - in `completed_set` only      → 0  (ran but failed)
    - in neither                   → None  (never reeval'd; drop)

    Without a reeval set, falls back to `traj.resolved`.
    """
    if reeval_sets is not None:
        resolved_set, completed_set = reeval_sets
        if traj.instance_id in resolved_set:
            return 1
        if traj.instance_id in completed_set:
            return 0
        return None
    if traj.resolved is None:
        return None
    return 1 if traj.resolved else 0


def _action_at_step(traj: LoadedTrajectory, step_idx: int) -> Optional[str]:
    """Return the agent action text at trajectory step `step_idx`.

    `step_idx` is the value emitted by `extract_compaction_events`
    (which is `step.step` from `_training.json`). Looks up by `.step`
    field (a few captures emit non-contiguous indices) and falls back
    to `_replay.json` tool_calls when `_training.json` left action empty
    (the server-side capture path elides the bash text from the per-step record).

    When `traj.steps` is empty (e.g. baseline-none trajectories loaded
    only from `_replay.json`) we skip the step-list scan entirely and go
    straight to the replay fallback, so that baseline message steps are
    still paired against OOD compaction events.
    """
    if not traj.steps:
        # No _training.json steps — try the replay message index directly.
        action = _action_from_replay(traj, step_idx)
        return action if action else None
    for s in traj.steps:
        if s.step == step_idx:
            return s.action or _action_from_replay(traj, step_idx)
    return None


def build_pairs_for_strategy(
    baseline_dir: Path,
    ood_dir: Path,
    strategy_name: str,
    out_dir: Path,
    *,
    baseline_reeval: Optional[Path] = None,
    ood_reeval: Optional[Path] = None,
    base_root: Optional[Path] = None,
) -> BuildStats:
    """Materialize one strategy-OOD pair file.

    Args:
        baseline_dir: Trajectory dir for the in-distribution baseline.
            Anything from /trajectories/sonnet45_llm_summarizing_* is
            the canonical sonnet45 baseline.
        ood_dir:      Trajectory dir for the held-out memory strategy.
        strategy_name: Short name encoded into `perturbation` and used
            downstream as the slice id (e.g. "sliding_window_w75").
        out_dir:       Where to write `pairs_partial.jsonl`.
        base_root:     Common parent for the two dirs if both are
            relative to one root. If None, each dir is loaded
            independently via its absolute path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairs_partial.jsonl"

    baseline_sets = load_resolved_set(baseline_reeval) if baseline_reeval else None
    ood_sets = load_resolved_set(ood_reeval) if ood_reeval else None
    if baseline_reeval and baseline_sets is None:
        raise FileNotFoundError(f"baseline_reeval not found: {baseline_reeval}")
    if ood_reeval and ood_sets is None:
        raise FileNotFoundError(f"ood_reeval not found: {ood_reeval}")

    # We use TrajectoryLoader twice with the parent of each dir as base.
    # `load_experiment` discovers instance_ids by scanning *_training.json
    # / *_replay.json under <base>/<dir_name>/[trajectories/]. Passing
    # the parent + last component handles both layouts uniformly.
    baseline_loader = TrajectoryLoader(baseline_dir.parent)
    ood_loader = TrajectoryLoader(ood_dir.parent)

    logger.info("loading baseline: %s", baseline_dir)
    baseline_trajs = baseline_loader.load_experiment(baseline_dir.name)
    logger.info("loading ood:      %s", ood_dir)
    ood_trajs = ood_loader.load_experiment(ood_dir.name)

    shared = sorted(set(baseline_trajs) & set(ood_trajs))
    logger.info(
        "shared instances: %d (baseline=%d, ood=%d)",
        len(shared), len(baseline_trajs), len(ood_trajs),
    )

    stats = BuildStats(
        instance_overlap=len(shared),
        instances_dropped_no_resolved=0,
        instances_kept=0,
        n_safe=0,
        n_harmful=0,
        n_pairs_written=0,
        events_dropped_no_baseline_step=0,
        events_dropped_empty_action=0,
    )

    with out_path.open("w") as fh:
        for iid in shared:
            base_traj = baseline_trajs[iid]
            ood_traj = ood_trajs[iid]

            r_base = _resolved_int(base_traj, baseline_sets)
            r_ood = _resolved_int(ood_traj, ood_sets)
            if r_base is None or r_ood is None:
                stats.instances_dropped_no_resolved += 1
                continue

            delta_r = r_ood - r_base
            label = _label_from_delta(delta_r)

            # Memory-trigger steps come from the OOD strategy — the v2
            # RM is being asked "was this memory decision safe?", so the
            # event source has to be the side whose memory we're scoring.
            events = _extract_trigger_steps(ood_traj)
            if not events:
                continue
            stats.instances_kept += 1

            for ev in events:
                ood_action = ev.agent_action or ""
                base_action = _action_at_step(base_traj, ev.step)
                if base_action is None:
                    stats.events_dropped_no_baseline_step += 1
                    continue
                if not ood_action.strip() or not base_action.strip():
                    # `aug_pairs._iter_pairs` filters empty
                    # `recorded_action`; mirror that here so the
                    # downstream count is honest.
                    stats.events_dropped_empty_action += 1
                    continue

                row = {
                    "instance_id": iid,
                    "step": int(ev.step),
                    "perturbation": f"strategy_swap__{strategy_name}",
                    "diverged": ood_action != base_action,
                    "label": int(label),
                    "split": "eval",
                    "recorded_action": base_action[:2000],
                    "predicted_action": ood_action[:2000],
                    # Bookkeeping for later analysis — these are extra
                    # keys aug_pairs ignores but downstream auditors
                    # can still read from the raw pairs file.
                    "delta_r": int(delta_r),
                    "baseline_reward": int(r_base),
                    "ood_reward": int(r_ood),
                    "ood_strategy": strategy_name,
                }
                fh.write(json.dumps(row) + "\n")
                stats.n_pairs_written += 1
                if label == 1:
                    stats.n_safe += 1
                else:
                    stats.n_harmful += 1

    logger.info(
        "wrote %d pairs to %s (safe=%d harmful=%d, %d instances)",
        stats.n_pairs_written, out_path, stats.n_safe, stats.n_harmful,
        stats.instances_kept,
    )
    return stats


def build_all(
    slate: List[Dict[str, str]],
    out_root: Path,
) -> Dict[str, BuildStats]:
    """Iterate a slate of {baseline_dir, ood_dir, strategy_name} dicts.

    Returns {strategy_name: BuildStats}.
    """
    results: Dict[str, BuildStats] = {}
    for entry in slate:
        baseline_dir = Path(entry["baseline_dir"])
        ood_dir = Path(entry["ood_dir"])
        strat = entry["strategy_name"]
        out_dir = out_root / f"heldout_{strat}"
        logger.info("=== %s ===", strat)
        stats = build_pairs_for_strategy(
            baseline_dir=baseline_dir,
            ood_dir=ood_dir,
            strategy_name=strat,
            out_dir=out_dir,
        )
        results[strat] = stats
    return results
