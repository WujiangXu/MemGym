"""WebArena long-context OOD pair builder.

Mirrors ``tau2_long_context_pairs.py``'s design but consumes the
WebArena trajectory schema discovered by ``probe_webarena_episode.py``.

Source layout (the dataset-augmentation phase-prime cohorts):
    <cohort_root>/<app>/<task_h*>/trajectory.json

Where ``trajectory.json`` has shape::

    {
        "num_episodes": 1,
        "episodes": [{
            "episode_id": ...,
            "metadata": {"example_id": "h10", "reward": float, ...},
            "steps": [
                {
                    "step_id": int,
                    "observation": {"accessibility_tree": str, "url": ...},
                    "action": str,                         # e.g. "click [123]"
                    "reward": float,
                    "memory_action": {
                        "was_compacted": bool,
                        "strategy": str,                   # "none" / "ms_struct_ms10" / ...
                        "tokens": int,
                        "compression_ratio": float,
                    },
                    ...
                },
                ...
            ]
        }],
        "metadata": {"example_id": "h10", "reward": float, "success": bool, ...}
    }

The pair-building logic — episode-reward labeling, per-step alignment by
index, prompt rendering with judgment turn — is identical to τ²'s flat
path. The only differences are the loader (which messages-list-shapes a
trajectory.json) and the `domain="webarena_<app>"` tag we set so per-source
sweeps surface the WebArena slice as its own bucket.

Usage::

    python -m memgym.training.scripts.build_webarena_long_context_pairs \\
        --baseline-cohort ${REPO_ROOT}/results/webarena/phase_d_prime/paypal_none \\
        --ood-cohorts paypal_struct_ms10 paypal_struct_ms20 paypal_summ_ms15 \\
        --webarena-root ${REPO_ROOT}/results/webarena/phase_d_prime \\
        --out data/world_model/training_output/reward_model_v2_scenario_ood/\\
              reward_model_pairs_webarena_longctx.jsonl
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from memgym.training.data.reward_model_pairs import (
    ACTION_JUDGMENT_TEMPLATE,
    DEFAULT_KEEP_FIRST,
    LABEL_HARMFUL,
    LABEL_SAFE,
    RESPONSE_PREFIX,
    SUMMARY_MARKER,
    _ROLE_HEADERS,
    _TrajCache,
    _reconstruct_view,
    _render_flat_prompt,
    _stringify_message,
)

logger = logging.getLogger("webarena_long_context_pairs")


# ---------------------------------------------------------------------------
# WebArena message shaping — convert episodes[0].steps[] into role-tagged msgs
# ---------------------------------------------------------------------------

def _stringify_observation(obs: Any) -> str:
    """Render one WebArena observation (accessibility tree + URL) as text.

    The accessibility tree is the dominant signal — typically 2-4K chars
    per step. URL/title give navigation context. We keep both because
    the v2 RM training distribution prizes context over surgical pruning;
    we trade verbosity for prompt-length parity.
    """
    if not isinstance(obs, dict):
        return str(obs) if obs is not None else ""
    parts: List[str] = []
    for k in ("url", "title"):
        v = obs.get(k)
        if v:
            parts.append(f"{k.capitalize()}: {v}")
    tree = obs.get("accessibility_tree")
    if isinstance(tree, str) and tree.strip():
        parts.append(f"Accessibility tree:\n{tree}")
    return "\n".join(parts)


def _stringify_action(action: Any, reasoning: Any = None) -> str:
    """Render one WebArena step's action.

    ``action`` is normally a flat string ("click [123]"). When the
    ``reasoning_action`` block is present, we surface its
    ``recorded_action``/``translated_action`` pair so that the rendered
    prompt mirrors what the agent actually emitted (some
    the dataset-augmentation phase-prime traces have non-trivial reasoning text we want to
    preserve in context).
    """
    text = action if isinstance(action, str) else json.dumps(action, default=str)
    if isinstance(reasoning, dict):
        rec = reasoning.get("recorded_action")
        trans = reasoning.get("translated_action")
        if isinstance(rec, str) and isinstance(trans, str) and rec != trans:
            text = f"recorded={rec}\ntranslated={trans}"
        elif isinstance(rec, str) and rec and rec != text:
            text = f"{text}  # recorded={rec}"
    return text


def _extract_summary_body_from_context(ctx: Any) -> Optional[str]:
    """Pull the [Summary] body out of a Step.context list.

    The WebArena runner persists ``filtered.content`` (the post-compaction
    head + summary_msg + tail) into ``step.context`` whenever
    ``was_compacted=True`` (webarena_runner.py:288, 408, 467). The summary
    body lives in a user-role message whose ``content`` starts with
    ``SUMMARY_MARKER``. We strip the marker prefix because downstream
    ``_render_flat_prompt`` re-prepends it.
    """
    if not isinstance(ctx, list):
        return None
    for m in ctx:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith(SUMMARY_MARKER):
            return content[len(SUMMARY_MARKER):].lstrip()
    return None


def _flatten_traj_messages(
    traj: dict,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[int, dict], Dict[int, str]]:
    """Flatten a WebArena trajectory.json into a messages list.

    Returns:
      messages: list of role-tagged dicts in transcript order.
      agent_msg_indices: agent_msg_indices[k] = index of the assistant
                        message in `messages` for step k.
      memory_action_by_step: step_id -> step.memory_action (for surfacing
                            `was_compacted` to the per-row metadata).
      summary_body_by_step: step_id -> [Summary] body (marker stripped) for
                            steps where ``was_compacted=True`` and the
                            runtime persisted the post-compaction context.
                            Empty for non-compaction steps.
    """
    eps = traj.get("episodes") or []
    if not eps:
        return [], [], {}, {}
    steps = eps[0].get("steps") or []

    msgs: List[Dict[str, Any]] = []
    agent_msg_indices: List[int] = []
    mem_by_step: Dict[int, dict] = {}
    summary_body_by_step: Dict[int, str] = {}

    # System / task framing — surface the user instruction first so the
    # RM has the goal in context. WebArena puts this on episodes[0].metadata
    # or on the top-level metadata.
    instruction = (
        eps[0].get("metadata", {}).get("instruction")
        or traj.get("metadata", {}).get("instruction")
        or ""
    )
    app_name = (
        eps[0].get("metadata", {}).get("app_name")
        or traj.get("metadata", {}).get("app_name")
        or ""
    )
    if instruction:
        framing = f"App: {app_name}\nGoal: {instruction}" if app_name else instruction
        msgs.append({"role": "user", "content": framing})

    for k, s in enumerate(steps):
        obs_text = _stringify_observation(s.get("observation"))
        if obs_text:
            msgs.append({"role": "user", "content": obs_text})
        action_text = _stringify_action(s.get("action"), s.get("reasoning_action"))
        agent_msg_indices.append(len(msgs))
        msgs.append({"role": "assistant", "content": action_text})
        mem_action = s.get("memory_action") or {}
        mem_by_step[k] = mem_action
        if mem_action.get("was_compacted"):
            body = _extract_summary_body_from_context(s.get("context"))
            if body:
                summary_body_by_step[k] = body

    return msgs, agent_msg_indices, mem_by_step, summary_body_by_step


# ---------------------------------------------------------------------------
# WebArena trajectory loader (single-format — flat trajectory.json only)
# ---------------------------------------------------------------------------

@dataclass
class _WebArenaLoadResult:
    cache: Optional[_TrajCache]
    n_messages: int
    n_steps: int
    n_compactions: int
    episode_reward: Optional[float]
    instruction: str
    app_name: str
    error: Optional[str] = None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.warning("read failed %s: %r", path, e)
        return None


def load_webarena_task(task_dir: Path) -> _WebArenaLoadResult:
    """Load a single ``<task>/trajectory.json``.

    Returns a result with ``cache=None`` on parse failure or empty
    trajectory — the caller treats that as an unrecoverable hit and
    increments the dropped-load counter.
    """
    traj_path = task_dir / "trajectory.json"
    if not traj_path.is_file():
        return _WebArenaLoadResult(None, 0, 0, 0, None, "", "", error="no_trajectory_json")
    traj = _read_json(traj_path)
    if traj is None:
        return _WebArenaLoadResult(None, 0, 0, 0, None, "", "", error="parse_failed")

    msgs, agent_idx, mem_by_step, summary_body_by_step = _flatten_traj_messages(traj)
    if not msgs or not agent_idx:
        return _WebArenaLoadResult(None, 0, 0, 0, None, "", "", error="empty_trajectory")

    eps = traj.get("episodes") or []
    ep_meta = (eps[0] if eps else {}).get("metadata", {}) or {}
    instruction = ep_meta.get("instruction") or traj.get("metadata", {}).get("instruction") or ""
    app_name = ep_meta.get("app_name") or traj.get("metadata", {}).get("app_name") or ""

    # Synthesize the per-step view metadata SWE/τ²'s _reconstruct_view needs.
    step_memory: Dict[int, Dict[str, Any]] = {}
    n_compactions = 0
    for k, msg_pos in enumerate(agent_idx):
        mem_action = mem_by_step.get(k, {}) or {}
        was_compacted = bool(mem_action.get("was_compacted", False))
        if was_compacted:
            n_compactions += 1
        step_memory[k] = {
            "original_msgs": msg_pos,
            "filtered_msgs": msg_pos,
            "summary": summary_body_by_step.get(k),
            "was_compacted": was_compacted,
            # WebArena-specific bookkeeping (surfaces in row metadata)
            "strategy": mem_action.get("strategy"),
            "tokens": mem_action.get("tokens"),
            "compression_ratio": mem_action.get("compression_ratio"),
        }

    # summary_events drives _latest_summary_at: the most recent active
    # summary at step ≤ N is what _reconstruct_view injects between head
    # and tail. Built from was_compacted=True steps with a recovered body.
    summary_events: List[Tuple[int, str]] = sorted(summary_body_by_step.items())

    cache = _TrajCache(
        training_path=traj_path,
        replay_path=traj_path,
        raw_messages=msgs,
        step_memory=step_memory,
        summary_events=summary_events,
    )
    # Pull episode reward from the canonical locations:
    # episodes[0].total_reward (per-episode), then top-level metadata.reward.
    ep_reward: Optional[float] = None
    if eps:
        try:
            ep_reward = float(eps[0].get("total_reward"))
        except (TypeError, ValueError):
            ep_reward = None
    if ep_reward is None:
        try:
            ep_reward = float(traj.get("metadata", {}).get("reward"))
        except (TypeError, ValueError):
            ep_reward = None

    return _WebArenaLoadResult(
        cache=cache,
        n_messages=len(msgs),
        n_steps=len(step_memory),
        n_compactions=n_compactions,
        episode_reward=ep_reward,
        instruction=instruction,
        app_name=app_name,
    )


# ---------------------------------------------------------------------------
# Pair builder — emits one row per aligned step (same shape as τ² builder)
# ---------------------------------------------------------------------------

@dataclass
class PairBuildResult:
    row: Optional[Dict[str, Any]]
    skip_reason: Optional[str] = None


def _action_at_step(cache: _TrajCache, step_k: int) -> str:
    """Return the assistant action string at step k (header stripped).

    Mirrors τ²'s ``_action_at_step`` so the judgment turn frames the
    action consistently across scenarios.
    """
    mem = cache.step_memory.get(step_k) or {}
    pos = mem.get("original_msgs")
    if not isinstance(pos, int) or pos < 0 or pos >= len(cache.raw_messages):
        return ""
    msg = cache.raw_messages[pos]
    rendered = _stringify_message(msg)
    header = _ROLE_HEADERS.get("assistant", "[assistant]")
    if rendered.startswith(f"{header}\n"):
        return rendered[len(header) + 1:]
    return rendered


def build_pair(
    *,
    baseline_cache: _TrajCache,
    ood_cache: _TrajCache,
    step_k: int,
    label: int,
    delta_r: float,
    baseline_reward: float,
    memory_reward: float,
    domain: str,
    task_id: str,
    cohort_name: str,
    ood_strategy: str,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> PairBuildResult:
    """Build one (recorded, predicted, label) pair at aligned step ``step_k``.

    The "view" rendered into the prompt is the **OOD trajectory's**
    pre-step view (matches what the OOD agent attended to). The recorded
    action is the **baseline trajectory's** action at the same step
    index — that's the counterfactual the v2 RM is asked to score.
    """
    if step_k not in ood_cache.step_memory:
        return PairBuildResult(None, "step_not_in_ood")
    if step_k not in baseline_cache.step_memory:
        return PairBuildResult(None, "step_not_in_baseline")

    view_msgs, stats = _reconstruct_view(ood_cache, step_k, keep_first)
    if not view_msgs:
        return PairBuildResult(None, "empty_view")

    recorded = _action_at_step(baseline_cache, step_k)
    predicted = _action_at_step(ood_cache, step_k)
    if not recorded.strip() and not predicted.strip():
        return PairBuildResult(None, "empty_actions")

    perturbation = f"scenario_swap__{ood_strategy}"
    judgment = ACTION_JUDGMENT_TEMPLATE.format(
        step=step_k,
        recorded=recorded,
        predicted=predicted,
        perturbation=perturbation,
    )
    full_messages = view_msgs + [{"role": "user", "content": judgment}]
    flat_prompt = _render_flat_prompt(full_messages)
    completion = LABEL_SAFE if label == 1 else LABEL_HARMFUL

    instance_id = f"webarena_{domain}_{task_id}"
    fork_event_id = f"{instance_id}_{step_k}_{perturbation}"
    ood_mem = ood_cache.step_memory.get(step_k, {}) or {}
    row = {
        "trajectory_id": instance_id,
        "instance_id": instance_id,
        "fork_event_id": fork_event_id,
        "source_dir": cohort_name,
        "source_model": ood_strategy,
        "step": step_k,
        "perturbation": perturbation,
        "label": int(label),
        "completion": completion,
        "target": completion,
        "messages": full_messages,
        "prompt": flat_prompt,
        "input": flat_prompt[: -len(RESPONSE_PREFIX)],
        "recorded_action": recorded,
        "predicted_action": predicted,
        "diverged": recorded != predicted,
        "n_compactions_active": stats["n_compactions_active"],
        "active_summary_chars": stats["active_summary_chars"],
        "n_messages_in_view": stats["n_messages_in_view"],
        "original_msgs": stats["original_msgs"],
        "filtered_msgs": stats["filtered_msgs"],
        # WebArena-specific bookkeeping
        "delta_r": float(delta_r),
        "baseline_reward": float(baseline_reward),
        "memory_reward": float(memory_reward),
        "domain": domain,
        "webarena_cohort": cohort_name,
        "ood_was_compacted_here": bool(ood_mem.get("was_compacted", False)),
        "ood_compression_ratio": ood_mem.get("compression_ratio"),
        "ood_step_strategy": ood_mem.get("strategy"),
        "split": "eval",
        "provenance": {
            "training_path": str(ood_cache.training_path),
            "replay_path": str(ood_cache.replay_path),
            "build_view_fn": "memgym.training.data.reward_model_pairs._reconstruct_view",
            "build_script": "src/memgym/training/data/webarena_long_context_pairs.py",
            "format": "flat",
            "keep_first": keep_first,
        },
    }
    return PairBuildResult(row)


# ---------------------------------------------------------------------------
# Cohort walker — iterate <cohort>/<app>/<task>/ pairs
# ---------------------------------------------------------------------------

def _iter_paired_tasks(
    baseline_cohort: Path, ood_cohort: Path,
) -> Iterator[Tuple[str, str, Path, Path]]:
    """Yield (app_name, task_id, baseline_task_dir, ood_task_dir) pairs.

    Both cohort dirs follow ``<cohort>/<app>/<task_h*>/`` layout. We join
    on the (app, task) tuple so cross-app contamination is impossible.
    """
    if not baseline_cohort.is_dir() or not ood_cohort.is_dir():
        return
    for app_dir in sorted(p for p in baseline_cohort.iterdir() if p.is_dir()):
        app = app_dir.name
        ood_app = ood_cohort / app
        if not ood_app.is_dir():
            continue
        base_tids = {p.name for p in app_dir.iterdir() if p.is_dir()}
        ood_tids = {p.name for p in ood_app.iterdir() if p.is_dir()}
        for tid in sorted(base_tids & ood_tids):
            yield app, tid, app_dir / tid, ood_app / tid


def _strategy_short_from_cohort(cohort_name: str) -> str:
    """Short, stable id for the OOD memory strategy.

    The the dataset-augmentation phase-prime cohort dirs are named like ``paypal_struct_ms10``
    or ``paypal_summ_ms15`` — we strip the leading app prefix so the
    bucket label is the strategy alone (``struct_ms10``).
    """
    parts = cohort_name.split("_", 1)
    return parts[1] if len(parts) == 2 else cohort_name


def build_pairs_for_cohort(
    *,
    baseline_cohort: Path,
    ood_cohort: Path,
    out_handle,
    keep_first: int = DEFAULT_KEEP_FIRST,
    gate_on_compacted: bool = True,
) -> Dict[str, int]:
    """Walk one (baseline, ood) cohort pair and emit pairs.

    Args:
        gate_on_compacted: When True (default), only emit a
            row for step k when the OOD trajectory has
            ``step_memory[k]['was_compacted'] == True``.  Set to False to
            recover the old every-step behaviour (used only for CI
            regression tests).
    """
    strategy = _strategy_short_from_cohort(ood_cohort.name)
    cohort_name = ood_cohort.name
    stats: Counter = Counter()

    for app, tid, base_path, ood_path in _iter_paired_tasks(baseline_cohort, ood_cohort):
        base_load = load_webarena_task(base_path)
        ood_load = load_webarena_task(ood_path)
        if base_load.cache is None or ood_load.cache is None:
            stats[f"dropped_load_{base_load.error or 'ok'}_{ood_load.error or 'ok'}"] += 1
            continue
        if base_load.episode_reward is None or ood_load.episode_reward is None:
            stats["dropped_missing_reward"] += 1
            continue

        delta_r = ood_load.episode_reward - base_load.episode_reward
        # Drop Δr=0 rows — episode-level no-signal censoring.
        if abs(delta_r) < 1e-9:
            stats["dropped_delta_zero"] += 1
            continue
        label = 1 if delta_r > 0 else 0

        n_aligned = min(len(base_load.cache.step_memory),
                        len(ood_load.cache.step_memory))
        if n_aligned == 0:
            stats["dropped_no_aligned_steps"] += 1
            continue
        stats["tasks_paired"] += 1

        domain = f"webarena_{app}"
        for k in range(n_aligned):
            # Gate emission on was_compacted so we only
            # produce rows where the memory strategy actually fired,
            # avoiding the 86-row diverged=False HARMFUL contamination
            # caused by emitting every step regardless of compaction.
            if gate_on_compacted:
                ood_mem = ood_load.cache.step_memory.get(k) or {}
                if not ood_mem.get("was_compacted", False):
                    stats["steps_skipped_not_compacted"] += 1
                    continue

            res = build_pair(
                baseline_cache=base_load.cache,
                ood_cache=ood_load.cache,
                step_k=k,
                label=label,
                delta_r=delta_r,
                baseline_reward=base_load.episode_reward,
                memory_reward=ood_load.episode_reward,
                domain=domain,
                task_id=tid,
                cohort_name=cohort_name,
                ood_strategy=strategy,
                keep_first=keep_first,
            )
            if res.row is None:
                stats[f"events_dropped_{res.skip_reason}"] += 1
                continue
            out_handle.write(json.dumps(res.row) + "\n")
            stats["pairs_written"] += 1
            if label == 1:
                stats["safe"] += 1
            else:
                stats["harmful"] += 1
    return dict(stats)
