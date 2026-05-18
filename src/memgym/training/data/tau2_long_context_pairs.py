"""τ2-bench long-context OOD pair builder.

Replaces the short-template ``build_tau2_ood_pairs.py`` output. Emits
pairs whose ``prompt`` matches the v2 RM training distribution (~22-38K
tokens of full conversation history + active compaction summary +
judgment turn) instead of the prior 600-token bare-actions-only string.

Two τ2 source-file formats are supported:

1. **Rich** (preferred):  ``<task>/0_training.json`` + ``<task>/0_replay.json``.
   ``_replay.json::messages[]`` carries the full role-tagged transcript,
   ``_replay.json::agent_steps[]`` maps ``step → msg_index`` for
   alignment, ``_training.json::steps[].memory`` carries
   ``was_compacted``/``summary``/``original_msgs``/``filtered_msgs`` per
   step. This is the format the in-process τ2 runner records and the
   only format that gives genuine 22-38K-token rendering.

2. **Flat** (fallback):  ``<task>/trajectory.json``. Used when the rich
   pair is missing — reconstructs a messages list from
   ``episodes[0].steps[]`` (each step has ``action`` + ``observation``)
   and applies a degenerate compaction view (no per-step summary
   available). Output prompts are richer than the old short template
   but typically shorter than format 1; the row's ``provenance.format``
   field records which path was used so downstream auditors can filter.

Why we don't just call the SWE builder: τ2 ``tool_calls`` have a flat
``{name, arguments, id, requestor}`` schema, while SWE/OpenAI uses the
nested ``{function: {name, arguments}}`` form. SWE's
``_stringify_message`` parses the nested form and falls back to a
generic ``"tool({...})"`` for τ2 — losing the function name. We
override the assistant-stringifier here while reusing every other piece
of ``reward_model_pairs`` machinery (the view-reconstruction,
flat-render, and judgment-turn template).

Output schema mirrors ``reward_model_pairs._build_row``'s row exactly
plus four τ2-specific keys (``domain``, ``tau2_run``, ``delta_r``,
``baseline_reward``/``memory_reward``), so the same ``eval_aug_sft.py``
+ per-source slicing pipeline that scores IID/strategy-OOD scores this
without code changes.

Usage::

    python -m memgym.training.scripts.build_tau2_long_context_pairs \\
        --tau2-root ${REPO_ROOT}/results/tau2_bench \\
        --runs HP_full_summ_ms10_kl2 HP_full_struct_ms10_kl4 \\
        --out ${REPO_ROOT}/data/world_model/training_output/\\
reward_model_v2_scenario_ood/reward_model_pairs_tau2_longctx.jsonl
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
    _ROLE_HEADERS,
    _TrajCache,
    _latest_summary_at,
    _reconstruct_view,
    _render_flat_prompt,
    _stringify_message,
)

logger = logging.getLogger("tau2_long_context_pairs")


# ---------------------------------------------------------------------------
# τ2 message stringifier (overrides the SWE/OpenAI-style assistant path)
# ---------------------------------------------------------------------------

def _stringify_tau2_message(msg: Dict[str, Any]) -> str:
    """Render one τ2 message as a printable role-tagged block.

    τ2's ``tool_calls`` use a flat ``{name, arguments, id, requestor}``
    schema (verified against
    ``tests/fixtures/trajectories/tau2_bench_run/...``); SWE's
    ``_stringify_message`` expects OpenAI's nested
    ``{function: {name, arguments}}`` form and silently falls back to
    ``tool(...)`` for τ2, losing the function name. We override here
    rather than monkey-patching so the SWE rendering path stays
    untouched for SWE callers.

    Other roles (``user``, ``tool``, ``system``) round-trip through
    SWE's renderer with no change in behavior, since they're plain-text
    and τ2 doesn't add structured fields there.
    """
    role = msg.get("role", "user")
    if role != "assistant":
        return _stringify_message(msg)

    header = _ROLE_HEADERS.get(role, f"[{role}]")
    parts: List[str] = []
    content = msg.get("content")
    if isinstance(content, str) and content:
        parts.append(content)

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name") or tc.get("function", {}).get("name") or "tool"
        args = tc.get("arguments")
        if args is None and isinstance(tc.get("function"), dict):
            args = tc["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        try:
            args_repr = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_repr = str(args)
        parts.append(f"{name}({args_repr})")

    body = "\n".join(p for p in parts if p)
    return f"{header}\n{body}"


def _render_tau2_flat_prompt(messages: List[Dict[str, Any]]) -> str:
    """Same shape as ``_render_flat_prompt`` but uses the τ2-aware
    stringifier for the assistant role."""
    blocks = [_stringify_tau2_message(m) for m in messages]
    body = "\n\n".join(b for b in blocks if b)
    return body + RESPONSE_PREFIX


# ---------------------------------------------------------------------------
# τ2 trajectory I/O — rich (training+replay) and flat (trajectory.json)
# ---------------------------------------------------------------------------

@dataclass
class _Tau2LoadResult:
    cache: Optional[_TrajCache]
    fmt: str             # "rich" | "flat" | "missing"
    n_messages: int
    n_steps: int
    n_compactions: int
    episode_reward: Optional[float]
    error: Optional[str] = None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.warning("read failed %s: %r", path, e)
        return None


def _load_rich(task_dir: Path) -> Optional[_Tau2LoadResult]:
    """Try the rich format: ``<task>/0_training.json`` + ``<task>/0_replay.json``.

    Returns None if either file is missing (caller falls back to the
    flat format). Returns a result with ``cache=None`` if the files
    exist but parse-fail or are empty — that's a hard error worth
    surfacing rather than silently degrading.
    """
    training_path = task_dir / "0_training.json"
    replay_path = task_dir / "0_replay.json"
    if not training_path.is_file() or not replay_path.is_file():
        return None

    training = _read_json(training_path)
    replay = _read_json(replay_path)
    if training is None or replay is None:
        return _Tau2LoadResult(None, "rich", 0, 0, 0, None, error="parse_failed")

    raw_messages = list(replay.get("messages") or [])
    step_memory: Dict[int, Dict[str, Any]] = {}
    summary_events: List[Tuple[int, str]] = []
    for s in training.get("steps") or []:
        try:
            step_i = int(s.get("step", -1))
        except (TypeError, ValueError):
            continue
        mem = s.get("memory") or {}
        step_memory[step_i] = mem
        summ = (mem.get("summary") or "").strip()
        if summ:
            summary_events.append((step_i, summ))
    summary_events.sort(key=lambda kv: kv[0])

    cache = _TrajCache(
        training_path=training_path,
        replay_path=replay_path,
        raw_messages=raw_messages,
        step_memory=step_memory,
        summary_events=summary_events,
    )
    ep_reward: Optional[float] = None
    try:
        ep_reward = float(training.get("episode_reward"))
    except (TypeError, ValueError):
        ep_reward = None

    return _Tau2LoadResult(
        cache=cache,
        fmt="rich",
        n_messages=len(raw_messages),
        n_steps=len(step_memory),
        n_compactions=len(summary_events),
        episode_reward=ep_reward,
    )


def _flatten_traj_messages(traj: dict) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Reconstruct a messages list from ``trajectory.json::episodes[0].steps[]``.

    Each step has ``action`` (assistant message dict) and optionally
    ``observation`` (tool/user response). We emit messages in step
    order and record the assistant message indices so the pair walker
    can align by step number.

    Returns (messages, agent_msg_indices) where ``agent_msg_indices[k]``
    is the position in ``messages`` of the assistant turn at step k.
    """
    eps = traj.get("episodes") or []
    if not eps:
        return [], []
    steps = eps[0].get("steps") or []
    msgs: List[Dict[str, Any]] = []
    agent_msg_indices: List[int] = []
    # Optional task / system prompt — some τ2 trajectories record this
    # on the episode itself rather than in steps[].
    initial = eps[0].get("initial_message") or eps[0].get("task")
    if isinstance(initial, dict):
        msgs.append({"role": "user", "content": initial.get("content") or json.dumps(initial)[:4000]})
    elif isinstance(initial, str) and initial:
        msgs.append({"role": "user", "content": initial})

    for s in steps:
        action = s.get("action")
        obs = s.get("observation")
        if isinstance(action, dict):
            agent_msg_indices.append(len(msgs))
            msgs.append(action)
        if isinstance(obs, dict):
            msgs.append(obs)
    return msgs, agent_msg_indices


def _load_flat(task_dir: Path) -> _Tau2LoadResult:
    """Fallback: reconstruct from ``<task>/trajectory.json`` only.

    No per-step ``memory.summary`` is available in this format, so we
    emit a degenerate cache (empty ``summary_events``) — the agent's
    "view" at step k is just messages[:agent_msg_index_k]. The output
    prompt shape is still long-context (full pre-step transcript +
    judgment turn) but the active-summary block will always be empty.
    """
    traj_path = task_dir / "trajectory.json"
    if not traj_path.is_file():
        return _Tau2LoadResult(None, "missing", 0, 0, 0, None, error="no_trajectory_json")
    traj = _read_json(traj_path)
    if traj is None:
        return _Tau2LoadResult(None, "missing", 0, 0, 0, None, error="parse_failed")

    msgs, agent_idx = _flatten_traj_messages(traj)
    step_memory: Dict[int, Dict[str, Any]] = {}
    for k, msg_pos in enumerate(agent_idx):
        # Synthesize the minimal mem dict SWE's _resolve_assistant_msg_count
        # needs: original_msgs = number of messages the agent saw before
        # producing its action at step k = msg_pos.
        step_memory[k] = {
            "original_msgs": msg_pos,
            "filtered_msgs": msg_pos,
            "summary": None,
            "was_compacted": False,
        }

    cache = _TrajCache(
        training_path=traj_path,
        replay_path=traj_path,
        raw_messages=msgs,
        step_memory=step_memory,
        summary_events=[],
    )
    # τ2 trajectory.json places the reward in the sibling result.json.
    result_path = task_dir / "result.json"
    ep_reward: Optional[float] = None
    if result_path.is_file():
        result = _read_json(result_path)
        if result is not None:
            try:
                ep_reward = float(result.get("reward"))
            except (TypeError, ValueError):
                ep_reward = None

    return _Tau2LoadResult(
        cache=cache,
        fmt="flat",
        n_messages=len(msgs),
        n_steps=len(step_memory),
        n_compactions=0,
        episode_reward=ep_reward,
    )


def load_tau2_task(task_dir: Path) -> _Tau2LoadResult:
    """Entry point: try rich format, fall back to flat."""
    rich = _load_rich(task_dir)
    if rich is not None and rich.cache is not None:
        return rich
    if rich is not None and rich.error:
        # Rich files were present but unparseable — surface the error
        # rather than masking it with a flat fallback.
        return rich
    return _load_flat(task_dir)


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

@dataclass
class PairBuildResult:
    row: Optional[Dict[str, Any]]
    skip_reason: Optional[str] = None


def _action_at_step(cache: _TrajCache, step_k: int) -> str:
    """Return the assistant action string at step k, serialized via the
    τ2-aware stringifier (drops the role header — the judgment template
    re-adds its own framing).
    """
    mem = cache.step_memory.get(step_k) or {}
    pos = mem.get("original_msgs")
    if not isinstance(pos, int) or pos < 0 or pos >= len(cache.raw_messages):
        return ""
    msg = cache.raw_messages[pos]
    rendered = _stringify_tau2_message(msg)
    # Strip the leading "[Assistant]\n" header so the judgment template's
    # framing is the only place where the action is labeled.
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
    tau2_run: str,
    ood_strategy: str,
    keep_first: int = DEFAULT_KEEP_FIRST,
    fmt: str = "rich",
) -> PairBuildResult:
    """Emit a single (recorded, predicted, label) pair at aligned step ``step_k``.

    The "view" the RM scores is the **OOD trajectory's** reconstructed
    pre-step view (so the prompt-length distribution matches what the
    OOD agent actually attended to). The recorded action comes from the
    baseline trajectory at the same step index — that's the
    counterfactual the RM is asked to compare against.
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
    flat_prompt = _render_tau2_flat_prompt(full_messages)
    completion = LABEL_SAFE if label == 1 else LABEL_HARMFUL

    instance_id = f"tau2_{domain}_{task_id}"
    fork_event_id = f"{instance_id}_{step_k}_{perturbation}"
    row = {
        "trajectory_id": instance_id,
        "instance_id": instance_id,
        "fork_event_id": fork_event_id,
        "source_dir": tau2_run,
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
        # τ2-specific bookkeeping
        "delta_r": float(delta_r),
        "baseline_reward": float(baseline_reward),
        "memory_reward": float(memory_reward),
        "domain": domain,
        "tau2_run": tau2_run,
        "split": "eval",
        "provenance": {
            "training_path": str(ood_cache.training_path),
            "replay_path": str(ood_cache.replay_path),
            "build_view_fn": "memgym.training.data.reward_model_pairs._reconstruct_view",
            "build_script": "src/memgym/training/data/tau2_long_context_pairs.py",
            "format": fmt,
            "keep_first": keep_first,
        },
    }
    return PairBuildResult(row)


# ---------------------------------------------------------------------------
# Run walker — iterate <run>/{baseline,memory}/<domain>/<task>/
# ---------------------------------------------------------------------------

def _iter_paired_tasks(run_dir: Path) -> Iterator[Tuple[str, str, Path, Path]]:
    base_root = run_dir / "baseline"
    mem_root = run_dir / "memory"
    if not base_root.is_dir() or not mem_root.is_dir():
        return
    for dom_dir in sorted(p for p in base_root.iterdir() if p.is_dir()):
        domain = dom_dir.name
        mem_dom = mem_root / domain
        if not mem_dom.is_dir():
            continue
        base_tids = {p.name for p in dom_dir.iterdir() if p.is_dir()}
        mem_tids = {p.name for p in mem_dom.iterdir() if p.is_dir()}
        for tid in sorted(base_tids & mem_tids):
            yield domain, tid, dom_dir / tid, mem_dom / tid


def _strategy_short_from_run(run_name: str) -> str:
    """Extract a stable short strategy id from a τ2 ``_both_`` run dirname.

    Matches the convention used by the legacy short-template builder so
    the new pairs use the same ``source_model`` bucket — keeping
    per-source sweep continuity if any downstream stats compare the
    two builders directly.
    """
    parts = run_name.split("_both_", 1)
    if len(parts) != 2:
        return run_name
    tail = parts[1]
    for cut in ("_HP_", "_T2_", "_T3_", "_phase", "_v"):
        i = tail.find(cut)
        if i > 0:
            tail = tail[:i]
            break
    return tail or run_name


def build_pairs_for_run(
    run_dir: Path,
    out_handle,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> Dict[str, int]:
    """Walk a single ``_both_`` τ2 run, emit pairs to ``out_handle``."""
    strategy = _strategy_short_from_run(run_dir.name)
    stats: Counter = Counter()

    for domain, tid, base_path, mem_path in _iter_paired_tasks(run_dir):
        base_load = load_tau2_task(base_path)
        mem_load = load_tau2_task(mem_path)
        if base_load.cache is None or mem_load.cache is None:
            stats[f"dropped_load_{base_load.fmt}_{mem_load.fmt}"] += 1
            continue
        if base_load.episode_reward is None or mem_load.episode_reward is None:
            stats["dropped_missing_reward"] += 1
            continue
        # Format mismatch within a pair is OK (one side rich, other flat) —
        # we just warn so it shows up in stats.
        if base_load.fmt != mem_load.fmt:
            stats[f"format_mismatch_{base_load.fmt}_{mem_load.fmt}"] += 1
        stats[f"format_{base_load.fmt}+{mem_load.fmt}"] += 1

        delta_r = mem_load.episode_reward - base_load.episode_reward
        label = 1 if delta_r >= 0 else 0
        n_aligned = min(len(base_load.cache.step_memory),
                        len(mem_load.cache.step_memory))
        if n_aligned == 0:
            stats["dropped_no_aligned_steps"] += 1
            continue
        stats["tasks_paired"] += 1

        for k in range(n_aligned):
            res = build_pair(
                baseline_cache=base_load.cache,
                ood_cache=mem_load.cache,
                step_k=k,
                label=label,
                delta_r=delta_r,
                baseline_reward=base_load.episode_reward,
                memory_reward=mem_load.episode_reward,
                domain=domain,
                task_id=tid,
                tau2_run=run_dir.name,
                ood_strategy=strategy,
                keep_first=keep_first,
                fmt=mem_load.fmt,
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
