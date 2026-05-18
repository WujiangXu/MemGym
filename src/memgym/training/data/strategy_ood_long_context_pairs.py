"""SWE-Gym strategy-OOD pair builder with full-context prompt rendering.

Replaces the stripped-prompt path used by ``memory_gain_pairs.py``. The
existing builder writes only ``(recorded_action, predicted_action)`` and
the downstream renderer produces a ~450-char bare-action prompt — that
distribution is nothing like the 32K-context prompts the v2 RM trained
on, which is why the strategy-OOD slice shows AUROC=0.952 / ECE=0.74:
it's measuring prompt-format transfer, not strategy transfer.

This builder:

  * Walks the trajectory dir recursively for ``<iid>_training.json`` +
    ``<iid>_replay.json`` pairs (handles ``<dir>/<run>/<files>`` layouts
    that exist locally — e.g. ``sonnet45_sliding_window_w75_kf3/
    run7c_n20/<iid>_*.json``).
  * Loads each side as ``_TrajCache`` via the same loader the IID
    builder uses (``reward_model_pairs._load_trajectory``).
  * Identifies compaction-trigger steps from the OOD trajectory by the
    ``step.memory.summary != None`` convention (matches
    ``compaction.extract_compaction_events`` — that's the **true**
    event marker; ``was_compacted`` is sticky-True afterwards).
  * For each trigger step k, reconstructs the OOD agent's view at step
    k via ``_reconstruct_view`` (head + ``[Summary]`` body + tail),
    appends an ``ACTION_JUDGMENT_TEMPLATE`` user turn comparing the
    baseline and OOD actions at the SAME step, and renders the result
    via ``_render_flat_prompt``.
  * Labels with sign(Δr) where Δr = OOD episode reward − baseline
    episode reward. Δr=0 instances are dropped (no signal). For
    SWE-Gym we read ``LoadedTrajectory.resolved`` (set from a re-eval
    report when supplied, else from the per-instance training JSON).

Output schema is byte-identical to ``tau2_long_context_pairs.py`` so
all downstream eval / sweep / audit scripts work unchanged.

Example invocation::

    python -m memgym.training.data.strategy_ood_long_context_pairs \\
        --baseline-dir trajectories/sonnet45_llm_summarizing_ms100_r075_kf3 \\
        --ood-dir      trajectories/sonnet45_sliding_window_w75_kf3 \\
        --strategy-name sonnet45_sliding_window \\
        --out          data/world_model/training_output/reward_model_v2_strategy_ood/reward_model_pairs_strategy_ood_v3.jsonl

Build all five sonnet45 cohorts at once via :func:`build_all`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from memgym.training.data.loader import (
    LoadedTrajectory,
    TrajectoryLoader,
    load_reeval_report,
)
from memgym.training.data.reward_model_pairs import (
    ACTION_JUDGMENT_TEMPLATE,
    DEFAULT_KEEP_FIRST,
    LABEL_HARMFUL,
    LABEL_SAFE,
    RESPONSE_PREFIX,
    SUMMARY_MARKER,
    _ROLE_HEADERS,
    _TrajCache,
    _load_trajectory,
    _reconstruct_view,
    _render_flat_prompt,
    _stringify_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path discovery — walk recursively so per-batch sub-run dirs are handled
# ---------------------------------------------------------------------------

def _iter_traj_paths(root: Path) -> Iterator[Tuple[str, Path, Path]]:
    """Yield ``(instance_id, training_path, replay_path)`` for each trajectory
    pair found anywhere under ``root``. Skips entries whose ``_replay.json``
    is missing (a few legacy dirs have only the training file)."""
    if not root.is_dir():
        return
    for training_path in sorted(root.rglob("*_training.json")):
        iid = training_path.name[: -len("_training.json")]
        replay_path = training_path.with_name(f"{iid}_replay.json")
        if not replay_path.is_file():
            forked_path = training_path.with_name(f"{iid}_forked.json")
            if forked_path.is_file():
                replay_path = forked_path
            else:
                continue
        yield iid, training_path, replay_path


@dataclass
class _StrategyLoadResult:
    """In-memory view of one trajectory side."""

    cache: Optional[_TrajCache]
    resolved: Optional[bool]
    error: Optional[str] = None


def _load_resolved_map_from_summary(summary_path: Path) -> Dict[str, bool]:
    """Read ``summary.json::instance_results`` and return ``{iid: bool}``.

    SWE-Gym runners write ``reward`` per instance (0.0 / 1.0). We treat
    ``reward >= 0.5`` as resolved=True. Returns {} if the file is absent or
    malformed (caller falls back to the next source).
    """
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    results = data.get("instance_results") or []
    out: Dict[str, bool] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        iid = r.get("instance_id")
        if not iid:
            continue
        try:
            val = float(r.get("reward", 0.0))
        except (TypeError, ValueError):
            continue
        out[iid] = bool(val >= 0.5)
    return out


def _load_resolved_map_from_reeval(reeval_path: Path) -> Dict[str, bool]:
    """Read a reeval JSON in either of two schemas:

    Legacy:  {"resolved": [...], "unresolved": [...]}
    Newer:   {"resolved_ids": [...], "unresolved_ids": [...]}

    Returns ``{iid: True}`` for resolved, ``{iid: False}`` for unresolved,
    {} if the file's malformed.
    """
    if not reeval_path.is_file():
        return {}
    try:
        data = json.loads(reeval_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    res = set(data.get("resolved") or data.get("resolved_ids") or [])
    unres = set(data.get("unresolved") or data.get("unresolved_ids") or [])
    out: Dict[str, bool] = {iid: True for iid in res}
    for iid in unres:
        out.setdefault(iid, False)
    return out


def _autodiscover_reeval(root: Path) -> Optional[Path]:
    """Find an in-tree reeval JSON, if any. Skips ``*_training``/``_replay``
    /``_forked``/``.traj``/``summary``/``preds`` files. Returns the
    most-recently-modified hit so the latest reeval wins."""
    candidates: List[Path] = []
    for p in root.rglob("*.json"):
        n = p.name
        if "reeval" not in n:
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_resolved_map(
    root: Path, *, explicit_reeval: Optional[Path] = None,
) -> Tuple[Dict[str, bool], str]:
    """Resolve a ``{iid: resolved_bool}`` map by trying (in order):

      1. ``explicit_reeval`` argument (highest authority).
      2. Auto-discovered ``*reeval*.json`` anywhere under ``root``.
      3. ``summary.json::instance_results`` (the runner's own grading).

    Returns the map plus a string label describing the source it
    actually used — useful for the per-cohort stats counter.
    """
    if explicit_reeval is not None:
        m = _load_resolved_map_from_reeval(explicit_reeval)
        if m:
            return m, f"reeval:{explicit_reeval.name}"

    auto = _autodiscover_reeval(root)
    if auto is not None:
        m = _load_resolved_map_from_reeval(auto)
        if m:
            return m, f"reeval-auto:{auto.name}"

    # Fall back to summary.json wherever it lives in the dir tree.
    merged: Dict[str, bool] = {}
    for sp in root.rglob("summary.json"):
        merged.update(_load_resolved_map_from_summary(sp))
    if merged:
        return merged, "summary.json"

    return {}, "none"


def _load_strategy_traj(
    training_path: Path,
    replay_path: Path,
    *,
    resolved_map: Dict[str, bool],
    iid: str,
) -> _StrategyLoadResult:
    """Load one SWE-Gym trajectory pair into a ``_TrajCache``. The resolved
    label comes from the pre-built ``resolved_map`` — whose authority order
    is set by ``_build_resolved_map`` (reeval > summary.json)."""
    try:
        cache = _load_trajectory(training_path, replay_path)
    except Exception as exc:
        logger.warning("load failed %s: %s", training_path, exc)
        return _StrategyLoadResult(None, None, error=f"load_failed:{type(exc).__name__}")
    return _StrategyLoadResult(cache=cache, resolved=resolved_map.get(iid))


# ---------------------------------------------------------------------------
# Compaction-trigger discovery — gate on summary != None (true-event marker)
# ---------------------------------------------------------------------------

def _trigger_steps(cache: _TrajCache) -> List[int]:
    """Return sorted step indices where a compaction *fires* (summary
    populated). Mirrors ``extract_compaction_events`` semantics — sticky
    ``was_compacted=True`` post-event steps are NOT included.
    """
    out: List[int] = []
    for step_i in sorted(cache.step_memory.keys()):
        mem = cache.step_memory[step_i] or {}
        summ = (mem.get("summary") or "").strip()
        if summ:
            out.append(step_i)
    return out


# ---------------------------------------------------------------------------
# Action extraction at a given step (header-stripped)
# ---------------------------------------------------------------------------

def _action_at_step(cache: _TrajCache, step_k: int) -> str:
    """Return the assistant action string at step ``step_k`` (role header
    stripped — the judgment template re-adds its own framing)."""
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


# ---------------------------------------------------------------------------
# Pair builder — emits one row per OOD compaction-trigger step
# ---------------------------------------------------------------------------

@dataclass
class PairBuildResult:
    row: Optional[Dict[str, Any]]
    skip_reason: Optional[str] = None


def build_pair(
    *,
    baseline_cache: _TrajCache,
    ood_cache: _TrajCache,
    step_k: int,
    label: int,
    delta_r: int,
    baseline_resolved: bool,
    ood_resolved: bool,
    instance_id: str,
    cohort_name: str,
    ood_strategy: str,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> PairBuildResult:
    """Emit one (recorded, predicted, label) pair at OOD compaction-trigger
    step ``step_k``.

    The view rendered into the prompt is the **OOD trajectory's**
    pre-step view at ``step_k`` (so the agent's actual context window
    + the active summary body are both in-prompt). The recorded action
    is the **baseline trajectory's** action at the same step index —
    that's the counterfactual the v2 RM is asked to score.
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

    perturbation = f"strategy_swap__{ood_strategy}"
    judgment = ACTION_JUDGMENT_TEMPLATE.format(
        step=step_k,
        recorded=recorded,
        predicted=predicted,
        perturbation=perturbation,
    )
    full_messages = view_msgs + [{"role": "user", "content": judgment}]
    flat_prompt = _render_flat_prompt(full_messages)
    completion = LABEL_SAFE if label == 1 else LABEL_HARMFUL

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
        # Strategy-OOD bookkeeping
        "delta_r": int(delta_r),
        "baseline_reward": int(baseline_resolved),
        "ood_reward": int(ood_resolved),
        "ood_strategy": ood_strategy,
        "ood_compression_ratio": ood_mem.get("compression_ratio"),
        "ood_step_strategy": ood_mem.get("strategy"),
        "split": "eval",
        "provenance": {
            "training_path": str(ood_cache.training_path),
            "replay_path": str(ood_cache.replay_path),
            "build_view_fn": "memgym.training.data.reward_model_pairs._reconstruct_view",
            "build_script": "src/memgym/training/data/strategy_ood_long_context_pairs.py",
            "format": "flat",
            "keep_first": keep_first,
        },
    }
    return PairBuildResult(row)


# ---------------------------------------------------------------------------
# Cohort builder — pair (baseline_dir, ood_dir) by shared instance_id
# ---------------------------------------------------------------------------

def build_pairs_for_strategy(
    baseline_dir: Path,
    ood_dir: Path,
    strategy_name: str,
    out_path: Path,
    *,
    baseline_reeval: Optional[Path] = None,
    ood_reeval: Optional[Path] = None,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> Dict[str, int]:
    """Walk one (baseline_dir, ood_dir) pair, emit full-context strategy-OOD
    pairs to ``out_path``. Returns a stats counter."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_resolved_map, baseline_src = _build_resolved_map(
        baseline_dir, explicit_reeval=baseline_reeval,
    )
    ood_resolved_map, ood_src = _build_resolved_map(
        ood_dir, explicit_reeval=ood_reeval,
    )
    logger.info(
        "[%s] baseline labels from %s (n=%d) | ood labels from %s (n=%d)",
        strategy_name, baseline_src, len(baseline_resolved_map),
        ood_src, len(ood_resolved_map),
    )

    baseline_paths: Dict[str, Tuple[Path, Path]] = {}
    for iid, tp, rp in _iter_traj_paths(baseline_dir):
        baseline_paths.setdefault(iid, (tp, rp))
    ood_paths: Dict[str, Tuple[Path, Path]] = {}
    for iid, tp, rp in _iter_traj_paths(ood_dir):
        ood_paths.setdefault(iid, (tp, rp))

    shared = sorted(set(baseline_paths) & set(ood_paths))
    logger.info(
        "[%s] shared instances: %d (baseline=%d ood=%d)",
        strategy_name, len(shared), len(baseline_paths), len(ood_paths),
    )

    stats: Counter = Counter()
    stats["instance_overlap"] = len(shared)
    stats["baseline_total"] = len(baseline_paths)
    stats["ood_total"] = len(ood_paths)

    cohort_name = f"heldout_{strategy_name}_v3"
    with out_path.open("w") as fh:
        for iid in shared:
            base_tp, base_rp = baseline_paths[iid]
            ood_tp, ood_rp = ood_paths[iid]

            base_load = _load_strategy_traj(base_tp, base_rp,
                                            resolved_map=baseline_resolved_map, iid=iid)
            ood_load = _load_strategy_traj(ood_tp, ood_rp,
                                           resolved_map=ood_resolved_map, iid=iid)
            if base_load.cache is None or ood_load.cache is None:
                stats["dropped_load"] += 1
                continue
            if base_load.resolved is None or ood_load.resolved is None:
                stats["dropped_missing_reward"] += 1
                continue

            r_base = int(base_load.resolved)
            r_ood = int(ood_load.resolved)
            delta_r = r_ood - r_base
            if delta_r == 0:
                stats["dropped_zero_delta"] += 1
                continue

            triggers = _trigger_steps(ood_load.cache)
            if not triggers:
                stats["dropped_no_triggers"] += 1
                continue

            label = 1 if delta_r > 0 else 0
            stats["instances_kept"] += 1

            for step_k in triggers:
                res = build_pair(
                    baseline_cache=base_load.cache,
                    ood_cache=ood_load.cache,
                    step_k=step_k,
                    label=label,
                    delta_r=delta_r,
                    baseline_resolved=bool(base_load.resolved),
                    ood_resolved=bool(ood_load.resolved),
                    instance_id=iid,
                    cohort_name=cohort_name,
                    ood_strategy=strategy_name,
                    keep_first=keep_first,
                )
                if res.row is None:
                    stats[f"events_dropped_{res.skip_reason}"] += 1
                    continue
                fh.write(json.dumps(res.row) + "\n")
                stats["pairs_written"] += 1
                if label == 1:
                    stats["safe"] += 1
                else:
                    stats["harmful"] += 1

    logger.info(
        "[%s] wrote %d pairs (safe=%d harmful=%d) → %s",
        strategy_name, stats["pairs_written"], stats["safe"], stats["harmful"],
        out_path,
    )
    return dict(stats)


def build_all(
    slate: List[Dict[str, str]],
    out_root: Path,
    *,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> Dict[str, Dict[str, int]]:
    """Build the full slate — one JSONL per cohort, plus a merged JSONL
    suitable for ``eval_aug_sft``."""
    out_root.mkdir(parents=True, exist_ok=True)
    per_strategy: Dict[str, Dict[str, int]] = {}
    merged_path = out_root / "reward_model_pairs_strategy_ood_v3.jsonl"

    with merged_path.open("w") as merged_fh:
        for entry in slate:
            strat = entry["strategy_name"]
            per_path = out_root / f"heldout_{strat}_v3.jsonl"
            stats = build_pairs_for_strategy(
                baseline_dir=Path(entry["baseline_dir"]),
                ood_dir=Path(entry["ood_dir"]),
                strategy_name=strat,
                out_path=per_path,
                baseline_reeval=(Path(entry["baseline_reeval"])
                                 if entry.get("baseline_reeval") else None),
                ood_reeval=(Path(entry["ood_reeval"])
                            if entry.get("ood_reeval") else None),
                keep_first=keep_first,
            )
            per_strategy[strat] = stats
            # Concatenate per-strategy file into the merged set.
            if per_path.is_file():
                with per_path.open() as src:
                    for line in src:
                        merged_fh.write(line)

    logger.info("merged → %s", merged_path)
    return per_strategy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path,
                   help="Trajectory dir for the in-distribution baseline strategy "
                        "(usually sonnet45_llm_summarizing_ms100_r075_kf3).")
    p.add_argument("--ood-dir", type=Path,
                   help="Trajectory dir for the held-out memory strategy.")
    p.add_argument("--strategy-name", type=str,
                   help="Short id for the OOD strategy (e.g. sonnet45_sliding_window).")
    p.add_argument("--out", type=Path,
                   help="Output JSONL (single-cohort mode).")
    p.add_argument("--baseline-reeval", type=Path, default=None,
                   help="Optional path to a re-eval report to override the baseline's "
                        "resolved labels with re-graded values.")
    p.add_argument("--ood-reeval", type=Path, default=None,
                   help="Optional path to a re-eval report for the OOD strategy.")
    p.add_argument("--keep-first", type=int, default=DEFAULT_KEEP_FIRST,
                   help="Number of head messages preserved in the reconstructed view "
                        "(should match the runtime's compaction config — default mirrors "
                        "the sonnet45 v2 training distribution).")
    p.add_argument("--slate", type=Path, default=None,
                   help="Path to a JSON file listing {baseline_dir, ood_dir, "
                        "strategy_name, [baseline_reeval, ood_reeval]} entries. When "
                        "supplied, runs all entries and concatenates into a merged "
                        "JSONL under --out-root.")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Output root dir (slate mode).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if args.slate is not None:
        if args.out_root is None:
            print("--out-root is required when --slate is set", file=sys.stderr)
            return 2
        slate = json.loads(args.slate.read_text())
        per_strat = build_all(slate, args.out_root, keep_first=args.keep_first)
        print(json.dumps({"per_strategy_stats": per_strat}, indent=2))
        return 0

    required = [args.baseline_dir, args.ood_dir, args.strategy_name, args.out]
    if any(v is None for v in required):
        print("--baseline-dir / --ood-dir / --strategy-name / --out are required "
              "in single-cohort mode (or pass --slate)", file=sys.stderr)
        return 2

    stats = build_pairs_for_strategy(
        baseline_dir=args.baseline_dir,
        ood_dir=args.ood_dir,
        strategy_name=args.strategy_name,
        out_path=args.out,
        baseline_reeval=args.baseline_reeval,
        ood_reeval=args.ood_reeval,
        keep_first=args.keep_first,
    )
    print(json.dumps({"stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
