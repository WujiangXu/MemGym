"""τ2-bench memory-gain OOD pair builder (runs server-side).

Walks `results/tau2_bench/<run>/{baseline,memory}/<domain>/<task_id>/`
for ``_both_`` runs (which execute baseline + memory on the **same**
instance set with the **same** seed), emits one pair per aligned
assistant turn between the two trajectories, labeled by the sign of
the per-episode reward delta.

Schema differences vs SWE-Bench (`memory_gain_pairs.py`):

- τ2 records `agent_compaction_count` at episode-stats level only —
  there is no per-step compaction marker (`step.memory_action` is None
  for all steps even on tasks with multiple compactions). We therefore
  cannot apply the "memory-trigger steps only" filter the SWE builder
  uses; instead we emit one pair per aligned assistant turn.
- τ2 actions are JSON tool-calls or natural-language replies, not bash
  commands. We serialize via a stable `json.dumps`.
- Per-task layout: `result.json` has `reward` ∈ {0,1}; `trajectory.json`
  has `episodes[0].steps[]` where each `step.action` is the assistant
  message dict.

Output schema matches `aug_pairs._iter_pairs` so the result file feeds
directly into `eval_aug_sft.py`. We set `split=eval` on every row so
the eval-loader's split filter passes.

Usage::

    python -m memgym.training.scripts.build_tau2_ood_pairs \
        --tau2-root ${REPO_ROOT}/results/tau2_bench \
        --runs HP_full_summ_ms10_kl2 HP_full_struct_ms10_kl4 \
        --out ${REPO_ROOT}/data/world_model/training_output/\
reward_model_v2_scenario_ood/reward_model_pairs_tau2.jsonl

The `--runs` filter matches against the suffix of each run dirname so
we don't have to type the timestamps. If omitted, all `_both_` runs are
included.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from memgym.training.data.aug_pairs import (
    DEFAULT_LABEL_HARMFUL,
    DEFAULT_LABEL_SAFE,
    RESPONSE_PREFIX,
    build_prompt_template,
)

logger = logging.getLogger("build_tau2_ood_pairs")


def _serialize_action(action: dict) -> str:
    """Convert a τ2 assistant action dict to a stable text representation.

    The v2 RM was trained on bash-shaped strings, so JSON tool-calls are
    structurally OOD regardless of what we do here. We keep the
    serialization deterministic so two runs of this script produce
    identical pair files (downstream auditors compare hashes).

    Format chosen: ``content + "\n[TOOL_CALLS] " + tool_calls_json`` —
    keeps the natural-language reasoning visible to the RM rather than
    burying it inside a JSON envelope.
    """
    if not isinstance(action, dict):
        return str(action) if action else ""
    parts: List[str] = []
    content = action.get("content")
    if content:
        parts.append(str(content))
    tcs = action.get("tool_calls")
    if tcs:
        try:
            parts.append("[TOOL_CALLS] " + json.dumps(tcs, sort_keys=True))
        except (TypeError, ValueError):
            parts.append("[TOOL_CALLS] " + repr(tcs))
    return "\n".join(parts)


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("skip %s: %r", path, e)
        return None


def _assistant_actions(traj: dict) -> List[dict]:
    """Return the list of assistant-role actions from a τ2 trajectory.

    τ2's `episodes[0].steps[]` interleaves agent and user turns. The
    assistant ones are the only places where memory affects what gets
    emitted, so we filter to those before alignment.
    """
    eps = traj.get("episodes") or []
    if not eps:
        return []
    steps = eps[0].get("steps") or []
    out: List[dict] = []
    for s in steps:
        a = s.get("action")
        if isinstance(a, dict) and a.get("role") == "assistant":
            out.append(a)
    return out


def _iter_paired_tasks(
    run_dir: Path,
) -> Iterator[Tuple[str, str, Path, Path]]:
    """Yield (domain, task_id, baseline_path, memory_path) tuples.

    Only tasks present in BOTH baseline/ and memory/ are yielded. Domains
    that exist on one side only are skipped silently.
    """
    base_root = run_dir / "baseline"
    mem_root = run_dir / "memory"
    if not base_root.is_dir() or not mem_root.is_dir():
        return
    for domain_dir in sorted(p for p in base_root.iterdir() if p.is_dir()):
        domain = domain_dir.name
        mem_dom = mem_root / domain
        if not mem_dom.is_dir():
            continue
        base_tids = {p.name for p in domain_dir.iterdir() if p.is_dir()}
        mem_tids = {p.name for p in mem_dom.iterdir() if p.is_dir()}
        for tid in sorted(base_tids & mem_tids):
            yield domain, tid, domain_dir / tid, mem_dom / tid


def _strategy_short_from_run(run_name: str) -> str:
    """Extract a stable short strategy id from a run dirname.

    Run names look like
    ``20260412-024137_all_both_tau2_summarizing_ms10kf1kl2r0.6_HP_full_summ_ms10_kl2``.
    The hyperparameter cluster (`tau2_summarizing_ms10kf1kl2r0.6`) is
    the most informative slice id without the timestamp noise. We
    extract the substring between `_both_` and the trailing `_HP_*`
    suffix so two runs with the same hyperparams (but different
    timestamps) get the same source_model bucket — important for the
    per-source variance audit downstream.
    """
    parts = run_name.split("_both_", 1)
    if len(parts) != 2:
        return run_name
    tail = parts[1]
    # strip experiment suffixes like _HP_full_*, _v2, etc.
    for cut in ("_HP_", "_T2_", "_T3_", "_phase", "_v"):
        i = tail.find(cut)
        if i > 0:
            tail = tail[:i]
            break
    return tail or run_name


def build_pairs_for_run(
    run_dir: Path,
    out_handle,
    label_safe: str = DEFAULT_LABEL_SAFE,
    label_harmful: str = DEFAULT_LABEL_HARMFUL,
) -> Dict[str, int]:
    """Append pairs from one τ2 ``_both_`` run to ``out_handle``.

    Emits **prompt-formatted rows** (matching `aug_pairs._to_sft_row`)
    so eval_aug_sft.py reads the JSONL directly without an aug_pairs
    pre-processing pass. Each row carries the rendered `prompt`, the
    `completion` target string ("Y"/"N"), and `split=eval` so the
    eval-loader's split filter passes.

    Returns a stats dict: {tasks_paired, dropped_*, pairs_written,
    safe, harmful}.
    """
    strategy = _strategy_short_from_run(run_dir.name)
    source_model = strategy  # use as bucket id for per-source slicing
    perturbation = f"scenario_swap__{strategy}"
    template = build_prompt_template(label_safe, label_harmful)
    stats = Counter()

    for domain, tid, base_path, mem_path in _iter_paired_tasks(run_dir):
        base_result = _read_json(base_path / "result.json")
        mem_result = _read_json(mem_path / "result.json")
        if base_result is None or mem_result is None:
            stats["dropped_missing_result"] += 1
            continue
        try:
            r_base = float(base_result.get("reward", 0))
            r_mem = float(mem_result.get("reward", 0))
        except (TypeError, ValueError):
            stats["dropped_bad_reward"] += 1
            continue

        # Binary label from sign of Δr (Δr=0 is SAFE; mirrors SWE
        # builder's `_label_from_delta`).
        delta_r = r_mem - r_base
        label = 1 if delta_r >= 0 else 0

        base_traj = _read_json(base_path / "trajectory.json")
        mem_traj = _read_json(mem_path / "trajectory.json")
        if base_traj is None or mem_traj is None:
            stats["dropped_missing_trajectory"] += 1
            continue

        base_assists = _assistant_actions(base_traj)
        mem_assists = _assistant_actions(mem_traj)
        n_aligned = min(len(base_assists), len(mem_assists))
        if n_aligned == 0:
            stats["dropped_no_assistant_turns"] += 1
            continue

        stats["tasks_paired"] += 1
        instance_id = f"tau2_{domain}_{tid}"
        target = label_safe if label == 1 else label_harmful
        for k in range(n_aligned):
            base_text = _serialize_action(base_assists[k])
            mem_text = _serialize_action(mem_assists[k])
            if not base_text.strip() or not mem_text.strip():
                stats["events_dropped_empty_action"] += 1
                continue
            base_text = base_text[:4000]
            mem_text = mem_text[:4000]
            rendered = template.format(
                step=k,
                perturbation=perturbation,
                recorded=base_text,
                predicted=mem_text,
            )
            fork_event_id = f"{instance_id}_{k}_{perturbation}"
            row = {
                # eval_aug_sft schema (matches aug_pairs._to_sft_row)
                "trajectory_id": instance_id,
                "fork_event_id": fork_event_id,
                "input": rendered,
                "prompt": rendered + RESPONSE_PREFIX,
                "completion": target,
                "target": target,
                "label": int(label),
                "source_model": source_model,
                "source_dir": run_dir.name,
                "perturbation": perturbation,
                "step": k,
                "split": "eval",
                # bookkeeping for episode-level aggregation downstream
                "instance_id": instance_id,
                "diverged": base_text != mem_text,
                "delta_r": float(delta_r),
                "baseline_reward": r_base,
                "memory_reward": r_mem,
                "domain": domain,
                "tau2_run": run_dir.name,
            }
            out_handle.write(json.dumps(row) + "\n")
            stats["pairs_written"] += 1
            if label == 1:
                stats["safe"] += 1
            else:
                stats["harmful"] += 1
    return dict(stats)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--tau2-root", type=Path, required=True,
                   help="root containing the τ2 _both_ runs (e.g. "
                        "${REPO_ROOT}/results/tau2_bench).")
    p.add_argument("--runs", nargs="+", default=None,
                   help="suffix substrings to filter run dirnames. "
                        "If omitted, all `_both_` runs are processed.")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL path (parent dir created if "
                        "missing).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    candidates = sorted(p for p in args.tau2_root.iterdir()
                        if p.is_dir() and "_both_" in p.name)
    if args.runs:
        wanted = [c for c in candidates
                  if any(r in c.name for r in args.runs)]
    else:
        wanted = candidates
    if not wanted:
        logger.error("no `_both_` runs matched filter %r in %s",
                     args.runs, args.tau2_root)
        return 1
    logger.info("processing %d runs", len(wanted))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grand: Counter = Counter()
    with args.out.open("w") as fh:
        for run_dir in wanted:
            logger.info("=== %s ===", run_dir.name)
            stats = build_pairs_for_run(run_dir, fh)
            for k, v in stats.items():
                grand[k] += v
            logger.info("  %s", dict(stats))

    logger.info("FINAL: %s", dict(grand))
    logger.info("wrote %d pairs to %s",
                grand.get("pairs_written", 0), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
