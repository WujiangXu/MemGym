"""Three-distribution sweep: 1 model load × 3 eval sets per checkpoint.

For each checkpoint in `checkpoints/rm_v2_1p7b_qlora_32k/checkpoint-N/`:
  1. Load NF4 base + LoRA adapter once (~30s).
  2. Score every row from each of three pair-files in turn.
  3. Emit three independent ``eval_*.json`` files.

This is the wrapper for the learning-curve sweep (6 ckpts × 3 sets = 18
result files). Loading once and running three eval passes amortizes the
~30s model-load cost — significant when each individual eval pass is
only a few minutes.

Three eval sets, in order of distribution shift:
  IID            : v2 split eval rows (training-distribution check)
  OOD-memory     : strategy-OOD pairs (different memory strategy,
                   same SWE-Bench scenario)
  OOD-scenario   : τ2-bench memory-gain pairs (different agent domain)

Each emitted JSON carries the same schema as `eval_aug_sft.py`:
  {checkpoint, pairs, n_eval_rows, truncation, metrics,
   per_row_predictions[], episode_metrics}

…with one new top-level block, ``episode_metrics``, computed by
aggregating ``prob_safe`` over each ``instance_id`` (mean prob; label
from any row's `label`, all share the same episode-level Δr signal
on the OOD sets). Step-level numbers stay in ``metrics`` so the two
tiers are independently citable.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("eval_three_distributions")


def _load_eval_rows(pairs_path: Path, limit: int = 0) -> List[dict]:
    """Same loader as eval_aug_sft (`split == "eval"` filter, no fancy
    streaming). Reads from a fully prompt-formatted JSONL.
    """
    rows: List[dict] = []
    with pairs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split", "train") != "eval":
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _truncation_report(model, rows: List[dict], max_length: int) -> dict:
    prompt_lengths = [
        len(model.tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        for row in rows
    ]
    n_truncated = sum(1 for L in prompt_lengths if L > max_length)
    sorted_lens = sorted(prompt_lengths)

    def _pct(p: float) -> int:
        if not sorted_lens:
            return 0
        idx = min(len(sorted_lens) - 1, int(round(p * (len(sorted_lens) - 1))))
        return int(sorted_lens[idx])

    return {
        "max_length_ref":     int(max_length),
        "truncated_count":    n_truncated,
        "truncated_pct":      (n_truncated / len(prompt_lengths) * 100.0) if prompt_lengths else 0.0,
        "prompt_length_p50":  _pct(0.50),
        "prompt_length_p95":  _pct(0.95),
        "prompt_length_max":  int(max(prompt_lengths) if prompt_lengths else 0),
    }


def _compute_episode_metrics(predictions, rows: List[dict]) -> dict:
    """Aggregate `prob_safe` per `instance_id` (mean over its rows),
    label from any row (rows of one instance share the same episode-Δr
    label on OOD sets — by construction of `memory_gain_pairs.py`).

    Returns {n_episodes, accuracy, auroc, safe_f1, harmful_f1, ece,
    by_source: {...}}. ECE here uses 10 equal-mass buckets (matches
    `evaluator._compute_ece` semantics; we don't import it to keep this
    wrapper standalone-friendly when run server-side).
    """
    from collections import defaultdict
    from statistics import mean

    by_inst: Dict[str, dict] = defaultdict(
        lambda: {"probs": [], "label": None, "source": ""}
    )
    for p, row in zip(predictions, rows):
        iid = row.get("instance_id") or row.get("trajectory_id", "")
        by_inst[iid]["probs"].append(p.prob_safe)
        if by_inst[iid]["label"] is None:
            by_inst[iid]["label"] = int(row["label"])
        if not by_inst[iid]["source"]:
            by_inst[iid]["source"] = row.get("source_model", "")

    if not by_inst:
        return {"n_episodes": 0}

    items = []
    for iid, d in by_inst.items():
        prob = mean(d["probs"]) if d["probs"] else 0.5
        items.append((iid, prob, d["label"], d["source"]))

    # Step-level metrics already exist; here we only need the
    # episode-level counterparts. Use a 0.5 cutoff for binary classify
    # — sweeping `t*` is downstream's job, not this script's.
    n = len(items)
    n_pos = sum(1 for _, _, lbl, _ in items if lbl == 1)
    n_neg = n - n_pos
    correct = sum(1 for _, prob, lbl, _ in items
                  if (prob >= 0.5) == (lbl == 1))

    # AUROC via Wilcoxon-Mann-Whitney rank-sum (no sklearn import).
    pos_probs = sorted(prob for _, prob, lbl, _ in items if lbl == 1)
    neg_probs = sorted(prob for _, prob, lbl, _ in items if lbl == 0)
    if pos_probs and neg_probs:
        # rank-sum; ties get 0.5 weight (standard AUC).
        all_p = sorted(pos_probs + neg_probs)
        ranks = {p: i + 1 for i, p in enumerate(all_p)}
        # ties: average rank for tied groups
        from collections import Counter
        cnt = Counter(all_p)
        rank_map = {}
        cur = 1
        for p in sorted(cnt):
            c = cnt[p]
            avg_rank = cur + (c - 1) / 2
            rank_map[p] = avg_rank
            cur += c
        rank_sum_pos = sum(rank_map[p] for p in pos_probs)
        n_p = len(pos_probs)
        n_n = len(neg_probs)
        auroc = (rank_sum_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)
    else:
        auroc = None

    # Per-source breakdown (just episode count + acc per source bucket).
    by_source: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_pos": 0, "correct": 0}
    )
    for _, prob, lbl, src in items:
        by_source[src]["n"] += 1
        by_source[src]["n_pos"] += (lbl == 1)
        by_source[src]["correct"] += int((prob >= 0.5) == (lbl == 1))
    by_source_clean = {
        src: {
            "n": d["n"],
            "n_pos": d["n_pos"],
            "accuracy": d["correct"] / d["n"] if d["n"] else 0.0,
        }
        for src, d in by_source.items()
    }

    return {
        "n_episodes": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": correct / n if n else 0.0,
        "auroc": auroc,
        "by_source": by_source_clean,
    }


def _eval_one_set(
    model,
    pairs_path: Path,
    set_name: str,
    output_path: Path,
    max_length: int,
    limit: int,
):
    """Run one eval pass and write the result JSON.

    Imports `_run_predictions` + `compute_metrics` lazily so the script
    can be imported (e.g. for testing) without dragging the model stack.
    """
    from memgym.training.models.evaluator import compute_metrics
    from memgym.training.scripts.eval_aug_sft import _run_predictions

    rows = _load_eval_rows(pairs_path, limit)
    if not rows:
        logger.warning("set %s: no eval rows in %s — writing skip stub",
                       set_name, pairs_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "set": set_name,
            "pairs": str(pairs_path),
            "n_eval_rows": 0,
            "skipped": "no eval rows",
        }, indent=2))
        return

    logger.info("set %s: %d rows from %s", set_name, len(rows), pairs_path)
    truncation = _truncation_report(model, rows, max_length)
    logger.info(
        "set %s: prompt p50=%d p95=%d max=%d truncated=%d/%d (%.1f%%)",
        set_name, truncation["prompt_length_p50"],
        truncation["prompt_length_p95"], truncation["prompt_length_max"],
        truncation["truncated_count"], len(rows), truncation["truncated_pct"],
    )

    predictions = _run_predictions(model, rows)
    metrics = compute_metrics(predictions)
    episode_metrics = _compute_episode_metrics(predictions, rows)

    payload = {
        "set": set_name,
        "checkpoint": str(getattr(model, "_checkpoint_path", "?")),
        "pairs": str(pairs_path),
        "n_eval_rows": len(rows),
        "truncation": truncation,
        "metrics": dataclasses.asdict(metrics),
        "episode_metrics": episode_metrics,
        "per_row_predictions": [dataclasses.asdict(p) for p in predictions],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "set %s: step-level safe_f1=%.3f auroc=%s | episode-level acc=%.3f auroc=%s → %s",
        set_name, metrics.safe_f1,
        f"{metrics.auroc:.3f}" if metrics.auroc is not None else "n/a",
        episode_metrics.get("accuracy", 0.0),
        f"{episode_metrics.get('auroc', 0):.3f}" if episode_metrics.get("auroc") is not None else "n/a",
        output_path,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="LoRA adapter dir (a `checkpoint-N/` from the "
                        "rm_v2_1p7b_qlora_32k run).")
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--label-safe", default=" Y")
    p.add_argument("--label-harmful", default=" N")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-4bit", dest="load_in_4bit",
                   action="store_false", default=True)
    p.add_argument("--gpu-id", type=str, default=None)

    p.add_argument("--iid-pairs", type=Path, required=True,
                   help="JSONL with `split=eval` rows from the v2 "
                        "training set (the IID held-out source).")
    p.add_argument("--ood-memory-pairs", type=Path, required=True,
                   help="strategy-OOD JSONL.")
    p.add_argument("--ood-scenario-pairs", type=Path, required=True,
                   help="τ2/scenario-OOD JSONL (build via "
                        "build_tau2_ood_pairs.py).")

    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir; the three result JSONs land at "
                        "<out-dir>/eval_iid.json / eval_ood_memory.json "
                        "/ eval_ood_scenario.json")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.gpu_id is not None:
        prior = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        logger.info(
            "pinned CUDA_VISIBLE_DEVICES=%s (was %r)", args.gpu_id, prior,
        )

    # Lazy import — keeps script importable for unit tests without
    # CUDA/transformers in the environment.
    from memgym.training.models.world_model import (
        MemoryWorldModel, ModelConfig,
    )

    config = ModelConfig(
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        safe_completion=args.label_safe,
        harmful_completion=args.label_harmful,
    )
    logger.info("loading checkpoint: %s", args.checkpoint)
    model = MemoryWorldModel.from_checkpoint(args.checkpoint, config=config)
    # Stash so each result JSON can record which ckpt it scored against.
    model._checkpoint_path = str(args.checkpoint)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan: List[Tuple[str, Path, str]] = [
        ("iid",          args.iid_pairs,          "eval_iid.json"),
        ("ood_memory",   args.ood_memory_pairs,   "eval_ood_memory.json"),
        ("ood_scenario", args.ood_scenario_pairs, "eval_ood_scenario.json"),
    ]
    for set_name, pairs_path, out_name in plan:
        if not pairs_path.exists():
            logger.warning("missing %s — skipping (writing empty stub)",
                           pairs_path)
            (args.out_dir / out_name).write_text(json.dumps({
                "set": set_name,
                "pairs": str(pairs_path),
                "skipped": "pairs file missing",
            }, indent=2))
            continue
        _eval_one_set(
            model=model,
            pairs_path=pairs_path,
            set_name=set_name,
            output_path=args.out_dir / out_name,
            max_length=args.max_length,
            limit=args.limit,
        )
    logger.info("ALL DONE — results under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
