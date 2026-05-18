"""Offline threshold sweep over per-row eval predictions.

Reads the `per_row_predictions` block saved by `eval_aug_sft.py` and
sweeps the SAFE/HARMFUL decision threshold `t` over a fine grid,
recomputing per-class P/R/F1 at each point by rebuilding
`PredictionResult` objects with `pred_label = 1 if prob_safe >= t else 0`
and calling the existing `evaluator.compute_metrics`.

The primary optimization is **constrained**: pick `t*` that maximizes
SAFE-F1 subject to `harmful_f1 >= --harmful-f1-floor`. This matches
the asymmetric cost structure of the downstream world-model gate,
where losing a true-HARMFUL signal (false SAFE) is much more expensive
than mis-flagging a SAFE compression as HARMFUL (we just keep the
memory an extra step).

As context, the unconstrained best-SAFE-F1 threshold is also reported.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import List

from memgym.training.models.evaluator import (
    PredictionResult,
    compute_metrics,
)

logger = logging.getLogger(__name__)


def _rebuild_predictions(rows: List[dict], threshold: float) -> List[PredictionResult]:
    out: List[PredictionResult] = []
    for r in rows:
        prob_safe = r.get("prob_safe")
        if prob_safe is None:
            raise ValueError(f"row missing prob_safe: {r}")
        pred_label = 1 if prob_safe >= threshold else 0
        confidence = prob_safe if pred_label == 1 else (1.0 - prob_safe)
        out.append(PredictionResult(
            instance_id=r.get("instance_id", ""),
            step=int(r.get("step", 0)),
            true_label=int(r["true_label"]),
            pred_label=pred_label,
            confidence=confidence,
            prob_safe=prob_safe,
            source=r.get("source", ""),
            perturbation=r.get("perturbation", ""),
            source_dir=r.get("source_dir", ""),
        ))
    return out


def _metrics_at(rows: List[dict], threshold: float) -> dict:
    preds = _rebuild_predictions(rows, threshold)
    m = compute_metrics(preds)
    return {
        "threshold": round(threshold, 4),
        "accuracy": m.accuracy,
        "safe_f1": m.safe_f1,
        "safe_precision": m.safe_precision,
        "safe_recall": m.safe_recall,
        "harmful_f1": m.harmful_f1,
        "harmful_precision": m.harmful_precision,
        "harmful_recall": m.harmful_recall,
        "confusion": m.confusion,
    }


def _sweep(rows: List[dict], start: float, stop: float, step: float) -> List[dict]:
    results = []
    t = start
    while t <= stop + 1e-9:
        results.append(_metrics_at(rows, t))
        t += step
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True,
                        help="eval_results*.json produced by eval_aug_sft (must contain "
                             "per_row_predictions)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write threshold_sweep.json "
                             "(default: <eval-json dir>/threshold_sweep.json)")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--harmful-f1-floor", type=float, default=0.90,
                        help="Constraint on HARMFUL F1 when picking the best SAFE-F1 threshold")
    parser.add_argument("--safe-precision-floor", type=float, default=0.0,
                        help="Optional additional constraint on SAFE precision")
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text())
    rows = payload.get("per_row_predictions")
    if not rows:
        raise ValueError(
            f"{args.eval_json} has no per_row_predictions; re-run eval_aug_sft with the "
            f"updated script that persists per-row prob_safe"
        )

    logger.info("sweeping threshold %.2f..%.2f step %.3f on %d rows",
                args.start, args.stop, args.step, len(rows))
    sweep = _sweep(rows, args.start, args.stop, args.step)

    feasible = [r for r in sweep
                if r["harmful_f1"] >= args.harmful_f1_floor
                and r["safe_precision"] >= args.safe_precision_floor]
    best_constrained = max(feasible, key=lambda r: r["safe_f1"]) if feasible else None
    best_unconstrained = max(sweep, key=lambda r: r["safe_f1"])

    # Header
    print(f"{'t':>6} {'acc':>7} {'SAFE-F1':>8} {'SAFE-P':>7} {'SAFE-R':>7} "
          f"{'H-F1':>6} {'H-P':>6} {'H-R':>6}")
    # Print a subset (every 5th point) for scanability, plus both best points
    for i, r in enumerate(sweep):
        if i % 5 != 0 and r not in (best_constrained, best_unconstrained):
            continue
        marker = ""
        if r is best_constrained:
            marker = "  * constrained best"
        elif r is best_unconstrained:
            marker = "  * unconstrained best"
        print(f"{r['threshold']:>6.2f} {r['accuracy']:>7.3f} "
              f"{r['safe_f1']:>8.3f} {r['safe_precision']:>7.3f} {r['safe_recall']:>7.3f} "
              f"{r['harmful_f1']:>6.3f} {r['harmful_precision']:>6.3f} {r['harmful_recall']:>6.3f}"
              f"{marker}")

    out_path = args.output or (args.eval_json.parent / "threshold_sweep.json")
    out_path.write_text(json.dumps({
        "eval_json": str(args.eval_json),
        "n_rows": len(rows),
        "harmful_f1_floor": args.harmful_f1_floor,
        "safe_precision_floor": args.safe_precision_floor,
        "sweep": sweep,
        "best_under_constraint": best_constrained,
        "best_unconstrained": best_unconstrained,
    }, indent=2))
    logger.info("wrote %s", out_path)

    # Gate signal for CI / orchestrator consumption.
    gate = {
        "constrained_best": best_constrained,
        "unconstrained_best": best_unconstrained,
    }
    print(json.dumps({"step": "threshold_sweep", **gate}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
