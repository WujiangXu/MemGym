"""Per-source threshold sweep — Phase A.0 of the world-model the training phase plan.

The the baseline-train phase cross-source audit found the gate's aggregate SAFE-F1 (0.844
at t=0.05) is haiku45-dominated: haiku45 = 0.863, gptoss120b = 0.857,
sonnet45 = 0.719. Before spending 40 min retraining with new labels,
check whether a per-teacher threshold closes the gap with only
arithmetic. If yes, we can deploy with per-source t* and skip Phase A/C.

Protocol:
  * Read the per-row eval dump (eval_results_v2.json).
  * Partition rows by `source` (sonnet45 / gptoss120b / haiku45 / unknown).
  * For each partition, reuse `threshold_sweep._sweep` to grid-search
    t* under the same HARMFUL-F1 >= floor constraint.
  * Report per-source t* + SAFE-F1 + HARMFUL-F1 + confusion. Compare
    against the shared-t*=0.05 numbers.

This is pure arithmetic — no GPU, no model load.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from memgym.training.scripts.threshold_sweep import _metrics_at, _sweep

logger = logging.getLogger(__name__)


def _best_under_constraint(sweep: List[dict], harmful_floor: float, safe_prec_floor: float) -> dict | None:
    feasible = [r for r in sweep
                if r["harmful_f1"] >= harmful_floor
                and r["safe_precision"] >= safe_prec_floor]
    return max(feasible, key=lambda r: r["safe_f1"]) if feasible else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.01)
    parser.add_argument("--stop", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--harmful-f1-floor", type=float, default=0.90)
    parser.add_argument("--safe-precision-floor", type=float, default=0.0)
    parser.add_argument("--shared-threshold", type=float, default=0.05,
                        help="Shared t* to compare per-source t* against")
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text())
    rows = payload.get("per_row_predictions")
    if not rows:
        raise ValueError(f"{args.eval_json} has no per_row_predictions")

    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_source[r.get("source") or "unknown"].append(r)

    logger.info("partitioned %d rows into %d sources: %s",
                len(rows), len(by_source),
                {s: len(rs) for s, rs in by_source.items()})

    per_source: Dict[str, dict] = {}
    for source, source_rows in sorted(by_source.items()):
        sweep = _sweep(source_rows, args.start, args.stop, args.step)
        best = _best_under_constraint(sweep, args.harmful_f1_floor, args.safe_precision_floor)
        at_shared = _metrics_at(source_rows, args.shared_threshold)
        per_source[source] = {
            "n": len(source_rows),
            "best_per_source": best,
            "at_shared_threshold": at_shared,
            "delta_safe_f1_vs_shared": (best["safe_f1"] - at_shared["safe_f1"]) if best else None,
        }

    aggregate_sweep = _sweep(rows, args.start, args.stop, args.step)
    aggregate_best = _best_under_constraint(aggregate_sweep, args.harmful_f1_floor, args.safe_precision_floor)
    aggregate_at_shared = _metrics_at(rows, args.shared_threshold)

    # Pretty-print for terminal scan
    print(f"\n{'source':<14} {'n':>5} {'t*':>6} {'S-F1*':>7} {'H-F1*':>7} "
          f"{'S-F1@0.05':>10} {'H-F1@0.05':>10} {'Δ S-F1':>8}")
    print("-" * 78)
    for source, r in per_source.items():
        b = r["best_per_source"]
        a = r["at_shared_threshold"]
        t_star = f"{b['threshold']:.3f}" if b else "—"
        sf = f"{b['safe_f1']:.3f}" if b else "—"
        hf = f"{b['harmful_f1']:.3f}" if b else "—"
        delta = f"+{r['delta_safe_f1_vs_shared']:.3f}" if r['delta_safe_f1_vs_shared'] is not None else "—"
        print(f"{source:<14} {r['n']:>5} {t_star:>6} {sf:>7} {hf:>7} "
              f"{a['safe_f1']:>10.3f} {a['harmful_f1']:>10.3f} {delta:>8}")
    b_agg = aggregate_best
    a_agg = aggregate_at_shared
    print(f"{'aggregate':<14} {len(rows):>5} "
          f"{b_agg['threshold']:>6.3f} {b_agg['safe_f1']:>7.3f} {b_agg['harmful_f1']:>7.3f} "
          f"{a_agg['safe_f1']:>10.3f} {a_agg['harmful_f1']:>10.3f} "
          f"{(b_agg['safe_f1'] - a_agg['safe_f1']):>+8.3f}")

    summary = {
        "eval_json": str(args.eval_json),
        "harmful_f1_floor": args.harmful_f1_floor,
        "safe_precision_floor": args.safe_precision_floor,
        "shared_threshold": args.shared_threshold,
        "per_source": per_source,
        "aggregate": {
            "n": len(rows),
            "best_per_source": aggregate_best,
            "at_shared_threshold": aggregate_at_shared,
        },
    }

    out_path = args.output or (args.eval_json.parent / "per_source_sweep.json")
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out_path)

    gap_source_best = max(
        (r["best_per_source"]["safe_f1"] for r in per_source.values() if r["best_per_source"]),
        default=0.0,
    )
    gap_source_worst = min(
        (r["best_per_source"]["safe_f1"] for r in per_source.values() if r["best_per_source"]),
        default=0.0,
    )
    print(json.dumps({
        "step": "per_source_threshold_sweep",
        "worst_source_safe_f1_at_source_t*": gap_source_worst,
        "best_source_safe_f1_at_source_t*": gap_source_best,
        "cross_source_spread_at_source_t*": gap_source_best - gap_source_worst,
    }), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
