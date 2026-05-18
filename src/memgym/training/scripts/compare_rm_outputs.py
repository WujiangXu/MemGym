"""Cross-checkpoint agreement between two `eval_results.json` runs.

Plan v2 Step 5: quantify how interchangeable the 8B and 1.7B classifiers
are on identical inputs. Pure reduction over `per_row_predictions` from
two `eval_aug_sft.py` outputs — no GPU, no model load.

Reports:

- **Pearson r** between the two `prob_safe` series. Captures continuous
  agreement; insensitive to threshold choice. r > 0.95 means the two
  models put almost-the-same row in almost-the-same place on the safe-
  ness scale.
- **Label-agreement rate** at each checkpoint's own re-swept t\*. The
  honest deployment question is "do they ship the same gate decision",
  which depends on each checkpoint's calibrated threshold, not a shared
  one. Threshold from `threshold_sweep.py`'s `best_under_constraint` if
  available, falls back to 0.5.
- **McNemar's test** for paired discordance — the only valid significance
  test for "do these two classifiers disagree more than chance" on the
  same data. The mid-p variant is reported alongside the standard form;
  the standard chi-square approximation breaks down when the off-diagonal
  count is small (a real concern at AUROC ≈ 0.99 — most rows agree).
- **Scatter plot** prob_safe_8b vs prob_safe_1p7b colored by ground-truth
  label. The eyeball check that complements the scalar agreement number:
  if the two probs are correlated but biased (one model systematically
  hotter), r and label-agreement can both look OK while the calibration
  shape diverges. The diagonal exposes that.

Pass criteria from plan v2: Pearson r > 0.90 AND label-agreement > 0.95.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _load_predictions(path: Path) -> List[dict]:
    with path.open() as f:
        data = json.load(f)
    return data.get("per_row_predictions") or []


def _row_key(r: dict) -> Tuple[str, int, str]:
    """Stable join key — `(instance_id, step, perturbation)`.

    Rows in `eval_aug_sft.py` output are pair-derived; the per-source
    augmentation can produce multiple rows per `(instance_id, step)`
    distinguished only by `perturbation`. Joining on the triple is what
    makes "same input, different model" a well-defined comparison.
    """
    return (
        str(r.get("instance_id", "")),
        int(r.get("step", -1)),
        str(r.get("perturbation", "")),
    )


def _join_rows(
    rows_a: List[dict], rows_b: List[dict]
) -> List[Tuple[dict, dict]]:
    idx_a = {_row_key(r): r for r in rows_a if r.get("prob_safe") is not None}
    paired: List[Tuple[dict, dict]] = []
    for r in rows_b:
        if r.get("prob_safe") is None:
            continue
        k = _row_key(r)
        if k in idx_a:
            paired.append((idx_a[k], r))
    return paired


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _mcnemar(b: int, c: int) -> Dict[str, float]:
    """McNemar's test on the discordant counts.

    `b` = rows where A predicts SAFE and B predicts HARMFUL.
    `c` = rows where A predicts HARMFUL and B predicts SAFE.

    Returns the standard chi-square p-value (with continuity correction)
    and an exact binomial mid-p-value — the latter is robust when
    `b + c` is small, which happens frequently when two strong RMs
    disagree only on the hardest few rows.
    """
    n = b + c
    if n == 0:
        return {"discordant": 0, "chi2": 0.0, "p_chi2": 1.0, "p_exact_mid": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival of chi-square(df=1) at `chi2`.
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))
    # Exact binomial two-sided mid-p — sum tail mass + half the point mass.
    k = min(b, c)
    cum = 0.0
    point_at_k = 0.0
    for i in range(0, k + 1):
        # P(X = i) under Binomial(n, 0.5)
        p_i = math.comb(n, i) * (0.5 ** n)
        if i < k:
            cum += p_i
        else:
            point_at_k = p_i
    p_exact_two_sided = 2 * cum + point_at_k  # mid-p variant
    p_exact_mid = min(1.0, p_exact_two_sided)
    return {
        "discordant": int(n),
        "b_a_safe_b_harmful": int(b),
        "c_a_harmful_b_safe": int(c),
        "chi2": float(chi2),
        "p_chi2": float(p_chi2),
        "p_exact_mid": float(p_exact_mid),
    }


def _threshold_for(eval_path: Path, fallback: float = 0.5) -> float:
    """Best-effort: read the corresponding threshold-sweep JSON.

    Convention: `eval_results_8b.json` ↔ `threshold_sweep_8b.json` in
    the same directory. Falls back to 0.5 (argmax) if missing — the
    appendix should cite the threshold-sweep file when present, so a
    fallback is annotated in the output JSON.
    """
    sibling = eval_path.parent / eval_path.name.replace(
        "eval_results", "threshold_sweep"
    )
    if not sibling.exists():
        return fallback
    try:
        data = json.loads(sibling.read_text())
        best = data.get("best_under_constraint") or data.get("best") or {}
        t = best.get("threshold")
        if t is None:
            return fallback
        return float(t)
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return fallback


def _scatter(
    pa: np.ndarray,
    pb: np.ndarray,
    labels: np.ndarray,
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    """Two-color scatter colored by ground truth.

    Down-samples to 5000 rows max so the figure stays legible at PDF
    print size. Both classes are sampled proportionally so the visual
    SAFE/HARMFUL ratio matches the dataset's ratio.
    """
    rng = np.random.default_rng(0)
    n = len(pa)
    if n > 5000:
        idx = rng.choice(n, size=5000, replace=False)
        pa, pb, labels = pa[idx], pb[idx], labels[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    safe_mask = labels == 1
    ax.scatter(
        pa[~safe_mask], pb[~safe_mask],
        s=4, alpha=0.4, color="#d62728", label="HARMFUL",
    )
    ax.scatter(
        pa[safe_mask], pb[safe_mask],
        s=4, alpha=0.6, color="#2ca02c", label="SAFE",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"prob_safe — {label_a}")
    ax.set_ylabel(f"prob_safe — {label_b}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Cross-checkpoint prob_safe (n={len(pa)})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def compare(
    eval_a: Path,
    eval_b: Path,
    label_a: str,
    label_b: str,
    threshold_a: Optional[float],
    threshold_b: Optional[float],
    out_json: Path,
    out_scatter: Optional[Path],
) -> Dict:
    rows_a = _load_predictions(eval_a)
    rows_b = _load_predictions(eval_b)
    paired = _join_rows(rows_a, rows_b)
    if not paired:
        raise SystemExit(
            f"no joinable rows between {eval_a} and {eval_b} — check that "
            "both eval runs cover the same split"
        )

    pa = np.array([a["prob_safe"] for a, _ in paired], dtype=float)
    pb = np.array([b["prob_safe"] for _, b in paired], dtype=float)
    labels = np.array([int(a["true_label"]) for a, _ in paired], dtype=int)

    t_a = threshold_a if threshold_a is not None else _threshold_for(eval_a)
    t_b = threshold_b if threshold_b is not None else _threshold_for(eval_b)
    t_a_source = "cli" if threshold_a is not None else (
        "sweep" if (eval_a.parent / eval_a.name.replace(
            "eval_results", "threshold_sweep")).exists() else "fallback_0.5"
    )
    t_b_source = "cli" if threshold_b is not None else (
        "sweep" if (eval_b.parent / eval_b.name.replace(
            "eval_results", "threshold_sweep")).exists() else "fallback_0.5"
    )

    pred_a = (pa >= t_a).astype(int)
    pred_b = (pb >= t_b).astype(int)
    agree = (pred_a == pred_b).mean()

    # Discordance broken down for McNemar.
    b_count = int(((pred_a == 1) & (pred_b == 0)).sum())
    c_count = int(((pred_a == 0) & (pred_b == 1)).sum())
    mc = _mcnemar(b_count, c_count)

    # Ground-truth-aware splits — where do disagreements actually land?
    disagree_mask = pred_a != pred_b
    n_dis = int(disagree_mask.sum())
    dis_safe = int((disagree_mask & (labels == 1)).sum())
    dis_harm = int((disagree_mask & (labels == 0)).sum())

    summary = {
        "eval_a": str(eval_a),
        "eval_b": str(eval_b),
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(paired),
        "threshold_a": float(t_a),
        "threshold_b": float(t_b),
        "threshold_a_source": t_a_source,
        "threshold_b_source": t_b_source,
        "pearson_r": _pearson(pa, pb),
        "spearman_rho": _pearson(
            np.argsort(np.argsort(pa)).astype(float),
            np.argsort(np.argsort(pb)).astype(float),
        ),
        "label_agreement_rate": float(agree),
        "discordance": {
            "n_disagree": n_dis,
            "disagree_on_safe": dis_safe,
            "disagree_on_harmful": dis_harm,
        },
        "mcnemar": mc,
        "pass_criteria": {
            "pearson_r_gt_0p90": _pearson(pa, pb) > 0.90,
            "label_agreement_gt_0p95": float(agree) > 0.95,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out_json)

    if out_scatter is not None:
        _scatter(pa, pb, labels, label_a, label_b, out_scatter)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-a", type=Path, required=True,
                        help="First eval_results.json (e.g. 8B).")
    parser.add_argument("--eval-b", type=Path, required=True,
                        help="Second eval_results.json (e.g. 1.7B).")
    parser.add_argument("--label-a", type=str, default="A",
                        help="Display label for eval-a (e.g. '8B').")
    parser.add_argument("--label-b", type=str, default="B",
                        help="Display label for eval-b (e.g. '1.7B').")
    parser.add_argument("--threshold-a", type=float, default=None,
                        help="Override threshold for eval-a; defaults to "
                             "sibling threshold_sweep_*.json or 0.5.")
    parser.add_argument("--threshold-b", type=float, default=None,
                        help="Override threshold for eval-b.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path for the summary.")
    parser.add_argument("--scatter", type=Path, default=None,
                        help="Optional output PNG path for the scatter plot.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = compare(
        args.eval_a, args.eval_b,
        args.label_a, args.label_b,
        args.threshold_a, args.threshold_b,
        args.output, args.scatter,
    )
    print(json.dumps({
        "n_paired": summary["n_paired"],
        "pearson_r": summary["pearson_r"],
        "label_agreement_rate": summary["label_agreement_rate"],
        "mcnemar_p_exact_mid": summary["mcnemar"]["p_exact_mid"],
        "pass_criteria": summary["pass_criteria"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
