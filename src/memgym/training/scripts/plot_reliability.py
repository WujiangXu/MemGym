"""Reliability diagram for one-or-more `eval_results.json` files.

Bins each row's `prob_safe` into equal-mass buckets and plots the
empirical SAFE rate per bucket vs the mean predicted probability,
with Wilson 95% confidence bands. A perfectly calibrated classifier
sits on the y=x diagonal.

This complements the scalar ECE already written to `eval_results.json`
by `_compute_ece` (`evaluator.py:247`). ECE summarizes calibration as
a single weighted-gap number; the reliability diagram exposes *where*
in [0, 1] the gap lives — high-confidence miscalibration looks
qualitatively different from low-confidence noise even at the same
ECE value, and the appendix's "lightweight option" claim is
weaker if the 1.7B's gap is concentrated at high confidence (the
regime that matters for gating).

Note on binning convention. The published ECE bins by `max(prob_safe,
1-prob_safe)` (confidence in the predicted class). This script bins
by `prob_safe` directly, which is the canonical reliability-diagram
axis for binary classification. Both views are valid; they expose
different facets and the diagram is not expected to algebraically
reproduce the scalar ECE.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)


def _load_predictions(path: Path) -> List[dict]:
    """Load per_row_predictions, dropping any row without prob_safe."""
    with path.open() as f:
        data = json.load(f)
    rows = data.get("per_row_predictions") or []
    return [r for r in rows if r.get("prob_safe") is not None]


def _wilson_band(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over normal-approximation when bin counts are small or
    the empirical rate is near 0/1 — both common at the edges of a
    well-calibrated reliability diagram.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _equal_mass_bins(
    rows: List[dict], n_bins: int
) -> List[Tuple[float, float, int, int]]:
    """Split rows into n_bins equal-mass buckets sorted by prob_safe.

    Returns one (mean_prob, empirical_rate, n_pos, n) tuple per bucket.
    """
    sorted_rows = sorted(rows, key=lambda r: r["prob_safe"])
    n = len(sorted_rows)
    bin_size = max(1, n // n_bins)
    out = []
    for i in range(0, n, bin_size):
        chunk = sorted_rows[i:i + bin_size]
        if not chunk:
            continue
        probs = [r["prob_safe"] for r in chunk]
        labels = [int(r["true_label"]) for r in chunk]
        mean_prob = sum(probs) / len(probs)
        n_pos = sum(labels)
        out.append((mean_prob, n_pos / len(chunk), n_pos, len(chunk)))
    return out


def _plot_one(ax, label: str, rows: List[dict], n_bins: int, color: str) -> None:
    bins = _equal_mass_bins(rows, n_bins)
    xs = [b[0] for b in bins]
    ys = [b[1] for b in bins]
    lows = []
    highs = []
    for _, _, k, n in bins:
        lo, hi = _wilson_band(k, n)
        lows.append(lo)
        highs.append(hi)
    ax.fill_between(xs, lows, highs, alpha=0.20, color=color)
    ax.plot(xs, ys, marker="o", linewidth=1.5, color=color, label=label)


def plot_reliability(
    eval_jsons: List[Tuple[str, Path]],
    out_path: Path,
    n_bins: int = 15,
    title_suffix: str = "",
) -> None:
    """Render side-by-side reliability diagrams for the given eval JSONs.

    `eval_jsons` is a list of (label, path) pairs — one panel per pair.
    """
    n_panels = len(eval_jsons)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), squeeze=False, sharey=True,
    )
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, (label, path) in enumerate(eval_jsons):
        rows = _load_predictions(path)
        if not rows:
            logger.warning("no prob_safe rows in %s", path)
            continue
        ax = axes[0, i]
        _plot_one(ax, label, rows, n_bins, palette[i % len(palette)])
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0,
                label="perfect calibration")
        ax.set_xlabel("predicted P(SAFE)")
        if i == 0:
            ax.set_ylabel("empirical SAFE rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{label}\n(n={len(rows)}, {n_bins} equal-mass bins)")
        ax.legend(loc="upper left", fontsize=8)
    if title_suffix:
        fig.suptitle(title_suffix)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval", action="append", required=True, metavar="LABEL:PATH",
        help="LABEL:PATH pair, e.g. '8B:training_output/.../eval_results_8b.json'. "
             "Pass multiple --eval flags to overlay panels.",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Output PNG path.")
    parser.add_argument("--bins", type=int, default=15,
                        help="Number of equal-mass bins (default 15, matches ECE).")
    parser.add_argument("--title", type=str, default="",
                        help="Optional figure suptitle.")
    args = parser.parse_args()

    pairs: List[Tuple[str, Path]] = []
    for spec in args.eval:
        if ":" not in spec:
            raise SystemExit(f"--eval expects LABEL:PATH, got {spec!r}")
        label, p = spec.split(":", 1)
        pairs.append((label, Path(p)))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plot_reliability(pairs, args.output, n_bins=args.bins, title_suffix=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
