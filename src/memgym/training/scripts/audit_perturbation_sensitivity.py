"""Audit: how much does the perturbation-name token drive the gate's decision?

Context (the baseline-train phase, narrowed). The original plan wanted a deployment-form
audit — replaying the gym compaction at each eval step — but the stored
pairs don't include `full_history` / `compressed_history`, only the
already-rendered (recorded_action, proposed_action). That replay needs
infrastructure we haven't built. What we CAN audit with the existing
artifact is: does the classifier hinge on the perturbation string? If
yes, the gate is leaking a label that deployment wouldn't know (the
perturbation is a train-time diagnostic tag, not a runtime signal).

Protocol:
  * Load eval rows from `aug_sft_pairs.jsonl`.
  * Per row, build a 2nd prompt where the perturbation in the template
    is rewritten to "unknown". Keep recorded_action + proposed_action
    verbatim (that's the part deployment would actually feed in).
  * Score both prompts through `MemoryWorldModel.predict_logits_text`.
  * Report: Δ-SAFE-F1 at t*, mean |Δ-prob_safe|, tail of largest swings.

Pass criterion: Δ-SAFE-F1 ≤ 0.03 AND P95(|Δ-prob_safe|) ≤ 0.10. A large
gap means the perturbation token is doing heavy lifting in the gate's
decision and we should expect offline metrics to overstate deployment
performance.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

from memgym.training.models.evaluator import PredictionResult, compute_metrics
from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

logger = logging.getLogger(__name__)

# Matches the literal substring the aug_pairs template writes:
#   "Trajectory step {step} with perturbation '{perturbation}'."
_PERT_LINE_RE = re.compile(r"with perturbation '([^']+)'")


def _rewrite_perturbation(prompt: str, replacement: str) -> str:
    """Replace only the perturbation-name occurrence in the first line,
    leaving recorded_action / proposed_action completely untouched.
    """
    new, n = _PERT_LINE_RE.subn(
        f"with perturbation '{replacement}'", prompt, count=1,
    )
    if n != 1:
        raise ValueError(f"could not locate perturbation line in prompt: {prompt[:200]!r}")
    return new


def _load_eval_rows(pairs_path: Path, n_samples: int, seed: int) -> List[dict]:
    all_rows: List[dict] = []
    with pairs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            all_rows.append(row)
    if not all_rows:
        raise ValueError(f"no eval rows in {pairs_path}")
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    return all_rows[:n_samples] if n_samples else all_rows


def _score(model: MemoryWorldModel, prompt: str) -> Tuple[str, float]:
    return model.predict_logits_text(prompt)


def _as_prediction(row: dict, prob_safe: float) -> PredictionResult:
    pred_label = 1 if prob_safe >= 0.5 else 0  # threshold rebound at metric time
    return PredictionResult(
        instance_id=row.get("trajectory_id", ""),
        step=int(row.get("step", 0)),
        true_label=int(row["label"]),
        pred_label=pred_label,
        confidence=prob_safe if pred_label == 1 else (1.0 - prob_safe),
        prob_safe=prob_safe,
        source=row.get("source_model", ""),
        perturbation=row.get("perturbation", ""),
        source_dir=row.get("source_dir", ""),
    )


def _metrics_at(preds: List[PredictionResult], threshold: float) -> dict:
    rethreshed = [
        PredictionResult(
            instance_id=p.instance_id,
            step=p.step,
            true_label=p.true_label,
            pred_label=1 if p.prob_safe >= threshold else 0,
            confidence=p.prob_safe if p.prob_safe >= threshold else (1.0 - p.prob_safe),
            prob_safe=p.prob_safe,
            source=p.source,
            perturbation=p.perturbation,
            source_dir=p.source_dir,
        )
        for p in preds
    ]
    m = compute_metrics(rethreshed)
    return {
        "safe_f1": m.safe_f1,
        "harmful_f1": m.harmful_f1,
        "safe_precision": m.safe_precision,
        "safe_recall": m.safe_recall,
        "accuracy": m.accuracy,
        "confusion": m.confusion,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path,
                        default=Path("training_output/aug_sft_full_yn/aug_sft_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="t* from Tier-1 sweep (constrained SAFE-F1 best)")
    parser.add_argument("--replacement", default="unknown",
                        help="Perturbation name to substitute in the counterfactual prompt")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", default=True)
    parser.add_argument("--gpu-id", type=str, default=None)
    args = parser.parse_args()

    if args.gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        logger.info("pinned CUDA_VISIBLE_DEVICES=%s", args.gpu_id)

    rows = _load_eval_rows(args.pairs, args.n_samples, args.seed)
    logger.info("audit: %d eval rows (replacement='%s')", len(rows), args.replacement)

    config = ModelConfig(base_model=args.base_model, load_in_4bit=args.load_in_4bit)
    model = MemoryWorldModel.from_checkpoint(args.checkpoint, config=config)

    orig_preds: List[PredictionResult] = []
    cf_preds: List[PredictionResult] = []
    deltas: List[dict] = []

    for i, row in enumerate(rows):
        orig_prompt = row["prompt"]
        cf_prompt = _rewrite_perturbation(orig_prompt, args.replacement)

        _, p_orig = _score(model, orig_prompt)
        _, p_cf = _score(model, cf_prompt)

        orig_preds.append(_as_prediction(row, p_orig))
        cf_preds.append(_as_prediction(row, p_cf))

        deltas.append({
            "trajectory_id": row.get("trajectory_id"),
            "step": row.get("step"),
            "perturbation": row.get("perturbation"),
            "source_model": row.get("source_model"),
            "true_label": int(row["label"]),
            "prob_safe_orig": p_orig,
            "prob_safe_cf": p_cf,
            "abs_delta": abs(p_orig - p_cf),
        })

        if (i + 1) % 50 == 0:
            logger.info("scored %d/%d", i + 1, len(rows))

    m_orig = _metrics_at(orig_preds, args.threshold)
    m_cf = _metrics_at(cf_preds, args.threshold)

    abs_deltas = sorted(d["abs_delta"] for d in deltas)
    n = len(abs_deltas)
    def pct(p: float) -> float:
        if not abs_deltas:
            return 0.0
        k = min(n - 1, int(p * n))
        return abs_deltas[k]

    summary = {
        "n_samples": len(rows),
        "threshold": args.threshold,
        "replacement": args.replacement,
        "metrics_original": m_orig,
        "metrics_counterfactual": m_cf,
        "delta_safe_f1": m_cf["safe_f1"] - m_orig["safe_f1"],
        "delta_harmful_f1": m_cf["harmful_f1"] - m_orig["harmful_f1"],
        "abs_prob_delta": {
            "mean": sum(abs_deltas) / n if n else 0.0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": abs_deltas[-1] if abs_deltas else 0.0,
        },
        "pass_fidelity": (
            abs(m_cf["safe_f1"] - m_orig["safe_f1"]) <= 0.03
            and pct(0.95) <= 0.10
        ),
    }
    print(json.dumps(summary, indent=2))

    out_path = args.output or (args.checkpoint / "perturbation_sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top20 = sorted(deltas, key=lambda d: d["abs_delta"], reverse=True)[:20]
    out_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "summary": summary,
        "largest_swings": top20,
        "per_row_deltas": deltas,
    }, indent=2))
    logger.info("wrote %s", out_path)

    print(json.dumps({"step": "perturbation_audit", **{
        k: summary[k] for k in ("delta_safe_f1", "delta_harmful_f1", "pass_fidelity")
    }}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
