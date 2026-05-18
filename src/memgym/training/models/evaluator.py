"""Evaluation metrics for the memory world model.

Computes binary classification metrics for SAFE/HARMFUL predictions,
with focus on HARMFUL recall (catching compression events that hurt
the agent) since the class distribution is heavily skewed toward SAFE.

Usage:
    from memgym.training.models.evaluator import WorldModelEvaluator

    evaluator = WorldModelEvaluator(model)
    metrics = evaluator.evaluate("training_output/dataset.jsonl")
    evaluator.print_report(metrics)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single prediction with ground truth.

    `confidence` is the calibrated probability of SAFE in [0,1] when
    produced by `MemoryWorldModel.predict_logits`; for legacy `predict`
    output it is the model's stated confidence in `pred_label`.
    """

    instance_id: str
    step: int
    true_label: int  # 1=SAFE, 0=HARMFUL
    pred_label: int  # 1=SAFE, 0=HARMFUL
    confidence: float = 0.5
    prob_safe: Optional[float] = None  # explicit P(SAFE) when available
    source: str = ""
    perturbation: str = ""
    source_dir: str = ""


@dataclass
class EvalMetrics:
    """Evaluation metrics for binary SAFE/HARMFUL classification."""

    # Counts
    total: int = 0
    correct: int = 0

    # Overall
    accuracy: float = 0.0

    # Per-class: HARMFUL (label=0, positive class for detection)
    harmful_precision: float = 0.0
    harmful_recall: float = 0.0
    harmful_f1: float = 0.0
    harmful_support: int = 0  # true HARMFUL count

    # Per-class: SAFE (label=1)
    safe_precision: float = 0.0
    safe_recall: float = 0.0
    safe_f1: float = 0.0
    safe_support: int = 0  # true SAFE count

    # Confusion matrix: [TN, FP, FN, TP] where positive=HARMFUL
    confusion: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # Optional: AUROC if confidence scores are available
    auroc: Optional[float] = None

    # Calibration: Expected Calibration Error (15 equal-mass bins)
    ece: Optional[float] = None

    # Breakdown by source / perturbation / source_model
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_perturbation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_source_model: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_metrics(predictions: List[PredictionResult]) -> EvalMetrics:
    """Compute binary classification metrics.

    Treats HARMFUL (label=0) as the positive class for detection,
    since catching harmful compressions is the primary objective.
    """
    if not predictions:
        return EvalMetrics()

    # Confusion matrix for HARMFUL detection
    # Positive = HARMFUL (label=0), Negative = SAFE (label=1)
    tp = fp = fn = tn = 0

    for p in predictions:
        is_harmful_true = p.true_label == 0
        is_harmful_pred = p.pred_label == 0

        if is_harmful_true and is_harmful_pred:
            tp += 1
        elif not is_harmful_true and is_harmful_pred:
            fp += 1
        elif is_harmful_true and not is_harmful_pred:
            fn += 1
        else:
            tn += 1

    total = len(predictions)
    correct = tp + tn

    # HARMFUL metrics (positive class)
    h_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    h_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0.0

    # SAFE metrics
    s_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    s_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    metrics = EvalMetrics(
        total=total,
        correct=correct,
        accuracy=correct / total,
        harmful_precision=h_prec,
        harmful_recall=h_rec,
        harmful_f1=h_f1,
        harmful_support=tp + fn,
        safe_precision=s_prec,
        safe_recall=s_rec,
        safe_f1=s_f1,
        safe_support=tn + fp,
        confusion=[tn, fp, fn, tp],
    )

    # AUROC if we have meaningful confidence scores
    confidences = [p.confidence for p in predictions]
    if len(set(confidences)) > 1:
        metrics.auroc = _compute_auroc(predictions)

    # ECE — only meaningful when prob_safe is populated (predict_logits path)
    if any(p.prob_safe is not None for p in predictions):
        metrics.ece = _compute_ece(predictions)

    metrics.by_source = _breakdown(predictions, key=lambda p: p.source)
    metrics.by_perturbation = _breakdown(predictions, key=lambda p: p.perturbation)
    metrics.by_source_model = _breakdown(
        predictions, key=lambda p: _source_model_key(p.source_dir),
    )

    return metrics


def _source_model_key(source_dir: str) -> str:
    """Collapse a source_dir like `sonnet_fork_gap554_scaleA_v1` to `sonnet`."""
    if not source_dir:
        return ""
    head = source_dir.split("_", 1)[0].lower()
    if head.startswith("haiku"):
        return "haiku"
    if head.startswith("gpt"):
        return "gptoss"
    if head.startswith("sonnet"):
        return "sonnet"
    return head


def _breakdown(
    predictions: List[PredictionResult],
    *,
    key,
    min_per_class: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Slice predictions by `key(p)` and compute per-slice F1 + AUROC + ECE.

    AUROC / ECE require both classes to be non-empty in the slice and need
    `prob_safe` populated. When either condition fails (or a class has fewer
    than `min_per_class` examples) we emit `nan` rather than a misleading
    point estimate. This catches the v1 OOD-failure shape: a slice where
    aggregate F1 looks fine but calibration on the tail is broken.
    """
    import math
    groups: Dict[str, List[PredictionResult]] = {}
    for p in predictions:
        k = key(p)
        if not k:
            continue
        groups.setdefault(k, []).append(p)
    out: Dict[str, Dict[str, float]] = {}
    for k, preds in groups.items():
        sub = compute_basic(preds)
        n_safe = sum(1 for p in preds if p.true_label == 1)
        n_harm = len(preds) - n_safe
        slice_auroc: float
        slice_ece: float
        try:
            if min(n_safe, n_harm) >= min_per_class and any(p.prob_safe is not None for p in preds):
                slice_auroc = _compute_auroc(preds)
                slice_ece = _compute_ece(preds)
            else:
                slice_auroc = float("nan")
                slice_ece = float("nan")
        except Exception:  # noqa: BLE001 — never break the report on one slice
            slice_auroc = float("nan")
            slice_ece = float("nan")
        out[k] = {
            "count": len(preds),
            "accuracy": sub["accuracy"],
            "harmful_f1": sub["harmful_f1"],
            "safe_f1": sub["safe_f1"],
            "auroc": slice_auroc,
            "ece": slice_ece,
            "harmful_count": sub["harmful_support"],
            "harmful_pct": sub["harmful_support"] / len(preds) * 100,
        }
    return out


def compute_basic(predictions: List[PredictionResult]) -> Dict[str, float]:
    """Lightweight metric computation used inside breakdowns (no recursion)."""
    if not predictions:
        return {"accuracy": 0.0, "harmful_f1": 0.0, "safe_f1": 0.0,
                "harmful_support": 0, "safe_support": 0}
    tp = fp = fn = tn = 0
    for p in predictions:
        if p.true_label == 0 and p.pred_label == 0: tp += 1
        elif p.true_label == 1 and p.pred_label == 0: fp += 1
        elif p.true_label == 0 and p.pred_label == 1: fn += 1
        else: tn += 1
    h_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    h_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    h_f1 = 2 * h_p * h_r / (h_p + h_r) if (h_p + h_r) > 0 else 0.0
    s_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    s_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    s_f1 = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
    return {
        "accuracy": (tp + tn) / len(predictions),
        "harmful_f1": h_f1,
        "safe_f1": s_f1,
        "harmful_support": tp + fn,
        "safe_support": tn + fp,
    }


def _compute_ece(
    predictions: List[PredictionResult],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error using equal-mass bins.

    Bins predictions by `prob_safe`, computes per-bin |mean_conf -
    accuracy_at_predicted_class|, weights by bin size. Lower is better
    (perfectly calibrated → 0). Only includes predictions with prob_safe
    set; assumes binary classification with SAFE=1 / HARMFUL=0.
    """
    scored = [p for p in predictions if p.prob_safe is not None]
    if not scored:
        return 0.0
    n = len(scored)
    # The "confidence in the predicted class" is max(prob_safe, 1-prob_safe).
    items = []
    for p in scored:
        conf_pred = max(p.prob_safe, 1.0 - p.prob_safe)
        is_correct = float(p.true_label == p.pred_label)
        items.append((conf_pred, is_correct))
    items.sort(key=lambda x: x[0])
    bin_size = max(1, n // n_bins)
    ece = 0.0
    for i in range(0, n, bin_size):
        chunk = items[i:i + bin_size]
        if not chunk:
            continue
        mean_conf = sum(c for c, _ in chunk) / len(chunk)
        acc = sum(corr for _, corr in chunk) / len(chunk)
        ece += abs(mean_conf - acc) * (len(chunk) / n)
    return ece


def _compute_auroc(predictions: List[PredictionResult]) -> float:
    """Compute Area Under ROC Curve using trapezoidal rule.

    For HARMFUL detection: higher confidence in HARMFUL prediction
    means lower confidence value (since confidence is for the predicted
    class). We use 1-confidence for SAFE predictions as the HARMFUL score.
    """
    # Score: probability of HARMFUL
    scored = []
    for p in predictions:
        if p.pred_label == 0:  # predicted HARMFUL
            harmful_score = p.confidence
        else:  # predicted SAFE
            harmful_score = 1.0 - p.confidence
        scored.append((harmful_score, p.true_label == 0))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    n_pos = sum(1 for _, is_pos in scored if is_pos)
    n_neg = len(scored) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return chance level

    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    auc = 0.0
    prev_score = None

    for score, is_pos in scored:
        if score != prev_score and prev_score is not None:
            # Trapezoidal rule
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if is_pos:
            tp += 1
        else:
            fp += 1
        prev_score = score

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


class WorldModelEvaluator:
    """Evaluate a trained world model on held-out data."""

    def __init__(self, model: Any = None):
        """Initialize evaluator.

        Args:
            model: MemoryWorldModel instance (optional — can also
                   evaluate from pre-computed predictions).
        """
        self.model = model

    def evaluate(
        self,
        dataset_path: str | Path,
        *,
        max_examples: Optional[int] = None,
        filter_split: Optional[str] = None,
    ) -> EvalMetrics:
        """Run model predictions on a dataset and compute metrics.

        Args:
            dataset_path: JSONL file from save_dataset_jsonl().
            max_examples: Limit evaluation size (for quick checks).

        Returns:
            EvalMetrics with all classification metrics.
        """
        if self.model is None:
            raise RuntimeError("Model required for evaluate(). Use "
                               "evaluate_predictions() for pre-computed results.")

        # Prefer predict_logits when available — single forward pass + real
        # calibrated probability. Fall back to legacy predict() otherwise.
        use_logits = hasattr(self.model, "predict_logits")
        predictions = []
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                record = json.loads(line.strip())
                meta = record.get("metadata", {})
                messages = record["messages"]
                # `split_field` filtering is left to callers (eval scripts);
                # this method evaluates whatever JSONL it's pointed at.
                if filter_split is not None and record.get("split") != filter_split:
                    continue

                true_label = meta.get("label", 1)
                predict_messages = [
                    m for m in messages if m["role"] in ("system", "user")
                ]
                if use_logits:
                    pred_str, prob_safe = self.model.predict_logits(predict_messages)
                    pred_label = 1 if pred_str == "SAFE" else 0
                    confidence = prob_safe if pred_label == 1 else (1.0 - prob_safe)
                    prob_safe_field: Optional[float] = prob_safe
                else:
                    pred_str, confidence = self.model.predict(predict_messages)
                    pred_label = 1 if pred_str == "SAFE" else 0
                    prob_safe_field = None

                predictions.append(PredictionResult(
                    instance_id=meta.get("instance_id", ""),
                    step=meta.get("step", 0),
                    true_label=true_label,
                    pred_label=pred_label,
                    confidence=confidence,
                    prob_safe=prob_safe_field,
                    source=meta.get("source", ""),
                    perturbation=record.get("perturbation", ""),
                    source_dir=record.get("source_dir", ""),
                ))

        logger.info("Evaluated %d examples", len(predictions))
        return compute_metrics(predictions)

    def evaluate_predictions(
        self,
        predictions: List[PredictionResult],
    ) -> EvalMetrics:
        """Compute metrics from pre-computed predictions.

        Useful when predictions were generated elsewhere (e.g., batch
        inference on GPU cluster).
        """
        return compute_metrics(predictions)

    @staticmethod
    def print_report(metrics: EvalMetrics) -> str:
        """Format metrics as a human-readable report.

        Returns the report string (also logs it).
        """
        lines = [
            "=" * 50,
            "World Model Evaluation Report",
            "=" * 50,
            f"Total examples:  {metrics.total}",
            f"Accuracy:        {metrics.accuracy:.3f} "
            f"({metrics.correct}/{metrics.total})",
            "",
            "--- HARMFUL Detection (positive class) ---",
            f"  Precision:  {metrics.harmful_precision:.3f}",
            f"  Recall:     {metrics.harmful_recall:.3f}",
            f"  F1:         {metrics.harmful_f1:.3f}",
            f"  Support:    {metrics.harmful_support}",
            "",
            "--- SAFE (negative class) ---",
            f"  Precision:  {metrics.safe_precision:.3f}",
            f"  Recall:     {metrics.safe_recall:.3f}",
            f"  F1:         {metrics.safe_f1:.3f}",
            f"  Support:    {metrics.safe_support}",
            "",
            "--- Confusion Matrix ---",
            f"  (HARMFUL detection: positive=HARMFUL, negative=SAFE)",
            f"                  Pred SAFE   Pred HARMFUL",
            f"  True SAFE       {metrics.confusion[0]:>8}   {metrics.confusion[1]:>12}",
            f"  True HARMFUL    {metrics.confusion[2]:>8}   {metrics.confusion[3]:>12}",
        ]

        if metrics.auroc is not None:
            lines.extend(["", f"AUROC:  {metrics.auroc:.3f}"])
        if metrics.ece is not None:
            lines.extend([f"ECE:    {metrics.ece:.3f} (lower is better)"])

        for header, breakdown in (
            ("--- By Source Model ---", metrics.by_source_model),
            ("--- By Perturbation ---", metrics.by_perturbation),
            ("--- By Source ---", metrics.by_source),
        ):
            if not breakdown:
                continue
            lines.extend(["", header])
            for k, stats in sorted(breakdown.items()):
                lines.append(
                    f"  {k}: {stats['count']:.0f} ex, "
                    f"acc={stats['accuracy']:.3f}, "
                    f"H-F1={stats.get('harmful_f1', 0):.3f}, "
                    f"S-F1={stats.get('safe_f1', 0):.3f}, "
                    f"harmful={stats['harmful_pct']:.1f}%"
                )

        lines.append("=" * 50)
        report = "\n".join(lines)
        logger.info("\n%s", report)
        return report


def make_trainer_compute_metrics(
    tokenizer: Any,
    label_safe_token: str = " Y",
    label_harmful_token: str = " N",
) -> Tuple[Any, Any]:
    """Factory for HF Trainer eval-time metric hooks.

    Returns ``(preprocess_logits_for_metrics, compute_metrics)`` to be passed
    to ``Trainer(...)`` / ``SFTTrainer(...)``. The preprocess hook runs on
    each eval batch on-device and reduces ``(B, T, V)`` logits to ``(B, 2)``
    so the full accumulation buffer fits in CPU RAM; the compute hook then
    converts those scores into a flat scalar dict that wandb can log.

    Tokenizer-asymmetry fix: under Qwen3 BPE the label ``" N"`` tokenizes to
    three sub-tokens while ``" Y"`` is a single token. Reading only the first
    sub-token logit makes ``P(HARMFUL)`` a partial-token signal vs.
    ``P(SAFE)``'s full-token signal — the v1→v2 OOD collapse partially
    traces back to this. Here we **sum logits across all sub-token IDs** of
    each label, then 2-way-softmax. Both classes go through the same
    aggregator, so the comparison is symmetric whether the answer string is
    one token or many.
    """
    import numpy as np
    import torch

    safe_ids = tokenizer.encode(label_safe_token, add_special_tokens=False)
    harm_ids = tokenizer.encode(label_harmful_token, add_special_tokens=False)
    if not safe_ids or not harm_ids:
        raise ValueError(
            f"Empty tokenization for label tokens "
            f"(safe={label_safe_token!r}→{safe_ids}, "
            f"harmful={label_harmful_token!r}→{harm_ids})."
        )
    safe_ids_t = torch.tensor(safe_ids, dtype=torch.long)
    harm_ids_t = torch.tensor(harm_ids, dtype=torch.long)
    safe_id_set = set(safe_ids)
    harm_id_set = set(harm_ids)

    def _preprocess(logits: Any, labels: Any) -> Any:
        """Per-batch on-device reduction: (B, T, V) → (B, 2).

        Column 0 = SAFE score (sum of logits over all " Y" sub-tokens).
        Column 1 = HARMFUL score (sum over all " N" sub-tokens).
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        non_masked = labels.ne(-100)
        # First non-masked label position per row; gather logits at t-1.
        first_idx = non_masked.float().argmax(dim=-1)
        pred_idx = (first_idx - 1).clamp(min=0)
        batch = logits.size(0)
        rows = torch.arange(batch, device=logits.device)
        gathered = logits[rows, pred_idx]  # (B, V)
        safe_ids_dev = safe_ids_t.to(gathered.device)
        harm_ids_dev = harm_ids_t.to(gathered.device)
        safe_score = gathered.index_select(-1, safe_ids_dev).sum(dim=-1)
        harm_score = gathered.index_select(-1, harm_ids_dev).sum(dim=-1)
        return torch.stack([safe_score, harm_score], dim=-1)  # (B, 2)

    def _compute(eval_pred: Any) -> Dict[str, float]:
        scores, labels = eval_pred  # scores: (N, 2)
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels)
        # 2-way softmax → P(SAFE)
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        prob = e / e.sum(axis=-1, keepdims=True)
        prob_safe = prob[:, 0]
        # Recover ground-truth label from the first non-(-100) target token.
        preds: List[PredictionResult] = []
        for i in range(labels.shape[0]):
            row = labels[i]
            mask = row != -100
            if not mask.any():
                continue  # nothing to score
            first_pos = int(mask.argmax())
            tok = int(row[first_pos])
            if tok in safe_id_set:
                tl = 1
            elif tok in harm_id_set:
                tl = 0
            else:
                # Unrecognized first-target token — skip rather than guess.
                continue
            ps = float(prob_safe[i])
            pl = 1 if ps >= 0.5 else 0
            conf = ps if pl == 1 else 1.0 - ps
            preds.append(PredictionResult(
                instance_id="", step=0,
                true_label=tl, pred_label=pl,
                confidence=conf, prob_safe=ps,
            ))
        if not preds:
            return {"safe_f1": 0.0, "harmful_f1": 0.0, "accuracy": 0.0,
                    "auroc": float("nan"), "ece": float("nan"),
                    "n_scored": 0}
        m_full = compute_metrics(preds)
        return {
            "accuracy":          m_full.accuracy,
            "safe_f1":           m_full.safe_f1,
            "safe_precision":    m_full.safe_precision,
            "safe_recall":       m_full.safe_recall,
            "harmful_f1":        m_full.harmful_f1,
            "harmful_precision": m_full.harmful_precision,
            "harmful_recall":    m_full.harmful_recall,
            "auroc": m_full.auroc if m_full.auroc is not None else float("nan"),
            "ece":   m_full.ece   if m_full.ece   is not None else float("nan"),
            "n_scored": len(preds),
        }

    return _preprocess, _compute
