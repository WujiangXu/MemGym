"""MemRM evaluation primitives: dataset registry, row loading, metrics.

This module is the implementation that backs the ``memgym-eval-rm``
console script. It builds on the existing primitives in
``memgym.training.models.evaluator`` (``compute_metrics``,
``PredictionResult``) and the inference loop in
``memgym.training.scripts.eval_aug_sft._run_predictions``, adding only
the four things those did not provide:

1. A registry mapping short dataset names (``iid-heldout``,
   ``strategy-ood``, ...) to their HuggingFace repo IDs + per-dataset
   defaults.
2. HuggingFace download helpers for both datasets and the model
   checkpoint (resolving repo IDs like ``MemGym/memgym-rm-1p7b`` to a
   local path).
3. Bootstrap percentile CIs over AUROC, ECE, and accuracy.
4. A coverage-at-threshold metric for the gate semantics described in
   the paper's MemRM appendix.

The CLI on top of this module is intentionally thin — it only handles
argparse, output formatting, and the ``--dataset all`` sweep loop.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from memgym.training.models.evaluator import (
    PredictionResult,
    compute_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """One row of the registry. Describes how to locate and score a dataset."""

    short_name: str
    hf_repo: str
    # When None, every ``.jsonl`` file in the snapshot is loaded.
    file: Optional[str]
    n_expected: int
    paper_auroc: Optional[float] = None
    paper_ece: Optional[float] = None
    paper_n_covered: Optional[int] = None
    # Minimum --max-length the dataset's prompts require. CLI emits a
    # warning if the user passes a smaller value.
    min_max_length: int = 4096
    # True if this dataset is the model's training set (for the
    # ``train-sanity`` row). The CLI emits a banner so users do not
    # confuse train-set accuracy with held-out generalization.
    is_train_split: bool = False
    # True if the released artifact is only the post-selection-rule
    # "covered" subset of the paper's broader sweep, and per-slice
    # numbers should not be compared to the paper headline.
    covered_subset_only: bool = False
    notes: str = ""


# The 6 RM-compatible datasets that ship binary ``label`` (int 0/1) +
# flat ``prompt`` text fields. The midtrain-sft set is intentionally
# absent — it is a generative SFT dataset with no classification label.
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "iid-heldout": DatasetSpec(
        short_name="iid-heldout",
        hf_repo="MemGym/memgym-rm-iid-heldout",
        file=None,
        n_expected=3007,
        paper_auroc=0.985,
        paper_ece=0.009,
        min_max_length=32768,
        notes="SWE-Gym IID held-out split, repo-grouped. Paper Tab. tab:memrm headline.",
    ),
    "train-sanity": DatasetSpec(
        short_name="train-sanity",
        hf_repo="MemGym/memgym-rm-train",
        file=None,
        n_expected=15630,
        min_max_length=32768,
        is_train_split=True,
        notes="Training split. Scoring here is a sanity check, not generalization.",
    ),
    "scenario-ood-tau2": DatasetSpec(
        short_name="scenario-ood-tau2",
        hf_repo="MemGym/memgym-rm-scenario-ood-extras",
        file="reward_model_pairs_tau2.jsonl",
        n_expected=6209,
        min_max_length=32768,
        notes="Scenario-OOD: tau2-bench compaction events.",
    ),
    "scenario-ood-wa-long": DatasetSpec(
        short_name="scenario-ood-wa-long",
        hf_repo="MemGym/memgym-rm-scenario-ood-extras",
        file="reward_model_pairs_webarena_longctx.jsonl",
        n_expected=111,
        min_max_length=32768,
        notes="Scenario-OOD: WebArena long-context events (small n).",
    ),
    "scenario-ood-webarena": DatasetSpec(
        short_name="scenario-ood-webarena",
        hf_repo="MemGym/memgym-rm-scenario-ood-webarena",
        file="pairs_paper_eval.jsonl",
        n_expected=426,
        paper_auroc=0.748,
        paper_n_covered=87,
        min_max_length=32768,
        notes=(
            "Scenario-OOD WebArena V2. Paper reports AUROC=0.748 on "
            "n=87 covered subset (linear cohorts). 487-row union has "
            "61 unsScored rows from a failed inference shard."
        ),
    ),
    "strategy-ood": DatasetSpec(
        short_name="strategy-ood",
        hf_repo="MemGym/memgym-rm-strategy-ood",
        file=None,
        n_expected=22,
        min_max_length=32768,
        covered_subset_only=True,
        notes=(
            "Strategy-OOD: 2 slices (obs_masking_w100, pipeline_mask) "
            "of 11 rows each. Paper Tab. tab:memrm reports AUROC=0.714 "
            "on n=166 broader sweep; the public 22-row artifact is the "
            "data-integrity-filtered subset and reproduces per-slice "
            "numbers, not the n=166 headline."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: True if ``value`` looks like ``user/repo`` and is not a path."""
    if "/" not in value:
        return False
    if value.startswith(("/", ".", "~")):
        return False
    if Path(value).exists():
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def resolve_checkpoint(
    checkpoint: str,
    *,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Return a local directory for the MemRM checkpoint.

    If ``checkpoint`` is a local path, return it unchanged. If it looks
    like an HF repo ID (``MemGym/memgym-rm-1p7b``), download via
    ``snapshot_download`` and return the cache path.
    """
    path = Path(checkpoint)
    if path.exists():
        return path
    if not looks_like_hf_repo_id(checkpoint):
        raise FileNotFoundError(
            f"checkpoint {checkpoint!r} is neither an existing path nor a "
            f"recognizable HF repo id (expected `user/repo` form)."
        )
    from huggingface_hub import snapshot_download

    logger.info("downloading checkpoint from HF: %s", checkpoint)
    local_dir = snapshot_download(
        repo_id=checkpoint,
        repo_type="model",
        token=hf_token,
        cache_dir=cache_dir,
    )
    return Path(local_dir)


def resolve_dataset(
    dataset_arg: str,
    *,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Path], Optional[DatasetSpec]]:
    """Return ``(jsonl_paths, spec)`` for a dataset argument.

    ``dataset_arg`` is either a registry short name, a path to a local
    JSONL, or a path to a directory containing JSONL files.
    """
    if dataset_arg in DATASET_REGISTRY:
        spec = DATASET_REGISTRY[dataset_arg]
        return _download_registry_dataset(spec, hf_token=hf_token,
                                          cache_dir=cache_dir), spec

    path = Path(dataset_arg)
    if path.is_file():
        return [path], None
    if path.is_dir():
        jsonl = sorted(path.glob("*.jsonl"))
        if not jsonl:
            raise FileNotFoundError(
                f"directory {path} contains no .jsonl files"
            )
        return jsonl, None
    raise FileNotFoundError(
        f"dataset {dataset_arg!r} is not a registry short name "
        f"({sorted(DATASET_REGISTRY)}), an existing file, or a directory."
    )


def _download_registry_dataset(
    spec: DatasetSpec,
    *,
    hf_token: Optional[str],
    cache_dir: Optional[str],
) -> List[Path]:
    from huggingface_hub import snapshot_download

    logger.info("downloading dataset %s from HF: %s", spec.short_name, spec.hf_repo)
    local_dir = Path(snapshot_download(
        repo_id=spec.hf_repo,
        repo_type="dataset",
        token=hf_token,
        cache_dir=cache_dir,
    ))

    if spec.file is not None:
        candidate = local_dir / spec.file
        if not candidate.exists():
            # Some datasets nest data files under data/.
            nested = list(local_dir.rglob(spec.file))
            if not nested:
                raise FileNotFoundError(
                    f"expected file {spec.file} in {spec.hf_repo}, not found"
                )
            candidate = nested[0]
        return [candidate]
    # No specific file: take every JSONL in the snapshot (recursive).
    jsonl = sorted(local_dir.rglob("*.jsonl"))
    if not jsonl:
        raise FileNotFoundError(
            f"no .jsonl files found in HF snapshot of {spec.hf_repo}"
        )
    return jsonl


# ---------------------------------------------------------------------------
# Row loading + normalisation
# ---------------------------------------------------------------------------

def load_rows(
    jsonl_paths: List[Path],
    *,
    limit: int = 0,
) -> List[dict]:
    """Concatenate rows from one or more JSONL files.

    Each row must carry ``label`` (int 0/1) and either ``prompt`` (flat
    text) or ``input`` (alternative serialization). Rows missing both
    are dropped with a warning; rows missing ``label`` raise.
    """
    rows: List[dict] = []
    n_dropped_no_prompt = 0
    for path in jsonl_paths:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "label" not in row:
                    raise ValueError(
                        f"row missing required `label` field in {path}: "
                        f"keys={sorted(row)[:8]}"
                    )
                if "prompt" not in row and "input" not in row:
                    n_dropped_no_prompt += 1
                    continue
                if "prompt" not in row:
                    row["prompt"] = row["input"]
                row["label"] = int(row["label"])
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
        if limit and len(rows) >= limit:
            break
    if n_dropped_no_prompt:
        logger.warning(
            "dropped %d rows missing both `prompt` and `input` fields",
            n_dropped_no_prompt,
        )
    if not rows:
        raise ValueError(f"no rows loaded from {jsonl_paths}")
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_predictions(
    model,
    rows: List[dict],
    *,
    prefix: str = "",
    progress_every: int = 100,
) -> List[PredictionResult]:
    """One forward pass per row using ``predict_logits_text``.

    Mirrors ``memgym.training.scripts.eval_aug_sft._run_predictions``
    but is dataset-schema-aware (gracefully handles WebArena and tau2
    rows that lack some optional fields).
    """
    out: List[PredictionResult] = []
    for i, row in enumerate(rows):
        prompt = (prefix + row["prompt"]) if prefix else row["prompt"]
        pred_str, prob_safe = model.predict_logits_text(prompt)
        pred_label = 1 if pred_str == "SAFE" else 0
        true_label = int(row["label"])
        confidence = prob_safe if pred_label == 1 else (1.0 - prob_safe)
        out.append(PredictionResult(
            instance_id=str(row.get("trajectory_id") or row.get("instance_id") or ""),
            step=int(row.get("step", 0) or 0),
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            prob_safe=prob_safe,
            source=str(row.get("source_model") or row.get("domain") or ""),
            perturbation=str(row.get("perturbation", "")),
            source_dir=str(row.get("source_dir", "")),
        ))
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("eval progress: %d/%d", i + 1, len(rows))
    return out


# ---------------------------------------------------------------------------
# Additional metrics (coverage, bootstrap CI)
# ---------------------------------------------------------------------------

def coverage_at_threshold(
    predictions: List[PredictionResult],
    threshold: float,
) -> Tuple[float, int]:
    """Fraction (and count) of rows where the gate is confident.

    A row is "covered" if ``prob_safe >= threshold`` (high-confidence
    SAFE) or ``prob_safe <= 1 - threshold`` (high-confidence HARMFUL).
    The complement is the abstention region.
    """
    if not predictions:
        return 0.0, 0
    lo = 1.0 - threshold
    covered = sum(
        1 for p in predictions
        if p.prob_safe is not None and (p.prob_safe >= threshold or p.prob_safe <= lo)
    )
    return covered / len(predictions), covered


def accuracy_at_threshold(
    predictions: List[PredictionResult],
    threshold: float,
) -> Optional[float]:
    """Accuracy on the covered subset at threshold ``t``.

    Returns ``None`` if no rows are covered.
    """
    lo = 1.0 - threshold
    correct = 0
    n = 0
    for p in predictions:
        if p.prob_safe is None:
            continue
        if p.prob_safe >= threshold:
            n += 1
            if p.true_label == 1:
                correct += 1
        elif p.prob_safe <= lo:
            n += 1
            if p.true_label == 0:
                correct += 1
    if n == 0:
        return None
    return correct / n


def bootstrap_ci(
    predictions: List[PredictionResult],
    metric_fn: Callable[[List[PredictionResult]], Optional[float]],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Bootstrap percentile CI for an arbitrary metric.

    Returns ``None`` if the metric is undefined on enough resamples
    that the CI cannot be computed (e.g. AUROC when one class is
    absent in most resamples).
    """
    if not predictions or n_resamples <= 0:
        return None
    rng = random.Random(rng_seed)
    n = len(predictions)
    samples: List[float] = []
    for _ in range(n_resamples):
        resampled = [predictions[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(resampled)
        if v is not None:
            samples.append(v)
    if len(samples) < max(20, n_resamples // 10):
        # Too many undefined resamples; CI is unreliable.
        return None
    samples.sort()
    lo_idx = int((1.0 - ci) / 2.0 * len(samples))
    hi_idx = int((1.0 - (1.0 - ci) / 2.0) * len(samples)) - 1
    lo_idx = max(0, lo_idx)
    hi_idx = min(len(samples) - 1, hi_idx)
    return samples[lo_idx], samples[hi_idx]


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

@dataclass
class DatasetReport:
    """Flat record summarising one dataset's MemRM evaluation."""

    dataset: str
    hf_repo: Optional[str]
    n: int
    n_safe: int
    n_harmful: int
    threshold: float
    accuracy: float
    safe_f1: float
    harmful_f1: float
    auroc: Optional[float]
    auroc_ci: Optional[Tuple[float, float]]
    ece: Optional[float]
    coverage: float
    n_covered: int
    accuracy_at_threshold: Optional[float]
    paper_auroc: Optional[float] = None
    paper_n_covered: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def evaluate_dataset(
    model,
    rows: List[dict],
    *,
    spec: Optional[DatasetSpec] = None,
    threshold: float = 0.88,
    n_bootstrap: int = 1000,
    prefix: str = "",
) -> Tuple[DatasetReport, List[PredictionResult]]:
    """Run inference, compute metrics, return a flat report + per-row predictions."""
    predictions = run_predictions(model, rows, prefix=prefix)
    metrics = compute_metrics(predictions)

    coverage, n_covered = coverage_at_threshold(predictions, threshold)
    acc_at_t = accuracy_at_threshold(predictions, threshold)

    def _auroc(preds: List[PredictionResult]) -> Optional[float]:
        # Re-uses compute_metrics; returns None when AUROC is undefined.
        m = compute_metrics(preds)
        return m.auroc

    auroc_ci = bootstrap_ci(predictions, _auroc, n_resamples=n_bootstrap)

    notes: List[str] = []
    if spec is not None:
        if spec.is_train_split:
            notes.append(
                "TRAIN SPLIT — model has seen these rows during training; "
                "treat metrics as a sanity check, not generalization."
            )
        if spec.covered_subset_only:
            notes.append(
                "PUBLIC ARTIFACT IS POST-SELECTION SUBSET — the paper's "
                "headline AUROC for this axis is reported on a broader "
                "sweep that is not publicly released. Use per-slice "
                "numbers, not aggregate, when comparing to the paper."
            )

    n_safe = sum(1 for p in predictions if p.true_label == 1)
    n_harm = len(predictions) - n_safe

    report = DatasetReport(
        dataset=spec.short_name if spec else "custom",
        hf_repo=spec.hf_repo if spec else None,
        n=len(predictions),
        n_safe=n_safe,
        n_harmful=n_harm,
        threshold=threshold,
        accuracy=metrics.accuracy,
        safe_f1=metrics.safe_f1,
        harmful_f1=metrics.harmful_f1,
        auroc=metrics.auroc,
        auroc_ci=auroc_ci,
        ece=metrics.ece,
        coverage=coverage,
        n_covered=n_covered,
        accuracy_at_threshold=acc_at_t,
        paper_auroc=spec.paper_auroc if spec else None,
        paper_n_covered=spec.paper_n_covered if spec else None,
        notes=notes,
    )
    return report, predictions


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_results_table(reports: List[DatasetReport]) -> str:
    """Render a markdown table matching the paper's MemRM table shape."""
    rows: List[str] = []
    header = (
        "| Dataset                | n     | AUROC (95% CI)       | ECE    | "
        "Cov@t  | Acc@t   | Paper AUROC |"
    )
    sep = (
        "|------------------------|-------|----------------------|--------|"
        "--------|---------|-------------|"
    )
    rows.append(header)
    rows.append(sep)
    for r in reports:
        if r.auroc is None:
            auroc_cell = "—"
        elif r.auroc_ci is None:
            auroc_cell = f"{r.auroc:.3f}"
        else:
            auroc_cell = f"{r.auroc:.3f} [{r.auroc_ci[0]:.3f}, {r.auroc_ci[1]:.3f}]"
        ece_cell = f"{r.ece:.3f}" if r.ece is not None else "—"
        cov_cell = f"{r.coverage * 100:.1f}%"
        acc_cell = (
            f"{r.accuracy_at_threshold:.3f}"
            if r.accuracy_at_threshold is not None
            else "—"
        )
        paper_cell = f"{r.paper_auroc:.3f}" if r.paper_auroc is not None else "—"
        rows.append(
            f"| {r.dataset:<22} | {r.n:>5} | {auroc_cell:<20} | "
            f"{ece_cell:<6} | {cov_cell:<6} | {acc_cell:<7} | {paper_cell:<11} |"
        )
    out = "\n".join(rows)
    footnotes = []
    for r in reports:
        for note in r.notes:
            footnotes.append(f"- **{r.dataset}**: {note}")
    if footnotes:
        out += "\n\nNotes:\n" + "\n".join(footnotes)
    return out


def report_to_dict(report: DatasetReport) -> dict:
    """Serialise a DatasetReport for JSON output (auroc_ci → list)."""
    payload = asdict(report)
    if report.auroc_ci is not None:
        payload["auroc_ci"] = list(report.auroc_ci)
    return payload
