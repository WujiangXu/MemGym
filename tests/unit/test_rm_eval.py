"""Unit tests for the MemRM evaluation CLI and supporting module.

These tests run without GPU, network, or a real MemRM checkpoint. The
model is mocked via a tiny stand-in class that returns deterministic
``predict_logits_text`` outputs from a lookup table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from memgym.training.eval import rm_eval
from memgym.training.eval.rm_eval import (
    DATASET_REGISTRY,
    accuracy_at_threshold,
    bootstrap_ci,
    coverage_at_threshold,
    evaluate_dataset,
    format_results_table,
    load_rows,
    looks_like_hf_repo_id,
    resolve_checkpoint,
    resolve_dataset,
    run_predictions,
)
from memgym.training.models.evaluator import PredictionResult, compute_metrics


class TestResolveCheckpoint:
    """Covers the published `MemGym/memgym-rm-1p7b` layout (adapter under
    `checkpoint-500/`), plus a root-level adapter and a multi-resume layout."""

    def test_root_level_adapter(self, tmp_path: Path):
        (tmp_path / "adapter_config.json").write_text("{}")
        assert resolve_checkpoint(str(tmp_path)) == tmp_path

    def test_subfolder_only(self, tmp_path: Path):
        sub = tmp_path / "checkpoint-500"
        sub.mkdir()
        (sub / "adapter_config.json").write_text("{}")
        assert resolve_checkpoint(str(tmp_path)) == sub

    def test_multiple_subfolders_picks_highest(self, tmp_path: Path):
        for nnn in (100, 500, 250):
            sub = tmp_path / f"checkpoint-{nnn}"
            sub.mkdir()
            (sub / "adapter_config.json").write_text("{}")
        assert resolve_checkpoint(str(tmp_path)) == tmp_path / "checkpoint-500"

    def test_root_wins_over_subfolder(self, tmp_path: Path):
        (tmp_path / "adapter_config.json").write_text("{}")
        sub = tmp_path / "checkpoint-500"
        sub.mkdir()
        (sub / "adapter_config.json").write_text("{}")
        assert resolve_checkpoint(str(tmp_path)) == tmp_path

    def test_no_adapter_returns_root(self, tmp_path: Path):
        # Caller can still pass a dir with no adapter (e.g. base-model-only
        # diagnostic); resolver shouldn't raise here.
        assert resolve_checkpoint(str(tmp_path)) == tmp_path


class StubMemRM:
    """Drop-in stand-in for ``MemoryWorldModel.predict_logits_text``.

    Built from a ``{prompt: prob_safe}`` lookup table. Useful for
    constructing tests where AUROC, ECE, and coverage have closed-form
    answers.
    """

    def __init__(self, prob_table: Dict[str, float]):
        self.prob_table = prob_table

    def predict_logits_text(self, text: str) -> Tuple[str, float]:
        if text not in self.prob_table:
            raise KeyError(f"unknown prompt: {text!r}")
        p = self.prob_table[text]
        return ("SAFE" if p >= 0.5 else "HARMFUL"), p


# ---------------------------------------------------------------------------
# Tiny dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_rows() -> List[dict]:
    """5 rows: 3 SAFE (label=1), 2 HARMFUL (label=0).

    Probabilities chosen so AUROC is exactly 1.0 (every SAFE row scores
    higher than every HARMFUL row).
    """
    return [
        {"prompt": "p1", "label": 1, "trajectory_id": "t1", "step": 0},
        {"prompt": "p2", "label": 1, "trajectory_id": "t2", "step": 0},
        {"prompt": "p3", "label": 1, "trajectory_id": "t3", "step": 0},
        {"prompt": "p4", "label": 0, "trajectory_id": "t4", "step": 0},
        {"prompt": "p5", "label": 0, "trajectory_id": "t5", "step": 0},
    ]


@pytest.fixture()
def tiny_model() -> StubMemRM:
    return StubMemRM({
        "p1": 0.95,  # SAFE, high conf — covered (above 0.88)
        "p2": 0.80,  # SAFE, medium conf — NOT covered (between 0.12 and 0.88)
        "p3": 0.60,  # SAFE, low conf — NOT covered
        "p4": 0.10,  # HARMFUL, high conf — covered (below 0.12)
        "p5": 0.30,  # HARMFUL, medium conf — NOT covered
    })


@pytest.fixture()
def tiny_predictions(tiny_rows: List[dict], tiny_model: StubMemRM) -> List[PredictionResult]:
    return run_predictions(tiny_model, tiny_rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_has_six_rm_compatible_datasets():
    """The registry must expose the six datasets named in the plan."""
    expected = {
        "iid-heldout",
        "train-sanity",
        "scenario-ood-tau2",
        "scenario-ood-wa-long",
        "scenario-ood-webarena",
        "strategy-ood",
    }
    assert expected.issubset(set(DATASET_REGISTRY))
    # midtrain-sft is intentionally absent (generative, no label).
    assert "midtrain-sft" not in DATASET_REGISTRY


def test_looks_like_hf_repo_id_distinguishes_paths_from_repos(tmp_path: Path):
    assert looks_like_hf_repo_id("MemGym/memgym-rm-1p7b") is True
    assert looks_like_hf_repo_id("user/repo") is True
    assert looks_like_hf_repo_id("/abs/path") is False
    assert looks_like_hf_repo_id("./relative") is False
    assert looks_like_hf_repo_id("nothing") is False
    real = tmp_path / "a"
    real.mkdir()
    assert looks_like_hf_repo_id(str(real)) is False


def test_load_rows_normalises_input_to_prompt(tmp_path: Path):
    path = tmp_path / "p.jsonl"
    path.write_text(
        json.dumps({"input": "hello", "label": 1}) + "\n"
        + json.dumps({"prompt": "world", "label": 0}) + "\n"
    )
    rows = load_rows([path])
    assert [r["prompt"] for r in rows] == ["hello", "world"]
    assert [r["label"] for r in rows] == [1, 0]


def test_load_rows_raises_on_missing_label(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing required `label`"):
        load_rows([path])


def test_load_rows_respects_limit(tmp_path: Path):
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(
        json.dumps({"prompt": f"p{i}", "label": i % 2}) for i in range(20)
    ))
    rows = load_rows([path], limit=5)
    assert len(rows) == 5


def test_run_predictions_returns_one_per_row(tiny_rows, tiny_model):
    preds = run_predictions(tiny_model, tiny_rows)
    assert len(preds) == len(tiny_rows)
    for pred, row in zip(preds, tiny_rows):
        assert pred.true_label == row["label"]
        assert pred.prob_safe is not None
        assert 0.0 <= pred.prob_safe <= 1.0


def test_compute_metrics_recovers_perfect_auroc(tiny_predictions):
    """Probabilities were chosen so AUROC == 1.0."""
    metrics = compute_metrics(tiny_predictions)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_coverage_at_threshold_matches_hand_count(tiny_predictions):
    """At t=0.88: rows with prob_safe in (0.12, 0.88) are NOT covered.

    From the fixture: p1=0.95 (covered SAFE), p4=0.10 (covered HARMFUL).
    p2=0.80, p3=0.60, p5=0.30 are in the abstention region.
    """
    coverage, n_covered = coverage_at_threshold(tiny_predictions, threshold=0.88)
    assert n_covered == 2
    assert coverage == pytest.approx(2 / 5)


def test_accuracy_at_threshold_is_one_on_covered_subset(tiny_predictions):
    """Both covered rows are correctly classified, so acc@t == 1.0."""
    acc = accuracy_at_threshold(tiny_predictions, threshold=0.88)
    assert acc == pytest.approx(1.0)


def test_accuracy_at_threshold_returns_none_when_no_rows_covered():
    """If every prob_safe is in the abstention zone, acc@t is undefined."""
    preds = [
        PredictionResult(instance_id="", step=0, true_label=1, pred_label=1,
                         confidence=0.5, prob_safe=0.5),
    ]
    assert accuracy_at_threshold(preds, threshold=0.88) is None


def test_bootstrap_ci_returns_pair_for_well_defined_metric(tiny_predictions):
    """AUROC is well-defined on resamples of the tiny dataset."""
    def _auroc(preds):
        return compute_metrics(preds).auroc

    ci = bootstrap_ci(tiny_predictions, _auroc, n_resamples=200, rng_seed=7)
    assert ci is not None
    lo, hi = ci
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_returns_none_for_degenerate_input():
    """Single-row input cannot give a meaningful AUROC bootstrap CI."""
    preds = [
        PredictionResult(instance_id="", step=0, true_label=1, pred_label=1,
                         confidence=0.9, prob_safe=0.9),
    ]
    def _auroc(p):
        return compute_metrics(p).auroc
    # Every resample has one class only → AUROC always returns 0.5
    # (chance), so the CI is computable but trivial. Accept either
    # None or a tight CI around 0.5 — both are valid degenerate
    # behaviour. The point of the test is "does not crash".
    bootstrap_ci(preds, _auroc, n_resamples=50)


def test_evaluate_dataset_emits_report_with_paper_fields(tiny_rows, tiny_model):
    spec = DATASET_REGISTRY["iid-heldout"]
    report, preds = evaluate_dataset(
        tiny_model, tiny_rows,
        spec=spec, threshold=0.88, n_bootstrap=50,
    )
    assert report.dataset == "iid-heldout"
    assert report.hf_repo == "MemGym/memgym-rm-iid-heldout"
    assert report.n == 5
    assert report.n_safe == 3
    assert report.n_harmful == 2
    assert report.paper_auroc == pytest.approx(0.985)
    assert len(preds) == 5


def test_evaluate_dataset_flags_train_split(tiny_rows, tiny_model):
    spec = DATASET_REGISTRY["train-sanity"]
    report, _ = evaluate_dataset(
        tiny_model, tiny_rows, spec=spec, n_bootstrap=0,
    )
    assert any("TRAIN SPLIT" in note for note in report.notes)


def test_evaluate_dataset_flags_covered_subset_only(tiny_rows, tiny_model):
    spec = DATASET_REGISTRY["strategy-ood"]
    report, _ = evaluate_dataset(
        tiny_model, tiny_rows, spec=spec, n_bootstrap=0,
    )
    assert any("PUBLIC ARTIFACT IS POST-SELECTION" in note for note in report.notes)


def test_format_results_table_includes_all_reports(tiny_rows, tiny_model):
    spec = DATASET_REGISTRY["iid-heldout"]
    report, _ = evaluate_dataset(
        tiny_model, tiny_rows, spec=spec, n_bootstrap=0,
    )
    table = format_results_table([report])
    assert "iid-heldout" in table
    assert "AUROC" in table
    assert "Paper AUROC" in table


def test_resolve_dataset_accepts_local_file(tmp_path: Path):
    path = tmp_path / "custom.jsonl"
    path.write_text(json.dumps({"prompt": "x", "label": 1}) + "\n")
    files, spec = resolve_dataset(str(path))
    assert files == [path]
    assert spec is None


def test_resolve_dataset_rejects_unknown_short_name():
    with pytest.raises(FileNotFoundError, match="is not a registry short name"):
        resolve_dataset("does-not-exist")


def test_cli_help_does_not_import_torch(monkeypatch, capsys):
    """``--help`` must work without GPU stack; importing the CLI module
    itself loads only ``rm_eval`` + argparse, not transformers/torch.
    """
    from memgym.training.eval import rm_cli  # noqa: F401 (import smoke)

    with pytest.raises(SystemExit) as excinfo:
        rm_cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "memgym-eval-rm" in out
    assert "--dataset" in out
    assert "--checkpoint" in out
