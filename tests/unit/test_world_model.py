"""Unit tests for ``MemoryWorldModel``, ``WorldModelGate``, and
``WorldModelEvaluator``.

We avoid loading real Qwen weights, downloading from HF, or touching a
GPU. Where the system under test reaches into ``transformers`` or
``peft``, we patch the entry point and inject deterministic stand-ins.

The tests cover:

* ``MemoryWorldModel.predict_logits_text`` — verifies the divergence-
  position decoding + softmax pipeline against a hand-computable
  fake tokenizer / fake forward.
* ``WorldModelGate`` — verifies it wires recorded/proposed actions
  into the prompt template and surfaces ``prob_safe``.
* ``WorldModelEvaluator.evaluate`` and
  ``WorldModelEvaluator.evaluate_predictions`` — verifies routing
  over a tiny JSONL with a stub ``predict_logits``-bearing model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# torch is a hard dep for ``world_model``; skip the whole module on
# bare-bones CI runners that omit it.
torch = pytest.importorskip("torch")

from memgym.training.models.evaluator import (
    PredictionResult,
    WorldModelEvaluator,
    compute_metrics,
)
from memgym.training.models.world_model import MemoryWorldModel, ModelConfig


# ---------------------------------------------------------------------------
# Fake transformers stand-ins
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer that maps each character to its ord() value.

    Crucially, ``" Y"`` and ``" N"`` encode to *different first tokens*
    after the shared prefix, exactly matching the divergence-decoding
    contract that ``predict_logits_text`` relies on.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) for c in text]


class _FakeOutputs:
    def __init__(self, logits: "torch.Tensor"):
        self.logits = logits


class _FakeForwardModel:
    """Pretends to be a ``transformers.PreTrainedModel`` forward call.

    The vocabulary is sized to fit ord(' Y') and ord(' N'); the logit at
    position ``safe_id`` is set high to force ``prob_safe ≈ 1.0`` so the
    test has a closed-form expected value.
    """

    def __init__(self, safe_id: int, harm_id: int, safe_logit: float, harm_logit: float):
        self._safe_id = safe_id
        self._harm_id = harm_id
        self._safe_logit = safe_logit
        self._harm_logit = harm_logit
        self.device = torch.device("cpu")
        # ``MemoryWorldModel`` calls ``.eval()`` on the model before each
        # forward; record the call so tests can assert it happened.
        self.eval_called = False

    def eval(self) -> "_FakeForwardModel":
        self.eval_called = True
        return self

    def __call__(self, input_ids: "torch.Tensor", **kwargs: Any) -> _FakeOutputs:
        vocab = max(self._safe_id, self._harm_id) + 1
        logits = torch.full((1, input_ids.shape[1], vocab), -1e3)
        logits[0, -1, self._safe_id] = self._safe_logit
        logits[0, -1, self._harm_id] = self._harm_logit
        return _FakeOutputs(logits)


def _make_wired_model(safe_logit: float, harm_logit: float) -> MemoryWorldModel:
    """Construct a ``MemoryWorldModel`` with fakes pre-wired in.

    Bypasses ``setup()`` / ``from_checkpoint`` so the test doesn't touch
    HF, torch CUDA, or PEFT.
    """
    cfg = ModelConfig(safe_completion=" Y", harmful_completion=" N", load_in_4bit=False)
    wm = MemoryWorldModel(cfg)
    wm.tokenizer = _FakeTokenizer()
    safe_id = ord(" Y"[-1])    # 'Y' = 89
    harm_id = ord(" N"[-1])    # 'N' = 78
    wm.model = _FakeForwardModel(safe_id, harm_id, safe_logit, harm_logit)
    wm._safe_token_id = safe_id
    wm._harmful_token_id = harm_id
    wm._is_setup = True
    return wm


# ---------------------------------------------------------------------------
# MemoryWorldModel.predict_logits_text
# ---------------------------------------------------------------------------

def test_predict_logits_text_returns_safe_when_safe_logit_dominates():
    """High SAFE logit ⇒ prob_safe ≈ 1.0, label='SAFE'."""
    wm = _make_wired_model(safe_logit=10.0, harm_logit=-10.0)
    label, prob_safe = wm.predict_logits_text("compress event:")
    assert label == "SAFE"
    assert prob_safe == pytest.approx(1.0, abs=1e-6)
    assert wm.model.eval_called is True


def test_predict_logits_text_returns_harmful_when_harm_logit_dominates():
    """High HARMFUL logit ⇒ prob_safe ≈ 0.0, label='HARMFUL'."""
    wm = _make_wired_model(safe_logit=-10.0, harm_logit=10.0)
    label, prob_safe = wm.predict_logits_text("compress event:")
    assert label == "HARMFUL"
    assert prob_safe == pytest.approx(0.0, abs=1e-6)


def test_predict_logits_text_softmax_is_symmetric():
    """Equal logits ⇒ prob_safe == 0.5 exactly."""
    wm = _make_wired_model(safe_logit=0.0, harm_logit=0.0)
    _, prob_safe = wm.predict_logits_text("compress event:")
    assert prob_safe == pytest.approx(0.5, abs=1e-6)


def test_predict_logits_text_respects_threshold():
    """``safe_threshold=0.9`` should reject borderline-SAFE probs."""
    # logit_diff=ln(2) ⇒ prob_safe = 2/3 ≈ 0.667. Below 0.9 threshold.
    wm = _make_wired_model(safe_logit=math.log(2.0), harm_logit=0.0)
    label, prob_safe = wm.predict_logits_text(
        "compress event:", safe_threshold=0.9,
    )
    assert prob_safe == pytest.approx(2 / 3, abs=1e-6)
    assert label == "HARMFUL"


def test_predict_logits_text_raises_if_not_setup():
    wm = MemoryWorldModel(ModelConfig())
    with pytest.raises(RuntimeError, match="setup"):
        wm.predict_logits_text("anything")


def test_predict_logits_text_raises_when_labels_share_full_prefix():
    """If SAFE and HARMFUL encodings never diverge, we cannot classify."""
    wm = _make_wired_model(safe_logit=0.0, harm_logit=0.0)

    class _DegenerateTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            # Both labels collapse to the same prefix → divergence loop
            # walks off the end of ``min_len``.
            return [1, 2, 3]

    wm.tokenizer = _DegenerateTokenizer()
    with pytest.raises(ValueError, match="share a full prefix"):
        wm.predict_logits_text("doesn't matter")


# ---------------------------------------------------------------------------
# WorldModelGate
# ---------------------------------------------------------------------------

class _GateStubMemRM:
    """Stand-in for ``MemoryWorldModel`` consumed by ``WorldModelGate``.

    Records the prompt the gate built so tests can assert that it
    includes the recorded + proposed actions verbatim.
    """

    def __init__(self, prob_safe: float):
        self.prob_safe = prob_safe
        self.last_text: str = ""

    def predict_logits_text(
        self, text: str, safe_threshold: float = 0.5,
    ) -> Tuple[str, float]:
        self.last_text = text
        return ("SAFE" if self.prob_safe >= safe_threshold else "HARMFUL"), self.prob_safe


@pytest.fixture()
def patched_gate(monkeypatch):
    """Factory that builds a ``WorldModelGate`` with a stub classifier."""
    from memgym.training.models import world_model_gate as wmg

    def _factory(prob_safe: float = 0.91, threshold: float = 0.78):
        stub = _GateStubMemRM(prob_safe)
        monkeypatch.setattr(
            wmg.MemoryWorldModel, "from_checkpoint",
            classmethod(lambda cls, ckpt, cfg: stub),
        )
        gate = wmg.WorldModelGate(checkpoint="fake-ckpt", threshold=threshold)
        return gate, stub

    return _factory


def test_world_model_gate_score_emits_prompt_with_actions(patched_gate):
    gate, stub = patched_gate(prob_safe=0.91)
    prob = gate.score(
        recorded_action="ls -la",
        proposed_action="rm -rf /",
        step=3,
        perturbation="redaction",
    )
    assert prob == pytest.approx(0.91)
    assert "ls -la" in stub.last_text
    assert "rm -rf /" in stub.last_text
    # The Answer: response prefix is appended so the classifier scores
    # the next-token slot. Without it the divergence position drifts.
    assert stub.last_text.rstrip().endswith("Answer:")


def test_world_model_gate_decision_respects_threshold(patched_gate):
    gate, _ = patched_gate(prob_safe=0.50, threshold=0.78)

    class _AgentModel:
        def __init__(self):
            self.calls = 0

        def query(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls += 1
            # Return a properly bracketed bash block so ``_extract_action``
            # produces a non-empty proposed/recorded action.
            return {"content": "```bash\necho hi\n```"}

    agent = _AgentModel()
    result = gate.gate(
        agent, full_history=[{"role": "user", "content": "x"}],
        compressed_history=[{"role": "user", "content": "x"}],
    )
    assert result["decision"] is False    # 0.50 < 0.78
    assert result["prob_safe"] == pytest.approx(0.50)
    assert result["recorded_action"] == "echo hi"
    assert result["proposed_action"] == "echo hi"
    assert agent.calls == 2
    # Latency dict has all four required keys.
    for key in ("agent_full", "agent_compressed", "classifier", "total"):
        assert key in result["latency_ms"]


def test_world_model_gate_returns_proposed_response_for_reuse(patched_gate):
    """The gate's caller skips a second agent call by reusing the
    ``proposed_response`` it already produced."""
    gate, _ = patched_gate(prob_safe=0.95)

    class _AgentModel:
        def query(self, _msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"content": "```bash\ntrue\n```", "stop_reason": "tool_calls"}

    out = gate.gate(_AgentModel(), full_history=[], compressed_history=[])
    assert out["proposed_response"]["stop_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# WorldModelEvaluator
# ---------------------------------------------------------------------------

class _LogitsStubModel:
    """Stand-in with ``predict_logits`` (chat-template path).

    Returns ``(label, prob_safe)`` from a fixed table keyed by the user
    turn so we can craft a JSONL with known ground-truth outcomes.
    """

    def __init__(self, table: Dict[str, float]):
        self.table = table

    def predict_logits(
        self, messages: List[Dict[str, str]],
    ) -> Tuple[str, float]:
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        prob_safe = self.table[user_msg]
        return ("SAFE" if prob_safe >= 0.5 else "HARMFUL"), prob_safe


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_evaluator_evaluate_routes_through_predict_logits(tmp_path: Path):
    """JSONL ➝ predict_logits ➝ EvalMetrics with populated AUROC."""
    records = [
        {"messages": [{"role": "user", "content": "u1"},
                      {"role": "assistant", "content": "SAFE"}],
         "metadata": {"instance_id": "i1", "step": 0, "label": 1}},
        {"messages": [{"role": "user", "content": "u2"},
                      {"role": "assistant", "content": "HARMFUL"}],
         "metadata": {"instance_id": "i2", "step": 0, "label": 0}},
    ]
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, records)

    model = _LogitsStubModel({"u1": 0.92, "u2": 0.07})
    evaluator = WorldModelEvaluator(model)
    metrics = evaluator.evaluate(str(path))
    assert metrics.total == 2
    assert metrics.accuracy == pytest.approx(1.0)
    # Both probabilities are decisive → AUROC = 1.0 (perfect separation).
    assert metrics.auroc == pytest.approx(1.0)
    # ECE is populated when prob_safe is available.
    assert metrics.ece is not None


def test_evaluator_evaluate_predictions_skips_model(tmp_path: Path):
    """``evaluate_predictions`` is the inference-free fast path."""
    preds = [
        PredictionResult(instance_id="i1", step=0, true_label=1, pred_label=1,
                         confidence=0.9, prob_safe=0.9),
        PredictionResult(instance_id="i2", step=0, true_label=0, pred_label=0,
                         confidence=0.9, prob_safe=0.1),
    ]
    metrics = WorldModelEvaluator(model=None).evaluate_predictions(preds)
    assert metrics.total == 2
    assert metrics.accuracy == pytest.approx(1.0)


def test_evaluator_evaluate_raises_without_model(tmp_path: Path):
    """Calling ``evaluate()`` with no model is a programmer error."""
    path = tmp_path / "x.jsonl"
    path.write_text("")
    with pytest.raises(RuntimeError, match="Model required"):
        WorldModelEvaluator(model=None).evaluate(str(path))


def test_evaluator_respects_filter_split(tmp_path: Path):
    """Only rows whose ``split`` matches the requested filter are scored."""
    records = [
        {"split": "train",
         "messages": [{"role": "user", "content": "u1"},
                      {"role": "assistant", "content": "SAFE"}],
         "metadata": {"label": 1}},
        {"split": "iid_test",
         "messages": [{"role": "user", "content": "u2"},
                      {"role": "assistant", "content": "HARMFUL"}],
         "metadata": {"label": 0}},
    ]
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, records)

    model = _LogitsStubModel({"u1": 0.9, "u2": 0.1})
    metrics = WorldModelEvaluator(model).evaluate(
        str(path), filter_split="iid_test",
    )
    assert metrics.total == 1
    assert metrics.confusion[3] == 1   # TP for HARMFUL


def test_evaluator_respects_max_examples(tmp_path: Path):
    records = [
        {"messages": [{"role": "user", "content": f"u{i}"},
                      {"role": "assistant", "content": "SAFE"}],
         "metadata": {"label": 1}}
        for i in range(10)
    ]
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, records)
    model = _LogitsStubModel({f"u{i}": 0.9 for i in range(10)})
    metrics = WorldModelEvaluator(model).evaluate(str(path), max_examples=3)
    assert metrics.total == 3


def test_compute_metrics_populates_ece_only_when_prob_safe_present():
    """No ``prob_safe`` ⇒ ECE stays ``None`` (it's the calibration metric)."""
    preds_no_prob = [
        PredictionResult(instance_id="i1", step=0, true_label=1, pred_label=1,
                         confidence=0.6, prob_safe=None),
    ]
    metrics = compute_metrics(preds_no_prob)
    assert metrics.ece is None
