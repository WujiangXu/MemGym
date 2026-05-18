"""Loss-shaped SFT trainers for class-imbalanced world-model SFT.

Two variants share one file because they diverge only in `compute_loss`:

* `WeightedTrainer` — per-row class-weighted causal-LM loss. Uses HF
  `Trainer` directly (not TRL `SFTTrainer`) because TRL's
  `_prepare_dataset` drops original columns, which would silently strip
  the per-row `class_weight` we depend on. Requires pretokenized rows
  with `input_ids`/`labels` already masked for completion-only loss.

* `FocalSFTTrainer` — TRL-native subclass applying `(1 - p_t)^γ` to
  per-token NLL on completion tokens only. Keeps TRL's
  `completion_only_loss=True` masking so no separate tokenization step.

Shared helpers (`find_subsequence`, `tokenize_with_completion_mask`,
`WeightedCollator`) power the class-weighted path; focal path reuses
TRL's own dataset preparation.

Ported from commit d8f83da; focal addition is new.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return the start index of `needle` in `haystack`, or -1 if not found."""
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def tokenize_with_completion_mask(
    tokenizer,
    text: str,
    response_template: str,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    """Tokenize `text` and mask everything before `response_template` to -100."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
    )
    input_ids: List[int] = list(enc["input_ids"])
    attn: List[int] = list(enc["attention_mask"])
    labels = [-100] * len(input_ids)

    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    start = find_subsequence(input_ids, template_ids)
    if start == -1:
        logger.warning(
            "response_template not found in input — row will contribute zero loss"
        )
    else:
        first_resp = start + len(template_ids)
        for i in range(first_resp, len(input_ids)):
            labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


@dataclass
class WeightedCollator:
    """Pad a batch and forward `class_weight` as a 1-D tensor."""

    pad_token_id: int

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attn, labels, weights = [], [], [], []
        for ex in examples:
            n = len(ex["input_ids"])
            pad = max_len - n
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            attn.append(ex["attention_mask"] + [0] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            weights.append(float(ex.get("class_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_weight": torch.tensor(weights, dtype=torch.float32),
        }


class WeightedTrainer(Trainer):
    """HF Trainer with per-row class-weighted causal-LM loss."""

    # Set externally (sft.py) to reduce eval logits before HF's
    # pad_across_processes; required at long context because the raw
    # (B, T, V) logits otherwise OOM at 32K seq + 152K vocab.
    eval_logit_reducer = None

    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Reduce (B, T, V) → (B, 2) before HF's pad_across_processes.

        HF Trainer's evaluation_loop pads raw logits along dim=1 *before*
        calling preprocess_logits_for_metrics — at 32K × 152K vocab that
        allocates 9.7 GiB per batch and OOMs on a 40 GiB GPU. By doing
        the reduction here (where we still own the model output) the
        tensor that escapes prediction_step is just (B, 2), so pad and
        gather are trivial.
        """
        if self.eval_logit_reducer is None:
            # No reducer wired — fall back to base behavior.
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys,
            )

        # Keep labels for the reducer + downstream pad_across_processes.
        labels = inputs.get("labels")
        # Strip our trainer-only column so the model forward doesn't choke.
        model_inputs = {k: v for k, v in inputs.items() if k != "class_weight"}
        # Pop labels for the same reason as compute_loss: prevents Qwen3
        # from running its built-in ForCausalLMLoss on full (B*T, V).
        model_inputs.pop("labels", None)

        with torch.no_grad():
            outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        reduced = self.eval_logit_reducer(logits, labels)
        # No loss returned — we're optimizing for memory, not eval-loss.
        # compute_metrics only consumes (predictions, labels).
        return (None, reduced, labels)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        weights = inputs.pop("class_weight", None)
        # Must `pop` labels (not `get`): if labels remain in inputs, Qwen3's
        # forward triggers its built-in ForCausalLMLoss which materializes
        # the full (B*T, V) fp32 cross_entropy and OOMs at long context,
        # before our custom loss path can replace it.
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        # Gather only the (row, position) pairs that actually contribute to
        # the loss. Cross-entropy upcasts logits to fp32 internally; doing
        # that over (B*T, V) OOMs at long context (e.g. 16K × 150K vocab),
        # while completion-only masking means almost every position is
        # already -100. Materializing only the kept rows keeps the fp32
        # working set proportional to the number of completion tokens.
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        B = labels.size(0)
        if not valid.any():
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        row_idx, col_idx = valid.nonzero(as_tuple=True)
        # logits at position t predicts labels at position t+1, so the
        # kept-prediction logits live at the same col indices as the
        # already-shifted labels.
        keep_logits = logits[row_idx, col_idx]
        keep_labels = shift_labels[row_idx, col_idx]

        per_tok_loss = torch.nn.functional.cross_entropy(
            keep_logits, keep_labels, reduction="none",
        )

        per_row_loss_sum = torch.zeros(
            B, device=per_tok_loss.device, dtype=per_tok_loss.dtype,
        )
        per_row_loss_sum.scatter_add_(0, row_idx, per_tok_loss)
        token_counts = valid.sum(dim=1).clamp(min=1).to(per_row_loss_sum.dtype)
        per_row_loss = per_row_loss_sum / token_counts

        if weights is not None:
            weights = weights.to(per_row_loss.device).to(per_row_loss.dtype)
            per_row_loss = per_row_loss * weights

        loss = per_row_loss.mean()
        return (loss, outputs) if return_outputs else loss


def _focal_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute `(1 - p_t)^γ * NLL` averaged over non-masked tokens.

    Matches the standard focal-loss formulation for multiclass CE: on
    each valid (non-`-100`) token position we gather `p_t = softmax(logits)[label]`,
    scale its NLL by `(1 - p_t)^γ`, and mean-reduce across the batch's
    valid positions. Shape invariants match the teacher-forced shift:
    `logits` is `(B, T-1, V)`, `labels` is `(B, T-1)`.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100).float()
    # Replace -100 with 0 for gather; masked positions contribute zero.
    safe_labels = shift_labels.clamp(min=0)

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    p_t = torch.exp(-nll).clamp(max=1.0)
    focal_weight = (1.0 - p_t).pow(gamma)

    loss_per_tok = focal_weight * nll * mask
    return loss_per_tok.sum() / mask.sum().clamp(min=1.0)


def build_focal_sft_trainer_cls(base_cls, gamma: float):
    """Return a subclass of `base_cls` (TRL SFTTrainer) with focal CE.

    We build the class at call time so sft.py can import TRL lazily
    without forcing it at module import (keeps `weighted_trainer.py`
    importable in environments that only need the class-weighted path).
    """

    class FocalSFTTrainer(base_cls):
        def compute_loss(
            self,
            model,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None,
        ):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = _focal_loss_from_logits(outputs.logits, labels, gamma)
            return (loss, outputs) if return_outputs else loss

    FocalSFTTrainer.__name__ = f"FocalSFTTrainer_gamma{gamma}"
    return FocalSFTTrainer
