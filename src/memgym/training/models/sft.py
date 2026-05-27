"""SFT trainer for the augmentation-pair dataset.

Plain supervised CE on the SAFE/HARMFUL completion
token via TRL ≥1.0's native `completion_only_loss=True`. Tokenizer
masks the prompt; loss fires only on the 1-token answer.

Why SFT here:
  * Deterministic ground-truth labels per pair — no scorer indirection.
  * CE loss is well-defined for every example; no policy-gradient
    `frac_reward_zero_std → 1.0` failure mode.
  * Class imbalance (≈88% HARMFUL) hurts SFT less than policy-gradient RL because
    each example contributes equally regardless of label; but eval
    must still check per-class F1 before declaring success.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

DEFAULT_PAIRS = Path("training_output/aug_sft_pairs.jsonl")
DEFAULT_OUT = Path("checkpoints/aug_sft")
DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"

# Must match the RESPONSE_PREFIX in aug_pairs.py. Kept fixed (not a
# chat-template) so the trainer is tokenizer-agnostic across base
# models — the recipe's design choice. No trailing space: the label
# strings (" Y"/" N") carry their own leading space so they stay
# single-token under Qwen3 byte-level BPE.
RESPONSE_PREFIX = "\n\nAnswer:"


def _validate_pairs(pairs_path: Path, limit: int = 0) -> List[dict]:
    """Load + validate the SFT-pairs JSONL.

    Required keys: `prompt`, `completion`, `label`, `trajectory_id`.
    (The adapter also emits `input`/`target` for legacy callers; those
    are not checked here.)
    """
    rows: List[dict] = []
    for line_idx, line in enumerate(pairs_path.read_text().splitlines()):
        if limit and len(rows) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for key in ("prompt", "completion", "label", "trajectory_id"):
            if key not in record:
                raise ValueError(f"aug_sft_pairs.jsonl row {line_idx} missing key {key!r}")
        rows.append(record)
    if not rows:
        raise ValueError(f"aug_sft_pairs.jsonl at {pairs_path} is empty")
    return rows


def _train_weighted(
    train_rows: List[dict],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    peft_config: Any,
    save_dir: Path,
    eval_rows: List[dict] | None = None,
) -> float:
    """HF Trainer path with per-row inverse-frequency class weights.

    Pre-tokenizes each row with `tokenize_with_completion_mask` (loss
    only on tokens after `RESPONSE_PREFIX`) and attaches a scalar
    `class_weight` per row, which the `WeightedCollator` forwards as a
    batched 1-D tensor into `WeightedTrainer.compute_loss`.

    `eval_rows` (optional) is tokenized with the same pipeline and used
    as `eval_dataset` so HF Trainer streams `eval/*` metrics live; in
    weighted-CE eval the row's class weight is forwarded but not used by
    the standard `Trainer.evaluate()` codepath (only `compute_loss` reads
    it, and we override the loss only at training time).
    """
    from datasets import Dataset
    from transformers import TrainingArguments

    from memgym.training.models.evaluator import make_trainer_compute_metrics
    from memgym.training.models.weighted_trainer import (
        WeightedCollator,
        WeightedTrainer,
    )

    cap = float(getattr(args, "class_weight_cap", 3.0))
    labels = [int(r.get("label", 0)) for r in train_rows]
    weights_by_label = _compute_class_weights(labels, cap)
    logger.info(
        "class-weighted CE: cap=%.2f weights={0: %.3f, 1: %.3f} "
        "(n_safe=%d n_harmful=%d)",
        cap, weights_by_label[0], weights_by_label[1],
        sum(1 for lb in labels if lb == 1),
        sum(1 for lb in labels if lb == 0),
    )

    # Apply LoRA wrapping manually — we're not going through TRL.
    # With gradient_checkpointing + frozen base weights, the reentrant
    # checkpoint boundary drops our input_require_grads hook's grad
    # modification, so loss.backward() returns all-zero grads and
    # training stalls at loss=0. Mirror TRL's pattern:
    #   1. LoRA-wrap the model
    #   2. Enable checkpointing with use_reentrant=False (stateless API
    #      respects forward hooks)
    #   3. Call enable_input_require_grads() AFTER ckpt wrapping so the
    #      hook sits on the outermost forward.
    # We then set `gradient_checkpointing=False` in TrainingArguments so
    # HF Trainer doesn't re-wrap the model and orphan our hook.
    ckpt_on = bool(getattr(args, "gradient_checkpointing", False))
    qlora_on = bool(getattr(args, "load_in_4bit", False))
    if peft_config is not None:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        if ckpt_on and not qlora_on:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            logger.info(
                "manually enabled gradient_checkpointing(use_reentrant=False) "
                "+ enable_input_require_grads() on PEFT model"
            )
        elif ckpt_on and qlora_on:
            logger.info(
                "QLoRA path: prepare_model_for_kbit_training already wrapped "
                "gradient_checkpointing + enable_input_require_grads; skipping "
                "the manual block."
            )
        # Sanity check visible even under tail-truncated logs.
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"WEIGHTED-PATH trainable={n_trainable:,} / total={n_total:,} "
            f"({100 * n_trainable / max(n_total, 1):.3f}%)",
            flush=True,
        )

    # Tokenize prompt and completion separately — the boundary is
    # `len(prompt_ids)`, deterministic under any BPE merges. Avoids the
    # failure mode where `"\n\nAnswer:"` tokenizes differently standalone
    # than in-context under Qwen3 byte-level BPE (response_template
    # search would miss and mask the completion out entirely).
    def _tok(rows: List[dict], tag: str) -> List[dict]:
        n_empty_completion = 0
        n_truncated = 0
        out_rows: List[dict] = []
        for r in rows:
            prompt_ids: List[int] = tokenizer(
                r["prompt"], add_special_tokens=False,
            )["input_ids"]
            completion_ids: List[int] = tokenizer(
                r["completion"], add_special_tokens=False,
            )["input_ids"]
            if not completion_ids:
                n_empty_completion += 1
                continue
            # Left-truncate the prompt if the concatenated length exceeds the
            # cap, so the single-token completion is always preserved.
            overflow = len(prompt_ids) + len(completion_ids) - args.max_length
            if overflow > 0:
                prompt_ids = prompt_ids[overflow:]
                n_truncated += 1
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + list(completion_ids)
            out_rows.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "class_weight": float(weights_by_label[int(r.get("label", 0))]),
            })
        logger.info(
            "weighted tokenization (%s): rows=%d empty_completion=%d truncated=%d "
            "sample_label_tokens=%d",
            tag, len(out_rows), n_empty_completion, n_truncated,
            sum(1 for x in out_rows[0]["labels"] if x != -100) if out_rows else 0,
        )
        return out_rows

    train_ds = Dataset.from_list(_tok(train_rows, "train"))
    eval_ds = None
    if eval_rows:
        eval_ds = Dataset.from_list(_tok(eval_rows, "eval"))

    wandb_project = getattr(args, "wandb_project", None)
    wandb_run_name = getattr(args, "wandb_run_name", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name
    report_to = ["wandb"] if wandb_project else []

    eval_strategy = getattr(args, "evaluation_strategy", "no")
    eval_steps = int(getattr(args, "eval_steps", 50))
    eval_batch_size = int(getattr(args, "eval_batch_size", 0)) or args.batch_size
    if eval_ds is None:
        eval_strategy = "no"

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        save_strategy="no" if args.no_save else "steps",
        save_steps=int(getattr(args, "save_steps", None) or args.max_steps),
        save_total_limit=getattr(args, "save_total_limit", None),
        bf16=args.bf16,
        report_to=report_to,
        run_name=wandb_run_name or None,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        # We already enabled ckpt manually above with use_reentrant=False.
        # If HF Trainer re-enables here it re-wraps the model and orphans
        # our enable_input_require_grads() hook.
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    collator = WeightedCollator(pad_token_id=tokenizer.pad_token_id)
    preprocess_fn, compute_fn = make_trainer_compute_metrics(tokenizer)
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
        trainer_kwargs["compute_metrics"] = compute_fn
        # NOTE: do NOT pass preprocess_logits_for_metrics here. We move the
        # reduction into WeightedTrainer.prediction_step (below). HF's
        # evaluation_loop would otherwise call the reducer a second time
        # on the already-(B, 2) tensor, indexing past dim=1's size-2 with
        # pred_idx values in [0, T) — which fires a CUDA index-out-of-
        # bounds assertion at the first eval.
    trainer = WeightedTrainer(**trainer_kwargs)
    # Long-context eval OOMs in HF's evaluation_loop because it pads raw
    # (B, T, V) logits before calling preprocess_logits_for_metrics. We
    # do the reduction inside prediction_step so the tensor that leaves
    # prediction_step is already (B, 2) — pad and gather then trivial.
    if eval_ds is not None:
        trainer.eval_logit_reducer = preprocess_fn
    train_result = trainer.train()

    if not args.no_save:
        trainer.save_model(str(save_dir))
        # HF Trainer.save_model saves tokenizer only if `processing_class`
        # was set; belt-and-braces: save explicitly so eval_aug_sft's
        # `AutoTokenizer.from_pretrained(checkpoint)` always succeeds.
        tokenizer.save_pretrained(str(save_dir))

    log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
    for entry in reversed(log_history):
        if "loss" in entry:
            return float(entry["loss"])
    metrics = getattr(train_result, "metrics", {}) or {}
    return float(metrics.get("train_loss", 0.0))


def _compute_class_weights(labels: List[int], cap: float) -> dict[int, float]:
    """sklearn-style `balanced` weights with symmetric cap.

    `w_c = clamp(N / (K * N_c), 1/cap, cap)`. Capping bounds both
    upweighting minority and downweighting majority, which keeps the
    loss scale comparable across labels at high imbalance (here ~88:12)
    without letting a single SAFE row dominate a step's gradient.
    """
    n_total = len(labels)
    n_safe = sum(1 for lbl in labels if int(lbl) == 1)
    n_harmful = n_total - n_safe
    if n_safe == 0 or n_harmful == 0:
        return {0: 1.0, 1: 1.0}
    raw = {0: n_total / (2 * n_harmful), 1: n_total / (2 * n_safe)}
    lo, hi = 1.0 / cap, cap
    return {k: max(lo, min(hi, v)) for k, v in raw.items()}


def _train_sft(rows: List[dict], args: argparse.Namespace) -> float:
    """Train TRL SFTTrainer on the provided rows. Returns final loss.

    Three loss paths, selected by CLI:
      * default: TRL `SFTTrainer` with `completion_only_loss=True`
      * `--focal-gamma > 0`: TRL subclass with `(1-p_t)^γ * NLL`
      * `--balance-classes`: HF `WeightedTrainer` with per-row
        inverse-frequency class weights (cap `--class-weight-cap`).
        Bypasses TRL because TRL drops non-standard columns.
    """
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SFT trainer requires `torch`, `transformers`, `datasets`. "
            "Install: pip install torch transformers trl datasets accelerate"
        ) from exc

    balance_classes = bool(getattr(args, "balance_classes", False))
    focal_gamma = float(getattr(args, "focal_gamma", 0.0) or 0.0)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = args.save_dir or out_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = bool(getattr(args, "load_in_4bit", False))
    model_kwargs: dict[str, Any] = {}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-4bit requires bitsandbytes; install with "
                "`pip install bitsandbytes`"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # NF4 + DDP: bnb materializes quantized blocks at load time onto
        # whatever device CUDA defaults to — without a per-rank pin, every
        # torchrun worker piles onto cuda:0 and OOMs before training starts.
        # The bf16-LoRA path is fine without this because bf16 weights stay
        # on CPU until Accelerate moves them; NF4 weights cannot move
        # post-quant. So for QLoRA we explicitly pin device_map={"": LOCAL_RANK}.
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            torch.cuda.set_device(local_rank)
            model_kwargs["device_map"] = {"": local_rank}
            logger.info("QLoRA+DDP: pinning rank %d to cuda:%d", local_rank, local_rank)
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if load_in_4bit:
        # kbit-prep handles gradient_checkpointing + enable_input_require_grads
        # in one call; the manual ckpt blocks below (TRL SFTConfig and the
        # weighted path) detect `args.load_in_4bit` and stand down.
        from peft import prepare_model_for_kbit_training
        if not getattr(args, "use_lora", False):
            raise ValueError(
                "--load-in-4bit requires --use-lora: NF4-quantized base weights "
                "are frozen, so trainable LoRA adapters are mandatory."
            )
        # use_reentrant=False is required for DDP + LoRA: with reentrant
        # checkpointing the LoRA-adapter params get autograd-hooks fired
        # twice per backward pass, and DDP's "mark variable ready" tracker
        # raises "marked as ready twice" on every param. Non-reentrant
        # checkpoint plays correctly with DDP because the autograd graph
        # is built statically and the gradient hook fires once.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        logger.info("loaded model in NF4 + ran prepare_model_for_kbit_training "
                    "(use_gradient_checkpointing=%s, use_reentrant=False)",
                    bool(getattr(args, "gradient_checkpointing", False)))

    peft_config = None
    if getattr(args, "use_lora", False):
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise RuntimeError(
                "--use-lora requires peft; install with `pip install peft`"
            ) from exc
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Filter to training rows only. The adapter stamps each row with
    # split ∈ {"train","eval"}; rows missing the field fall back to
    # train (back-compat with older adapter output).
    train_rows = [r for r in rows if r.get("split", "train") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    logger.info(
        "sft rows: total=%d train=%d eval=%d unsplit=%d",
        len(rows), len(train_rows), len(eval_rows),
        len(rows) - len(train_rows) - len(eval_rows),
    )
    if not train_rows:
        raise ValueError("no train rows after split filter")

    if balance_classes:
        return _train_weighted(
            train_rows, args, model, tokenizer, peft_config, save_dir,
            eval_rows=eval_rows,
        )

    # TRL path (default or focal). Lazy import so the weighted path
    # stays usable in TRL-less environments.
    from memgym.training.models.evaluator import make_trainer_compute_metrics
    from trl import SFTConfig, SFTTrainer

    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in train_rows]
    )
    eval_ds = None
    if eval_rows:
        eval_ds = Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in eval_rows]
        )

    wandb_project = getattr(args, "wandb_project", None)
    wandb_run_name = getattr(args, "wandb_run_name", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name
    report_to = ["wandb"] if wandb_project else []

    eval_strategy = getattr(args, "evaluation_strategy", "no")
    eval_steps = int(getattr(args, "eval_steps", 50))
    eval_batch_size = int(getattr(args, "eval_batch_size", 0)) or args.batch_size
    if eval_ds is None:
        eval_strategy = "no"

    import inspect
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    candidate_kwargs: dict[str, Any] = {
        "output_dir": str(save_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": 1,
        "save_strategy": "no" if args.no_save else "steps",
        "save_steps": int(getattr(args, "save_steps", None) or args.max_steps),
        "save_total_limit": getattr(args, "save_total_limit", None),
        "bf16": args.bf16,
        "report_to": report_to,
        "run_name": wandb_run_name or None,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "completion_only_loss": True,
        "packing": False,
        # Under QLoRA, prepare_model_for_kbit_training already enabled
        # checkpointing on the base; letting SFTConfig re-enable it would
        # re-wrap the model and orphan the input-require-grads hook (the
        # same failure mode handled in the weighted path).
        "gradient_checkpointing": (
            False if load_in_4bit
            else getattr(args, "gradient_checkpointing", False)
        ),
    }
    cfg = SFTConfig(**{k: v for k, v in candidate_kwargs.items() if k in accepted})

    preprocess_fn, compute_fn = make_trainer_compute_metrics(tokenizer)
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=train_ds,
    )
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds
        trainer_kwargs["compute_metrics"] = compute_fn
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_fn
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    if focal_gamma > 0.0:
        from memgym.training.models.weighted_trainer import build_focal_sft_trainer_cls
        trainer_cls = build_focal_sft_trainer_cls(SFTTrainer, focal_gamma)
        logger.info("using focal-loss SFTTrainer (gamma=%.2f)", focal_gamma)
    else:
        trainer_cls = SFTTrainer

    trainer = trainer_cls(**trainer_kwargs)
    train_result = trainer.train()

    if not args.no_save:
        trainer.save_model(str(save_dir))

    log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
    for entry in reversed(log_history):
        if "loss" in entry:
            return float(entry["loss"])
    metrics = getattr(train_result, "metrics", {}) or {}
    return float(metrics.get("train_loss", 0.0))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Phase-C cell 1: per-row class-weighted CE via HF Trainer "
                             "(bypasses TRL to keep the class_weight column).")
    parser.add_argument("--class-weight-cap", type=float, default=3.0,
                        help="Symmetric cap on balanced weights; 3.0 turns ~88:12 into "
                             "a 3:0.33 ratio (minority boost capped to avoid a single "
                             "SAFE row dominating a step's gradient).")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Phase-C cell 2: TRL SFTTrainer subclass applying "
                             "(1-p_t)^gamma * NLL. 0 disables (default path).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _validate_pairs(args.pairs, args.limit)
    print(json.dumps({"step": "sft_validate", "rows": len(rows)}), file=sys.stderr)

    if args.dry_run:
        print(0.0)
        return 0

    final_loss = _train_sft(rows, args)
    print(
        json.dumps({"step": "sft_trainer", "final_loss": final_loss, "rows": len(rows)}),
        file=sys.stderr,
    )
    print(final_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
