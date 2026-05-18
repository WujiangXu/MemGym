"""Orchestrator: SFT the Qwen3-8B-Base summarizer on SAFE pairs (H.2).

Wraps `memgym.training.models.sft._train_sft` without touching it —
summarizer SFT is exactly the default TRL path with
`completion_only_loss=True`, LoRA wrapping, and the classifier's
proven gradient-checkpointing order. The knobs that change from
classifier SFT are just defaults:

* `--max-length 8192` (vs classifier's 1024; summarizer inputs run
  2-6 k tokens and its completions 300-800 tokens).
* `--use-lora` on by default (explicit re-affirmation of the recipe).
* No `--balance-classes`, no `--focal-gamma` — all rows are SAFE by
  construction, so per-class weighting is not meaningful.

Usage:

    python -m memgym.training.scripts.train_summarizer_sft \
        --pairs data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl \
        --model Qwen/Qwen3-8B-Base \
        --save-dir checkpoints/summarizer_sft_8b_v1 \
        --max-steps 1200 --bf16 --flash-attn \
        --use-lora --gradient-checkpointing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from memgym.training.data.summarizer_dataset import validate_summarizer_pairs
from memgym.training.models.sft import _train_sft

logger = logging.getLogger(__name__)


DEFAULT_PAIRS = Path("data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl")
DEFAULT_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_SAVE = Path("checkpoints/summarizer_sft_8b_v1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="summarizer_pairs.jsonl from H.1.1")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="base HF model id or local checkpoint dir")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="TrainingArguments.output_dir; defaults to --save-dir")
    parser.add_argument("--max-steps", type=int, default=1200,
                        help="2x the classifier's 600 because summarizer SFT has ~2x effective "
                             "tokens/row even with smaller row count (completions are longer)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="small default — 8k-token rows + LoRA already consume ~35 GB/GPU at bs=1")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="effective batch 8 on a single GPU; scales down with --num-gpus via DDP")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="prompt + completion token cap; bumped from the classifier's 1024")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap validated rows for smoke tests (0 = all)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="enabled by default for A100s; use --no-bf16 to disable")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--flash-attn", action="store_true", default=False)
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="LoRA (r=16 α=32 q/k/v/o_proj); on by default. --no-lora to disable")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="on by default; --no-gradient-checkpointing to disable")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate pairs + print stats; skip training")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    rows = validate_summarizer_pairs(args.pairs, args.limit)
    print(json.dumps({
        "step": "summarizer_sft_validate",
        "rows": len(rows),
        "train": sum(1 for r in rows if r["split"] == "train"),
        "eval": sum(1 for r in rows if r["split"] == "eval"),
    }), file=sys.stderr)

    if args.dry_run:
        print(0.0)
        return 0

    # `_train_sft` reads args attributes directly, so build a namespace
    # that mirrors the classifier trainer's attribute set. `balance_classes`
    # and `focal_gamma` stay off — summarizer SFT is the plain TRL path.
    sft_args = SimpleNamespace(
        out_dir=args.out_dir or args.save_dir,
        save_dir=args.save_dir,
        model=args.model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        flash_attn=args.flash_attn,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        no_save=args.no_save,
        gradient_checkpointing=args.gradient_checkpointing,
        balance_classes=False,
        focal_gamma=0.0,
    )

    final_loss = _train_sft(rows, sft_args)
    print(json.dumps({
        "step": "summarizer_sft_train",
        "final_loss": final_loss,
        "rows": len(rows),
        "save_dir": str(args.save_dir),
    }), file=sys.stderr)
    print(final_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
