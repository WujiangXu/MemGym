"""Orchestrator: aug_pairs adapter → SFT trainer.

Single-process and torchrun-multi-GPU dispatch in one file:

  * launcher process (no RANK env) runs `convert_aug_dirs()` once,
  * then either trains in-process (`--num-gpus 1`) or re-execs under
    `torchrun --nproc_per_node=N --module memgym.training.scripts.train_sft`.

`--dry-run` validates the adapter output + prints GPU summary without
touching the trainer.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Reduce CUDA-allocator fragmentation. At 32K seq + 152K vocab the (B,T,V)
# logits tensor (9.7 GiB bf16) is right at the memory cliff; without
# expandable_segments, transient clones (e.g. DDP's _DDPSink output clone)
# can fail because reserved-but-unallocated blocks are wrong-sized.
# Must be set before torch imports.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Disable HuggingFace tokenizers' rayon thread pool. With 8 torchrun
# worker processes each forking from the parent (which has already
# initialized tokenizers internals), the rayon pool deadlocks on
# Mutex acquisition when CPU cores are contended — Diagnosis hang on
# May 6 traced 4 hung jobs all to the _tok() loop inside _train_weighted
# (sft.py:189), 0% GPU util, no NCCL output, before training began.
# Sub-second tokenization with parallelism off, vs. silent deadlock on.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Surface NCCL hangs as timeouts (5 min) instead of silent stalls. Three
# runs in a row hung at accelerator.prepare() — the model-broadcast NCCL
# allreduce — with 0% GPU util and no log output. With these env vars
# the next hang prints communicator init details and aborts at the
# timeout instead of waiting forever.
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
# Per-PR diagnosis (May 6 hang): /dev/shm has months-old stale cuda.shm
# segments + a Ray cluster occupying IPC slots; NCCL can fail P2P init
# silently in this state. Disabling P2P forces TCP fallback for the
# initial broadcast. ~5-10% throughput hit, fully recoverable signal.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

logger = logging.getLogger(__name__)

DEFAULT_AUG_ROOT = Path("augmentation_output")
DEFAULT_DIRS = [
    DEFAULT_AUG_ROOT / "sonnet_fork_gap554_scaleA_v1",
    DEFAULT_AUG_ROOT / "gptoss_fork_gap554_scaleA_v1",
    DEFAULT_AUG_ROOT / "haiku45_fork_diverse500_scaleA_v1",
]


def _patch_accelerate_skip_large_fp32_upcast(skip_bytes: int = 2 * 1024**3) -> None:
    """Skip Accelerate's bf16→fp32 upcast for outputs above ``skip_bytes``.

    HF Trainer with ``bf16=True`` calls ``Accelerator.prepare(model)`` which
    wraps ``model.forward`` in ``ConvertOutputsToFp32``. That wrapper walks
    the entire output dict (a ``CausalLMOutputWithPast``) and calls
    ``.float()`` on every fp16/bf16 tensor — including the (B, T, V) logits.

    At 32K seq with Qwen3's 152K vocab, the logits are ~9.7 GiB bf16; the
    upcast doubles that to ~19.4 GiB fp32 for a tensor we never consume in
    fp32 (our compute_loss gathers valid completion positions and lets
    cross_entropy do its own fp32 upcast on the slice). On 40GB A100s this
    is the difference between fitting and OOM at long context.

    The patch leaves smaller tensors (loss scalars, hidden states at
    pre-projection scale) untouched so HF Trainer's downstream math
    stays correct.
    """
    import accelerate.utils.operations as _ops  # noqa: PLC0415

    _orig = _ops.convert_to_fp32

    def _convert_skip_large(tensor):
        # Reuse Accelerate's own recursion machinery so dict/list/tuple/
        # dataclass containers are unwrapped exactly as before.
        import torch  # noqa: PLC0415

        def _convert(t):
            if t.element_size() * t.numel() > skip_bytes:
                return t  # leave large bf16 logit slabs alone
            return t.float()

        def _is_fp16_bf16_tensor(t):
            return (hasattr(t, "dtype") and t.dtype in (torch.float16, torch.bfloat16))

        return _ops.recursively_apply(_convert, tensor, test_type=_is_fp16_bf16_tensor)

    _ops.convert_to_fp32 = _convert_skip_large
    logger.info(
        "patched accelerate.utils.operations.convert_to_fp32 to skip tensors "
        ">%.1f GiB (was %s)",
        skip_bytes / 1024**3, _orig,
    )


def _gpu_summary() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": "missing"}
    info = {
        "torch": getattr(torch, "__version__", "?"),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if info["cuda_available"]:
        info["device_0"] = torch.cuda.get_device_name(0)
        try:
            info["cuda_version"] = torch.version.cuda
        except AttributeError:
            pass
    return info


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aug-dir", action="append", dest="aug_dirs", type=Path, default=None,
        help="Augmentation output dir (repeatable). Defaults to the 3 scaleA EC2 dirs.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("training_output/aug_sft"))
    parser.add_argument("--pairs-out", type=Path, default=None,
                        help="Path for combined aug_sft_pairs.jsonl (default: <out-dir>/aug_sft_pairs.jsonl)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Trainer checkpoint dir (default: <out-dir>/final)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--limit", type=int, default=20000,
                        help="Cap rows from the combined adapter output")
    parser.add_argument("--eval-frac", type=float, default=0.10,
                        help="Fraction of pairs (group-shuffled by repo prefix) held out for eval")
    parser.add_argument("--label-safe", default=" Y",
                        help="Single-token completion string for label=1 (e.g. ' Y', ' yes', ' good'). "
                             "Drives both the completion field and the answer hint in the rendered prompt.")
    parser.add_argument("--label-harmful", default=" N",
                        help="Single-token completion string for label=0 (e.g. ' N', ' no', ' bad')")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--flash-attn", action="store_true", default=False)
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                        help="Quantize the base model with NF4 (QLoRA). Required for "
                             "long-context (16K+) training on memory-constrained GPUs; "
                             "default off keeps the bf16 LoRA path unchanged. "
                             "Implies --use-lora at the trainer level (kbit-prep + "
                             "LoRA adapters); the trainer asserts --use-lora is set.")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Phase-C cell 1: per-row class-weighted CE (HF Trainer path).")
    parser.add_argument("--class-weight-cap", type=float, default=3.0,
                        help="Symmetric cap on sklearn-balanced weights.")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Phase-C cell 2: TRL focal-loss subclass. 0 disables.")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    # WandB + in-loop eval. WandB is opt-in: `--wandb-project` activates the
    # integration; without it, `report_to=[]` is preserved (no behavior change).
    # In-loop eval defaults to "no" so existing callers see no change in wall
    # time; `--evaluation-strategy steps --eval-steps N` enables it.
    # Caller must export WANDB_API_KEY (or WANDB_MODE=offline) before launch.
    parser.add_argument("--wandb-project", default=None,
                        help="WandB project name. None disables wandb (default). "
                             "Requires WANDB_API_KEY or WANDB_MODE=offline.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="WandB run name (defaults to HF Trainer's auto-name).")
    parser.add_argument("--evaluation-strategy", default="no",
                        choices=("no", "steps", "epoch"),
                        help="HF Trainer evaluation_strategy. 'no' (default) preserves "
                             "the historical behavior; 'steps' triggers in-loop eval.")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Eval frequency when --evaluation-strategy=steps")
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="per_device_eval_batch_size; 0 → fall back to --batch-size")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Trainer save_steps. Default None → save once at "
                             "max_steps (historical behavior). Set to e.g. 100 "
                             "to get periodic checkpoints — survives mid-run "
                             "SIGTERM at the cost of extra disk + ~1-2min per "
                             "save. Pair with --save-total-limit to cap usage.")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="HF Trainer save_total_limit. None keeps all "
                             "checkpoints (default); set to e.g. 3 to keep "
                             "the 3 most recent and prune older ones.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="If >1, re-launch via torchrun --nproc_per_node=N")
    parser.add_argument("--master-port", type=int, default=29500,
                        help="torchrun rendezvous port; bump if a previous run left "
                             "29500 in use (lingering torchrun agent or TIME_WAIT)")
    parser.add_argument("--gpu-id", type=str, default=None,
                        help="Pin to a single GPU by setting CUDA_VISIBLE_DEVICES before "
                             "torch import. Useful on shared boxes where GPU 0 is busy. "
                             "Ignored if CUDA_VISIBLE_DEVICES is already set.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate adapter + print GPU summary; skip trainer")
    parser.add_argument("--skip-adapter", action="store_true",
                        help="Use --pairs-out as a pre-built JSONL instead of running "
                             "convert_aug_dirs(). Required for v2 retraining where "
                             "reward_model_pairs.py already produced the JSONL offline.")
    args = parser.parse_args()

    if args.gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        logger.info("pinned CUDA_VISIBLE_DEVICES=%s", args.gpu_id)

    # Long-context (16K+) bf16 training under HF Trainer otherwise OOMs in
    # accelerate's mixed-precision output-cast wrap; gate the patch on the
    # QLoRA flag because that's the documented long-context entry point.
    if args.load_in_4bit:
        _patch_accelerate_skip_large_fp32_upcast()

    # WandB-key fallback: when launch.py wasn't started with WANDB_API_KEY
    # in its env, read it from the bootstrap file written by
    # `memgym.training.scripts.bootstrap_wandb_key`. Keeps the key off
    # the trainer's argv (and out of every job's `command` record) once
    # the one-time bootstrap has run.
    if "WANDB_API_KEY" not in os.environ:
        _key_file = Path.home() / ".config" / "memgym" / "wandb_api_key"
        if _key_file.exists():
            try:
                _key = _key_file.read_text().strip()
                if _key:
                    os.environ["WANDB_API_KEY"] = _key
                    logger.info("loaded WANDB_API_KEY from %s", _key_file)
            except OSError as exc:
                logger.warning("could not read %s: %s", _key_file, exc)

    # wandb's internal IPC daemon (spawned by wandb.init on rank 0) writes
    # its port file under /tmp; under load (8-GPU torchrun, post-tokenize)
    # the default 30s deadline can be missed, surfacing as
    # `ServicePollForTokenError`. Triple it here. Setdefault so an explicit
    # env override still wins.
    if args.wandb_project:
        os.environ.setdefault("WANDB__SERVICE_WAIT", "180")
        os.environ.setdefault("WANDB_X_SERVICE_WAIT", "180")

    aug_dirs = args.aug_dirs or DEFAULT_DIRS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = args.pairs_out or (args.out_dir / "aug_sft_pairs.jsonl")

    in_worker = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if args.skip_adapter:
        if not pairs_out.exists():
            print(json.dumps({"step": "aug_pairs_adapter_skipped",
                              "error": f"pairs_out missing: {pairs_out}"}), file=sys.stderr)
            return 2
        print(json.dumps({"step": "aug_pairs_adapter_skipped",
                          "pairs_out": str(pairs_out),
                          "reason": "--skip-adapter; using pre-built JSONL"}),
              file=sys.stderr)
    elif not in_worker:
        from memgym.training.data.aug_pairs import convert_aug_dirs
        adapter_stats = convert_aug_dirs(
            aug_dirs, pairs_out, args.limit, args.eval_frac,
            label_safe=args.label_safe, label_harmful=args.label_harmful,
        )
        print(json.dumps({"step": "aug_pairs_adapter", **adapter_stats}), file=sys.stderr)
    else:
        print(json.dumps({"step": "aug_pairs_adapter_skipped_worker",
                          "rank": os.environ.get("RANK")}), file=sys.stderr)

    gpu = _gpu_summary()
    print(json.dumps({"step": "gpu_summary", **gpu}), file=sys.stderr)

    if args.dry_run:
        if not gpu.get("cuda_available"):
            print("CUDA not available; refusing dry-run gate", file=sys.stderr)
            return 2
        print(0.0)
        return 0

    if not gpu.get("cuda_available"):
        print("WARNING: CUDA not available; will fall back to CPU", file=sys.stderr)

    # Multi-GPU mode: re-exec under torchrun. Prefer the active venv's
    # torchrun so we don't pick up a sibling venv's CUDA-mismatched wheel.
    if args.num_gpus > 1 and not in_worker:
        venv_torchrun = Path(sys.executable).parent / "torchrun"
        torchrun = str(venv_torchrun) if venv_torchrun.exists() else (
            shutil.which("torchrun") or str(venv_torchrun)
        )
        worker_argv = [
            torchrun,
            f"--nproc_per_node={args.num_gpus}",
            f"--master-port={args.master_port}",
            "--module", "memgym.training.scripts.train_sft",
            "--out-dir", str(args.out_dir),
            "--pairs-out", str(pairs_out),
            "--save-dir", str(args.save_dir or (args.out_dir / "final")),
            "--model", args.model,
            "--max-steps", str(args.max_steps),
            "--limit", str(args.limit),
            "--eval-frac", str(args.eval_frac),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--max-length", str(args.max_length),
            "--lr", str(args.lr),
            "--num-gpus", "1",
        ]
        if args.bf16:
            worker_argv.append("--bf16")
        else:
            worker_argv.append("--no-bf16")
        if args.flash_attn:
            worker_argv.append("--flash-attn")
        if args.use_lora:
            worker_argv.extend(["--use-lora", "--lora-r", str(args.lora_r),
                                "--lora-alpha", str(args.lora_alpha)])
        if args.gradient_checkpointing:
            worker_argv.append("--gradient-checkpointing")
        if args.load_in_4bit:
            worker_argv.append("--load-in-4bit")
        if args.save_steps is not None:
            worker_argv.extend(["--save-steps", str(args.save_steps)])
        if args.save_total_limit is not None:
            worker_argv.extend(["--save-total-limit", str(args.save_total_limit)])
        worker_argv.extend(["--lr-scheduler-type", args.lr_scheduler_type,
                            "--warmup-ratio", str(args.warmup_ratio)])
        worker_argv.extend(["--label-safe", args.label_safe,
                            "--label-harmful", args.label_harmful])
        if args.balance_classes:
            worker_argv.extend(["--balance-classes",
                                "--class-weight-cap", str(args.class_weight_cap)])
        if args.focal_gamma and args.focal_gamma > 0.0:
            worker_argv.extend(["--focal-gamma", str(args.focal_gamma)])
        if args.skip_adapter:
            worker_argv.append("--skip-adapter")
        for d in (args.aug_dirs or []):
            worker_argv.extend(["--aug-dir", str(d)])
        # WandB + in-loop eval flags propagate to every torchrun rank — HF
        # Trainer auto-gates wandb.init to local_rank==0 once the env vars
        # are visible, so simply mirroring the flags is sufficient.
        if args.wandb_project:
            worker_argv.extend(["--wandb-project", args.wandb_project])
        if args.wandb_run_name:
            worker_argv.extend(["--wandb-run-name", args.wandb_run_name])
        worker_argv.extend(["--evaluation-strategy", args.evaluation_strategy,
                            "--eval-steps", str(args.eval_steps),
                            "--eval-batch-size", str(args.eval_batch_size)])
        print(json.dumps({"step": "torchrun_dispatch", "argv": worker_argv}), file=sys.stderr)
        return subprocess.call(worker_argv)

    from memgym.training.models import sft as sft_trainer

    trainer_args = argparse.Namespace(
        pairs=pairs_out,
        out_dir=args.out_dir / "final",
        save_dir=args.save_dir or (args.out_dir / "final"),
        model=args.model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        limit=args.limit,
        bf16=args.bf16,
        flash_attn=args.flash_attn,
        no_save=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.gradient_checkpointing,
        load_in_4bit=args.load_in_4bit,
        balance_classes=args.balance_classes,
        class_weight_cap=args.class_weight_cap,
        focal_gamma=args.focal_gamma,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_batch_size=(args.eval_batch_size or args.batch_size),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dry_run=False,
    )

    rows = sft_trainer._validate_pairs(trainer_args.pairs, trainer_args.limit)
    print(json.dumps({"step": "validate_pairs", "rows": len(rows)}), file=sys.stderr)

    final_loss = sft_trainer._train_sft(rows, trainer_args)
    print(
        json.dumps({"step": "sft_trainer", "final_loss": final_loss, "rows": len(rows)}),
        file=sys.stderr,
    )
    print(final_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
