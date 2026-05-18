"""FSDP wrap helper used by Way B's GRPO trainer + ref-policy KL anchor.

Way A's VeRL trainer has its own FSDP path (driven by
`actor_rollout_ref.actor.fsdp_config` in the YAML config) — this module
is only consumed by `train_rl_online.py --mode memory --num-gpus N` and
the same path's ref-policy clone.

The wrap is opinionated for the 8B Qwen3 case:

* Full-shard strategy (FULL_SHARD) — every parameter is sharded across
  every rank. ZeRO-3 equivalent. At 8B-bf16 / 8 GPUs that's ~2 GiB of
  params per rank; with grads + Adam state it lands around 8 GiB / rank
  in full-FT mode (LoRA mode is dominated by activations, not params).
* Mixed precision in bf16 — params + reduce + buffer all bf16. Matches
  the SFT recipe + the rollout backend's default `torch_dtype`.
* Optional activation checkpointing — gated on `activation_checkpoint`
  because LoRA-mode doesn't need it (only full-FT does at our 6K-token
  prompt budget). The hook re-runs each transformer block's forward
  during backward, trading FLOPs for activation memory.

Critically, the ref policy and the trainable model are wrapped on the
**same** process group (`dist.group.WORLD`). If they differ, ranks
1..N-1 see ref-logprobs of 0 and the KL term silently zeroes — that's
the plan §"Way B FSDP rank desync" risk row.
"""
from __future__ import annotations

from typing import Any


def wrap_for_fsdp(
    model: Any,
    dtype: str = "bfloat16",
    full_shard: bool = True,
    activation_checkpoint: bool = False,
) -> Any:
    """Wrap a model with FullyShardedDataParallel for the Way B trainer.

    Caller must have already called `torch.distributed.init_process_group`
    (the orchestrator does this when launched under `torchrun`). Returns
    the wrapped model; subsequent forward/backward calls go through the
    FSDP collective ops automatically.

    `dtype` is the mixed-precision compute dtype, separate from the
    model's storage dtype (which the caller controls via
    `LocalHFSummarizerBackend(torch_dtype=...)`). For the standard
    bf16 SFT recipe, set both to "bfloat16".
    """
    import functools
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if not dist.is_initialized():
        raise RuntimeError(
            "wrap_for_fsdp requires torch.distributed to be initialized; "
            "launch via `torchrun --nproc-per-node N` so the world size "
            "and process group are set up before this call."
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype!r} for FSDP mixed precision")
    compute_dtype = dtype_map[dtype]

    mp_policy = MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=compute_dtype,
        buffer_dtype=compute_dtype,
    )
    strategy = ShardingStrategy.FULL_SHARD if full_shard else ShardingStrategy.SHARD_GRAD_OP

    # PEFT inserts lora_A/lora_B in fp32 while the base 8B sits in bf16.
    # FSDP's FlatParamHandle still enforces uniform dtype across each
    # flat-param even with use_orig_params=True (the kwarg controls
    # *exposure* of params, not the flat-param construction validator
    # in `_validate_tensors_to_flatten`). The standard PEFT+FSDP fix
    # (HF accelerate guide, PyTorch issue #108277) is to cast adapter
    # params to compute_dtype before wrap. Only LoRA params get cast;
    # full-FT mode has no fp32 params and is a no-op here.
    cast_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != compute_dtype:
            p.data = p.data.to(compute_dtype)
            cast_count += 1
    if cast_count:
        import logging
        logging.getLogger(__name__).info(
            "wrap_for_fsdp: cast %d trainable params to %s before FSDP wrap",
            cast_count, compute_dtype,
        )

    # Per-transformer-block auto-wrap is the binding fix for 8B+ on
    # A100-40GB: without it, FSDP builds one flat-param covering the full
    # 8B model and needs ~2× the model's memory transiently (original +
    # flat copy) before sharding — that's 32 GiB on top of the 16 GiB
    # resident model = OOM at the 39.5 GiB device limit. With per-block
    # wrap, FSDP only ever has ~one block × 2 of transient memory
    # (~200 MiB), and forward/backward overlap all-gather/reduce-scatter
    # across blocks. We discover the block class by name suffix to keep
    # the helper HF-architecture-agnostic (Qwen3DecoderLayer, GPT2Block,
    # LlamaDecoderLayer, etc all match).
    block_classes: set[type] = set()
    for module in model.modules():
        cls = type(module)
        if cls.__name__.endswith("DecoderLayer") or cls.__name__.endswith("Block"):
            block_classes.add(cls)
    if block_classes:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        auto_wrap_policy = None

    # `use_orig_params=True` preserves per-param identity (required for
    # PEFT optimizers keyed off original parameter objects + matches the
    # accelerate-FSDP defaults).
    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        process_group=dist.group.WORLD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        auto_wrap_policy=auto_wrap_policy,
    )

    if activation_checkpoint:
        # Apply per-transformer-block activation checkpointing. This is
        # the cheapest form: re-run each block's forward during backward.
        # Required for full-FT at 6144-token prompts on A100-40GB (saves
        # ~14 GiB activations) and harmless in LoRA mode (caller passes
        # activation_checkpoint=False to avoid the recompute cost).
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        from functools import partial

        def _is_transformer_block(module: Any) -> bool:
            # Cover both Qwen3 (`Qwen3DecoderLayer`) and the generic
            # HF naming pattern. Avoids importing the model class
            # directly so this helper stays HF-arch-agnostic.
            cls_name = type(module).__name__
            return cls_name.endswith("DecoderLayer") or cls_name.endswith("Block")

        wrap_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=wrap_fn,
            check_fn=_is_transformer_block,
        )

    return wrapped


__all__ = ["wrap_for_fsdp"]
