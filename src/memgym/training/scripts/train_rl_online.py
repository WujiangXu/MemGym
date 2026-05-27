"""Online RL orchestrator for the summarizer (memory-RL).

EC2 validation: a single-node, single-policy driver that exercises all
pieces — rollout pool, advantage prep, old-logprob snapshot, GRPO step
— so we can confirm the abstraction works end-to-end before friends
scale it across more GPUs.

`--mode memory` (default) runs Way B: frozen agent + trained
summarizer. `--mode agent` runs Way A and currently raises — the
agent path needs vLLM weight-sync (hot-reload of LoRA adapter into
vLLM after each step), which is a friend-cluster engineering task per
the 2026-04-27 plan revision.

Concurrency rules of thumb (plan §H.3'' Way B):
- API agent (gpt-5-nano / gpt-4o-mini / Bedrock Sonnet) → 4–8 on EC2.
- Self-hosted vLLM agent → 1–4, capped by vLLM batch.
- LocalHFSummarizerBackend in the loop → effectively 1 for the
  summarize critical section; agent calls still parallelize via the
  module-level lock in `online_rollout.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("train_rl_online")


def _build_env_factory(args: argparse.Namespace, memory_model_factory):
    """Closure that builds a SWEMemoryEnv per-rollout.

    The memory_model_factory builds the memory manager fresh each call
    — same SFT ckpt, same compaction settings, but a fresh
    `summarizer_backend` slot that `run_episode_with_reward` will
    hot-swap with a `RecordingSummarizerBackend`.
    """
    from memgym.envs import SWEMemoryEnv

    def factory(instance: Dict[str, Any], rollout_idx: int):
        memory_model = memory_model_factory()
        # Group variance comes from sampling stochasticity at a fixed
        # temperature (set once at startup in main(), not here — mutating
        # the shared generation_kwargs from rollout threads would race).
        agent_model_kwargs = {
            "max_tokens": args.agent_max_tokens,
            "temperature": args.agent_temperature,
            "top_p": args.agent_top_p,
        }
        if os.environ.get("OPENAI_API_BASE"):
            agent_model_kwargs["extra_body"] = {
                "top_k": args.agent_top_k,
                "min_p": args.agent_min_p,
            }

        from memgym.gym.swe_bench.evaluate import DATASET_MAP
        return SWEMemoryEnv(
            dataset=DATASET_MAP.get(args.dataset, args.dataset),
            split=args.split,
            instance_id=instance["instance_id"],
            agent_llm=args.agent_llm,
            agent_llm_args={"model_kwargs": agent_model_kwargs},
            memory_model=memory_model,
            use_context_management=True,
            use_docker=args.env_type in ("docker", "swerex_docker"),
            environment_class=args.env_type,
            docker_image_dir=args.docker_image_dir,
            docker_image_source=args.docker_image_source,
            max_steps=args.step_limit,
            agent_type="mini-swe",
            skip_eval=args.skip_eval,
            memory_gate=None,
        )

    return factory


def _load_instances(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load the SWE-Gym training pool.

    Mirrors `eval_memory_ab.py:main` instance-loading so the training
    distribution matches H.4' eval exactly: load HF dataset, optionally
    filter by SWE-Smith subcategory, then either diverse-by-repo sample
    or random-shuffle to `n_instances`.
    """
    from datasets import load_dataset
    from memgym.gym.swe_bench.evaluate import DATASET_MAP
    from memgym.training.scripts.eval_memory_ab import (
        _filter_by_subcategory,
        _select_diverse_by_repo,
    )

    # Resolve project aliases (`swe-gym`, `swe-smith`, `lite`, etc.) to
    # their Hub paths; pass through anything that's already `org/name`.
    dataset_path = DATASET_MAP.get(args.dataset, args.dataset)
    dataset = load_dataset(dataset_path, split=args.split)
    instances = list(dataset)
    if args.subcategory_include:
        allowed = [s.strip() for s in args.subcategory_include.split(",") if s.strip()]
        instances = _filter_by_subcategory(instances, allowed)
    if args.diverse_by_repo and args.n_instances < len(instances):
        instances = _select_diverse_by_repo(instances, args.n_instances)
    elif args.n_instances < len(instances):
        random.Random(args.seed).shuffle(instances)
        instances = instances[: args.n_instances]
    return instances


def _build_memory_factory(args: argparse.Namespace):
    """Returns a function that builds a NaiveSummarizationMemory wired
    to the *shared* policy backend. The same backend instance is reused
    across rollouts because the policy weights are shared; the recording
    wrapper inside `run_episode_with_reward` is what differentiates one
    rollout's calls from another's.

    When `--num-gpus > 1`, the trainable backend's underlying HF model
    is wrapped with FSDP after construction (full-shard, bf16). LoRA
    mode skips activation checkpointing; full-FT mode enables it (the
    activation memory at 6144-token prompts is the binding constraint
    on 8×A100-40GB at 8B-bf16).
    """
    from memgym.memory.naive_summarization import NaiveSummarizationMemory
    from memgym.memory.summarizer_backend import LocalHFSummarizerBackend

    # FSDP requires every rank's pre-wrap module to live on a single device.
    # HF's default `device_map="auto"` spreads the 8B across all visible
    # GPUs per rank, which fails the `FSDP only supports single device
    # modules` check. Pin each rank to its own `cuda:LOCAL_RANK` so the
    # wrap can shard cleanly across the world.
    device_map = "auto"
    if args.num_gpus > 1:
        device_map = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    backend = LocalHFSummarizerBackend(
        model_name_or_path=args.policy_model,
        torch_dtype=args.policy_dtype,
        max_prompt_tokens=args.summarizer_max_prompt_tokens,
        use_lora=args.use_lora,
        device_map=device_map,
    )
    if args.num_gpus > 1:
        import copy as _copy

        import torch as _torch
        import torch.nn as _nn

        from memgym.training.rl.fsdp_wrap import wrap_for_fsdp

        # Build the per-rank non-sharded inference copy *before* FSDP-wrapping
        # the trainable model. FSDP1 installs forward hooks on the wrapped
        # module in place, so deep-copying afterwards would inherit those
        # hooks and re-introduce the per-rank-collective bug we're fixing.
        #
        # Route the deepcopy through CPU. `Parameter.__deepcopy__` clones on
        # the *source* device, so a naive `deepcopy(model_on_gpu)` peaks at
        # 2× the model on GPU before the `.to()` move. CPU intermediate keeps
        # the GPU peak at 1× the model. CPU RAM cost: ~16 GB per rank
        # transient — fine on EC2 nodes with 256+ GB RAM. V4e OOM'd here on
        # 40 GB A100 because both copies briefly co-resided on cuda:LOCAL_RANK.
        rank_device = _torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        backend.model = backend.model.to("cpu")
        inference_model = _copy.deepcopy(backend.model)
        backend.model = backend.model.to(rank_device)
        inference_model = inference_model.to(rank_device)
        inference_model.eval()

        # PEFT LoRA dropout (typically p=0.05 from SFT) interacts badly with
        # `apply_activation_checkpointing(NO_REENTRANT)`: the forward and
        # recomputed-backward graphs save a different number of tensors when
        # dropout is sampled, raising
        #   CheckpointError: A different number of tensors was saved during
        #   the original forward and recomputation (forward=72, recompute=70).
        # V4p crashed here. Replacing every `lora_dropout` submodule with
        # Identity removes the stochastic path; harmless during RL fine-tune
        # of a small adapter where deterministic forwards are desirable.
        n_dropouts_zeroed = 0
        for module in backend.model.modules():
            if hasattr(module, "lora_dropout") and isinstance(module.lora_dropout, _nn.ModuleDict):
                for key in list(module.lora_dropout.keys()):
                    if not isinstance(module.lora_dropout[key], _nn.Identity):
                        module.lora_dropout[key] = _nn.Identity()
                        n_dropouts_zeroed += 1
        if n_dropouts_zeroed:
            import logging
            logging.getLogger(__name__).info(
                "_build_memory_factory: replaced %d LoRA dropout modules with Identity "
                "(activation-checkpoint compatibility, V4p fix)",
                n_dropouts_zeroed,
            )

        # Activation checkpointing is *off* for LoRA mode at our 2048-token
        # prompt cap. V4q proved that with `apply_activation_checkpointing
        # (NO_REENTRANT)` + PEFT LoRA + SDPA, the forward and recomputed
        # backward graphs save a different number of tensors (68 vs 66
        # post-LoRA-dropout-fix), raising CheckpointError. The bug stems
        # from SDPA's kernel selection diverging between forward and
        # recompute under FSDP — not solvable at the wrap level. With the
        # `--summarizer-max-prompt-tokens 2048` cap, the activation peak
        # is ~3 GiB / rank (SDPA O(seq²) at 2048 vs 6144 = 9× smaller),
        # well under the 39.5 GB A100-40GB budget without checkpointing.
        # Full-FT mode keeps activation checkpointing on (the param count
        # alone needs the savings). Friends on H100/H200 80GB+ at long
        # contexts should use HF's
        #   model.gradient_checkpointing_enable(
        #       gradient_checkpointing_kwargs={"use_reentrant": False})
        # before FSDP wrap (not FSDP's apply_activation_checkpointing) —
        # HF's path coordinates with PEFT's LoRA hooks cleanly.
        backend.model = wrap_for_fsdp(
            backend.model,
            dtype=args.policy_dtype,
            full_shard=True,
            activation_checkpoint=not args.use_lora,
        )
        backend._inference_model = inference_model

    def factory():
        return NaiveSummarizationMemory(
            summarization_model=args.policy_model,
            summarizer_backend=backend,
            max_tokens=args.max_tokens,
            env_type="swe",
        )

    return backend, factory


def _build_ref_backend(args: argparse.Namespace):
    """Frozen SFT ckpt for the KL anchor. None if --kl-coef==0.

    The ref policy and the trainable model share the same FSDP process
    group (`dist.group.WORLD` inside `wrap_for_fsdp`). If they didn't,
    ranks 1..N-1 would see ref-logprobs of 0 and the KL term would
    silently zero — see the "Way B FSDP rank desync" risk row.
    """
    if args.kl_coef <= 0:
        return None
    from memgym.memory.summarizer_backend import LocalHFSummarizerBackend

    # Same `cuda:LOCAL_RANK` pin as `_build_memory_factory` — both the
    # trainable model and the ref share the same FSDP process group, so
    # they must live on the same single device per rank pre-wrap.
    device_map = "auto"
    if args.num_gpus > 1:
        device_map = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    backend = LocalHFSummarizerBackend(
        model_name_or_path=args.ref_model or args.policy_model,
        torch_dtype=args.policy_dtype,
        max_prompt_tokens=args.summarizer_max_prompt_tokens,
        use_lora=args.use_lora,
        device_map=device_map,
    )
    if args.num_gpus > 1:
        from memgym.training.rl.fsdp_wrap import wrap_for_fsdp

        # No `_inference_model` for the ref. The ref is invoked only during
        # the GRPO step (all ranks in lockstep on `mode="train"`), so the
        # rank-desync bug we're fixing on the trainable side never applies
        # here. `LocalHFSummarizerBackend._inference_model` defaults to
        # None, and `compute_logprobs_for_completion` falls back to
        # `self.model` in that case — exactly the FSDP forward we want for
        # the synchronized KL anchor. Skipping the deepcopy saves ~16 GiB
        # per rank, which V4e proved is the binding constraint on A100-40GB.
        # Mirror the trainable: activation checkpointing off in LoRA mode
        # (the seq=2048 cap keeps activations under budget without it, and
        # the V4q CheckpointError reproduced on this side too whenever a
        # gradient happens to flow). In full-FT mode keep it on.
        backend.model = wrap_for_fsdp(
            backend.model,
            dtype=args.policy_dtype,
            full_shard=True,
            activation_checkpoint=not args.use_lora,
        )
    return backend


def _save_checkpoint(backend, save_dir: Path, step: int) -> Path:
    """Save policy weights + tokenizer to `save_dir/step-{step}/`.

    Under FSDP, only rank 0 writes — every rank holds a shard, so
    `save_pretrained` from each rank would race on the same files.
    The `state_dict_type=FULL_STATE_DICT` context gathers shards onto
    rank 0 before the write; non-rank-0 calls are no-ops (returning
    empty dicts), preserving the original API for non-FSDP runs.
    """
    out = save_dir / f"step-{step}"

    # Detect FSDP wrap; non-FSDP path is the original behavior.
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        is_fsdp = isinstance(backend.model, FSDP)
    except Exception:  # noqa: BLE001 — torch.distributed missing is fine
        is_fsdp = False

    if is_fsdp:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(backend.model, StateDictType.FULL_STATE_DICT, cfg):
            state = backend.model.state_dict()
            if dist.get_rank() == 0:
                out.mkdir(parents=True, exist_ok=True)
                # `backend.model.module` is the unwrapped HF model (or
                # PeftModel when use_lora=True); `save_pretrained` knows
                # how to serialize either flavor.
                inner = backend.model.module
                inner.save_pretrained(out, state_dict=state)
                backend.tokenizer.save_pretrained(out)
        dist.barrier()
    else:
        out.mkdir(parents=True, exist_ok=True)
        backend.model.save_pretrained(out)
        backend.tokenizer.save_pretrained(out)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["memory", "agent"], default="memory")
    parser.add_argument("--policy-model", required=True,
                        help="Path/HF-id of the SFT checkpoint to start from.")
    parser.add_argument("--ref-model", default=None,
                        help="Path/HF-id of the frozen reference for KL. Defaults to policy_model.")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--save-dir", required=True)

    parser.add_argument("--dataset", default="swe-gym")
    parser.add_argument("--docker-image-source", default="swe-gym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--env-type", default="docker")
    parser.add_argument("--docker-image-dir", default=None)

    parser.add_argument("--n-instances", type=int, default=200)
    parser.add_argument("--rollouts-per-instance", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--instances-per-batch", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)

    parser.add_argument("--agent-llm", default="openai/gpt-5-nano")
    parser.add_argument("--agent-max-tokens", type=int, default=8000)
    parser.add_argument("--agent-temperature", type=float, default=0.6)
    parser.add_argument("--agent-top-p", type=float, default=0.95)
    parser.add_argument("--agent-top-k", type=int, default=20)
    parser.add_argument("--agent-min-p", type=float, default=0.0)

    parser.add_argument("--max-tokens", type=int, default=3000,
                        help="Memory compaction trigger threshold.")
    parser.add_argument("--summary-temperature", type=float, default=0.7)
    parser.add_argument("--summarizer-max-prompt-tokens", type=int, default=6144)
    parser.add_argument("--step-limit", type=int, default=75)
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--use-compression-guardrail", action="store_true", default=True)

    parser.add_argument("--subcategory-include", default=None,
                        help="Comma-separated list, e.g. func_basic,lm_modify,lm_rewrite")
    parser.add_argument("--diverse-by-repo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")

    # LoRA vs full fine-tune. Default LoRA matches the H.2 SFT recipe
    # and the existing summarizer_sft_8b_v1 checkpoint shape. Friends
    # with H100/80GB pick --full-finetune; on 8×A100-40GB this requires
    # FSDP full-shard + activation checkpointing (both auto-enabled
    # when --num-gpus>1 + --full-finetune).
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora", dest="use_lora", action="store_true", default=True,
        help="Train LoRA adapter only (default; matches H.2 SFT shape).",
    )
    lora_group.add_argument(
        "--full-finetune", dest="use_lora", action="store_false",
        help="Disable PEFT; train all 8B params. Requires matching full-FT SFT ckpt.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="When >1, wrap the policy + ref model with FSDP. Launch via "
             "`torchrun --nproc-per-node N` so the process group is set up.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.mode == "agent":
        # Way A is implemented separately on VeRL — its FSDP+vLLM
        # weight-sync model doesn't fit this script's single-policy
        # driver shape. We surface the pointer rather than dispatching
        # in-process so Way B users don't have to install verl/ray.
        logger.error(
            "Way A (agent-RL) is implemented on VeRL, not via this script. "
            "Run instead: python -m memgym.training.rl.way_a_verl.train "
            "--config src/memgym/training/rl/way_a_verl/config/way_a_qwen3_8b.yaml "
            "[--use-lora|--full-finetune] [--num-gpus 8] ... "
            "."
        )
        return 3

    # Imports deferred so --help works without torch/transformers.
    import torch
    from memgym.training.rl.grpo_loop import (
        grpo_step,
        prepare_advantages,
        snapshot_old_logprobs,
    )
    from memgym.training.rl.online_rollout import RolloutRequest, run_batch

    # FSDP requires `torch.distributed` initialized before model wrap.
    # `torchrun` exports RANK/WORLD_SIZE/LOCAL_RANK; we just call init
    # if it hasn't already been done (idempotent under torchrun's setup).
    # V4t fix: separate Gloo group for the rollout-broadcast path.
    #
    # Rank 0 runs all cloud-agent rollouts while ranks 1..N idle at
    # `broadcast_object_list`. NCCL's default collective watchdog is
    # 10 min — at gpt-5-nano speeds + step-limit=10 we already exceed
    # this on the V4t smoke (~11 min rank-0 rollouts → V4t SIGTERM at
    # `store->get('0')` timeout). Friend-scale rollouts run hours.
    #
    # Solution: keep the NCCL group strict (catches real FSDP deadlocks
    # in `grpo_step` quickly) and create a Gloo subgroup with a long
    # timeout specifically for the cross-rank Python-object broadcast.
    # `_OBJECT_GROUP` is consumed at the broadcast call site below.
    global _OBJECT_GROUP
    _OBJECT_GROUP = None
    if args.num_gpus > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        # 4-hour Gloo collective timeout covers any realistic rollout
        # batch. Friends running >4h batches should bump this further
        # or chunk rollouts; that's a scale concern, not a smoke one.
        _OBJECT_GROUP = dist.new_group(
            backend="gloo", timeout=timedelta(hours=4),
        )
        logger.info(
            "FSDP mode: world_size=%d local_rank=%d (gloo object group ready)",
            dist.get_world_size(), local_rank,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.jsonl"

    instances = _load_instances(args)
    if not instances:
        logger.error("No instances after filtering — check --dataset / --subcategory-include")
        return 2
    logger.info("Loaded %d instances", len(instances))

    policy_backend, memory_factory = _build_memory_factory(args)
    ref_backend = _build_ref_backend(args)
    # Enable sampling once on the shared backend — no thread races, and
    # GRPO group variance is supplied by softmax sampling stochasticity.
    policy_backend.generation_kwargs.update(
        {
            "do_sample": True,
            "temperature": float(args.summary_temperature),
            "top_p": 0.95,
        }
    )
    env_factory = _build_env_factory(args, memory_factory)

    def summarizer_factory(rollout_idx: int):
        return policy_backend

    optimizer = torch.optim.AdamW(
        [p for p in policy_backend.model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    rng = random.Random(args.seed)
    global_step = 0
    t_start = time.time()

    for epoch in range(args.n_epochs):
        rng.shuffle(instances)
        for batch_start in range(0, len(instances), args.instances_per_batch):
            batch_insts = instances[batch_start : batch_start + args.instances_per_batch]
            requests: List[RolloutRequest] = []
            for inst in batch_insts:
                for k in range(args.rollouts_per_instance):
                    requests.append(RolloutRequest(instance=inst, rollout_idx=k))

            # Rollouts run on rank 0 only, broadcast to all ranks.
            #
            # Why: `summarize()` uses each rank's per-rank `_inference_model`
            # which is byte-identical across ranks (deepcopy of same SFT
            # weights, post-step sync via `sync_inference_from_fsdp`), but
            # the cloud agent (gpt-5-nano / Sonnet) is stochastic so each
            # rank running its own rollouts gets a different number of
            # summary calls / different completions. Downstream `grpo_step`
            # iterates over `valid_pairs` doing per-call FSDP forward, and
            # the FSDP all-gather collective deadlocks the moment one rank
            # has fewer calls than another. V4s died here on a 10-min
            # ALLREDUCE watchdog (NumelIn=1 → `dist.barrier`). Running
            # rollouts on rank 0 only and broadcasting outcomes guarantees
            # every rank sees the same advantage list and calls FSDP
            # collectives in lockstep.
            #
            # Cost: rank 0 does 8× the rollout work, ranks 1..N idle on
            # CPU during rollouts. Cloud agent doesn't use rank GPUs, so
            # this is wasted wall-clock on N-1 ranks but no GPU waste.
            # At our smoke scale (1 inst × 2 rollouts) it's seconds; at
            # friend-scale we'll revisit with chunked rollout-per-rank +
            # all_gather + advantage-list-padding instead of broadcast.
            t0 = time.time()
            try:
                import torch.distributed as _dist
                _is_dist = _dist.is_available() and _dist.is_initialized()
                _rank = _dist.get_rank() if _is_dist else 0
            except Exception:  # noqa: BLE001
                _is_dist = False
                _rank = 0
            if not _is_dist or _rank == 0:
                outcomes = run_batch(
                    requests,
                    env_factory=env_factory,
                    summarizer_factory=summarizer_factory,
                    concurrency=args.concurrency,
                    verbose=args.verbose,
                )
            else:
                outcomes = None  # placeholder; filled by broadcast below
            if _is_dist:
                obj_list = [outcomes] if _rank == 0 else [None]
                # Use the Gloo object-group (4h timeout) — NOT the
                # default NCCL group (10-min watchdog) which would
                # SIGTERM ranks 1..N during long rollouts. See V4t
                # diagnosis above the init site.
                _dist.broadcast_object_list(
                    obj_list, src=0, group=_OBJECT_GROUP,
                )
                outcomes = obj_list[0]
            rollout_seconds = time.time() - t0

            results = [o.result for o in outcomes if o.result is not None]
            rollout_indices = [o.rollout_idx for o in outcomes if o.result is not None]
            n_failed = sum(1 for o in outcomes if o.result is None)

            advantages = prepare_advantages(
                results,
                rollout_indices=rollout_indices,
                use_compression_guardrail=args.use_compression_guardrail,
            )

            old_logprobs = snapshot_old_logprobs(advantages, policy_backend)

            metrics = grpo_step(
                advantages,
                old_logprobs,
                policy_backend,
                ref_backend,
                optimizer,
                clip_ratio=args.clip_ratio,
                kl_coef=args.kl_coef,
            )

            metrics.update(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "n_rollouts": len(outcomes),
                    "n_failed": n_failed,
                    "n_advantages": len(advantages),
                    "rollout_seconds": rollout_seconds,
                    "elapsed_seconds": time.time() - t_start,
                    "resolve_rate": (
                        sum(1 for r in results if r.episode_resolved) / max(1, len(results))
                    ),
                    "ctx_fail_rate": (
                        sum(1 for r in results if r.hit_context_limit) / max(1, len(results))
                    ),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
            with log_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            logger.info(
                "epoch=%d step=%d resolve=%.2f ctx_fail=%.2f loss=%.4f ratio=%.3f clip=%.2f n_adv=%d",
                epoch, global_step, metrics["resolve_rate"], metrics["ctx_fail_rate"],
                metrics.get("loss_total", 0.0), metrics.get("mean_ratio", 0.0),
                metrics.get("frac_clipped", 0.0), metrics["n_advantages"],
            )

            global_step += 1
            if global_step % args.save_every == 0:
                ckpt = _save_checkpoint(policy_backend, save_dir, global_step)
                logger.info("Saved checkpoint to %s", ckpt)

    final_ckpt = _save_checkpoint(policy_backend, save_dir, global_step)
    logger.info("Final checkpoint: %s (total %.1fs)", final_ckpt, time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
