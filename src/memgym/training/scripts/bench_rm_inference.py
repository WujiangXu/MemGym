"""Reward-model inference benchmark — latency + VRAM + throughput.

The published RM datacard reports AUROC and ECE; nothing measures the
classifier *forward pass* in isolation. The Phase-G gate latency at
`world_model_phase3.md:316,328` is end-to-end gate including two
agent re-generation calls — useful for runtime budget claims, but it
does not separate the classifier cost from the agent cost.

This script answers the cost question that backs the "lightweight
option" framing: at matched hardware and matched quantization, how
much faster (and how much smaller in VRAM) is the 1.7B QLoRA RM than
the 8B QLoRA RM, on real eval prompts?

What we measure (per checkpoint × per quant mode):
  * Forward latency p50 / p95 / p99 at three prompt-length buckets
    (p10 / p50 / p90 of the eval set's prompt-length distribution)
  * Tokens / second prefill throughput
  * Peak VRAM via `torch.cuda.max_memory_allocated`
  * Cold-start time (model load + adapter attach)
  * Parameter count, total and trainable
  * Optional batched-throughput at bs=8 (offline-eval cost)

Output: a single JSON. Cells in the appendix's cost table cite this.

Hardware note: results depend on the GPU SKU and on whether
flash-attention is in the wheel. Both are recorded in the JSON so
appendix readers can audit.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _BenchCell:
    checkpoint: str
    base_model: str
    quant_mode: str  # "nf4" | "bf16"
    cold_start_s: float
    n_params_total: int
    n_params_trainable: int
    bucket_results: List[Dict] = field(default_factory=list)
    batched_results: Optional[Dict] = None
    peak_vram_gb: float = 0.0


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _gpu_info() -> Dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        return {"name": "cpu", "memory_gb": "0"}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": props.name,
        "memory_gb": f"{props.total_memory / (1024 ** 3):.1f}",
        "cuda_capability": f"{props.major}.{props.minor}",
    }


def _sample_prompts(
    pairs_jsonl: Path,
    n_total: int,
    rng_seed: int = 0,
) -> List[str]:
    """Sample n_total prompts spread across prompt-length percentiles.

    We collect *all* prompts, sort by char-length, then return the
    rows nearest p10 / p50 / p90 (a third each, so latency-by-bucket
    is interpretable). The eval JSONL is the authoritative source —
    prompt strings already have the production aggregator prefix.
    """
    import random

    rng = random.Random(rng_seed)
    rows: List[Tuple[int, str]] = []
    with pairs_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            prompt = row.get("prompt") or ""
            if prompt:
                rows.append((len(prompt), prompt))
    if not rows:
        raise SystemExit(f"no eval-split rows with non-empty prompt in {pairs_jsonl}")
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    per_bucket = max(1, n_total // 3)
    p10_idx = int(n * 0.05)
    p50_idx = int(n * 0.45)
    p90_idx = int(n * 0.85)
    out: List[str] = []
    for center in (p10_idx, p50_idx, p90_idx):
        lo = max(0, center - per_bucket // 2)
        hi = min(n, lo + per_bucket)
        bucket = [r[1] for r in rows[lo:hi]]
        rng.shuffle(bucket)
        out.extend(bucket[:per_bucket])
    return out


def _bucket_label(prompt_len: int, p10: int, p50: int, p90: int) -> str:
    if prompt_len <= (p10 + p50) // 2:
        return "p10"
    if prompt_len <= (p50 + p90) // 2:
        return "p50"
    return "p90"


def _load_model(
    checkpoint: str,
    base_model: str,
    quant_mode: str,
):
    """Load the RM via the canonical MemoryWorldModel path.

    Forces single-GPU (`device_map={"":0}`) — this is the eval-time
    pattern from `world_model.py:setup`, NOT the train-time DDP path.
    """
    from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

    config = ModelConfig(
        base_model=base_model,
        load_in_4bit=(quant_mode == "nf4"),
        torch_dtype=("bfloat16" if quant_mode == "bf16" else None),
    )
    return MemoryWorldModel.from_checkpoint(Path(checkpoint), config)


def _bench_one(
    checkpoint: str,
    base_model: str,
    quant_mode: str,
    prompts: Sequence[str],
    *,
    warmup: int = 3,
    do_batched: bool = False,
    batch_size: int = 8,
) -> _BenchCell:
    """Benchmark one (checkpoint, quant_mode) cell."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = _load_model(checkpoint, base_model, quant_mode)
    model.setup()
    cold_start_s = time.perf_counter() - t0

    n_total = sum(p.numel() for p in model.model.parameters())
    n_train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    lengths = [len(p) for p in prompts]
    p10 = int(_percentile(lengths, 10))
    p50 = int(_percentile(lengths, 50))
    p90 = int(_percentile(lengths, 90))

    # Warmup — first kernel launches always include autotuner overhead.
    for prompt in prompts[:warmup]:
        with torch.no_grad():
            _ = model.predict_logits_text(prompt)
        torch.cuda.synchronize()

    by_bucket: Dict[str, List[Tuple[float, int]]] = {"p10": [], "p50": [], "p90": []}

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for prompt in prompts:
        bucket = _bucket_label(len(prompt), p10, p50, p90)
        torch.cuda.synchronize()
        start_evt.record()
        with torch.no_grad():
            _ = model.predict_logits_text(prompt)
        end_evt.record()
        torch.cuda.synchronize()
        ms = start_evt.elapsed_time(end_evt)
        by_bucket[bucket].append((ms, len(prompt)))

    bucket_results: List[Dict] = []
    for bucket, samples in by_bucket.items():
        if not samples:
            continue
        latencies = [s[0] for s in samples]
        total_chars = sum(s[1] for s in samples)
        bucket_results.append({
            "bucket": bucket,
            "n": len(samples),
            "char_p50": int(_percentile([s[1] for s in samples], 50)),
            "latency_ms_p50": _percentile(latencies, 50),
            "latency_ms_p95": _percentile(latencies, 95),
            "latency_ms_p99": _percentile(latencies, 99),
            "chars_per_sec": total_chars / (sum(latencies) / 1000.0)
                if sum(latencies) > 0 else 0.0,
        })

    batched_results: Optional[Dict] = None
    if do_batched:
        batched_results = _bench_batched(model, prompts, batch_size)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    cell = _BenchCell(
        checkpoint=str(checkpoint),
        base_model=base_model,
        quant_mode=quant_mode,
        cold_start_s=cold_start_s,
        n_params_total=n_total,
        n_params_trainable=n_train,
        bucket_results=bucket_results,
        batched_results=batched_results,
        peak_vram_gb=peak_vram_gb,
    )

    # Be explicit about cleanup so successive cells don't OOM.
    del model
    torch.cuda.empty_cache()
    return cell


def _bench_batched(model, prompts: Sequence[str], batch_size: int) -> Dict:
    """Throughput at fixed batch size — uses the model's tokenizer to pad."""
    import torch

    tok = model.tokenizer
    device = next(model.model.parameters()).device
    n_batches = max(1, len(prompts) // batch_size)
    total_rows = n_batches * batch_size
    total_chars = 0
    t0 = time.perf_counter()
    for i in range(n_batches):
        batch = list(prompts[i * batch_size:(i + 1) * batch_size])
        total_chars += sum(len(p) for p in batch)
        enc = tok(batch, padding=True, truncation=True, max_length=32768,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.model(**enc)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "batch_size": batch_size,
        "n_rows": total_rows,
        "wall_s": elapsed,
        "rows_per_sec": total_rows / elapsed if elapsed > 0 else 0.0,
        "chars_per_sec": total_chars / elapsed if elapsed > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", action="append", required=True,
        metavar="CKPT,BASE,QUANT",
        help="One cell to benchmark. Comma-separated triple of "
             "checkpoint path, base model id, quant mode (nf4|bf16). "
             "Repeat the flag for each cell, e.g. "
             "--cell checkpoints/rm_v2_8b_yn_cw3,Qwen/Qwen3-8B-Base,nf4 "
             "--cell checkpoints/rm_v2_8b_yn_cw3,Qwen/Qwen3-8B-Base,bf16 "
             "--cell checkpoints/rm_v2_1p7b_qlora_32k,Qwen/Qwen3-1.7B-Base,nf4",
    )
    parser.add_argument("--pairs", type=Path, required=True,
                        help="reward_model_pairs_v2.jsonl — used to sample "
                             "real eval prompts at p10/p50/p90 length.")
    parser.add_argument("--n-prompts", type=int, default=99,
                        help="Total prompts to time (split across 3 buckets, "
                             "default 99 → 33 per bucket).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup forward passes before timing.")
    parser.add_argument("--batched", action="store_true", default=False,
                        help="Also measure throughput at bs=8 (offline-eval cost).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON output path.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    prompts = _sample_prompts(args.pairs, args.n_prompts, args.seed)
    logger.info("sampled %d prompts (target %d) from %s",
                len(prompts), args.n_prompts, args.pairs)

    cells: List[_BenchCell] = []
    for spec in args.cell:
        try:
            ckpt, base, quant = spec.split(",")
        except ValueError:
            raise SystemExit(f"--cell expects CKPT,BASE,QUANT — got {spec!r}")
        if quant not in ("nf4", "bf16"):
            raise SystemExit(f"quant must be nf4 or bf16 — got {quant!r}")
        logger.info("benchmarking ckpt=%s base=%s quant=%s",
                    ckpt, base, quant)
        cell = _bench_one(
            ckpt, base, quant, prompts,
            warmup=args.warmup,
            do_batched=args.batched,
            batch_size=args.batch_size,
        )
        cells.append(cell)
        logger.info(
            "  cold_start=%.2fs n_params_total=%.1fM peak_vram=%.2fGB",
            cell.cold_start_s, cell.n_params_total / 1e6, cell.peak_vram_gb,
        )
        for bucket in cell.bucket_results:
            logger.info(
                "  %s: n=%d char_p50=%d lat_p50=%.1fms lat_p95=%.1fms",
                bucket["bucket"], bucket["n"], bucket["char_p50"],
                bucket["latency_ms_p50"], bucket["latency_ms_p95"],
            )

    out = {
        "gpu": _gpu_info(),
        "n_prompts": len(prompts),
        "warmup": args.warmup,
        "cells": [asdict(c) for c in cells],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
