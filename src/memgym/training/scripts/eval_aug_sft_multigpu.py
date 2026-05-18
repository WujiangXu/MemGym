"""Multi-GPU eval orchestrator for `eval_aug_sft.py`.

Eval is embarrassingly parallel — each row is an independent forward
pass — so we don't need DDP/NCCL. This wrapper:

1. Spawns N subprocesses (one per GPU) each calling `eval_aug_sft.py`
   with `--shard-id i --num-shards N --output <stem>_shard{i}.json`
   and `CUDA_VISIBLE_DEVICES=i` in the env.
2. Waits for all N to finish.
3. Concatenates `per_row_predictions` from every shard JSON,
   recomputes metrics with the shared `compute_metrics`, and writes
   the merged `eval_results.json` to `--output`.

The shard JSONs are kept on disk for traceability — useful when one
shard OOMs and you want to relaunch only the failed slice.

Usage:
    python -m memgym.training.scripts.eval_aug_sft_multigpu \
        --checkpoint checkpoints/rm_v2_1p7b_qlora_32k/checkpoint-400 \
        --base-model Qwen/Qwen3-1.7B-Base \
        --pairs data/.../reward_model_pairs_v2.jsonl \
        --max-length 32768 \
        --num-gpus 8 \
        --output training_output/lightweight_comparison/eval_results_1p7b_ckpt400_iid.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from memgym.training.models.evaluator import (
    PredictionResult,
    compute_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _spawn_shard(
    shard_id: int,
    num_shards: int,
    gpu_id: int,
    forwarded_args: list[str],
    out_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m", "memgym.training.scripts.eval_aug_sft",
        "--shard-id", str(shard_id),
        "--num-shards", str(num_shards),
        "--output", str(out_path),
        *forwarded_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = out_path.parent / f"{out_path.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = log_path.open("w")
    logger.info(
        "spawn shard %d/%d on GPU %d  -> %s  (log: %s)",
        shard_id, num_shards, gpu_id, out_path.name, log_path.name,
    )
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)


def _load_predictions(shard_json: Path) -> list[PredictionResult]:
    d = json.loads(shard_json.read_text())
    preds = []
    for r in d.get("per_row_predictions", []):
        preds.append(PredictionResult(
            instance_id=r.get("instance_id", ""),
            step=int(r.get("step", 0) or 0),
            true_label=int(r["true_label"]),
            pred_label=int(r["pred_label"]),
            confidence=float(r.get("confidence", 0.5)),
            prob_safe=r.get("prob_safe"),
            source=r.get("source", ""),
            perturbation=r.get("perturbation", ""),
            source_dir=r.get("source_dir", ""),
        ))
    return preds


def _merge_truncation(shards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Recombine per-shard truncation stats into one unified report.

    Each shard's truncation block has counts on its slice; we sum the
    integer fields and recompute pct/percentiles from raw lengths if
    they were preserved (they may not be — we degrade gracefully).
    """
    blocks = [s.get("truncation") for s in shards if s.get("truncation")]
    if not blocks:
        return None
    total_truncated = sum(int(b.get("truncated_count", 0) or 0) for b in blocks)
    total_n = sum(int(b.get("prompt_length_n", 0) or 0) for b in blocks)
    # Some writers don't emit prompt_length_n; fall back to per-shard pct.
    if total_n == 0:
        # Best-effort: weight pcts by truncated_count proxy
        pct = (
            sum(float(b.get("truncated_pct", 0) or 0) for b in blocks) / len(blocks)
        )
    else:
        pct = 100.0 * total_truncated / total_n if total_n else 0.0
    out: dict[str, Any] = {
        "truncated_count": total_truncated,
        "truncated_pct": round(pct, 4),
        "n_shards": len(blocks),
    }
    # Forward p50/p95/max as the max across shards (conservative — surfaces
    # the longest prompt observed anywhere).
    for k in ("prompt_length_p50", "prompt_length_p95", "prompt_length_max"):
        vals = [b.get(k) for b in blocks if b.get(k) is not None]
        if vals:
            out[k] = max(vals) if "max" in k or "p95" in k else sorted(vals)[len(vals) // 2]
    out["max_length"] = blocks[0].get("max_length")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="Number of GPUs to fan out across (one shard per GPU).")
    ap.add_argument("--gpu-ids", type=str, default=None,
                    help="Comma-separated GPU ids to use (overrides --num-gpus). "
                         "E.g. '0,1,2,3'.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Final merged eval JSON path.")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Keep per-shard JSONs after merging (default: keep).")
    # Pass-through args for eval_aug_sft.py
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label-safe", default=" Y")
    ap.add_argument("--label-harmful", default=" N")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--contamination-note", default=None)
    ap.add_argument("--system-prompt-prefix", default="",
                    help="Forwarded verbatim to eval_aug_sft.py --system-prompt-prefix.")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Alias for --num-gpus (for callers that prefer shard terminology).")
    args = ap.parse_args()

    # --num-shards is an alias for --num-gpus; --num-shards takes precedence.
    if args.num_shards is not None:
        args.num_gpus = args.num_shards

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = list(range(args.num_gpus))
    n_shards = len(gpu_ids)
    if n_shards < 1:
        logger.error("no GPUs specified")
        return 2

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        out_path.with_name(f"{out_path.stem}.shard{i}{out_path.suffix}")
        for i in range(n_shards)
    ]

    forwarded = [
        "--checkpoint", args.checkpoint,
        "--base-model", args.base_model,
        "--pairs", args.pairs,
        "--max-length", str(args.max_length),
        "--label-safe", args.label_safe,
        "--label-harmful", args.label_harmful,
    ]
    if args.limit:
        forwarded += ["--limit", str(args.limit)]
    if args.no_4bit:
        forwarded += ["--no-4bit"]
    if args.contamination_note:
        forwarded += ["--contamination-note", args.contamination_note]
    if args.system_prompt_prefix:
        forwarded += ["--system-prompt-prefix", args.system_prompt_prefix]

    t0 = time.time()
    procs = [
        _spawn_shard(i, n_shards, gpu_ids[i], forwarded, shard_paths[i])
        for i in range(n_shards)
    ]
    rcs = []
    for i, p in enumerate(procs):
        rc = p.wait()
        rcs.append(rc)
        logger.info("shard %d (gpu %d) exited rc=%d", i, gpu_ids[i], rc)
    elapsed = time.time() - t0

    landed: list[dict] = []
    for sp in shard_paths:
        if sp.exists():
            try:
                landed.append(json.loads(sp.read_text()))
            except Exception as e:
                logger.warning("could not parse %s: %s", sp, e)
    if not landed:
        logger.error("no shard JSONs produced; aborting merge")
        return 3
    if len(landed) < n_shards:
        logger.warning(
            "only %d/%d shard JSONs produced — merging partial", len(landed), n_shards
        )

    all_preds: list[PredictionResult] = []
    for sj_path in shard_paths:
        if sj_path.exists():
            all_preds += _load_predictions(sj_path)
    metrics = compute_metrics(all_preds)
    payload = {
        "checkpoint": args.checkpoint,
        "pairs": args.pairs,
        "n_eval_rows": len(all_preds),
        "max_length": args.max_length,
        "system_prompt_prefix": args.system_prompt_prefix,
        "shards": {
            "n_planned": n_shards,
            "n_landed": len(landed),
            "rcs": rcs,
            "wall_time_sec": round(elapsed, 1),
            "gpu_ids": gpu_ids,
        },
        "truncation": _merge_truncation(landed),
        "metrics": dataclasses.asdict(metrics),
        "per_row_predictions": [dataclasses.asdict(p) for p in all_preds],
    }
    if args.contamination_note:
        payload["contamination_note"] = args.contamination_note
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "merged %d shards (%d rows) -> %s  (wall=%.1fs, AUROC=%s, safe_f1=%.3f, harmful_f1=%.3f, ece=%s)",
        len(landed), len(all_preds), out_path, elapsed,
        f"{metrics.auroc:.4f}" if metrics.auroc is not None else "n/a",
        metrics.safe_f1 or 0.0, metrics.harmful_f1 or 0.0,
        f"{metrics.ece:.4f}" if metrics.ece is not None else "n/a",
    )
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
