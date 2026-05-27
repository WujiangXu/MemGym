"""``memgym-eval-memory`` console script entry point.

Evaluate a memory strategy by applying it to raw trajectory steps and
scoring each (memory-compressed context, action) pair with the MemRM.
The output is a *relative* memory-quality metric — see the
:mod:`memgym.training.eval.memory_eval` module docstring for the
calibration caveat.

Examples
--------

Score the baseline ``passthrough`` memory on a custom JSONL::

    memgym-eval-memory \\
        --memory-model passthrough \\
        --trajectories sample_trajectories.jsonl \\
        --checkpoint MemGym/memgym-rm-1p7b

Score a built-in summarizing memory with custom args::

    memgym-eval-memory \\
        --memory-model naive_summarization \\
        --memory-args '{"max_tokens": 4000, "keep_first": 1}' \\
        --trajectories sample_trajectories.jsonl \\
        --checkpoint MemGym/memgym-rm-1p7b

Score a third-party memory registered from a user-provided module::

    memgym-eval-memory \\
        --memory-model my_memory \\
        --memory-module mypkg.my_memory \\
        --trajectories sample_trajectories.jsonl \\
        --checkpoint MemGym/memgym-rm-1p7b

Compare two memories on the same JSONL (``--output`` writes the JSON
side-by-side so a sibling script can diff them)::

    memgym-eval-memory --memory-model passthrough          --trajectories t.jsonl --checkpoint MemGym/memgym-rm-1p7b --output a.json
    memgym-eval-memory --memory-model naive_summarization  --trajectories t.jsonl --checkpoint MemGym/memgym-rm-1p7b --output b.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from memgym.training.eval.memory_eval import (
    MemoryEvalReport,
    RowResult,
    evaluate_memory,
    format_memory_report,
    load_trajectory_rows,
    report_to_dict,
)
from memgym.training.eval.rm_eval import resolve_checkpoint

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memgym-eval-memory",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--memory-model", required=True,
        help="Registered memory name (e.g. 'passthrough', "
             "'naive_summarization', or a name registered via "
             "register_memory_model() in --memory-module).",
    )
    p.add_argument(
        "--memory-args", default="{}",
        help="JSON dict of kwargs passed to the memory constructor "
             "(e.g. '{\"max_tokens\": 4000}'). Default: '{}'.",
    )
    p.add_argument(
        "--memory-module", default=None,
        help="Optional Python module path that registers a custom memory "
             "via register_memory_model() at import time. Imported BEFORE "
             "looking up --memory-model.",
    )
    p.add_argument(
        "--trajectories", required=True, type=Path,
        help="Path to a JSONL file (or directory of JSONLs) with one "
             "trajectory step per line. See module docstring for schema.",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Local path to the MemRM checkpoint OR an HF model repo id "
             "(e.g. MemGym/memgym-rm-1p7b).",
    )
    p.add_argument(
        "--base-model", default="Qwen/Qwen3-1.7B-Base",
        help="Base model the LoRA adapters attach to.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.88,
        help="Decision threshold t* for coverage and acc@t.",
    )
    p.add_argument(
        "--max-context-chars", type=int, default=24000,
        help="Truncate the memory's serialized output to this many "
             "characters before rendering the RM prompt (defense against "
             "a degenerate memory blowing past the tokenizer cap).",
    )
    p.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu | cuda:N.",
    )
    p.add_argument(
        "--no-4bit", dest="load_in_4bit", action="store_false", default=True,
        help="Disable NF4 quantization (load adapters in bf16).",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples for AUROC 95%% CI. 0 disables.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Write per-row results + report JSON here.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Cap rows (0 = all).",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="HF read token. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--hf-cache-dir", default=None,
        help="Override the HuggingFace cache directory.",
    )
    p.add_argument(
        "--label-safe", default=" Y",
        help="Single-token SAFE completion the checkpoint was trained on.",
    )
    p.add_argument(
        "--label-harmful", default=" N",
        help="Single-token HARMFUL completion.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress logs.",
    )
    return p


def _maybe_pin_gpu(device: str) -> None:
    if device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]


def _resolve_trajectory_paths(arg: Path) -> List[Path]:
    if arg.is_dir():
        paths = sorted(arg.glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"no .jsonl files found under {arg}")
        return paths
    if not arg.exists():
        raise SystemExit(f"--trajectories {arg} does not exist")
    return [arg]


def _parse_memory_args(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--memory-args is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"--memory-args must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _load_model(args: argparse.Namespace):
    # Lazy imports so --help works without torch / transformers installed.
    from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

    ckpt_path = resolve_checkpoint(
        args.checkpoint,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        cache_dir=args.hf_cache_dir,
    )
    config = ModelConfig(
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        safe_completion=args.label_safe,
        harmful_completion=args.label_harmful,
    )
    logger.info("loading MemRM from %s (base=%s, 4bit=%s)",
                ckpt_path, args.base_model, args.load_in_4bit)
    return MemoryWorldModel.from_checkpoint(ckpt_path, config=config)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.memory_module:
        # Import side-effect: the user's module is expected to call
        # `register_memory_model(...)` at import time.
        logger.info("importing user memory module: %s", args.memory_module)
        importlib.import_module(args.memory_module)

    memory_args = _parse_memory_args(args.memory_args)
    paths = _resolve_trajectory_paths(args.trajectories)
    rows = load_trajectory_rows(paths, limit=args.limit)
    logger.info("loaded %d trajectory rows from %d file(s)", len(rows), len(paths))

    _maybe_pin_gpu(args.device)
    model = _load_model(args)

    report, row_results = evaluate_memory(
        model, rows,
        memory_name=args.memory_model,
        memory_args=memory_args,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        max_context_chars=args.max_context_chars,
    )

    print(format_memory_report(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(report_to_dict(report, row_results), fh, indent=2)
        logger.info("wrote %d rows + report to %s", len(row_results), args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
