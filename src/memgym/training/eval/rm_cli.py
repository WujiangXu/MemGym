"""``memgym-eval-rm`` console script entry point.

Loads a MemGym Memory Reward Model checkpoint and scores it on one or
more of the publicly released RM evaluation datasets, reproducing the
paper's MemRM table (Tab. tab:memrm) shape.

Examples
--------

Reproduce the paper's IID-heldout AUROC (≈0.985) from a fresh clone::

    memgym-eval-rm --dataset iid-heldout \\
        --checkpoint MemGym/memgym-rm-1p7b \\
        --output iid_results.json

Sweep every RM-compatible dataset and print a markdown table::

    memgym-eval-rm --dataset all \\
        --checkpoint MemGym/memgym-rm-1p7b \\
        --output all_results.json

Score on a custom JSONL (any file with ``prompt`` + ``label`` columns)::

    memgym-eval-rm --dataset path/to/my_pairs.jsonl \\
        --checkpoint ./checkpoints/my_rm

Pass ``--dataset train-sanity`` to score the model on its training set
as a sanity check — the CLI prints a banner so you do not mistake the
near-perfect numbers for generalization.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from memgym.training.eval.rm_eval import (
    DATASET_REGISTRY,
    DatasetReport,
    DatasetSpec,
    evaluate_dataset,
    format_results_table,
    load_rows,
    report_to_dict,
    resolve_checkpoint,
    resolve_dataset,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memgym-eval-rm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True,
        help=(
            "Registry short name, 'all', or a path to a JSONL/directory. "
            f"Short names: {', '.join(sorted(DATASET_REGISTRY))}"
        ),
    )
    p.add_argument(
        "--checkpoint", required=True,
        help=(
            "Local path to a LoRA adapter dir OR an HF model repo id "
            "(e.g. MemGym/memgym-rm-1p7b). HF repos are auto-downloaded "
            "via huggingface_hub.snapshot_download."
        ),
    )
    p.add_argument(
        "--base-model", default="Qwen/Qwen3-1.7B-Base",
        help="Base model the LoRA adapters attach to. Default matches "
             "the public MemRM checkpoint.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.88,
        help="Decision threshold t* for coverage and acc@t metrics. "
             "Default 0.88 matches the paper's pre-declared threshold.",
    )
    p.add_argument(
        "--max-length", type=int, default=32768,
        help="Tokenizer truncation reference cap. Default 32768 matches "
             "the MemRM training context.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Reserved for future batched inference. Values >1 currently "
             "emit a warning and fall back to single-row forward passes.",
    )
    p.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu | cuda:N. 'auto' lets accelerate decide. "
             "When set to 'cuda:N', the script also pins CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--no-4bit", dest="load_in_4bit", action="store_false", default=True,
        help="Disable NF4 quantization (load adapters in bf16). Higher VRAM.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples for AUROC 95%% CI. 0 disables CI.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write per-row predictions + metrics JSON. "
             "If omitted, only the stdout report is produced.",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="HF read token for private repos. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--hf-cache-dir", default=None,
        help="Override the HuggingFace cache directory.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Cap rows per dataset (0 = all). Useful for smoke checks.",
    )
    p.add_argument(
        "--system-prompt-prefix", default="",
        help="Optional string prepended to every prompt before scoring.",
    )
    p.add_argument(
        "--label-safe", default=" Y",
        help="Single-token SAFE completion string. Must match what the "
             "checkpoint was trained against.",
    )
    p.add_argument(
        "--label-harmful", default=" N",
        help="Single-token HARMFUL completion string. Must match training.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress logs (errors still print).",
    )
    return p


def _maybe_pin_gpu(device: str) -> None:
    if device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_id = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        logger.info("pinned CUDA_VISIBLE_DEVICES=%s", gpu_id)


def _load_model(args: argparse.Namespace):
    # Imported lazily so --help works without torch / transformers installed.
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


def _maybe_warn_max_length(spec: Optional[DatasetSpec], max_length: int) -> None:
    if spec is not None and max_length < spec.min_max_length:
        logger.warning(
            "dataset %s expects prompts up to ~%d tokens; --max-length=%d "
            "may truncate.",
            spec.short_name, spec.min_max_length, max_length,
        )


def _selected_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "all":
        return [
            "iid-heldout",
            "train-sanity",
            "scenario-ood-tau2",
            "scenario-ood-wa-long",
            "scenario-ood-webarena",
            "strategy-ood",
        ]
    return [dataset_arg]


def _emit_report(
    reports: List[DatasetReport],
    predictions_per_dataset: dict,
    output_path: Optional[Path],
) -> None:
    table = format_results_table(reports)
    print(table)
    if output_path is None:
        return
    payload = {
        "reports": [report_to_dict(r) for r in reports],
        "per_row_predictions": predictions_per_dataset,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", output_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.batch_size > 1:
        logger.warning(
            "--batch-size=%d ignored: batched forward passes are not yet "
            "implemented for predict_logits_text. Falling back to bs=1.",
            args.batch_size,
        )

    _maybe_pin_gpu(args.device)
    selections = _selected_datasets(args.dataset)
    model = _load_model(args)

    reports: List[DatasetReport] = []
    predictions_per_dataset = {}

    for sel in selections:
        jsonl_paths, spec = resolve_dataset(
            sel,
            hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
            cache_dir=args.hf_cache_dir,
        )
        _maybe_warn_max_length(spec, args.max_length)
        rows = load_rows(jsonl_paths, limit=args.limit)
        logger.info("scoring %s: %d rows from %d file(s)",
                    sel, len(rows), len(jsonl_paths))

        report, predictions = evaluate_dataset(
            model, rows,
            spec=spec,
            threshold=args.threshold,
            n_bootstrap=args.n_bootstrap,
            prefix=args.system_prompt_prefix,
        )
        reports.append(report)
        predictions_per_dataset[report.dataset] = [
            dataclasses.asdict(p) for p in predictions
        ]

    _emit_report(reports, predictions_per_dataset, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
