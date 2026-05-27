#!/usr/bin/env python3
"""Re-evaluate existing patches using the official SWE-bench harness.

Uses swebench.harness.run_evaluation — the same pipeline as sb-cli and the
SWE-bench leaderboard. No custom test command building or output parsing.

Usage:
    python scripts/reeval_patches.py \
        --preds results/gpt5_baseline_50/preds.json \
        --dataset lite --slice 0:50 --workers 4 \
        --run-id memgym-reeval

    # Evaluate specific instances
    python scripts/reeval_patches.py \
        --preds results/gpt5_baseline_50/preds.json \
        --dataset lite --instance-ids django__django-11133 django__django-12747

    # Train split (no pre-built images on DockerHub, builds locally)
    python scripts/reeval_patches.py \
        --preds results/train100_sonnet45_none/preds.json \
        --dataset full --split train --slice 0:100 --workers 4 \
        --run-id train100-sonnet45-none --clean
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for MemGym imports

from datasets import load_dataset as hf_load
from swebench.harness.run_evaluation import main as swebench_eval


def convert_preds_format(preds_path: str, output_path: str) -> int:
    """Convert MemGym preds.json (dict keyed by instance_id) to SWE-bench format (list)."""
    with open(preds_path) as f:
        preds = json.load(f)

    if isinstance(preds, dict):
        swebench_preds = [
            {
                "instance_id": iid,
                "model_patch": p.get("model_patch", ""),
                "model_name_or_path": p.get("model_name_or_path", "unknown"),
            }
            for iid, p in preds.items()
        ]
    elif isinstance(preds, list):
        swebench_preds = preds
    else:
        raise ValueError(f"Unexpected preds format: {type(preds)}")

    with open(output_path, "w") as f:
        json.dump(swebench_preds, f, indent=2)
    return len(swebench_preds)


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate patches with official SWE-bench harness")
    parser.add_argument("--preds", required=True, help="Path to MemGym preds.json")
    parser.add_argument("--dataset", default="lite", choices=["lite", "verified", "full", "swe-gym", "swe-smith"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--slice", default=None, help="Instance slice e.g. 0:50")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--run-id", default="memgym-reeval", help="Run ID for logs")
    parser.add_argument("--timeout", type=int, default=300, help="Per-instance timeout")
    parser.add_argument("--instance-ids", nargs="+", default=None, help="Specific instance IDs")
    parser.add_argument("--report-dir", default=None, help="Custom report directory")
    parser.add_argument("--cache-level", default=None,
                        choices=["none", "base", "env", "instance"],
                        help="Docker image cache level (default: 'instance' for remote images, "
                             "'env' for local builds). 'instance' keeps pulled images to avoid "
                             "re-downloading. 'env' deletes instance images after each eval.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove instance images after each evaluation (saves storage)")
    parser.add_argument("--namespace", default=None,
                        help="Docker namespace for pre-built images (default: auto-detect. "
                             "'swebench' for test split, None for train split which builds locally)")
    args = parser.parse_args()

    # SWE-Gym and SWE-smith only have train split
    if args.dataset in ("swe-gym", "swe-smith") and args.split == "test":
        args.split = "train"

    ds_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "swe-gym": "SWE-Gym/SWE-Gym",
        "swe-smith": "SWE-bench/SWE-smith",
    }
    dataset_name = ds_map[args.dataset]

    # Convert predictions to SWE-bench format (list of dicts)
    converted = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    n_preds = convert_preds_format(args.preds, converted.name)
    converted.close()
    print(f"Converted {n_preds} predictions to SWE-bench format")

    # Resolve instance IDs from slice
    instance_ids = args.instance_ids
    if args.slice and not instance_ids:
        data = list(hf_load(dataset_name, split=args.split))
        parts = args.slice.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(data)

        with open(args.preds) as f:
            preds_data = json.load(f)
        pred_ids = set(preds_data.keys()) if isinstance(preds_data, dict) else {p["instance_id"] for p in preds_data}

        instance_ids = [d["instance_id"] for d in data[start:end] if d["instance_id"] in pred_ids]
        print(f"Evaluating {len(instance_ids)} instances (slice {args.slice})")

    # Auto-detect namespace: pre-built images exist on DockerHub
    # SWE-Gym uses xingyaoww namespace, SWE-bench test split uses swebench namespace
    namespace = args.namespace
    if namespace is None:
        if args.dataset in ("swe-gym",):
            namespace = "xingyaoww"
            print(f"Auto-detected namespace='xingyaoww' (SWE-Gym pre-built images)")
        elif args.dataset in ("swe-smith",):
            namespace = "jyangballin"
            print(f"Auto-detected namespace='jyangballin' (SWE-smith pre-built images)")
        elif args.split == "test":
            namespace = "swebench"
            print(f"Auto-detected namespace='swebench' (pre-built test images on DockerHub)")
        else:
            print(f"No namespace (split={args.split}): building images locally")

    # Auto-detect cache level: use 'instance' for remote images (avoids re-pulling
    # 500MB-2GB images that were just downloaded), 'env' for local builds.
    if args.cache_level is None:
        if namespace is not None:
            args.cache_level = "instance"
            print(f"Auto-detected cache_level='instance' (remote images — avoids re-pulling)")
        else:
            args.cache_level = "env"
            print(f"Auto-detected cache_level='env' (local builds)")

    # Patch swebench with SWE-Gym repo specs if needed
    if args.dataset in ("swe-gym",):
        from memgym.utils.swegym_specs import patch_swebench_specs
        patch_swebench_specs()

    # Call the official SWE-bench evaluation
    print(f"\nRunning official SWE-bench harness (run_id={args.run_id})...")
    print(f"  cache_level={args.cache_level}, clean={args.clean}, namespace={namespace}")
    print(f"  Logs: ./logs/run_evaluation/{args.run_id}/\n")

    start = time.time()
    swebench_eval(
        dataset_name=dataset_name,
        split=args.split,
        instance_ids=instance_ids,
        predictions_path=converted.name,
        max_workers=args.workers,
        force_rebuild=False,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=4096,
        run_id=args.run_id,
        timeout=args.timeout,
        namespace=namespace,
        rewrite_reports=False,
        modal=False,
        report_dir=args.report_dir or ".",
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results in: ./logs/run_evaluation/{args.run_id}/")

    # Copy report JSON to preds directory for API access
    import shutil, glob
    report_dir = args.report_dir or "."
    preds_dir = Path(args.preds).parent
    for report_file in glob.glob(f"*.{args.run_id}.json") + glob.glob(f"{report_dir}/*.{args.run_id}.json"):
        dest = preds_dir / Path(report_file).name
        if Path(report_file).resolve() != dest.resolve():
            shutil.copy2(report_file, dest)
            print(f"Copied report to {dest}")

    # Clean up temp file
    Path(converted.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
