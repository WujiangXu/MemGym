"""CLI shim around ``memgym.training.data.tau2_long_context_pairs``.

Walks one or more τ2 ``_both_`` run dirs (each containing
``baseline/<domain>/<task>/`` and ``memory/<domain>/<task>/`` subtrees)
and emits a single long-context pairs JSONL whose rows match the v2 RM
training distribution (full conversation history + active compaction
summary + judgment turn).

Usage::

    python -m memgym.training.scripts.build_tau2_long_context_pairs \\
        --tau2-root ${REPO_ROOT}/results/tau2_bench \\
        --runs 20260412-024137_HP_full_summ_ms10_kl2 \\
               20260412-213744_HP_full_struct_ms10_kl4 \\
        --out data/world_model/training_output/reward_model_v2_scenario_ood/\\
              reward_model_pairs_tau2_longctx.jsonl

The output replaces the prior short-template
``reward_model_pairs_tau2.jsonl`` (62-1286-token prompts) with rows
whose prompt-length distribution overlaps the v2 RM training
distribution (~22-38K tokens), so ``eval_aug_sft.py --max-length 32768``
scores it without the prompt-distribution-mismatch confound that
inflated/deflated the prior τ2 OOD AUROC numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from memgym.training.data.reward_model_pairs import DEFAULT_KEEP_FIRST
from memgym.training.data.tau2_long_context_pairs import build_pairs_for_run

logger = logging.getLogger("build_tau2_long_context_pairs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build long-context τ2 OOD pairs (v2-RM-distribution-shaped).",
    )
    parser.add_argument(
        "--tau2-root", required=True, type=Path,
        help="Parent directory containing one subdir per τ2 _both_ run.",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run dirnames under --tau2-root to process (each must have "
             "baseline/ and memory/ children).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output JSONL path. Overwritten if it exists.",
    )
    parser.add_argument(
        "--keep-first", type=int, default=DEFAULT_KEEP_FIRST,
        help="Number of leading messages to always retain in the "
             "reconstructed view (matches the SWE pair builder).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.tau2_root.is_dir():
        logger.error("tau2-root not a directory: %s", args.tau2_root)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overall: Counter = Counter()
    n_runs_ok = 0
    with args.out.open("w") as fh:
        for run_name in args.runs:
            run_dir = args.tau2_root / run_name
            if not run_dir.is_dir():
                logger.warning("skipping missing run dir: %s", run_dir)
                overall["runs_skipped_missing"] += 1
                continue
            logger.info("building pairs for run: %s", run_name)
            stats = build_pairs_for_run(run_dir, fh, keep_first=args.keep_first)
            for k, v in stats.items():
                overall[k] += v
            overall["runs_processed"] += 1
            n_runs_ok += 1
            logger.info("  run %s stats: %s", run_name,
                        {k: v for k, v in stats.items() if v})

    summary = {
        "out_path": str(args.out),
        "runs_processed": n_runs_ok,
        "stats": dict(overall),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if n_runs_ok > 0 and overall.get("pairs_written", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
