#!/usr/bin/env python3
"""Benchmark memory methods across hop-depth strata.

Runs 4 memory methods × 3 hop strata (3/4/5-6 hop) and produces
a comparison table for the paper.

Usage:
    python -m scripts.run_ir_benchmark \
        --data-3hop data/memgym_ir_3hop/verified.jsonl \
        --data-4hop data/paper_run_final/verified.jsonl \
        --data-56hop data/memgym_ir_hop56_clean/combined.jsonl \
        --strategies ir_naive_rag,ir_bm25,ir_amem,ir_structured \
        --limit 50 \
        --output data/benchmark_results.json
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memgym.pipelines.memgym_ir.types.schemas import MemGymIRInstance
from memgym.pipelines.memgym_ir.llm.client import LLMClient
from memgym.pipelines.memgym_ir.eval.agent import evaluate_with_manager

# Trigger registration of all IR memory methods
import memgym.memory.ir  # noqa: F401

logger = logging.getLogger(__name__)

STRATA = ["3hop", "4hop", "56hop"]
DEFAULT_STRATEGIES = [
    "ir_passthrough",
    "ir_naive_rag",
    "ir_bm25",
    "ir_structured",
    "ir_amem",
    "ir_lightmem",
]


def load_instances(path: str, limit: int = 50) -> List[MemGymIRInstance]:
    """Load instances from JSONL file."""
    instances = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            instances.append(MemGymIRInstance(**json.loads(line)))
    return instances


def run_benchmark(
    data_paths: Dict[str, str],
    strategies: List[str],
    eval_client: LLMClient,
    limit: int = 50,
    summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    workers: int = 8,
) -> Dict:
    """Run full benchmark: strategies × strata."""
    results = {
        "config": {
            "strategies": strategies,
            "limit_per_stratum": limit,
            "summarization_model": summarization_model,
        },
        "per_stratum": {},
        "summary_table": [],
    }

    for stratum, path in data_paths.items():
        if not path or not Path(path).exists():
            logger.warning(f"Skipping {stratum}: file not found at {path}")
            continue

        instances = load_instances(path, limit=limit)
        logger.info(f"\n{'='*60}")
        logger.info(f"Stratum: {stratum} ({len(instances)} instances)")
        logger.info(f"{'='*60}")

        stratum_results = {
            "num_instances": len(instances),
            "strategies": {},
        }

        for strategy in strategies:
            logger.info(f"\n  Strategy: {strategy} (workers={workers})")
            scores = []
            recalls = []
            per_instance = []
            t0 = time.time()

            def _run_one(inst):
                try:
                    result = evaluate_with_manager(
                        inst, eval_client,
                        strategy=strategy,
                        summarization_model=summarization_model,
                    )
                    return {
                        "instance_id": result["instance_id"],
                        "score": result["score"],
                        "recall": result["note_fact_recall"],
                    }
                except Exception as e:
                    logger.error(f"    Error on {inst.instance_id}: {e}")
                    return {
                        "instance_id": inst.instance_id,
                        "score": 0.0,
                        "recall": 0.0,
                        "error": str(e),
                    }

            if workers <= 1:
                results_iter = (_run_one(inst) for inst in instances)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_run_one, inst) for inst in instances]
                    results_iter = [f.result() for f in as_completed(futures)]

            for r in results_iter:
                scores.append(r["score"])
                recalls.append(r["recall"])
                per_instance.append(r)

            elapsed = time.time() - t0
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_recall = sum(recalls) / len(recalls) if recalls else 0

            logger.info(f"    avg_score={avg_score:.3f}, avg_recall={avg_recall:.3f}, time={elapsed:.1f}s")

            stratum_results["strategies"][strategy] = {
                "avg_score": avg_score,
                "avg_recall": avg_recall,
                "elapsed_seconds": elapsed,
                "per_instance": per_instance,
            }

            results["summary_table"].append({
                "stratum": stratum,
                "strategy": strategy,
                "avg_score": avg_score,
                "avg_recall": avg_recall,
                "n": len(instances),
            })

        results["per_stratum"][stratum] = stratum_results

    return results


def print_table(results: Dict):
    """Print a formatted comparison table."""
    table = results.get("summary_table", [])
    if not table:
        print("No results to display.")
        return

    strategies = sorted(set(r["strategy"] for r in table))
    strata = sorted(set(r["stratum"] for r in table))

    # Header
    header = f"{'Strategy':<20}"
    for s in strata:
        header += f" {s:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for strategy in strategies:
        row = f"{strategy:<20}"
        for stratum in strata:
            match = [r for r in table if r["strategy"] == strategy and r["stratum"] == stratum]
            if match:
                row += f" {match[0]['avg_score']:>10.3f}"
            else:
                row += f" {'--':>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark IR memory methods across hop strata")
    parser.add_argument("--data-3hop", type=str, default=None)
    parser.add_argument("--data-4hop", type=str, default=None)
    parser.add_argument("--data-56hop", type=str, default=None)
    parser.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--summarization-model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel instance workers per strategy (I/O-bound Bedrock calls).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_paths = {}
    if args.data_3hop:
        data_paths["3hop"] = args.data_3hop
    if args.data_4hop:
        data_paths["4hop"] = args.data_4hop
    if args.data_56hop:
        data_paths["56hop"] = args.data_56hop

    if not data_paths:
        print("Error: provide at least one --data-* path")
        sys.exit(1)

    strategies = [s.strip() for s in args.strategies.split(",")]
    eval_client = LLMClient(model=args.model)

    results = run_benchmark(
        data_paths=data_paths,
        strategies=strategies,
        eval_client=eval_client,
        limit=args.limit,
        summarization_model=args.summarization_model,
        workers=args.workers,
    )

    # Print table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print_table(results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
