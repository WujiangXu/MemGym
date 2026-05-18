#!/usr/bin/env python
"""
Compare results across memory strategies.

Reads summary.json from each strategy's output directory and prints
a comparison table with resolve rates, compression ratios, and costs.

Usage:
    python scripts/compare_results.py results/comparison_20250204_120000
    python scripts/compare_results.py results/comparison_*  # latest
"""

import json
import sys
from pathlib import Path


def load_strategy_results(base_dir: Path) -> dict:
    """Load summary.json from each strategy subdirectory."""
    results = {}
    for summary_path in sorted(base_dir.glob("*/summary.json")):
        strategy_name = summary_path.parent.name
        with open(summary_path) as f:
            results[strategy_name] = json.load(f)
    return results


def print_comparison_table(results: dict):
    """Print formatted comparison table."""
    if not results:
        print("No results found.")
        return

    # Header
    print()
    print("=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)

    # Config from first result
    first = next(iter(results.values()))
    config = first.get("config", {})
    print(f"Dataset: {config.get('dataset', '?')}  |  Model: {config.get('model', '?')}  |  Slice: {config.get('slice', '?')}")
    print()

    # Table header
    header = f"{'Strategy':<32} | {'Resolved':>10} | {'Submit%':>8} | {'Compression':>12} | {'Summarizations':>15} | {'Avg Time':>9}"
    print(header)
    print("-" * len(header))

    # Table rows
    for strategy, data in results.items():
        r = data.get("results", {})
        m = data.get("memory_stats", {})
        instances = data.get("instance_results", [])

        total = r.get("total_instances", 0)
        resolved = r.get("resolved", 0)
        submit_rate = r.get("submission_rate", 0) * 100
        compression = m.get("avg_compression_ratio", 1.0)
        summarizations = m.get("total_summarizations", 0)

        # Average elapsed time
        elapsed_times = [i.get("elapsed_seconds", 0) for i in instances]
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

        resolve_str = f"{resolved}/{total}"
        print(f"{strategy:<32} | {resolve_str:>10} | {submit_rate:>7.1f}% | {compression:>11.2f}x | {summarizations:>15} | {avg_time:>8.1f}s")

    print()

    # Per-instance breakdown
    print_instance_breakdown(results)


def print_instance_breakdown(results: dict):
    """Print per-instance results showing which strategies solved which instances."""
    # Collect all instance IDs
    all_instances = set()
    for data in results.values():
        for inst in data.get("instance_results", []):
            all_instances.add(inst["instance_id"])

    if not all_instances:
        return

    all_instances = sorted(all_instances)
    strategies = list(results.keys())

    print("=" * 100)
    print("PER-INSTANCE BREAKDOWN")
    print("=" * 100)

    # Truncate instance IDs for display
    max_id_len = 40

    # Header
    strat_short = [s[:8] for s in strategies]
    header = f"{'Instance':<{max_id_len}} | " + " | ".join(f"{s:>8}" for s in strat_short)
    print(header)
    print("-" * len(header))

    for instance_id in all_instances:
        display_id = instance_id if len(instance_id) <= max_id_len else instance_id[:max_id_len-3] + "..."
        cells = []
        for strategy in strategies:
            data = results[strategy]
            inst_results = {i["instance_id"]: i for i in data.get("instance_results", [])}
            inst = inst_results.get(instance_id, {})
            reward = inst.get("reward", 0)
            if inst.get("error"):
                cells.append("ERROR")
            elif reward > 0:
                cells.append("PASS")
            else:
                cells.append("FAIL")

        row = f"{display_id:<{max_id_len}} | " + " | ".join(f"{c:>8}" for c in cells)
        print(row)

    # Summary row
    print("-" * len(header))
    cells = []
    for strategy in strategies:
        data = results[strategy]
        resolved = data.get("results", {}).get("resolved", 0)
        total = data.get("results", {}).get("total_instances", 0)
        cells.append(f"{resolved}/{total}")
    row = f"{'TOTAL':<{max_id_len}} | " + " | ".join(f"{c:>8}" for c in cells)
    print(row)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_results.py <results_directory>")
        print("Example: python scripts/compare_results.py results/comparison_20250204_120000")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)

    results = load_strategy_results(base_dir)
    if not results:
        print(f"No summary.json files found in {base_dir}/*/")
        sys.exit(1)

    print(f"Found {len(results)} strategies: {', '.join(results.keys())}")
    print_comparison_table(results)


if __name__ == "__main__":
    main()
