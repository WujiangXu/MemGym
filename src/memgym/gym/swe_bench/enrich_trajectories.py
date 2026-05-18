"""
Enrich SWE-bench trajectories with reeval results and token cost analysis.

Post-processes existing result directories to produce an enrichment.json that
links reeval outcomes (resolved/unresolved) to individual instances and computes
true compression ratios accounting for summarizer LLM overhead.

Non-destructive: never modifies original trajectory files.

Usage:
    python -m memgym.gym.swe_bench.enrich_trajectories \
        --result-dir results/gpt_oss_120b_llmsumm_ms100_100 \
        --baseline-dir results/gpt_oss_120b_baseline_100
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# File Discovery
# =============================================================================

def discover_files(result_dir: Path) -> Tuple[Path, List[Path], Optional[Path], Optional[Path]]:
    """Discover trajectory files, reeval reports, and summary in a result dir.

    Handles two conventions:
      - results/<dir>/trajectories/<instance>_*.json  (Convention B)
      - trajectories/<dir>/<instance>_*.json          (Convention A)

    Returns:
        (traj_dir, reeval_files, summary_path, preds_path)
    """
    traj_subdir = result_dir / "trajectories"
    traj_dir = traj_subdir if traj_subdir.is_dir() else result_dir

    reeval_files = sorted(result_dir.glob("*reeval*.json"))
    if not reeval_files:
        reeval_files = sorted(traj_dir.glob("*reeval*.json"))

    summary_path = None
    for candidate in [result_dir / "summary.json"] + sorted(result_dir.glob("run*_summary.json")):
        if candidate.exists():
            summary_path = candidate
            break

    preds_path = None
    for candidate in [result_dir / "preds.json"] + sorted(result_dir.glob("run*_preds.json")):
        if candidate.exists():
            preds_path = candidate
            break

    return traj_dir, reeval_files, summary_path, preds_path


def find_instance_ids(traj_dir: Path, preds_path: Optional[Path]) -> List[str]:
    """Collect all instance IDs from trajectory files or preds.json."""
    ids = set()
    for pattern in ["*_replay.json", "*_training.json"]:
        for f in traj_dir.glob(pattern):
            suffix = "_replay.json" if pattern.startswith("*_replay") else "_training.json"
            ids.add(f.name.removesuffix(suffix))
    for f in traj_dir.glob("*.traj.json"):
        ids.add(f.name.removesuffix(".traj.json"))

    if not ids and preds_path and preds_path.exists():
        with open(preds_path) as f:
            ids.update(json.load(f).keys())

    return sorted(ids)


# =============================================================================
# Reeval Parsing
# =============================================================================

def parse_reeval(reeval_files: List[Path]) -> Dict[str, bool]:
    """Parse reeval reports into instance_id → resolved mapping.

    Handles swebench harness format (completed_ids / incomplete_ids).
    """
    resolved_map: Dict[str, bool] = {}
    for path in reeval_files:
        with open(path) as f:
            data = json.load(f)

        # Schema v2 (swebench harness): resolved_ids + unresolved_ids are definitive
        if "resolved_ids" in data:
            for iid in data["resolved_ids"]:
                resolved_map[iid] = True
            for iid in data.get("unresolved_ids", []):
                resolved_map[iid] = False
            # error_ids are also unresolved
            for iid in data.get("error_ids", []):
                resolved_map.setdefault(iid, False)
            continue

        # Legacy format with results list
        if "results" in data and isinstance(data["results"], list):
            for r in data["results"]:
                iid = r.get("instance_id", "")
                if iid:
                    resolved_map[iid] = r.get("reward", 0) > 0

    return resolved_map


# =============================================================================
# Token Extraction
# =============================================================================

def extract_replay_tokens(replay_path: Path) -> Optional[Dict[str, Any]]:
    """Extract API token usage from _replay.json."""
    if not replay_path.exists():
        return None

    with open(replay_path) as f:
        data = json.load(f)

    total_prompt = 0
    total_completion = 0
    num_queries = 0

    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        usage = (msg.get("extra") or {}).get("response", {}).get("usage", {})
        if not usage:
            continue
        total_prompt += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        num_queries += 1

    return {
        "api_prompt_tokens": total_prompt,
        "api_completion_tokens": total_completion,
        "api_total_tokens": total_prompt + total_completion,
        "num_queries": num_queries,
        "status": data.get("status", "unknown"),
    }


def extract_training_metrics(training_path: Path) -> Optional[Dict[str, Any]]:
    """Extract memory metrics from _training.json (memory runs only)."""
    if not training_path.exists():
        return None

    with open(training_path) as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        return None

    total_original = 0
    total_filtered = 0
    num_condensations = 0
    total_summarizer_prompt = 0
    total_summarizer_completion = 0

    for s in steps:
        mem = s.get("memory", {})
        total_original += mem.get("original_tokens", 0)
        total_filtered += mem.get("filtered_tokens", 0)
        if mem.get("was_compacted"):
            num_condensations += 1
            # Per-condensation summarizer tokens (available after agent.py fix)
            total_summarizer_prompt += mem.get("summarizer_prompt_tokens", 0)
            total_summarizer_completion += mem.get("summarizer_completion_tokens", 0)

    return {
        "num_steps": len(steps),
        "total_original_tokens": total_original,
        "total_filtered_tokens": total_filtered,
        "num_condensations": num_condensations,
        "memory_strategy": data.get("memory_strategy", "unknown"),
        "summarizer_prompt_tokens": total_summarizer_prompt,
        "summarizer_completion_tokens": total_summarizer_completion,
        "summarizer_total_tokens": total_summarizer_prompt + total_summarizer_completion,
    }


# =============================================================================
# Instance Enrichment
# =============================================================================

def enrich_instance(
    instance_id: str,
    resolved: Optional[bool],
    replay_data: Optional[Dict],
    training_data: Optional[Dict],
    summary_instance: Optional[Dict],
    tokens_per_condensation: float,
) -> Dict[str, Any]:
    """Combine all sources into enriched metrics for one instance."""
    enriched: Dict[str, Any] = {"instance_id": instance_id, "resolved": resolved}

    # From summary.json instance_results
    if summary_instance:
        enriched["status"] = summary_instance.get("status")
        enriched["elapsed_seconds"] = summary_instance.get("elapsed_seconds")
        enriched["patch_size"] = summary_instance.get("patch_size")

    # From _replay.json (available for both baseline and memory)
    if replay_data:
        enriched["api_prompt_tokens"] = replay_data["api_prompt_tokens"]
        enriched["api_completion_tokens"] = replay_data["api_completion_tokens"]
        enriched["api_total_tokens"] = replay_data["api_total_tokens"]
        enriched["num_steps"] = replay_data["num_queries"]
        if not summary_instance:
            enriched["status"] = replay_data.get("status")

    # From _training.json (memory runs only)
    if training_data:
        enriched["original_tokens"] = training_data["total_original_tokens"]
        enriched["filtered_tokens"] = training_data["total_filtered_tokens"]
        enriched["num_condensations"] = training_data["num_condensations"]
        enriched["num_steps"] = training_data["num_steps"]
        enriched["memory_strategy"] = training_data["memory_strategy"]

        # Per-instance summarizer tokens: prefer exact (from _training.json per-step data
        # or summary.json per-instance memory_stats), fall back to estimation
        exact_summarizer = training_data.get("summarizer_total_tokens", 0)
        if not exact_summarizer and summary_instance:
            inst_ms = summary_instance.get("memory_stats", {})
            exact_summarizer = inst_ms.get("summarizer_total_tokens", 0)
        if exact_summarizer > 0:
            summarizer_tokens = exact_summarizer
            enriched["summarizer_tokens"] = summarizer_tokens
            enriched["summarizer_source"] = "exact"
        else:
            summarizer_tokens = int(training_data["num_condensations"] * tokens_per_condensation)
            enriched["summarizer_tokens"] = summarizer_tokens
            enriched["summarizer_source"] = "estimated"

        original = training_data["total_original_tokens"]
        filtered = training_data["total_filtered_tokens"]
        total_cost = filtered + summarizer_tokens

        enriched["naive_compression_ratio"] = round(original / filtered, 2) if filtered > 0 else None
        enriched["true_compression_ratio"] = round(original / total_cost, 2) if total_cost > 0 else None
        enriched["memory_overhead_pct"] = round(summarizer_tokens / total_cost * 100, 1) if total_cost > 0 else 0
        enriched["net_token_savings"] = original - total_cost
    else:
        # Baseline defaults
        enriched["num_condensations"] = 0
        enriched["summarizer_tokens"] = 0
        enriched["summarizer_source"] = "none"
        enriched["naive_compression_ratio"] = 1.0
        enriched["true_compression_ratio"] = 1.0
        enriched["memory_overhead_pct"] = 0.0
        enriched["net_token_savings"] = 0

    return enriched


def add_baseline_comparison(enriched: Dict, baseline: Dict) -> Dict:
    """Add baseline comparison fields to an enriched instance."""
    enriched["baseline_api_prompt_tokens"] = baseline.get("api_prompt_tokens")
    enriched["baseline_num_steps"] = baseline.get("num_steps")
    enriched["baseline_resolved"] = baseline.get("resolved")

    bl_tokens = baseline.get("api_prompt_tokens", 0)
    mem_tokens = enriched.get("api_prompt_tokens", 0)
    if bl_tokens and bl_tokens > 0:
        enriched["api_token_ratio_vs_baseline"] = round(mem_tokens / bl_tokens, 3)
        enriched["api_token_savings_vs_baseline"] = bl_tokens - mem_tokens

    bl_resolved = baseline.get("resolved")
    mem_resolved = enriched.get("resolved")
    if bl_resolved is not None and mem_resolved is not None:
        if mem_resolved and not bl_resolved:
            enriched["outcome_vs_baseline"] = "capability_win"
        elif not mem_resolved and bl_resolved:
            enriched["outcome_vs_baseline"] = "regression"
        elif mem_resolved and bl_resolved:
            enriched["outcome_vs_baseline"] = "both_resolved"
        else:
            enriched["outcome_vs_baseline"] = "neither_resolved"

    return enriched


# =============================================================================
# Main Pipeline
# =============================================================================

def enrich_result_dir(
    result_dir: Path,
    baseline_dir: Optional[Path] = None,
    reeval_override: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a result directory and produce enrichment data."""
    traj_dir, reeval_files, summary_path, preds_path = discover_files(result_dir)

    if reeval_override:
        reeval_files = [reeval_override]

    # Parse reeval
    resolved_map = parse_reeval(reeval_files) if reeval_files else {}
    if verbose:
        print(f"Reeval: {len(resolved_map)} instances, {sum(resolved_map.values())} resolved")

    # Load summary.json for config + instance_results + summarizer stats
    summary_data: Dict[str, Any] = {}
    instance_results_map: Dict[str, Dict] = {}
    if summary_path:
        with open(summary_path) as f:
            summary_data = json.load(f)
        for ir in summary_data.get("instance_results", []):
            instance_results_map[ir["instance_id"]] = ir

    # Compute tokens_per_condensation from summary aggregates
    ms = summary_data.get("memory_stats", {})
    total_summarizer_tokens = ms.get("summarizer_total_tokens", 0)
    total_summarizations = ms.get("total_summarizations", 0)
    tokens_per_condensation = total_summarizer_tokens / total_summarizations if total_summarizations > 0 else 0

    # Find all instance IDs
    instance_ids = find_instance_ids(traj_dir, preds_path)
    if verbose:
        print(f"Instances: {len(instance_ids)}")

    # Enrich each instance
    instances: Dict[str, Dict] = {}
    for iid in instance_ids:
        replay_data = extract_replay_tokens(traj_dir / f"{iid}_replay.json")
        training_data = extract_training_metrics(traj_dir / f"{iid}_training.json")
        resolved = resolved_map.get(iid)
        summary_inst = instance_results_map.get(iid)

        enriched = enrich_instance(iid, resolved, replay_data, training_data, summary_inst, tokens_per_condensation)
        instances[iid] = enriched

    # Baseline comparison
    baseline_instances: Dict[str, Dict] = {}
    if baseline_dir:
        bl_enrichment = enrich_result_dir(baseline_dir, verbose=False)
        baseline_instances = bl_enrichment.get("instances", {})
        for iid, enriched in instances.items():
            bl = baseline_instances.get(iid)
            if bl:
                add_baseline_comparison(enriched, bl)

    # Compute aggregates
    resolved_count = sum(1 for e in instances.values() if e.get("resolved") is True)
    total = len(instances)
    all_api_prompt = [e.get("api_prompt_tokens", 0) for e in instances.values() if e.get("api_prompt_tokens")]
    all_api_compl = [e.get("api_completion_tokens", 0) for e in instances.values() if e.get("api_completion_tokens")]
    all_true_cr = [e["true_compression_ratio"] for e in instances.values() if e.get("true_compression_ratio") is not None]
    all_overhead = [e["memory_overhead_pct"] for e in instances.values() if e.get("memory_overhead_pct", 0) > 0]

    config = summary_data.get("config", {})

    return {
        "metadata": {
            "result_dir": str(result_dir),
            "baseline_dir": str(baseline_dir) if baseline_dir else None,
            "generated_at": datetime.now().isoformat(),
            "memory_type": config.get("memory_type", "unknown"),
            "model": config.get("model", "unknown"),
            "reeval_source": str(reeval_files[0]) if reeval_files else None,
            "tokens_per_condensation": round(tokens_per_condensation, 1),
        },
        "aggregate": {
            "total_instances": total,
            "resolved": resolved_count,
            "resolve_rate": round(resolved_count / total, 4) if total > 0 else 0,
            "total_api_prompt_tokens": sum(all_api_prompt),
            "total_api_completion_tokens": sum(all_api_compl),
            "total_summarizer_tokens": total_summarizer_tokens,
            "total_summarizations": total_summarizations,
            "avg_true_compression_ratio": round(sum(all_true_cr) / len(all_true_cr), 2) if all_true_cr else 1.0,
            "avg_memory_overhead_pct": round(sum(all_overhead) / len(all_overhead), 1) if all_overhead else 0,
        },
        "instances": instances,
    }


def main():
    parser = argparse.ArgumentParser(description="Enrich trajectories with reeval results and token cost analysis")
    parser.add_argument("--result-dir", required=True, type=Path, help="Path to result directory")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Path to baseline result directory for comparison")
    parser.add_argument("--reeval", type=Path, default=None, help="Explicit reeval report path (auto-discovered if omitted)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: <result-dir>/enrichment.json)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.result_dir.exists():
        print(f"Error: result dir not found: {args.result_dir}")
        return

    enrichment = enrich_result_dir(
        args.result_dir,
        baseline_dir=args.baseline_dir,
        reeval_override=args.reeval,
        verbose=args.verbose,
    )

    output_path = args.output or (args.result_dir / "enrichment.json")
    with open(output_path, "w") as f:
        json.dump(enrichment, f, indent=2)

    agg = enrichment["aggregate"]
    print(f"Enriched {agg['total_instances']} instances → {output_path}")
    print(f"  Resolved: {agg['resolved']}/{agg['total_instances']} ({agg['resolve_rate']:.1%})")
    print(f"  API prompt tokens: {agg['total_api_prompt_tokens']:,}")
    print(f"  Summarizer tokens: {agg['total_summarizer_tokens']:,}")
    print(f"  Avg true compression: {agg['avg_true_compression_ratio']:.2f}x")
    if agg['avg_memory_overhead_pct'] > 0:
        print(f"  Avg memory overhead: {agg['avg_memory_overhead_pct']:.1f}%")

    if args.baseline_dir:
        outcomes = {}
        for e in enrichment["instances"].values():
            o = e.get("outcome_vs_baseline", "unknown")
            outcomes[o] = outcomes.get(o, 0) + 1
        print(f"  vs baseline: {outcomes}")


if __name__ == "__main__":
    main()
