#!/usr/bin/env python
"""
SWE-bench Evaluation with Memory Management.

Official mini-swe-agent settings for proper benchmark performance.
Supports baseline (no memory) and memory-managed runs.

Usage:
    # Run BASELINE (no memory) - 5 instances test
    python scripts/evaluate_swe_bench.py \
        --dataset lite --model gpt-4o-mini --memory none \
        --slice 0:5 --output results/baseline --verbose

    # Run WITH MEMORY - 5 instances test
    python scripts/evaluate_swe_bench.py \
        --dataset lite --model gpt-4o-mini --memory naive \
        --max-tokens 4000 --slice 0:5 --output results/naive_memory --verbose

    # Full evaluation (50 instances)
    python scripts/evaluate_swe_bench.py \
        --dataset verified --model gpt-4o --memory naive \
        --workers 2 --slice 0:50 --output results/verified_naive

    # Submit for evaluation (after running)
    sb-cli submit swe-bench_lite test \
        --predictions_path results/baseline/preds.json --run_id baseline-v1
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# MemGym imports

from datasets import load_dataset

# === OFFICIAL mini-swe-agent SETTINGS ===
# Source: https://mini-swe-agent.com/latest/usage/swebench/
OFFICIAL_SETTINGS = {
    "step_limit": 250,       # Max steps per instance
    "cost_limit": 3.0,       # Max cost USD per instance
    "timeout": 60,           # Seconds per command
    "working_dir": "/testbed",
    "env_vars": {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1"
    }
}

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
    "swe-gym": "SWE-Gym/SWE-Gym",
    "swe-smith": "SWE-bench/SWE-smith",
}

# Thread-safe lock for incremental prediction saves (matches mini-swe-agent pattern)
_PREDS_LOCK = threading.Lock()


def update_preds_file(output_dir: Path, model: str, result: Dict) -> None:
    """Thread-safe incremental prediction save.

    Reads existing preds.json, merges the new result, and writes back.
    Safe for concurrent access from multiple worker threads.
    """
    preds_path = output_dir / "preds.json"
    with _PREDS_LOCK:
        preds = {}
        if preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)
        preds[result["instance_id"]] = {
            "model_name_or_path": model,
            "instance_id": result["instance_id"],
            "model_patch": result["patch"],
            "status": result.get("status", "Submitted"),
        }
        with open(preds_path, "w") as f:
            json.dump(preds, f, indent=2)


def create_memory_model(memory_type: str, args):
    """Create memory model based on type with strategy-specific parameters."""
    from memgym.memory import (
        AdaptiveTokenBudgetMemory,
        NaiveSummarizationMemory,
        PassThroughMemory,
        ObservationMaskingMemory,
        LLMSummarizingMemory,
        StructuredSummaryMemory,
        PipelineMemory,
        SlidingWindowSummaryMemory,
    )

    max_tokens = args.max_tokens

    if memory_type == "naive":
        return NaiveSummarizationMemory(
            max_tokens=max_tokens,
            env_type="swe",
            summarization_model=args.summarization_model,
        )
    elif memory_type in ("none", "passthrough"):
        return PassThroughMemory(max_tokens=max_tokens)
    elif memory_type == "observation_masking":
        return ObservationMaskingMemory(
            attention_window=args.attention_window,
            keep_first=args.keep_first,
            selective=getattr(args, 'selective_masking', True),
            max_tokens=max_tokens,
        )
    elif memory_type == "llm_summarizing":
        return LLMSummarizingMemory(
            max_size=args.max_size,
            keep_first=args.keep_first,
            condensation_ratio=args.condensation_ratio,
            summarization_model=args.summarization_model,
            max_tokens=max_tokens,
            max_token_size=getattr(args, 'max_token_size', None),
        )
    elif memory_type == "structured_summary":
        return StructuredSummaryMemory(
            max_size=args.max_size,
            keep_first=args.keep_first,
            condensation_ratio=args.condensation_ratio,
            summarization_model=args.summarization_model,
            max_tokens=max_tokens,
            max_token_size=getattr(args, 'max_token_size', None),
        )
    elif memory_type == "pipeline_masking_summarizing":
        return PipelineMemory(
            stages=[
                ObservationMaskingMemory(
                    attention_window=args.attention_window,
                    keep_first=args.keep_first,
                    selective=getattr(args, 'selective_masking', True),
                    max_tokens=max_tokens,
                ),
                LLMSummarizingMemory(
                    max_size=args.max_size,
                    keep_first=args.keep_first,
                    condensation_ratio=args.condensation_ratio,
                    summarization_model=args.summarization_model,
                    max_tokens=max_tokens,
                ),
            ],
            max_tokens=max_tokens,
        )
    elif memory_type == "pipeline_masking_structured":
        return PipelineMemory(
            stages=[
                ObservationMaskingMemory(
                    attention_window=args.attention_window,
                    keep_first=args.keep_first,
                    selective=getattr(args, 'selective_masking', True),
                    max_tokens=max_tokens,
                ),
                StructuredSummaryMemory(
                    max_size=args.max_size,
                    keep_first=args.keep_first,
                    condensation_ratio=args.condensation_ratio,
                    summarization_model=args.summarization_model,
                    max_tokens=max_tokens,
                ),
            ],
            max_tokens=max_tokens,
        )
    elif memory_type == "sliding_window":
        return SlidingWindowSummaryMemory(
            window_size=getattr(args, 'window_size', 20),
            keep_first=args.keep_first,
            summarization_model=args.summarization_model,
            env_type="swe",
            max_tokens=max_tokens,
        )
    elif memory_type == "adaptive_token_budget":
        return AdaptiveTokenBudgetMemory(
            max_tokens=max_tokens,
            keep_recent=args.keep_recent,
            keep_first=args.keep_first,
            preserve_first_user=args.preserve_first_user,
            summarization_model=args.summarization_model,
            env_type="swe",
        )
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")


def process_instance(
    instance_id: str,
    dataset: str,
    model: str,
    memory_type: str,
    args,
    env_type: str,
    docker_image_dir: str,
    docker_image_source: str,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Process single SWE-bench instance."""
    from memgym.envs import SWEMemoryEnv

    # Create fresh memory model for this instance
    memory_model = create_memory_model(memory_type, args)
    use_context_management = memory_type != "none"

    # Determine if using Docker-based environment
    use_docker = env_type in ("docker", "swerex_docker", "singularity", "bubblewrap")

    # Build agent LLM args (supports local model via --api-base)
    agent_llm_args = {}
    if getattr(args, 'api_base', None):
        agent_llm_args["model_kwargs"] = {"base_url": args.api_base, "api_key": "local"}
        # litellm needs openai/ prefix to route to OpenAI-compatible endpoints
        if not model.startswith("openai/"):
            model = f"openai/{model}"

    # Create environment with official settings
    env = SWEMemoryEnv(
        dataset=dataset,
        split=args.split,
        instance_id=instance_id,
        agent_llm=model,
        agent_llm_args=agent_llm_args,
        memory_model=memory_model,
        use_context_management=use_context_management,
        use_docker=use_docker,
        environment_class=env_type,
        docker_image_dir=docker_image_dir,
        docker_image_source=docker_image_source,
        max_steps=OFFICIAL_SETTINGS["step_limit"],
        agent_type=getattr(args, 'agent', 'mini-swe'),
        skip_eval=getattr(args, 'skip_eval', False),
        bwrap_rootfs=getattr(args, 'bwrap_rootfs', None),
        bwrap_executable=getattr(args, 'bwrap_executable', None),
        bwrap_wrapper_args=getattr(args, 'bwrap_wrapper_args', None),
    )

    start_time = datetime.now()

    try:
        # Run episode
        result = env.run_episode(verbose=verbose)

        # Extract patch
        patch = result.get("patch", "")
        status = result.get("status", "Unknown")

        # Save trajectory
        traj_dir = output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"{instance_id}.traj.json"
        env.save_trajectory(str(traj_path))

        # Save step-based training trajectory
        training_traj = result.get("training_trajectory")
        if training_traj:
            training_path = traj_dir / f"{instance_id}_training.json"
            with open(training_path, "w") as f:
                json.dump(training_traj, f, indent=2)

        # Save replay data (full untruncated messages for replay mechanism)
        replay_data = result.get("replay_data")
        if replay_data:
            replay_path = traj_dir / f"{instance_id}_replay.json"
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2, default=str)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "instance_id": instance_id,
            "status": status,
            "patch": patch,
            "patch_size": len(patch),
            "reward": result.get("reward", 0),
            "elapsed_seconds": elapsed,
            "memory_stats": memory_model.get_stats() if memory_model else {},
            "wrapper_stats": result.get("wrapper_stats", {}),
            "error": None
        }

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_msg = f"{type(e).__name__}: {str(e)}"
        if verbose:
            traceback.print_exc()

        return {
            "instance_id": instance_id,
            "status": "Error",
            "patch": f"# Error: {error_msg}",
            "patch_size": 0,
            "reward": 0,
            "elapsed_seconds": elapsed,
            "memory_stats": {},
            "wrapper_stats": {},
            "error": error_msg
        }
    finally:
        env.close()


def save_predictions(results: List[Dict], output_dir: Path, model: str) -> Path:
    """Save predictions in official SWE-bench format for sb-cli."""
    preds = {}
    for r in results:
        preds[r["instance_id"]] = {
            "model_name_or_path": model,
            "instance_id": r["instance_id"],
            "model_patch": r["patch"]
        }

    preds_path = output_dir / "preds.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)

    return preds_path


def save_summary(results: List[Dict], args, output_dir: Path) -> Path:
    """Save evaluation summary with statistics."""
    # Calculate statistics
    total = len(results)
    submitted = sum(1 for r in results if r["status"] == "Submitted")
    resolved = sum(1 for r in results if r.get("reward", 0) > 0)
    errors = sum(1 for r in results if r["status"] == "Error")
    limits_exceeded = sum(1 for r in results if r["status"] == "LimitsExceeded")
    empty_patches = sum(1 for r in results if r["patch_size"] == 0)

    total_summarizations = 0
    total_filtered = 0
    compression_ratios = []
    avg_raw_before = []
    avg_raw_after = []
    peak_raw_before = []
    peak_raw_after = []
    total_summarizer_prompt_tokens = 0
    total_summarizer_completion_tokens = 0

    for r in results:
        memory_stats = r.get("memory_stats") or {}
        if memory_stats:
            total_summarizations += memory_stats.get(
                "summary_trigger_count",
                memory_stats.get("summarization_count", 0),
            )
            total_summarizer_prompt_tokens += memory_stats.get("summarizer_total_prompt_tokens", 0)
            total_summarizer_completion_tokens += memory_stats.get("summarizer_total_completion_tokens", 0)
            if "avg_raw_tokens_before" in memory_stats:
                avg_raw_before.append(memory_stats.get("avg_raw_tokens_before", 0))
            if "avg_raw_tokens_after" in memory_stats:
                avg_raw_after.append(memory_stats.get("avg_raw_tokens_after", 0))
            if "peak_raw_tokens_before" in memory_stats:
                peak_raw_before.append(memory_stats.get("peak_raw_tokens_before", 0))
            if "peak_raw_tokens_after" in memory_stats:
                peak_raw_after.append(memory_stats.get("peak_raw_tokens_after", 0))

        agent_wrapper = (r.get("wrapper_stats") or {}).get("agent")
        if agent_wrapper:
            total_filtered += agent_wrapper.get("times_filtered", 0)
            if agent_wrapper.get("avg_compression_ratio"):
                compression_ratios.append(agent_wrapper["avg_compression_ratio"])

    avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
    avg_prompt_tokens_before = sum(avg_raw_before) / len(avg_raw_before) if avg_raw_before else 0
    avg_prompt_tokens_after = sum(avg_raw_after) / len(avg_raw_after) if avg_raw_after else 0
    peak_prompt_tokens_before = max(peak_raw_before) if peak_raw_before else 0
    peak_prompt_tokens_after = max(peak_raw_after) if peak_raw_after else 0

    summary = {
        "config": {
            "dataset": args.dataset,
            "agent": getattr(args, 'agent', 'mini-swe'),
            "model": args.model,
            "memory_type": args.memory,
            "max_tokens": args.max_tokens,
            "slice": args.slice,
            "env_type": args.env_type,
            "docker_image_source": args.docker_image_source,
            "timestamp": datetime.now().isoformat()
        },
        "results": {
            "total_instances": total,
            "submitted": submitted,
            "resolved": resolved,
            "resolve_rate": resolved / total if total > 0 else 0,
            "errors": errors,
            "limits_exceeded": limits_exceeded,
            "empty_patches": empty_patches,
            "submission_rate": submitted / total if total > 0 else 0
        },
        "memory_stats": {
            "total_summarizations": total_summarizations,
            "total_filtered": total_filtered,
            "avg_compression_ratio": avg_compression,
            "avg_prompt_tokens_before": avg_prompt_tokens_before,
            "avg_prompt_tokens_after": avg_prompt_tokens_after,
            "peak_prompt_tokens_before": peak_prompt_tokens_before,
            "peak_prompt_tokens_after": peak_prompt_tokens_after,
            "summarizer_total_prompt_tokens": total_summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": total_summarizer_completion_tokens,
            "summarizer_total_tokens": total_summarizer_prompt_tokens + total_summarizer_completion_tokens,
        },
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "status": r["status"],
                "reward": r.get("reward", 0),
                "patch_size": r["patch_size"],
                "elapsed_seconds": r["elapsed_seconds"],
                "error": r["error"],
                "memory_stats": {
                    k: v for k, v in (r.get("memory_stats") or {}).items()
                    if k in ("summarizer_total_prompt_tokens",
                             "summarizer_total_completion_tokens",
                             "summarizer_total_tokens",
                             "compaction_count", "summarization_count")
                },
            }
            for r in results
        ]
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path


def print_summary(results: List[Dict], args):
    """Print evaluation summary to console."""
    total = len(results)
    submitted = sum(1 for r in results if r["status"] == "Submitted")
    resolved = sum(1 for r in results if r.get("reward", 0) > 0)
    errors = sum(1 for r in results if r["status"] == "Error")
    limits_exceeded = sum(1 for r in results if r["status"] == "LimitsExceeded")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Memory: {args.memory}")
    print(f"Max tokens: {args.max_tokens}")
    print()
    print(f"Total instances: {total}")
    if total > 0:
        print(f"  Submitted: {submitted} ({100*submitted/total:.1f}%)")
        print(f"  Resolved (tests pass): {resolved}/{total} ({100*resolved/total:.1f}%)")
        print(f"  LimitsExceeded: {limits_exceeded}")
        print(f"  Errors: {errors}")
    else:
        print("  (no new instances processed)")
    print()

    # Memory stats
    if args.memory != "none":
        total_summarizations = sum(
            r.get("memory_stats", {}).get("summary_trigger_count", r.get("memory_stats", {}).get("summarization_count", 0))
            for r in results
        )
        avg_prompt_before = [
            r.get("memory_stats", {}).get("avg_raw_tokens_before")
            for r in results
            if r.get("memory_stats", {}).get("avg_raw_tokens_before") is not None
        ]
        avg_prompt_after = [
            r.get("memory_stats", {}).get("avg_raw_tokens_after")
            for r in results
            if r.get("memory_stats", {}).get("avg_raw_tokens_after") is not None
        ]
        print(f"Memory Statistics:")
        print(f"  Total summarizations: {total_summarizations}")
        if avg_prompt_before and avg_prompt_after:
            print(f"  Avg prompt tokens: {sum(avg_prompt_before)/len(avg_prompt_before):.0f} -> {sum(avg_prompt_after)/len(avg_prompt_after):.0f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Evaluation with Memory Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no memory)
  python scripts/evaluate_swe_bench.py --dataset lite --memory none --slice 0:5 -o results/baseline

  # With naive summarization memory
  python scripts/evaluate_swe_bench.py --dataset lite --memory naive --max-tokens 4000 --slice 0:5 -o results/naive

  # Full verified dataset
  python scripts/evaluate_swe_bench.py --dataset verified --memory naive --workers 2 -o results/verified
        """
    )

    # Dataset options
    parser.add_argument("--dataset", choices=["lite", "verified", "full", "swe-gym", "swe-smith"],
                        default="lite", help="SWE-bench dataset (default: lite)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--slice", default=None,
                        help="Slice instances (e.g., '0:50' for first 50)")
    parser.add_argument("--instances", nargs="+",
                        help="Specific instance IDs to evaluate")
    parser.add_argument("--instance-file",
                        help="Text file with instance IDs (one per line)")

    # Model options
    parser.add_argument("--model", "-m", default="openai/gpt-5-mini",
                        help="Agent LLM (default: openai/gpt-5-mini)")

    # Memory options
    parser.add_argument("--memory", choices=[
                            "naive", "none", "passthrough",
                            "observation_masking", "llm_summarizing", "structured_summary",
                            "pipeline_masking_summarizing", "pipeline_masking_structured",
                            "sliding_window", "adaptive_token_budget",
                        ],
                        default="naive", help="Memory model type (default: naive)")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="Max tokens before summarization (default: 4000)")

    # Strategy-specific options (OpenHands condenser params)
    parser.add_argument("--max-size", type=int, default=400,
                        help="Max events in view before condensation (llm_summarizing/structured_summary, default: 400)")
    parser.add_argument("--keep-first", type=int, default=3,
                        help="Initial messages to always preserve (default: 3)")
    parser.add_argument("--condensation-ratio", type=float, default=0.75,
                        help="Fraction of max_size to keep after condensation (default: 0.75)")
    parser.add_argument("--attention-window", type=int, default=100,
                        help="Recent messages to leave unmasked (observation_masking, default: 100)")
    parser.add_argument("--summarization-model", default=None,
                        help="Model for LLM summarization (default: same as --model)")
    parser.add_argument("--max-token-size", type=int, default=None,
                        help="Token-based trigger for condensation (llm_summarizing/structured_summary). "
                             "When set, triggers on token count instead of message count.")
    parser.add_argument("--window-size", type=int, default=20,
                        help="Recent messages to keep verbatim (sliding_window, default: 20)")
    parser.add_argument("--keep-recent", type=int, default=8,
                        help="Recent raw messages to keep verbatim (adaptive_token_budget, default: 8)")
    parser.add_argument("--preserve-first-user", action="store_true", default=True,
                        help="Preserve the first user issue statement when compacting (adaptive_token_budget)")
    parser.add_argument("--no-preserve-first-user", dest="preserve_first_user", action="store_false",
                        help="Do not pin the first user issue statement in adaptive_token_budget")
    parser.add_argument("--selective-masking", action="store_true", default=True,
                        help="Preserve error-containing observations outside attention window (observation_masking)")
    parser.add_argument("--no-selective-masking", dest="selective_masking", action="store_false",
                        help="Disable selective masking (mask all observations outside window)")

    # Agent options
    parser.add_argument("--agent", choices=["mini-swe", "codeact"],
                        default="mini-swe",
                        help="Agent scaffold (default: mini-swe)")

    # Local model options
    parser.add_argument("--api-base", default=None,
                        help="Custom API base URL for local model (e.g., http://localhost:30000/v1)")

    # Environment options
    parser.add_argument("--env-type", choices=["docker", "swerex_docker", "singularity", "bubblewrap", "local"],
                        default="docker", help="Environment type (default: docker)")
    parser.add_argument("--docker-image-dir", default="${HOME}/swe-bench/dockers",
                        help="Directory to save/cache Docker images")
    parser.add_argument("--docker-image-source", choices=["swebench", "epoch", "swe-gym", "swe-smith"],
                        default="swebench", help="Docker image source (default: swebench)")
    parser.add_argument("--bwrap-rootfs", default=None,
                        help="Path to pre-exported rootfs directory for bubblewrap env-type")
    parser.add_argument("--bwrap-executable", default=None,
                        help="Path to the bubblewrap executable (default: use MSWEA_BUBBLEWRAP_EXECUTABLE or 'bwrap')")
    parser.add_argument("--bwrap-wrapper-arg", action="append", default=[],
                        help="Extra shell-style wrapper arg fragment appended to the default bubblewrap args. "
                             "Repeatable, e.g. --bwrap-wrapper-arg='--bind /host/path /sandbox/path'")

    # Execution options
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--step-limit", type=int, default=None,
                        help="Override step limit (default: 250)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results (skip completed)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip Docker patch evaluation (use sb-cli submit instead)")

    args = parser.parse_args()

    # SWE-Gym and SWE-smith only have train split
    if args.dataset in ("swe-gym", "swe-smith") and args.split == "test":
        args.split = "train"

    # Auto-default docker image source for new datasets
    if args.dataset == "swe-gym" and args.docker_image_source == "swebench":
        args.docker_image_source = "swe-gym"
    elif args.dataset == "swe-smith" and args.docker_image_source == "swebench":
        args.docker_image_source = "swe-smith"

    # Override step limit if specified
    if args.step_limit is not None:
        OFFICIAL_SETTINGS["step_limit"] = args.step_limit

    # Default summarizer = same as reasoning model (user can override via --summarization-model)
    if args.summarization_model is None:
        args.summarization_model = args.model

    # Check API keys for all models being used (skip when using local model)
    if not args.api_base:
        providers_needed = set()
        for model_name in [args.model, args.summarization_model]:
            provider = model_name.split("/")[0] if "/" in model_name else "openai"
            providers_needed.add(provider)
            
        # breakpoint()

        for provider in providers_needed:
            if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
                print(f"ERROR: ANTHROPIC_API_KEY not set (required for {provider}/ models)")
                sys.exit(1)
            elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
                print(f"ERROR: OPENAI_API_KEY not set (required for {provider}/ models)")
                sys.exit(1)
            elif provider == "bedrock" and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
                print(f"ERROR: AWS_BEARER_TOKEN_BEDROCK not set (required for {provider}/ models)")
                sys.exit(1)

    # Auto-enforce --skip-eval for non-Docker backends (swebench harness requires Docker)
    if args.env_type in ("singularity", "bubblewrap") and not args.skip_eval:
        print(f"WARNING: --skip-eval forced for {args.env_type} env-type (swebench harness requires Docker)")
        args.skip_eval = True

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SWE-bench Evaluation with Memory Management")
    print("=" * 70)
    print(f"Dataset: {args.dataset} ({DATASET_MAP[args.dataset]})")
    print(f"Agent: {args.agent}")
    print(f"Model: {args.model}" + (f" (local: {args.api_base})" if args.api_base else ""))
    print(f"Memory: {args.memory}" + (f" (max_tokens={args.max_tokens})" if args.memory != "none" else ""))
    print(f"Environment: {args.env_type}" + (f" (images: {args.docker_image_source})" if args.env_type != "local" else ""))
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    dataset_path = DATASET_MAP[args.dataset]
    dataset = load_dataset(dataset_path, split=args.split)
    instances = list(dataset)

    # Apply slice
    if args.slice:
        start, end = map(int, args.slice.split(":"))
        instances = instances[start:end]
        print(f"Sliced to instances [{start}:{end}]")

    # Load instance IDs from file
    if args.instance_file:
        with open(args.instance_file) as f:
            file_ids = [line.strip() for line in f if line.strip()]
        if args.instances:
            args.instances = list(set(args.instances) | set(file_ids))
        else:
            args.instances = file_ids

    # Apply specific instances filter
    if args.instances:
        instance_set = set(args.instances)
        instances = [inst for inst in instances if inst["instance_id"] in instance_set]
        print(f"Filtered to {len(instances)} instances (from {len(args.instances)} requested)")

    # Resume: skip successfully submitted instances, retry errors and limits_exceeded
    if args.resume:
        preds_path = output_dir / "preds.json"
        if preds_path.exists():
            with open(preds_path) as f:
                existing_preds = json.load(f)
            # Only skip instances that were successfully submitted
            submitted_ids = set()
            retry_ids = set()
            for iid, pred in existing_preds.items():
                status = pred.get("status")
                if status is None:
                    # Legacy entries without status field: infer from patch content
                    patch = pred.get("model_patch", "")
                    if patch.startswith("# Error") or not patch.strip():
                        status = "Error"
                    else:
                        status = "Submitted"
                if status == "Submitted":
                    submitted_ids.add(iid)
                else:
                    retry_ids.add(iid)
            instances = [inst for inst in instances if inst["instance_id"] not in submitted_ids]
            if retry_ids:
                print(f"Resuming: skipping {len(submitted_ids)} submitted, retrying {len(retry_ids)} failed/limited instances")
            else:
                print(f"Resuming: skipping {len(submitted_ids)} submitted instances")

    print(f"\nEvaluating {len(instances)} instances...")

    # Run evaluation
    results = []

    total = len(instances)
    run_start = time.time()

    if args.workers == 1:
        # Sequential execution
        for i, inst in enumerate(instances):
            instance_id = inst["instance_id"]
            print(f"\n[{i+1}/{total}] {instance_id}")

            result = process_instance(
                instance_id=instance_id,
                dataset=dataset_path,
                model=args.model,
                memory_type=args.memory,
                args=args,
                env_type=args.env_type,
                docker_image_dir=args.docker_image_dir,
                docker_image_source=args.docker_image_source,
                output_dir=output_dir,
                verbose=args.verbose
            )
            results.append(result)

            # Thread-safe incremental save (read-then-merge, preserves resumed results)
            update_preds_file(output_dir, args.model, result)

            # Print progress
            status = result["status"]
            elapsed = result["elapsed_seconds"]
            result_str = "PASS" if result["reward"] > 0 else "FAIL"
            print(f"  Status: {status}, Result: {result_str}, Elapsed: {elapsed:.1f}s")

    else:
        # Parallel execution
        if args.workers > 16:
            print(f"\nWARNING: --workers {args.workers} exceeds recommended max of 16.")
            print("Docker network pool exhaustion may occur above 16 concurrent containers.")
            print("Consider using --workers 4-8 for reliability.\n")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance_id=inst["instance_id"],
                    dataset=dataset_path,
                    model=args.model,
                    memory_type=args.memory,
                    args=args,
                    env_type=args.env_type,
                    docker_image_dir=args.docker_image_dir,
                    docker_image_source=args.docker_image_source,
                    output_dir=output_dir,
                    verbose=args.verbose
                ): inst
                for inst in instances
            }

            completed = 0
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    update_preds_file(output_dir, args.model, result)
                    completed += 1
                    elapsed_total = time.time() - run_start
                    result_str = "PASS" if result["reward"] > 0 else "FAIL"
                    print(f"[{completed}/{total}] {result['instance_id']}: "
                          f"{result_str} ({result['elapsed_seconds']:.0f}s) "
                          f"[total: {elapsed_total:.0f}s]")
            except KeyboardInterrupt:
                print("\nInterrupt received. Cancelling pending jobs...")
                for fut in futures:
                    if not fut.running() and not fut.done():
                        fut.cancel()
                # Let running futures finish and save their results
                for fut in as_completed(futures):
                    if not fut.cancelled():
                        try:
                            result = fut.result(timeout=300)
                            results.append(result)
                            update_preds_file(output_dir, args.model, result)
                            completed += 1
                            print(f"  Saved running job: {result['instance_id']}")
                        except Exception:
                            pass
                print(f"Saved {completed}/{total} results before exit.")

    # Save final results (preds.json already saved incrementally via update_preds_file)
    preds_path = output_dir / "preds.json"
    print(f"\nSaving results...")
    summary_path = save_summary(results, args, output_dir)

    print(f"  Predictions: {preds_path} ({len(results)} from this run)")
    if preds_path.exists():
        with open(preds_path) as f:
            total_preds = len(json.load(f))
        print(f"  Total predictions on disk: {total_preds}")
    print(f"  Summary: {summary_path}")

    # Print summary
    print_summary(results, args)

    # Print next steps
    print("\nNext steps:")
    print(f"  # Submit for evaluation (free, fast)")
    print(f"  sb-cli submit swe-bench_{args.dataset} {args.split} \\")
    print(f"      --predictions_path {preds_path} \\")
    print(f"      --run_id memgym-{args.memory}-v1")


if __name__ == "__main__":
    main()
