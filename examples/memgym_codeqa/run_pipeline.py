#!/usr/bin/env python3
"""Run the coding_synthetic (MemGymCodeQA) pipeline (one-shot, no server needed).

Requires ``pip install -e .`` first.

Usage:
    python examples/memgym_codeqa/run_pipeline.py --limit 20 \
        --worker-model "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
"""
import argparse

from memgym.pipelines.coding_synthetic.pipeline import run_pipeline_from_args


def main():
    parser = argparse.ArgumentParser(description="Run coding_synthetic pipeline")
    parser.add_argument("--output", default="results/pipeline_output")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--worker-model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--verifier-model", default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--skip-verification", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--num-distractors", type=int, default=5)
    parser.add_argument("--max-lines-changed", type=int, default=500)
    parser.add_argument("--target-context-tokens", type=int, default=12000,
                        help="Target total context tokens per instance")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--expand-target-tokens", type=int, default=None,
                        help="Expand instances to this token count (e.g., 50000, 100000)")
    parser.add_argument("--exclude-file", default=None,
                        help="JSONL file with instance_ids to skip (for multi-batch runs)")
    args = parser.parse_args()

    run_pipeline_from_args(
        output=args.output,
        limit=args.limit,
        worker_model=args.worker_model,
        verifier_model=args.verifier_model,
        difficulty_preset=args.difficulty,
        skip_verification=args.skip_verification,
        no_resume=args.no_resume,
        num_distractors=args.num_distractors,
        max_lines_changed=args.max_lines_changed,
        target_context_tokens=args.target_context_tokens,
        num_workers=args.num_workers,
        expand_target_tokens=args.expand_target_tokens,
        exclude_file=args.exclude_file,
    )


if __name__ == "__main__":
    main()
