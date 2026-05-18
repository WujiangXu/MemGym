"""CLI entry point for coding_synthetic pipeline.

Usage:
    python -m coding_synthetic.pipeline --limit 10 --worker-model gpt-4o-mini
"""
import argparse
from .pipeline import run_pipeline_from_args


def main():
    parser = argparse.ArgumentParser(description="Coding-Synthetic Pipeline v2")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-model", default="gpt-4o-mini")
    parser.add_argument("--verifier-model", default="gpt-4o-mini")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--num-distractors", type=int, default=5)
    parser.add_argument("--max-lines-changed", type=int, default=500)
    parser.add_argument("--target-context-tokens", type=int, default=12000,
                        help="Target total context tokens per instance")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (1 = sequential)")
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
    )


if __name__ == "__main__":
    main()
