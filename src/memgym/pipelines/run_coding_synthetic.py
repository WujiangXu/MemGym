#!/usr/bin/env python3
"""
CLI entry point for Coding-Synthetic pipeline v2.

Usage:
    # Full pipeline with limit
    python pipelines/run_coding_synthetic.py --output ./output --limit 10

    # Use different Worker/Verifier models
    python pipelines/run_coding_synthetic.py -o ./output --worker-model gpt-5-mini --verifier-model claude-opus-4-20250514

    # Skip verification (faster, no ablation check)
    python pipelines/run_coding_synthetic.py -o ./output --skip-verification

    # Enable Docker-based verification
    python pipelines/run_coding_synthetic.py -o ./output --docker

    # Hard difficulty preset
    python pipelines/run_coding_synthetic.py -o ./output --difficulty hard
"""

import argparse
import sys
from pathlib import Path

from memgym.pipelines.coding_synthetic import run_pipeline_from_args


def main():
    parser = argparse.ArgumentParser(
        description="Generate Coding-Synthetic benchmark v2 from SWE-smith instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with 10 instances
    python run_coding_synthetic.py -o ./output --limit 10

    # Use specific models
    python run_coding_synthetic.py -o ./output --worker-model gpt-5-mini --verifier-model claude-opus-4-20250514

    # Hard difficulty with Docker verification
    python run_coding_synthetic.py -o ./output --difficulty hard --docker

    # Skip verification for faster iteration
    python run_coding_synthetic.py -o ./output --limit 5 --skip-verification
        """
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process"
    )

    # Model settings
    model_group = parser.add_argument_group("Model settings")

    model_group.add_argument(
        "--worker-model",
        type=str,
        default="gpt-5-mini",
        help="Worker LLM model — cheap, generative (default: gpt-5-mini)"
    )

    model_group.add_argument(
        "--verifier-model",
        type=str,
        default="claude-opus-4-20250514",
        help="Verifier LLM model — expensive, judgment (default: claude-opus-4-20250514)"
    )

    # Filter settings
    filter_group = parser.add_argument_group("Filter settings")

    filter_group.add_argument(
        "--max-lines-changed",
        type=int,
        default=500,
        help="Maximum patch size in lines (default: 500)"
    )

    # Difficulty settings
    difficulty_group = parser.add_argument_group("Difficulty settings")

    difficulty_group.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty preset (default: medium)"
    )

    difficulty_group.add_argument(
        "--num-distractors",
        type=int,
        default=5,
        help="Target number of distractors per instance (default: 5)"
    )

    # Verification settings
    verify_group = parser.add_argument_group("Verification settings")

    verify_group.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip ablation verification stage"
    )

    verify_group.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help="Enable Docker-based test execution for verification"
    )

    verify_group.add_argument(
        "--adaptive-rounds",
        type=int,
        default=1,
        help="Number of adaptive difficulty feedback rounds (0 to disable, default: 1)"
    )

    # General settings
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoints (start fresh)"
    )

    args = parser.parse_args()

    # Run pipeline
    try:
        results = run_pipeline_from_args(
            output=args.output,
            limit=args.limit,
            worker_model=args.worker_model,
            verifier_model=args.verifier_model,
            max_lines_changed=args.max_lines_changed,
            difficulty_preset=args.difficulty,
            skip_verification=args.skip_verification,
            docker_available=args.docker,
            adaptive_rounds=args.adaptive_rounds,
            no_resume=args.no_resume,
            num_distractors=args.num_distractors,
        )

        print(f"\nGenerated {len(results)} Coding-Synthetic v2 instances")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
