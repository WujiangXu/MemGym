#!/usr/bin/env python
"""
Run a single episode and collect trajectory.

Usage:
    python scripts/run_episode.py --env tau2 --domain airline --task-id book_flight_1
    python scripts/run_episode.py --env swe --dataset princeton-nlp/SWE-bench_Lite --seed 42
"""

import sys
from pathlib import Path
import argparse

# Add src to path
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from memgym.envs import get_env


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single episode")
    parser.add_argument("--env", type=str, required=True, choices=["tau2", "swe"],
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="./trajectories",
                        help="Output directory for trajectories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Tau2 arguments
    parser.add_argument("--domain", type=str, help="Tau2 domain")
    parser.add_argument("--task-id", type=str, help="Tau2 task ID")
    parser.add_argument("--agent-llm", type=str, help="LLM model for agent")

    # SWE arguments
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite",
                        help="SWE-bench dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--instance-id", type=str, help="Specific instance ID")

    # Memory options
    parser.add_argument("--context-management", action="store_true",
                        help="Enable LLM context summarization")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="Max context tokens before summarization")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build environment kwargs
    if args.env == "tau2":
        if not args.domain or not args.task_id:
            print("Error: --domain and --task-id required for tau2")
            sys.exit(1)

        env_kwargs = {
            "domain": args.domain,
            "task_id": args.task_id,
            "use_context_management": args.context_management,
            "max_context_tokens": args.max_tokens,
        }
        if args.agent_llm:
            env_kwargs["agent_llm"] = args.agent_llm

    elif args.env == "swe":
        env_kwargs = {
            "dataset": args.dataset,
            "split": args.split,
            "use_context_management": args.context_management,
            "max_context_tokens": args.max_tokens,
            "use_docker": False,  # Default off for quick testing
        }
        if args.instance_id:
            env_kwargs["instance_id"] = args.instance_id
        if args.agent_llm:
            env_kwargs["agent_llm"] = args.agent_llm

    # Create environment
    print(f"Creating {args.env} environment...")
    env = get_env(args.env, **env_kwargs)

    # Reset
    print("Resetting environment...")
    obs, info = env.reset(seed=args.seed)

    if args.verbose:
        print(f"Instance: {info.get('instance_id', info.get('task_id', 'unknown'))}")
        print(f"Initial tokens: {info.get('memory_info', {}).get('tokens', 0)}")

    # Run episode (if LLM configured)
    if args.agent_llm:
        print("Running episode with LLM agent...")
        result = env.run_episode(seed=args.seed, verbose=args.verbose)
        print(f"\nResult: reward={result['reward']}, steps={result['steps']}")
    else:
        print("No --agent-llm specified. Use step() to interact manually.")

    # Save trajectory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_path / f"trajectory_{args.env}.json"
    env.save_trajectory(str(trajectory_file))
    print(f"Saved trajectory to {trajectory_file}")

    env.close()


if __name__ == "__main__":
    main()
