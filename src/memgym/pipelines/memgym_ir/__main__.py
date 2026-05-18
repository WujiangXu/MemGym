"""CLI entry point for MemGym-IR pipeline.

Subcommands:
    python -m memgym.pipelines.memgym_ir dataset        --config configs/smoke_test.yaml
    python -m memgym.pipelines.memgym_ir research        --topic "attention" --num-questions 10
    python -m memgym.pipelines.memgym_ir deep-research   --topic "attention" --target-tokens 100000
    python -m memgym.pipelines.memgym_ir build           --dataset 2wikimultihop --min-hops 4
    python -m memgym.pipelines.memgym_ir eval            instances.jsonl --strategy ir_structured
    python -m memgym.pipelines.memgym_ir review          instances.jsonl

Backward compat (no subcommand = dataset):
    python -m memgym.pipelines.memgym_ir --config configs/smoke_test.yaml
    python -m memgym.pipelines.memgym_ir --dataset musique --limit 10 --dry-run
"""

import argparse
import sys
from pathlib import Path


def _add_dataset_args(parser: argparse.ArgumentParser):
    """Add v2 dataset pipeline arguments to a parser."""
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: musique, hotpotqa, synthetic (default: musique)")
    parser.add_argument("--worker-model", type=str, default=None)
    parser.add_argument("--verifier-model", type=str, default=None)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Difficulty preset: easy, medium, hard")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Token budget per instance (default: 10000)")
    parser.add_argument("--min-hops", type=int, default=None,
                        help="Minimum hop count (default: 2)")
    parser.add_argument("--skip-verification", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max parallel LLM calls (default: 10)")

    # Fine-grained difficulty overrides
    parser.add_argument("--adversarial-count", type=int, default=None)
    parser.add_argument("--contradiction-count", type=int, default=None)
    parser.add_argument("--query-fuzz", type=str, default=None,
                        choices=["none", "light", "heavy"])
    parser.add_argument("--distractor-scale", type=float, default=None)
    parser.add_argument("--min-a-score", type=float, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument("--red-herring-turns", type=int, default=None)
    parser.add_argument("--reveal-mode", type=str, default=None,
                        choices=["immediate", "delayed"])

    # Deep research noise
    parser.add_argument("--expand-paragraphs", action="store_true", default=None)
    parser.add_argument("--expansion-target-tokens", type=int, default=None)
    parser.add_argument("--overlap-count", type=int, default=None)
    parser.add_argument("--conflicting-sources", type=int, default=None)

    # Memory eviction
    parser.add_argument("--eviction-mode", type=str, default=None,
                        choices=["none", "full_eviction", "sliding_window"])
    parser.add_argument("--eviction-window", type=int, default=None)
    parser.add_argument("--agent-notes-budget", type=int, default=None)
    parser.add_argument("--no-agent-notes", action="store_true")
    parser.add_argument("--min-memory-facts", type=int, default=None)

    # Hop upgrade
    parser.add_argument("--upgrade-hops", type=int, default=None)


def _run_dataset(args):
    """Run the v2 dataset pipeline."""
    import yaml
    from .config import PipelineConfig, DifficultyConfig
    from .orchestrators.dataset_pipeline import run_pipeline

    config_dict = {}

    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f) or {}
            print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}")

    # CLI overrides
    if args.output is not None:
        config_dict["output_dir"] = args.output
    if args.limit is not None:
        config_dict["limit"] = args.limit
    if args.dataset is not None:
        config_dict["dataset"] = args.dataset
    if args.worker_model is not None:
        config_dict["worker_model"] = args.worker_model
    if args.verifier_model is not None:
        config_dict["verifier_model"] = args.verifier_model
    if args.difficulty is not None:
        config_dict["difficulty_preset"] = args.difficulty
    if args.target_tokens is not None:
        config_dict["target_tokens"] = args.target_tokens
    if args.skip_verification is True:
        config_dict["skip_verification"] = True
    if args.min_hops is not None:
        config_dict["min_hops"] = args.min_hops
    if args.no_resume:
        config_dict["resume"] = False
    if args.dry_run:
        config_dict["dry_run"] = True
    if args.max_concurrent is not None:
        config_dict["max_concurrent"] = args.max_concurrent
    if args.min_memory_facts is not None:
        config_dict["min_memory_facts"] = args.min_memory_facts
    if args.upgrade_hops is not None:
        config_dict["upgrade_hops"] = args.upgrade_hops

    if "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    # Build DifficultyConfig from preset + CLI overrides
    preset = config_dict.get("difficulty_preset", PipelineConfig.difficulty_preset)
    diff_config = DifficultyConfig.from_preset(preset)

    if args.target_tokens is not None:
        diff_config.target_tokens = args.target_tokens
    if args.adversarial_count is not None:
        diff_config.adversarial_count = args.adversarial_count
    if args.contradiction_count is not None:
        diff_config.contradiction_count = args.contradiction_count
    if args.query_fuzz is not None:
        diff_config.query_fuzz_intensity = args.query_fuzz
    if args.distractor_scale is not None:
        diff_config.distractor_scale_factor = args.distractor_scale
    if args.min_a_score is not None:
        diff_config.min_a_score = args.min_a_score
    if args.min_gap is not None:
        diff_config.min_gap_ab = args.min_gap
    if args.red_herring_turns is not None:
        diff_config.red_herring_turns = args.red_herring_turns
    if args.reveal_mode is not None:
        diff_config.reveal_mode = args.reveal_mode
    if args.expand_paragraphs:
        diff_config.expand_paragraphs = True
    if args.expansion_target_tokens is not None:
        diff_config.expansion_target_tokens = args.expansion_target_tokens
    if args.overlap_count is not None:
        diff_config.overlap_count = args.overlap_count
    if args.conflicting_sources is not None:
        diff_config.conflicting_sources_count = args.conflicting_sources
    if args.eviction_mode is not None:
        diff_config.eviction_mode = args.eviction_mode
    if args.eviction_window is not None:
        diff_config.eviction_window_size = args.eviction_window
    if args.agent_notes_budget is not None:
        diff_config.agent_notes_budget = args.agent_notes_budget
    if args.no_agent_notes:
        diff_config.allow_agent_notes = False

    config_dict["difficulty_config"] = diff_config
    config = PipelineConfig(**config_dict)
    run_pipeline(config)


def _run_research(args):
    """Run the v3 research pipeline."""
    from .orchestrators.research_pipeline import main as research_main
    research_main()


def _run_build(args):
    """Run the structural instance builder."""
    from .generators.build_instances import main as build_main
    build_main()


def _run_eval(args):
    """Run the evaluation suite."""
    from .eval.agent import main as eval_main
    eval_main()


def _run_deep_research(args):
    """Run the deep research pipeline."""
    from .orchestrators.deep_research_pipeline import main as deep_main
    deep_main()


def _run_review(args):
    """Run quality review."""
    from .eval.quality_review import main as review_main
    review_main()


def _run_generate_topics(args):
    """Generate topic pool for scaling."""
    from .topics.generator import main as topics_main
    topics_main()


def main():
    # Detect backward compat: if first arg is not a subcommand, default to 'dataset'
    subcommands = {"dataset", "research", "deep-research", "build", "eval", "review", "generate-topics"}
    if len(sys.argv) < 2 or sys.argv[1] not in subcommands:
        # Backward compat: no subcommand → dataset pipeline
        parser = argparse.ArgumentParser(
            description="MemGym-IR Pipeline (default: dataset mode)"
        )
        _add_dataset_args(parser)
        args = parser.parse_args()
        _run_dataset(args)
        return

    # Subcommand mode
    parser = argparse.ArgumentParser(
        description="MemGym-IR Pipeline",
        usage="python -m memgym.pipelines.memgym_ir {dataset,research,build,eval,review} [options]",
    )
    parser.add_argument("command", choices=subcommands,
                        help="Pipeline command to run")

    # Parse just the subcommand, pass rest to the handler
    args = parser.parse_args(sys.argv[1:2])

    # Rewrite sys.argv for the sub-handler's own argparse
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if args.command == "dataset":
        sub_parser = argparse.ArgumentParser(description="v2 dataset pipeline")
        _add_dataset_args(sub_parser)
        sub_args = sub_parser.parse_args()
        _run_dataset(sub_args)
    elif args.command == "research":
        _run_research(args)
    elif args.command == "build":
        _run_build(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "deep-research":
        _run_deep_research(args)
    elif args.command == "review":
        _run_review(args)
    elif args.command == "generate-topics":
        _run_generate_topics(args)


if __name__ == "__main__":
    main()
