"""Unified research-first pipeline for MemGym-IR.

3 stages: Grow → Craft → Calibrate

Usage:
    # Research mode (default) — grow questions through real search
    python -m memgym.pipelines.memgym_ir.research_pipeline \
        --topic "attention mechanisms" --num-questions 10 \
        --search-backend semantic_scholar+wikipedia

    # Dataset mode (v2 compat) — load from existing QA datasets
    python -m memgym.pipelines.memgym_ir.research_pipeline \
        --source dataset --dataset 2wikimultihop --limit 20
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from ..backends import get_backend, SUGGESTED_ORDER
from ..llm.client import LLMClient
from ..types.schemas import MemGymIRInstance, GrowthResult

logger = logging.getLogger(__name__)


def run_research_pipeline(
    topic: str = "machine learning",
    num_questions: int = 10,
    target_hops: int = 4,
    min_hops: int = 3,
    search_backend_spec: str = "semantic_scholar+openalex+wikipedia",
    worker_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    verifier_model: str = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    target_tokens: int = 100000,
    near_miss_per_hop: int = 3,
    expand_paragraphs: bool = True,
    skip_calibration: bool = False,
    resume: bool = True,
    output_dir: str = "output_ir",
) -> List[MemGymIRInstance]:
    """Run the full research-first pipeline: Grow → Craft → Calibrate.

    With resume=True (default), skips stages whose checkpoint files
    already exist. Use --no-resume to force a fresh run.
    """
    from ..generators.question_grower import grow_batch
    from ..stages.generate import craft_from_growth
    from ..stages.calibrate import harden_and_verify

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not resume:
        # Clean old checkpoints for a fresh run
        for f in ["growths.jsonl", "crafted.jsonl", "calibrated.jsonl"]:
            p = output_path / f
            if p.exists():
                p.unlink()
                logger.info(f"  Removed old checkpoint: {p}")

    search_backend = get_backend(search_backend_spec)
    worker_client = LLMClient(model=worker_model)
    verifier_client = LLMClient(model=verifier_model)

    t0 = time.time()

    # Checkpoint paths
    growth_path = output_path / "growths.jsonl"
    crafted_path = output_path / "crafted.jsonl"
    calibrated_path = output_path / "calibrated.jsonl"

    # ── Stage 1: Grow questions through search ──
    if growth_path.exists():
        logger.info(f"Stage 1: Resuming from checkpoint {growth_path}")
        growths = []
        with open(growth_path) as f:
            for line in f:
                growths.append(GrowthResult(**json.loads(line)))
        logger.info(f"  Loaded {len(growths)} growths from checkpoint")
    else:
        logger.info(f"Stage 1: Growing {num_questions} questions on '{topic}'")
        logger.info(f"  Backend: {search_backend_spec}")
        logger.info(f"  Target hops: {target_hops}, min hops: {min_hops}")

        growths = grow_batch(
            topic=topic,
            search_backend=search_backend,
            llm=worker_client,
            num_questions=num_questions,
            target_hops=target_hops,
            min_hops=min_hops,
        )
        logger.info(f"Stage 1 complete: {len(growths)} questions grown")

        with open(growth_path, "w") as f:
            for g in growths:
                f.write(json.dumps(g.to_jsonl_dict()) + "\n")
        logger.info(f"  Checkpoint saved: {growth_path}")

    # ── Stage 2: Craft instances (3-tier distractors) ──
    if crafted_path.exists():
        logger.info(f"\nStage 2: Resuming from checkpoint {crafted_path}")
        crafted = []
        with open(crafted_path) as f:
            for line in f:
                crafted.append(MemGymIRInstance(**json.loads(line)))
        logger.info(f"  Loaded {len(crafted)} crafted instances from checkpoint")
    else:
        logger.info(f"\nStage 2: Crafting {len(growths)} instances")
        crafted = []
        for i, growth in enumerate(growths):
            logger.info(f"  [{i+1}/{len(growths)}] Crafting '{growth.question[:60]}...'")
            instance = craft_from_growth(
                growth=growth,
                worker_client=worker_client,
                target_tokens=target_tokens,
                near_miss_per_hop=near_miss_per_hop,
                expand_paragraphs=expand_paragraphs,
                verifier_client=verifier_client,  # Invariant C
            )
            if instance:
                crafted.append(instance)
                logger.info(
                    f"    → {len(instance.turns)} turns, "
                    f"{len(instance.grounding_facts)} facts, "
                    f"{instance.total_tokens} tokens"
                )
            else:
                logger.info(f"    → Skipped (no memory-required facts)")

        logger.info(f"Stage 2 complete: {len(crafted)} instances crafted")

        with open(crafted_path, "w") as f:
            for inst in crafted:
                f.write(json.dumps(inst.to_jsonl_dict()) + "\n")
        logger.info(f"  Checkpoint saved: {crafted_path}")

    # ── Stage 3: Harden + Verify (calibration) ──
    if skip_calibration:
        calibrated = crafted
        logger.info(f"\nStage 3: Skipped (--skip-calibration)")
    elif calibrated_path.exists():
        logger.info(f"\nStage 3: Resuming from checkpoint {calibrated_path}")
        calibrated = []
        with open(calibrated_path) as f:
            for line in f:
                calibrated.append(MemGymIRInstance(**json.loads(line)))
        logger.info(f"  Loaded {len(calibrated)} calibrated instances from checkpoint")
    else:
        logger.info(f"\nStage 3: Calibrating {len(crafted)} instances")
        calibrated = []
        for i, inst in enumerate(crafted):
            logger.info(f"  [{i+1}/{len(crafted)}] Calibrating '{inst.question[:50]}...'")
            result = harden_and_verify(inst, verifier_client, worker_client)
            if result:
                calibrated.append(result)
                v = result.verification or {}
                logger.info(
                    f"    → PASS: gap={v.get('memory_gap', 0):.2f}, "
                    f"long_ctx={v.get('score_long_context', 0):.2f}"
                )
            else:
                logger.info(f"    → FAIL: calibration failed")

            # Incremental checkpoint after each calibration
            with open(calibrated_path, "w") as f:
                for c in calibrated:
                    f.write(json.dumps(c.to_jsonl_dict()) + "\n")

        logger.info(f"Stage 3 complete: {len(calibrated)}/{len(crafted)} passed calibration")

    # ── Save final output ──
    final_path = output_path / "memgym_ir_instances.jsonl"
    with open(final_path, "w") as f:
        for inst in calibrated:
            f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    elapsed = time.time() - t0
    logger.info(f"\nPipeline complete in {elapsed:.0f}s")
    logger.info(f"  {len(growths)} grown → {len(crafted)} crafted → {len(calibrated)} calibrated")
    logger.info(f"  Output: {final_path}")

    return calibrated


def main():
    parser = argparse.ArgumentParser(
        description="Unified research-first MemGym-IR pipeline"
    )

    # Source mode
    parser.add_argument("--source", default="research", choices=["research", "dataset"],
                        help="Generation mode: research (grow-by-search) or dataset (v2 compat)")

    # Research mode args
    parser.add_argument("--topic", type=str, default="machine learning")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--target-hops", type=int, default=4)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--search-backend", type=str, default="semantic_scholar+openalex+wikipedia")

    # Dataset mode args (v2 compat)
    parser.add_argument("--dataset", type=str, default="2wikimultihop")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="memory_hard")

    # LLM config
    parser.add_argument("--worker-model", type=str,
                        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--verifier-model", type=str,
                        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    # Craft config
    parser.add_argument("--target-tokens", type=int, default=100000)
    parser.add_argument("--near-miss-per-hop", type=int, default=3,
                        help="Sibling-seeded near-miss distractors per hop (Invariant B)")
    parser.add_argument("--expand-paragraphs", action="store_true", default=True)
    parser.add_argument("--no-expand", dest="expand_paragraphs", action="store_false")

    # Calibration
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force fresh run, ignore existing checkpoints")

    # Output
    parser.add_argument("--output-dir", type=str, default="output_ir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.source == "research":
        run_research_pipeline(
            topic=args.topic,
            num_questions=args.num_questions,
            target_hops=args.target_hops,
            min_hops=args.min_hops,
            search_backend_spec=args.search_backend,
            worker_model=args.worker_model,
            verifier_model=args.verifier_model,
            target_tokens=args.target_tokens,
            near_miss_per_hop=args.near_miss_per_hop,
            expand_paragraphs=args.expand_paragraphs,
            skip_calibration=args.skip_calibration,
            resume=not args.no_resume,
            output_dir=args.output_dir,
        )
    else:
        # Dataset mode — delegate to existing build_instances with hardening
        from ..generators.build_instances import main as build_main
        import sys
        sys.argv = [
            "build_instances",
            "--dataset", args.dataset,
            "--min-hops", str(args.min_hops),
            "--limit", str(args.limit),
            "--difficulty", args.difficulty,
            "--model", args.worker_model,
            "--output", f"{args.output_dir}/dataset_{args.dataset}_instances.jsonl",
        ]
        build_main()


if __name__ == "__main__":
    main()
