"""Structural Instance Builder for MemGym-IR v3 (zero LLM, offline).

Builds MemGymIRInstance objects directly from 2WikiMultihopQA / MuSiQue
dataset structure without any LLM calls. Useful for:
- Quick testing when no search API is available
- Generating large batches instantly at zero cost
- Offline development and debugging

With --difficulty, applies v2 hardening (LLM-based) for production-quality
instances: adversarial distractors, contradictions, paragraph expansion,
filler noise, red herrings, and memory requirements.

Reuses existing dataset adapters and _compute_memory_requirements().

Usage:
    # Zero LLM (default, instant)
    python -m memgym.pipelines.memgym_ir.build_instances \
        --dataset 2wikimultihop --min-hops 4 --limit 1000

    # With v2 hardening (requires LLM)
    python -m memgym.pipelines.memgym_ir.build_instances \
        --dataset 2wikimultihop --min-hops 4 --limit 20 \
        --difficulty memory_hard --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from ..types.schemas import (
    MemGymIRInstance,
    SearchTurn,
    GroundingFactIR,
    EvictionPolicy,
    RubricIR,
    FilteredIRInstance,
)
from ..stages.adapters import ADAPTERS
from ..stages.generate import _build_search_turns, _compute_memory_requirements

logger = logging.getLogger(__name__)


def _structural_facts(instance: FilteredIRInstance) -> List[GroundingFactIR]:
    """Extract grounding facts structurally from decomposition (no LLM)."""
    facts = []
    for i, hop in enumerate(instance.decomposition):
        sub_answer = hop.get("sub_answer", "")
        # Bridge if not the last hop, terminal if last
        is_last = (i == len(instance.decomposition) - 1)
        fact_type = "terminal_fact" if is_last else "bridge_fact"
        needed_at = min(i + 1, len(instance.decomposition) - 1) if not is_last else i

        facts.append(GroundingFactIR(
            fact_id=f"F{i+1}",
            content=f"{hop.get('paragraph_title', '')}: {sub_answer}",
            fact_type=fact_type,
            source_paragraph_title=hop.get("paragraph_title", ""),
            needed_at_hop=needed_at,
            first_available_at_hop=i,
            retention_span=needed_at - i,
            intermediate_answer=sub_answer if not is_last else "",
        ))
    return facts


def build_instance_structural(
    filtered: FilteredIRInstance,
    eviction_mode: str = "full_eviction",
    window_size: int = 1,
    notes_budget: int = 150,
) -> Optional[MemGymIRInstance]:
    """Build a MemGymIRInstance from a FilteredIRInstance, zero LLM calls."""
    # Build search turns from decomposition
    turns = _build_search_turns(filtered, {})

    # Add distractor paragraphs to turns (round-robin)
    for i, dp in enumerate(filtered.distractor_paragraphs):
        target_turn = i % len(turns) if turns else 0
        if target_turn < len(turns):
            turns[target_turn].documents.append({
                "title": dp.get("title", f"Distractor {i}"),
                "text": dp.get("text", ""),
                "is_supporting": False,
            })

    # Extract facts structurally
    facts = _structural_facts(filtered)

    # Compute memory requirements
    memory_required_ids, per_turn_retention = _compute_memory_requirements(
        turns, facts, eviction_mode, window_size,
    )

    if not memory_required_ids:
        return None

    # Build eviction policy
    eviction_policy = EvictionPolicy(
        mode=eviction_mode,
        window_size=window_size,
        allow_notes=True,
        notes_budget_tokens=notes_budget,
    )

    # Build memory files
    memory_files = {}
    for turn in turns:
        docs_text = "\n\n".join(
            f"--- {d['title']} ---\n{d['text']}"
            for d in turn.documents
        )
        memory_files[f"search_results_turn_{turn.turn_index}.txt"] = docs_text

    # Build rubric
    rubric = RubricIR(
        expected_answer=filtered.answer,
        retention_facts=[f.fact_id for f in facts if f.retention_span > 0],
        noise_contradictions=[],
        temporal_facts=[],
        per_hop_answers=[
            {"hop": i, "answer": hop.get("sub_answer", "")}
            for i, hop in enumerate(filtered.decomposition)
        ],
        bridge_dependencies=[
            {"fact_id": f.fact_id, "from_hop": f.first_available_at_hop, "to_hop": f.needed_at_hop}
            for f in facts if f.fact_type == "bridge_fact"
        ],
        per_turn_retention=per_turn_retention,
    )

    return MemGymIRInstance(
        instance_id=f"memgym_ir__{filtered.source_dataset}__{filtered.instance_id}",
        base_instance=filtered.instance_id,
        source_dataset=filtered.source_dataset,
        question=filtered.question,
        original_question=filtered.question,
        answer=filtered.answer,
        num_hops=filtered.num_hops,
        decomposition=filtered.decomposition,
        turns=turns,
        grounding_facts=facts,
        distractors=[],
        memory_files=memory_files,
        gold_supporting_paragraphs=filtered.supporting_paragraphs,
        contradictions=[],
        transforms_applied=["structural_build"],
        difficulty_preset="structural",
        reveal_mode="immediate",
        eviction_policy=eviction_policy,
        memory_required_facts=memory_required_ids,
        total_tokens=sum(
            len(d.get("text", "").split()) for turn in turns for d in turn.documents
        ),
        rubric=rubric,
    )


def harden_instance(
    filtered: FilteredIRInstance,
    difficulty_preset: str,
    worker_client,
    all_instances: List[FilteredIRInstance],
) -> Optional[MemGymIRInstance]:
    """Apply v2 hardening pipeline to a structurally-parsed instance.

    Runs LLM-based fact extraction then craft_instance() with the
    difficulty preset's full set of hardening features.
    """
    from ..config import DifficultyConfig
    from ..stages.extract import extract_grounding_facts
    from ..stages.generate import craft_instance
    from ..llm.client import VerifierClient

    config = DifficultyConfig.from_preset(difficulty_preset)

    # Extract grounding facts (LLM-based for better quality)
    extraction = extract_grounding_facts(
        filtered, worker_client, VerifierClient(model=worker_client.model), dry_run=False,
    )
    if not extraction.get("quality_gate_passed"):
        logger.info(f"  {filtered.instance_id}: extraction quality gate failed, trying dry_run")
        extraction = extract_grounding_facts(
            filtered, worker_client, VerifierClient(model=worker_client.model), dry_run=True,
        )
        # Force quality gate for dry_run (structural facts are always valid)
        extraction["quality_gate_passed"] = True

    return craft_instance(
        instance=filtered,
        extraction=extraction,
        all_instances=all_instances,
        worker_client=worker_client,
        target_tokens=config.target_tokens,
        adversarial_count=config.adversarial_count,
        contradiction_count=config.contradiction_count,
        red_herring_count=config.red_herring_turns,
        reveal_mode=config.reveal_mode,
        expand_paragraphs=config.expand_paragraphs,
        expansion_target_tokens=config.expansion_target_tokens,
        overlap_count=config.overlap_count,
        conflicting_sources_count=config.conflicting_sources_count,
        eviction_mode=config.eviction_mode,
        eviction_window_size=config.eviction_window_size,
        allow_agent_notes=config.allow_agent_notes,
        agent_notes_budget=config.agent_notes_budget,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Structural Instance Builder (zero LLM, offline)"
    )
    parser.add_argument("--dataset", type=str, default="2wikimultihop",
                        choices=list(ADAPTERS.keys()),
                        help="Source dataset")
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--eviction-mode", type=str, default="full_eviction",
                        choices=["full_eviction", "sliding_window"])
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--notes-budget", type=int, default=150)
    parser.add_argument("--output", type=str, default=None)

    # v2 hardening (requires LLM)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="v2 difficulty preset for hardening (e.g. memory_hard). "
                             "Requires --model. Without this flag, zero-LLM mode.")
    parser.add_argument("--model", type=str,
                        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Worker LLM model (only used with --difficulty)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.output is None:
        suffix = f"_{args.difficulty}" if args.difficulty else ""
        args.output = f"output_ir/structural_{args.dataset}{suffix}_instances.jsonl"

    # Load and filter instances
    adapter_cls = ADAPTERS[args.dataset]
    adapter = adapter_cls()
    logger.info(f"Loading {args.dataset} dataset...")
    raw_instances = adapter.load_raw(limit=args.limit * 5)
    logger.info(f"Loaded {len(raw_instances)} raw instances")

    # Parse and filter by hop count
    filtered = []
    for raw in raw_instances:
        parsed = adapter.parse(raw)
        if parsed and parsed.num_hops >= args.min_hops:
            filtered.append(parsed)
        if len(filtered) >= args.limit:
            break
    logger.info(f"After parse + min_hops={args.min_hops} filter: {len(filtered)} instances")

    # Set up LLM client if hardening
    worker_client = None
    if args.difficulty:
        from ..llm.client import LLMClient
        worker_client = LLMClient(model=args.model)
        logger.info(f"Hardening with difficulty='{args.difficulty}', model={args.model}")

    # Build instances
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_built = 0
    num_skipped = 0
    t0 = time.time()

    with open(output_path, "w") as f:
        for i, inst in enumerate(filtered):
            if args.difficulty:
                logger.info(f"[{i+1}/{len(filtered)}] Hardening {inst.instance_id}...")
                result = harden_instance(
                    inst, args.difficulty, worker_client, filtered,
                )
            else:
                result = build_instance_structural(
                    inst,
                    eviction_mode=args.eviction_mode,
                    window_size=args.window_size,
                    notes_budget=args.notes_budget,
                )
            if result:
                f.write(json.dumps(result.to_jsonl_dict()) + "\n")
                num_built += 1
            else:
                num_skipped += 1

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s: {num_built} instances built, "
        f"{num_skipped} skipped (no memory-required facts)"
    )
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
