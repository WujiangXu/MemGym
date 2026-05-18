"""Grow-by-search question builder for MemGym-IR.

Grows multi-hop research questions step-by-step through real search.
Each hop = real search → extract fact → extend question. Bridge facts
are guaranteed by construction (F_n enables query for F_n+1).

Usage:
    python -m memgym.pipelines.memgym_ir.question_grower \
        --topic "attention mechanisms in NLP" \
        --target-hops 4 --num-questions 10 \
        --search-backend arxiv+semantic_scholar+wikipedia
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from ..backends import SearchBackend, SearchResult, get_backend
from ..llm.client import LLMClient
from ..llm.prompts import (
    SEED_QUESTION_PROMPT,
    SEED_QUESTION_SCHEMA,
    HOP_ANALYZE_PROMPT,
    HOP_ANALYZE_SCHEMA,
    HOP_EXTEND_PROMPT,
    HOP_EXTEND_SCHEMA,
    SYNTHESIZE_QUESTION_PROMPT,
    SYNTHESIZE_QUESTION_SCHEMA,
    _DEPTH_GUIDANCE,
)
from ..types.schemas import GroundingFactIR, GrowthResult, HopRecord

logger = logging.getLogger(__name__)


def grow_question(
    topic: str,
    search_backend: SearchBackend,
    llm: LLMClient,
    target_hops: int = 4,
    min_hops: int = 3,
    search_results_per_hop: int = 5,
    inter_search_delay: float = 1.5,
) -> Optional[GrowthResult]:
    """Grow a multi-hop question step by step through real search.

    Returns GrowthResult with the synthesized question, answer, and
    grounding facts, or None if growth failed.
    """
    # Step 0: Generate seed question
    seed = llm.get_json_completion(
        prompt=SEED_QUESTION_PROMPT.format(topic=topic),
        response_format=SEED_QUESTION_SCHEMA,
        temperature=0.7,
    )
    seed_question = seed.get("question", "")
    if not seed_question:
        logger.warning(f"Failed to generate seed question for topic '{topic}'")
        return None

    logger.info(f"  Seed: {seed_question}")

    hops: List[HopRecord] = []
    facts: List[GroundingFactIR] = []
    current_question = seed_question
    current_search_query = seed.get("search_query", seed_question)

    for hop_idx in range(target_hops):
        # Rate limiting between searches
        if hop_idx > 0:
            time.sleep(inter_search_delay)

        # Search using the short search query (not the full question)
        results = search_backend.search(current_search_query, num_results=search_results_per_hop)
        if not results:
            logger.warning(f"  Hop {hop_idx}: no results for query '{current_search_query}'")
            # Retry with just the topic
            results = search_backend.search(topic, num_results=search_results_per_hop)
            if not results:
                logger.warning(f"  Hop {hop_idx}: retry with topic also failed, stopping")
                break

        logger.info(f"  Hop {hop_idx}: {len(results)} results for '{current_search_query}'")

        # Analyze: extract key fact
        results_text = _format_search_results(results)
        previous_facts_text = _format_facts(facts)

        analysis = llm.get_json_completion(
            prompt=HOP_ANALYZE_PROMPT.format(
                topic=topic,
                question=current_question,
                results=results_text,
                previous_facts=previous_facts_text or "(no previous facts)",
            ),
            response_format=HOP_ANALYZE_SCHEMA,
            temperature=0.1,
        )

        key_fact = analysis.get("key_fact", "")
        if not key_fact:
            logger.warning(f"  Hop {hop_idx}: no key fact extracted, stopping")
            break

        supporting_indices = analysis.get("supporting_doc_indices", [])
        logger.info(f"  Hop {hop_idx}: fact = '{key_fact[:80]}...'")

        # Create GroundingFactIR (bridge by construction)
        fact = GroundingFactIR(
            fact_id=f"F{hop_idx + 1}",
            content=key_fact,
            fact_type="bridge_fact",
            source_paragraph_title=_get_source_title(results, supporting_indices),
            first_available_at_hop=hop_idx,
            needed_at_hop=hop_idx + 1,  # needed at next hop (by construction)
            retention_span=1,
        )
        facts.append(fact)

        # Store hop record with results as dicts
        hop_record = HopRecord(
            hop_index=hop_idx,
            query=current_question,
            fact=fact,
            supporting_doc_indices=supporting_indices,
            results=[_result_to_dict(r) for r in results],
        )
        hops.append(hop_record)

        # Extend: generate next question (unless this is the last hop)
        if hop_idx < target_hops - 1:
            depth_guide = _DEPTH_GUIDANCE.get(
                hop_idx + 2,
                _DEPTH_GUIDANCE.get(4, "Look for results that synthesize your findings."),
            )
            extension = llm.get_json_completion(
                prompt=HOP_EXTEND_PROMPT.format(
                    topic=topic,
                    current_fact=key_fact,
                    previous_facts=previous_facts_text or "(none)",
                    hop_number=hop_idx + 2,
                    target_hops=target_hops,
                    depth_guidance=depth_guide,
                ),
                response_format=HOP_EXTEND_SCHEMA,
                temperature=0.4,
            )
            next_q = extension.get("next_question", "")
            next_search = extension.get("search_query", next_q)
            if not next_q:
                logger.warning(f"  Hop {hop_idx}: failed to extend, stopping")
                break
            current_question = next_q
            current_search_query = next_search
            logger.info(f"  Hop {hop_idx}: → next: '{next_q[:60]}' [query: '{next_search}']")

    # Check minimum hops
    if len(hops) < min_hops:
        logger.warning(
            f"  Only {len(hops)} hops (need {min_hops}), discarding"
        )
        return None

    # Mark last fact as terminal
    if facts:
        facts[-1].fact_type = "terminal_fact"
        facts[-1].needed_at_hop = len(hops) - 1

    # Update retention spans: all bridge facts needed at final step
    for f in facts[:-1]:
        f.needed_at_hop = len(hops) - 1
        f.retention_span = f.needed_at_hop - f.first_available_at_hop

    # Synthesize final multi-hop question
    facts_text = "\n".join(
        f"  Hop {i}: [F{i+1}] {f.content}" for i, f in enumerate(facts)
    )
    trajectory_text = "\n".join(
        f"  Step {h.hop_index}: searched '{h.query[:60]}' → found F{h.hop_index+1}"
        for h in hops
    )

    synthesis = llm.get_json_completion(
        prompt=SYNTHESIZE_QUESTION_PROMPT.format(
            topic=topic,
            facts_with_hops=facts_text,
            trajectory_summary=trajectory_text,
        ),
        response_format=SYNTHESIZE_QUESTION_SCHEMA,
        temperature=0.2,
    )

    final_question = synthesis.get("question", "")
    final_answer = synthesis.get("answer", "")
    if not final_question or not final_answer:
        logger.warning("  Synthesis failed — no question or answer")
        return None

    logger.info(f"  Final Q: {final_question[:80]}...")
    logger.info(f"  Final A: {final_answer[:80]}...")

    return GrowthResult(
        question=final_question,
        answer=final_answer,
        topic=topic,
        hops=hops,
        facts=facts,
        reasoning_trace=synthesis.get("reasoning_trace", ""),
        seed_question=seed_question,
        num_hops=len(hops),
    )


def grow_batch(
    topic: str,
    search_backend: SearchBackend,
    llm: LLMClient,
    num_questions: int = 10,
    target_hops: int = 4,
    min_hops: int = 3,
    inter_question_delay: float = 2.0,
    max_workers: int = 4,
    hop_choices: Optional[List[int]] = None,
) -> List[GrowthResult]:
    """Grow multiple questions for a topic using parallel workers.

    If ``hop_choices`` is given, each question samples its target hop count
    uniformly from the list. Otherwise all questions use ``target_hops``.
    """
    import random as _random

    def _grow_one(idx: int) -> Optional[GrowthResult]:
        per_question_hops = (
            _random.choice(hop_choices) if hop_choices else target_hops
        )
        logger.info(
            f"\n[{idx+1}/{num_questions}] Growing question (target_hops={per_question_hops})..."
        )
        for attempt in range(2):  # allow 1 retry per slot
            growth = grow_question(
                topic=topic,
                search_backend=search_backend,
                llm=llm,
                target_hops=per_question_hops,
                min_hops=min_hops,
            )
            if growth:
                logger.info(
                    f"  [{idx+1}] Success: {growth.num_hops} hops, "
                    f"{len(growth.facts)} facts"
                )
                return growth
            logger.info(f"  [{idx+1}] Failed (attempt {attempt+1}), retrying...")
        return None

    workers = min(max_workers, num_questions)
    growths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_grow_one, i): i for i in range(num_questions)}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                growths.append(result)

    return growths


# =========================================================================
# Helpers
# =========================================================================


def _format_search_results(results: List[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results):
        text = r.text  # full_text if available, else snippet
        parts.append(f"[{i}] {r.title}\nSource: {r.source}\n{text}")
    return "\n\n".join(parts)


def _format_facts(facts: List[GroundingFactIR]) -> str:
    if not facts:
        return ""
    return "\n".join(f"  [F{i+1}] {f.content}" for i, f in enumerate(facts))


def _get_source_title(results: List[SearchResult], indices: List[int]) -> str:
    if indices and 0 <= indices[0] < len(results):
        return results[indices[0]].title
    return results[0].title if results else ""


def _result_to_dict(r: SearchResult) -> dict:
    return {
        "title": r.title,
        "url": r.url,
        "text": r.text,
        "source": r.source,
        "metadata": r.metadata,
    }


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Grow multi-hop research questions through real search"
    )
    parser.add_argument("--topic", type=str, required=True,
                        help="Research topic to generate questions about")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--target-hops", type=int, default=4)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--search-backend", type=str, default="arxiv+semantic_scholar+wikipedia",
                        help="Search backend spec (e.g., arxiv+semantic_scholar+openalex+wikipedia)")
    parser.add_argument("--model", type=str,
                        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--output", type=str, default="output_ir/growths.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm = LLMClient(model=args.model)
    backend = get_backend(args.search_backend)

    growths = grow_batch(
        topic=args.topic,
        search_backend=backend,
        llm=llm,
        num_questions=args.num_questions,
        target_hops=args.target_hops,
        min_hops=args.min_hops,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for g in growths:
            f.write(json.dumps(g.to_jsonl_dict()) + "\n")

    logger.info(f"\nDone: {len(growths)} questions grown, saved to {output_path}")


if __name__ == "__main__":
    main()
