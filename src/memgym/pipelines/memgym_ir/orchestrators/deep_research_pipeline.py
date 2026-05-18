"""Deep Research Pipeline for MemGym-IR.

Generates synthetic benchmark instances that mock the deep research process:
  1. Generate memory facts (multi-hop reasoning chains via real search)
  2. Add highly relevant distractors (near-miss documents that confuse)
  3. Scale length from 10K to 1M tokens (hierarchical filler)
  4. Verify: solvable WITH memory facts, NOT hackable WITHOUT them

Architecture:
  Stage 1: Grow  — build fact chains through real search (reuses question_grower)
  Stage 2: Craft — create instances with 4-tier distractors (reuses stages/generate)
  Stage 3: Scale — hierarchical filler to reach target token count
  Stage 4: Verify — memory ablation + adversarial hack check

Usage:
    python -m memgym.pipelines.memgym_ir.orchestrators.deep_research_pipeline \
        --topic "attention mechanisms" --num-questions 10 \
        --target-tokens 100000

    # 1M stress test
    python -m memgym.pipelines.memgym_ir.orchestrators.deep_research_pipeline \
        --topic "reinforcement learning" --num-questions 5 \
        --target-tokens 1000000 --target-hops 6
"""

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..backends import get_backend
from ..backends.fictionalized import ENGLISH_STOP_WORDS
from ..llm.client import LLMClient
from ..llm.prompts import (
    ADVERSARIAL_HACK_CHECK_PROMPT,
    ADVERSARIAL_HACK_CHECK_SCHEMA,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
    FICTIONALIZE_EXTRACT_PROMPT,
    FICTIONALIZE_EXTRACT_SCHEMA,
    FICTIONALIZE_REWRITE_PROMPT,
    FICTIONALIZE_REWRITE_SCHEMA,
    SUBTOPIC_GENERATOR_PROMPT,
    SUBTOPIC_GENERATOR_SCHEMA,
)
from ..types.schemas import MemGymIRInstance, GrowthResult

logger = logging.getLogger(__name__)


# =========================================================================
# Stage 3: Hierarchical token scaling
# =========================================================================

# Token thresholds for filler tiers
_TIER_THRESHOLDS = {
    "simple": 100_000,    # < 100K: simple filler paragraphs
    "academic": 500_000,  # 100K-500K: academic survey sections
    "research": 1_000_000,  # 500K-1M: full research paper sections
}


def _count_instance_tokens(instance: MemGymIRInstance) -> int:
    """Count total tokens in an instance's documents."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for turn in instance.turns:
            for doc in turn.documents:
                total += len(enc.encode(doc.get("text", "")))
        return total
    except ImportError:
        return sum(
            len(doc.get("text", "").split()) * 4 // 3
            for turn in instance.turns for doc in turn.documents
        )


def _generate_subtopics(
    topic: str,
    question: str,
    count: int,
    client: LLMClient,
) -> List[str]:
    """Generate related but distinct sub-topics for filler content."""
    result = client.get_json_completion(
        prompt=SUBTOPIC_GENERATOR_PROMPT.format(
            count=count, topic=topic, question=question,
        ),
        response_format=SUBTOPIC_GENERATOR_SCHEMA,
        temperature=0.7,
    )
    return result.get("subtopics", [topic] * count)


def _generate_one_filler(
    client: LLMClient,
    topic: str,
    subtopic: str,
    target_words: int,
    num_subsections: int,
    question: str,
    entities_str: str,
) -> Optional[Dict]:
    """Generate a single filler document. Used by ThreadPoolExecutor.

    Only called when --legacy-llm-filler is active.
    """
    from ..llm.prompts import ACADEMIC_SURVEY_FILLER_PROMPT, ACADEMIC_SURVEY_FILLER_SCHEMA
    try:
        result = client.get_json_completion(
            prompt=ACADEMIC_SURVEY_FILLER_PROMPT.format(
                topic=topic,
                subtopic=subtopic,
                target_words=target_words,
                num_subsections=num_subsections,
                question=question,
                entities=entities_str,
            ),
            response_format=ACADEMIC_SURVEY_FILLER_SCHEMA,
            temperature=0.7,
        )
        text = result.get("text", "")
        title = result.get("title", f"Survey: {subtopic[:50]}")
        if text and len(text.split()) >= 50:
            return {"title": title, "text": text, "is_supporting": False}
    except Exception as e:
        logger.debug(f"  Filler generation failed: {e}")
    return None


def _fetch_search_filler(
    search_backend,
    subtopics: List[str],
    token_budget: int,
    num_turns: int,
    concurrency: int = 4,
) -> List[Dict]:
    """Fetch real papers from search APIs as filler documents.

    Returns a list of {title, text, is_supporting} dicts, distributed
    across turns round-robin style.
    """
    docs = []
    tokens_fetched = 0
    seen_titles = set()

    def _search_one(subtopic: str) -> List[Dict]:
        try:
            results = search_backend.search(subtopic, num_results=10)
            out = []
            for r in results:
                text = r.text
                if not text or len(text.split()) < 30:
                    continue
                if r.title in seen_titles:
                    continue
                out.append({
                    "title": r.title,
                    "text": text,
                    "is_supporting": False,
                })
            return out
        except Exception as e:
            logger.debug(f"  Search filler failed for '{subtopic[:40]}': {e}")
            return []

    # Fire parallel search requests
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_search_one, sub): sub for sub in subtopics}
        for fut in as_completed(futures):
            for doc in fut.result():
                if doc["title"] in seen_titles:
                    continue
                seen_titles.add(doc["title"])
                doc_tokens = len(doc["text"].split()) * 4 // 3
                docs.append(doc)
                tokens_fetched += doc_tokens
                if tokens_fetched >= token_budget:
                    break
            if tokens_fetched >= token_budget:
                break

    logger.info(f"  Search filler: {len(docs)} papers, ~{tokens_fetched} tokens")
    return docs


def scale_to_target(
    instance: MemGymIRInstance,
    target_tokens: int,
    topic: str,
    client: LLMClient,
    search_backend=None,
    search_filler_ratio: float = 1.0,
    max_rounds: int = 100,
    concurrency: int = 8,
    legacy_llm_filler: bool = False,
) -> MemGymIRInstance:
    """Scale instance to target token count using search-based filler.

    Phase A: Fetch real papers from search APIs (fast, free, realistic).
    the baseline-train phase (legacy, off by default): LLM-generate remaining filler.
             Disabled because LLM filler re-establishes topic vocabulary,
             inflating score_long_context and leaking answers into the
             no-memory ablation condition.  Use --legacy-llm-filler to
             re-enable for backwards-compatibility testing.

    The ratio between phases is controlled by search_filler_ratio:
      1.0 = pure search (default), 0.0 = pure LLM, values between = hybrid
    """
    current = _count_instance_tokens(instance)
    if current >= target_tokens:
        logger.info(f"  Already at {current} tokens (target {target_tokens}), no scaling needed")
        return instance

    remaining = target_tokens - current
    logger.info(f"  Scaling: {current} → {target_tokens} tokens ({remaining} to fill)")

    # Extract entities for content generation
    entities = list({
        f.source_paragraph_title
        for f in instance.grounding_facts if f.source_paragraph_title
    })
    entities_str = ", ".join(entities[:10]) if entities else topic

    # Generate sub-topics for variety
    num_subtopics = min(60, remaining // 1500 + 1)
    subtopics = _generate_subtopics(topic, instance.question, num_subtopics, client)

    # ── Phase A: Search-based filler (fast, free) ──
    search_budget = int(remaining * search_filler_ratio)
    if search_backend and search_budget > 0:
        logger.info(f"  Phase A: Filling ~{search_budget} tokens from search APIs")
        search_docs = _fetch_search_filler(
            search_backend, subtopics, search_budget,
            num_turns=len(instance.turns), concurrency=concurrency,
        )
        for i, doc in enumerate(search_docs):
            turn_idx = i % len(instance.turns)
            instance.turns[turn_idx].documents.append(doc)
            remaining -= len(doc["text"].split()) * 4 // 3
    else:
        logger.info(f"  Phase A: Skipped (no search backend or ratio=0)")

    # ── the baseline-train phase: LLM-generated filler (legacy, off by default) ──
    # Disabled: LLM filler re-establishes topic vocabulary, inflating
    # score_long_context and leaking answers into no-memory ablation.
    if legacy_llm_filler and remaining > 500:
        logger.info(f"  the baseline-train phase [legacy]: LLM-filling ~{remaining} remaining tokens")
        subtopic_idx = 0
        round_num = 0
        while remaining > 500 and round_num < max_rounds:
            round_num += 1

            if remaining > 200_000:
                target_words, num_subsections = 8000, 10
            elif remaining > 50_000:
                target_words, num_subsections = 3000, 6
            elif remaining > 10_000:
                target_words, num_subsections = 1000, 3
            else:
                target_words, num_subsections = 300, 1

            expected_tokens_per_call = target_words * 4 // 3
            calls_needed = max(1, min(concurrency, remaining // max(expected_tokens_per_call, 1) + 1))

            batch_results = []
            with ThreadPoolExecutor(max_workers=calls_needed) as pool:
                futures = []
                for _ in range(calls_needed):
                    sub = subtopics[subtopic_idx % len(subtopics)]
                    subtopic_idx += 1
                    futures.append(pool.submit(
                        _generate_one_filler,
                        client, topic, sub, target_words, num_subsections,
                        instance.question, entities_str,
                    ))
                for fut in as_completed(futures):
                    doc = fut.result()
                    if doc:
                        batch_results.append(doc)

            added_this_round = 0
            for doc in batch_results:
                turn_idx = (round_num + added_this_round) % len(instance.turns)
                instance.turns[turn_idx].documents.append(doc)
                added_tokens = len(doc["text"].split()) * 4 // 3
                remaining -= added_tokens
                added_this_round += added_tokens

            logger.info(
                f"  LLM round {round_num}: {len(batch_results)} docs, "
                f"+{added_this_round} tokens (~{remaining} remaining)"
            )
            if remaining <= 0:
                break
    elif remaining > 500:
        logger.info(
            f"  the baseline-train phase: SKIPPED (legacy LLM filler disabled). "
            f"Under-fill by ~{remaining} tokens — search APIs didn't "
            f"fully cover budget. Instance proceeds at actual size."
        )

    # Recount and update
    final_tokens = _count_instance_tokens(instance)
    instance.total_tokens = final_tokens
    logger.info(f"  Scaling complete: {final_tokens} tokens (target was {target_tokens})")

    # Shuffle documents within each turn
    for turn in instance.turns:
        random.shuffle(turn.documents)

    return instance


# =========================================================================
# Stage 2.5: Fictionalize — replace real entities with fictional ones
# =========================================================================

# Synthesis prompt instructs the model to produce 'F1 tells us X ...' style
# reasoning, which often bleeds into the final answer. Strip parenthesized
# forms only — bare 'F1' can be a legitimate science term (genetics, ATPase).
_FACT_ID_CITATION = re.compile(r'\s*\(F\d+(?:\s*[,\-]\s*F?\d+)*\)\s*')


def _compile_substitution_regex(
    substitutions: List[Dict],
) -> Tuple[Optional[re.Pattern], Dict[str, str]]:
    """Compile substitutions into a single alternation regex + lookup dict.

    Returns (compiled_pattern, lookup). A single regex pass is O(m) vs
    O(n*m) for n sequential re.sub calls — 100× faster for n > 1000.
    """
    valid = [
        (s["original"], s["replacement"])
        for s in substitutions
        if s.get("original", "").strip() and len(s["original"]) >= 2
        and s["original"].lower() not in ENGLISH_STOP_WORDS
    ]
    if not valid:
        return None, {}
    valid.sort(key=lambda x: len(x[0]), reverse=True)
    lookup = {orig.lower(): repl for orig, repl in valid}
    escaped = [re.escape(orig) for orig, _ in valid]
    pattern = re.compile(
        r"\b(" + "|".join(escaped) + r")s?\b",
        re.IGNORECASE,
    )
    return pattern, lookup


def _apply_substitutions_regex(text: str, substitutions: List[Dict]) -> str:
    """Apply entity substitutions using a single compiled alternation regex.

    Word boundaries (`\\b`) on both sides prevent mid-word contamination.
    A trailing optional `s?` catches simple plurals. Longer originals take
    precedence in the alternation (sorted by length descending).
    """
    pattern, lookup = _compile_substitution_regex(substitutions)
    if not pattern:
        return text
    return pattern.sub(lambda m: lookup.get(m.group(1).lower(), m.group(0)), text)


def sanitize_instance(
    instance: MemGymIRInstance,
    substitutions: List[Dict],
) -> MemGymIRInstance:
    """Post-sanitizer: apply cached fictional-first substitutions to catch leaks.

    Under Invariant A, search results are fictionalized at retrieval time.
    But LLM-generated text (from HOP_ANALYZE, PARAGRAPH_EXPANSION, etc.)
    may reintroduce real entities from pretraining knowledge despite
    anti-leak preambles. This function sweeps all text fields and applies
    the wrapper's accumulated substitution registry.

    The regex is compiled ONCE and reused across all fields — critical for
    performance when there are thousands of substitutions.
    """
    if not substitutions:
        return instance

    # Compile the regex once for the entire instance
    pattern, lookup = _compile_substitution_regex(substitutions)
    if not pattern:
        return instance

    def _sub(text: str) -> str:
        if not text:
            return text
        return pattern.sub(
            lambda m: lookup.get(m.group(1).lower(), m.group(0)), text
        )

    # Question and answer
    instance.question = _sub(instance.question)
    instance.answer = _sub(instance.answer)

    # Grounding facts
    for fact in instance.grounding_facts:
        fact.content = _sub(fact.content)
        fact.source_paragraph_title = _sub(fact.source_paragraph_title)
        if fact.intermediate_answer:
            fact.intermediate_answer = _sub(fact.intermediate_answer)

    # All documents in all turns
    for turn in instance.turns:
        turn.sub_query = _sub(turn.sub_query)
        turn.sub_answer = _sub(turn.sub_answer)
        for doc in turn.documents:
            doc["title"] = _sub(doc.get("title", ""))
            doc["text"] = _sub(doc.get("text", ""))

    # Decomposition — sub-questions and sub-answers were built from
    # pre-fictionalization growth output, so they need the same substitution
    # pass as everything else.
    for hop in instance.decomposition:
        if hop.get("sub_question"):
            hop["sub_question"] = _sub(hop["sub_question"])
        if hop.get("sub_answer"):
            hop["sub_answer"] = _sub(hop["sub_answer"])

    # Strip parenthesized fact-id citations like "(F1)", "(F2, F3)" that
    # the synthesis prompt tends to leak into the final answer.
    instance.answer = _FACT_ID_CITATION.sub(' ', instance.answer)
    instance.answer = re.sub(r'\s{2,}', ' ', instance.answer).strip()

    # Gold supporting paragraphs
    for para in instance.gold_supporting_paragraphs:
        if "title" in para:
            para["title"] = _sub(para["title"])
        if "text" in para:
            para["text"] = _sub(para["text"])

    # Contradictions
    for contra in instance.contradictions:
        for key in ("content", "text", "title"):
            if key in contra:
                contra[key] = _sub(contra[key])

    # Distractors
    for dist in instance.distractors:
        dist.content = _sub(dist.content)

    instance.transforms_applied.append("sanitize_v1_fictional_first")
    return instance


def fictionalize_instance(
    instance: MemGymIRInstance,
    client: LLMClient,
) -> MemGymIRInstance:
    """Replace real-world entities/numbers with fictional ones.

    This prevents pretrained models from answering questions using memorized
    knowledge. After fictionalization, the ONLY way to know the facts is
    through the context documents and memory notes.

    Two-phase approach:
      the training phase: LLM extracts entities from grounding facts, generates
               fictional replacements + rewritten question/answer
      the training phase: Apply substitutions across all documents via regex
               (fast, deterministic, consistent)
    """
    # the training phase: Extract entities and generate substitution map
    facts_text = "\n".join(
        f"[{f.fact_id}] {f.content}" for f in instance.grounding_facts
    )

    logger.info("    the training phase: Extracting entities and generating substitutions")

    # Retry up to 2 times if too few substitutions (< 8), since low counts
    # often leave enough real-world terms for pretraining leakage
    min_substitutions = 8
    substitutions = []
    for fict_attempt in range(2):
        result = client.get_json_completion(
            prompt=FICTIONALIZE_EXTRACT_PROMPT.format(
                facts=facts_text,
                question=instance.question,
                answer=instance.answer,
            ),
            response_format=FICTIONALIZE_EXTRACT_SCHEMA,
            temperature=0.7 + fict_attempt * 0.2,  # increase creativity on retry
        )
        substitutions = result.get("substitutions", [])
        if len(substitutions) >= min_substitutions:
            break
        logger.info(
            f"    Only {len(substitutions)} substitutions (need {min_substitutions}), "
            f"retrying with higher temperature"
        )

    if not substitutions:
        logger.info("    No entities to fictionalize (skipping)")
        return instance

    # Sort by length descending to avoid partial replacement issues
    # (e.g., replace "Transformer XL" before "Transformer")
    substitutions.sort(key=lambda s: len(s["original"]), reverse=True)

    logger.info(f"    Found {len(substitutions)} substitutions:")
    for s in substitutions[:10]:
        logger.info(f"      {s['type']:8s}: {s['original']!r} → {s['replacement']!r}")
    if len(substitutions) > 10:
        logger.info(f"      ... and {len(substitutions) - 10} more")

    # the training phase: Apply substitutions across all text fields

    # Update question and answer — use regex for consistency with facts/docs.
    # LLM-rewritten Q/A may use slightly different fictional terms than the
    # substitution map, causing verification to fail (facts say "polycore
    # attention" but answer says "poly-kernel cross-focus").
    instance.original_question = instance.question
    regex_question = _apply_substitutions_regex(instance.question, substitutions)
    regex_answer = _apply_substitutions_regex(instance.answer, substitutions)

    # Use regex result if it changed something; fall back to LLM rewrite
    if regex_question != instance.question:
        instance.question = regex_question
    else:
        instance.question = result.get("new_question", instance.question)

    if regex_answer != instance.answer:
        instance.answer = regex_answer
    else:
        instance.answer = result.get("new_answer", instance.answer)

    # Update grounding facts
    for fact in instance.grounding_facts:
        fact.content = _apply_substitutions_regex(fact.content, substitutions)
        fact.source_paragraph_title = _apply_substitutions_regex(
            fact.source_paragraph_title, substitutions
        )
        if fact.intermediate_answer:
            fact.intermediate_answer = _apply_substitutions_regex(
                fact.intermediate_answer, substitutions
            )

    # Update all documents in all turns
    doc_count = 0
    for turn in instance.turns:
        turn.sub_query = _apply_substitutions_regex(turn.sub_query, substitutions)
        turn.sub_answer = _apply_substitutions_regex(turn.sub_answer, substitutions)
        for doc in turn.documents:
            doc["title"] = _apply_substitutions_regex(
                doc.get("title", ""), substitutions
            )
            doc["text"] = _apply_substitutions_regex(
                doc.get("text", ""), substitutions
            )
            doc_count += 1

    # Update gold supporting paragraphs
    for para in instance.gold_supporting_paragraphs:
        if "title" in para:
            para["title"] = _apply_substitutions_regex(para["title"], substitutions)
        if "text" in para:
            para["text"] = _apply_substitutions_regex(para["text"], substitutions)

    # Update contradictions
    for contra in instance.contradictions:
        for key in ("content", "text", "title"):
            if key in contra:
                contra[key] = _apply_substitutions_regex(contra[key], substitutions)

    # Update distractors
    for dist in instance.distractors:
        dist.content = _apply_substitutions_regex(dist.content, substitutions)

    # Store the substitution map in the instance for reference
    instance.transforms_applied.append("fictionalize_v2_entity_number")

    logger.info(
        f"    the training phase: Applied {len(substitutions)} substitutions across "
        f"{doc_count} documents"
    )

    return instance


# =========================================================================
# Stage 4: Verification (memory ablation + adversarial hack check)
# =========================================================================


def verify_instance(
    instance: MemGymIRInstance,
    verifier_client: LLMClient,
    worker_client: LLMClient,
    min_memory_gap: float = 0.3,
    max_hack_score: float = 0.45,
    max_retries: int = 2,
) -> Optional[MemGymIRInstance]:
    """Verify an instance passes both memory ablation and adversarial hack check.

    Returns the verified instance with verification metadata, or None if it fails.
    """
    from ..stages.calibrate import memory_fact_ablation

    for attempt in range(max_retries):
        logger.info(f"  Verification attempt {attempt + 1}/{max_retries}")

        # Test 1: Memory ablation (existing)
        ablation = memory_fact_ablation(instance, verifier_client)

        logger.info(
            f"    Ablation: no_mem={ablation.score_no_memory:.2f}, "
            f"all_mem={ablation.score_all_memory:.2f}, "
            f"long_ctx={ablation.score_long_context:.2f}, "
            f"gap={ablation.memory_gap:.2f}"
        )

        if not ablation.passed:
            if ablation.score_all_memory < 0.5:
                logger.info("    FAIL: too hard (even with all facts, score < 0.5)")
            elif ablation.memory_gap < min_memory_gap:
                logger.info(f"    FAIL: memory gap too small ({ablation.memory_gap:.2f} < {min_memory_gap})")
            elif ablation.score_no_memory > 0.5:
                logger.info(f"    FAIL: answerable without memory (no_mem={ablation.score_no_memory:.2f} > 0.5)")
            elif ablation.score_long_context > 0.90:
                logger.info(f"    FAIL: long context too easy (long_ctx={ablation.score_long_context:.2f} > 0.90)")
            elif ablation.score_all_memory < ablation.score_long_context:
                logger.info(f"    FAIL: notes worse than raw dump (all_mem={ablation.score_all_memory:.2f} < long_ctx={ablation.score_long_context:.2f})")
            continue

        # Test 2: Adversarial hack check (NEW)
        hack_score = _adversarial_hack_check(instance, worker_client)
        logger.info(f"    Hack check: score={hack_score:.2f} (max allowed: {max_hack_score})")

        if hack_score > max_hack_score:
            logger.info(f"    FAIL: hackable without memory (score {hack_score:.2f} > {max_hack_score})")
            continue

        # Both tests passed
        instance.verification = {
            **ablation.to_dict(),
            "hack_score": hack_score,
            "hack_passed": True,
        }
        return instance

    logger.info("  Verification failed after all retries")
    return None


def _adversarial_hack_check(
    instance: MemGymIRInstance,
    client: LLMClient,
) -> float:
    """Test if the question can be answered without memory (by hacking).

    Presents ONLY the distractors and non-supporting documents from the
    last turn — no notes, no memory facts. If the model can still answer
    correctly, the instance needs harder distractors.
    """
    # Collect only non-supporting documents
    non_supporting_docs = []
    for turn in instance.turns:
        for doc in turn.documents:
            if not doc.get("is_supporting", False):
                non_supporting_docs.append(doc)

    # Sample up to 20 docs to keep prompt manageable
    if len(non_supporting_docs) > 20:
        non_supporting_docs = random.sample(non_supporting_docs, 20)

    context = "\n\n".join(
        f"[{i+1}] {doc.get('title', 'Untitled')}\n{doc.get('text', '')[:1000]}"
        for i, doc in enumerate(non_supporting_docs)
    )

    if not context:
        return 0.0  # No distractors means we can't check

    try:
        result = client.get_json_completion(
            prompt=ADVERSARIAL_HACK_CHECK_PROMPT.format(
                question=instance.question,
                context=context,
            ),
            response_format=ADVERSARIAL_HACK_CHECK_SCHEMA,
            temperature=0.0,
        )
        predicted = result.get("answer", "")
        if not predicted or predicted.lower().startswith("cannot"):
            return 0.0

        # Judge the hacked answer
        judge = client.get_json_completion(
            prompt=ANSWER_JUDGE_PROMPT.format(
                gold_answer=instance.answer,
                predicted_answer=predicted,
                question=instance.question,
            ),
            response_format=ANSWER_JUDGE_SCHEMA,
            temperature=0.0,
        )
        return judge.get("score", 0.0)
    except Exception as e:
        logger.debug(f"Hack check failed: {e}")
        return 0.0


# =========================================================================
# Pipeline orchestrator
# =========================================================================


def run_deep_research_pipeline(
    topic: str = "machine learning",
    num_questions: int = 10,
    target_hops: int = 4,
    min_hops: int = 3,
    hop_choices: Optional[List[int]] = None,
    target_tokens: int = 100_000,
    search_backend_spec: str = "",  # empty = auto-detect local DB
    worker_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    verifier_model: str = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    near_miss_per_hop: int = 3,
    expand_paragraphs: bool = True,
    min_memory_gap: float = 0.3,
    max_hack_score: float = 0.45,
    skip_verification: bool = False,
    skip_fictionalize: bool = False,
    search_filler_ratio: float = 1.0,
    legacy_llm_filler: bool = False,
    concurrency: int = 4,
    resume: bool = True,
    output_dir: str = "output_ir/deep_research",
) -> List[MemGymIRInstance]:
    """Run the full deep research pipeline: Grow → Craft → Scale → Fictionalize → Verify.

    Produces instances that mock a deep research process with:
    - Multi-hop fact chains built from real search results
    - Highly relevant distractors (near-miss, adversarial, contradiction)
    - Search-based scaling: real papers from search APIs (pure retrieval)
    - Entity fictionalization (prevents pretraining knowledge exploitation)
    - Controllable length from 10K to 1M tokens
    - Dual verification: memory ablation + adversarial hack resistance
    """
    from ..generators.question_grower import grow_batch
    from ..stages.generate import craft_from_growth

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not resume:
        for f in ["growths.jsonl", "crafted.jsonl", "fictionalized.jsonl", "scaled.jsonl", "verified.jsonl"]:
            p = output_path / f
            if p.exists():
                p.unlink()
                logger.info(f"  Removed old checkpoint: {p}")

    # Auto-detect local arXiv DB if no backend specified
    if not search_backend_spec:
        import os as _os
        _db = _os.environ.get("LOCAL_ARXIV_DB", _os.path.expanduser("~/.memgym/arxiv_papers.db"))
        if _os.path.exists(_db):
            search_backend_spec = "local_arxiv+semantic_scholar+openalex+wikipedia"
            logger.info(f"Auto-detected local arXiv DB at {_db}")
        else:
            search_backend_spec = "arxiv+semantic_scholar+openalex+wikipedia"

    raw_backend = get_backend(search_backend_spec)
    worker_client = LLMClient(model=worker_model)
    verifier_client = LLMClient(model=verifier_model)

    # Invariant A: wrap backend with fictional-first layer
    from ..backends.fictionalized import FictionalizedBackend
    fictional_backend = FictionalizedBackend(
        raw_backend, worker_client, enabled=not skip_fictionalize,
    )
    search_backend = fictional_backend

    t0 = time.time()

    # Checkpoint paths
    growth_path = output_path / "growths.jsonl"
    crafted_path = output_path / "crafted.jsonl"
    scaled_path = output_path / "scaled.jsonl"
    verified_path = output_path / "verified.jsonl"

    print("=" * 60)
    print("MemGym-IR Deep Research Pipeline")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Questions: {num_questions}, Hops: {target_hops}, Min: {min_hops}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Distractors: sibling-seeded near-miss ({near_miss_per_hop}/hop, Invariant B)")
    print(f"Fictional-first wrapper: {'ON' if not skip_fictionalize else 'OFF'} (Invariant A)")
    print(f"Search filler ratio: {search_filler_ratio:.0%} search / {1-search_filler_ratio:.0%} LLM")
    print(f"Concurrency: {concurrency} workers")
    print(f"Worker: {worker_model}")
    print(f"Verifier: {verifier_model}")
    print(f"Output: {output_path}")
    print()

    # ── Stage 1: Grow fact chains through real search ──
    if resume and growth_path.exists():
        logger.info(f"Stage 1 [Grow]: Resuming from checkpoint")
        growths = [GrowthResult(**json.loads(line)) for line in open(growth_path)]
        logger.info(f"  Loaded {len(growths)} growths")
    else:
        logger.info(f"Stage 1 [Grow]: Building {num_questions} fact chains on '{topic}'")
        growths = grow_batch(
            topic=topic,
            search_backend=search_backend,
            llm=worker_client,
            num_questions=num_questions,
            target_hops=target_hops,
            min_hops=min_hops,
            max_workers=concurrency,
            hop_choices=hop_choices,
        )
        logger.info(f"Stage 1 complete: {len(growths)} fact chains")
        with open(growth_path, "w") as f:
            for g in growths:
                f.write(json.dumps(g.to_jsonl_dict()) + "\n")

    # ── Stage 2: Craft with 4-tier distractors ──
    if resume and crafted_path.exists():
        logger.info(f"\nStage 2 [Craft]: Resuming from checkpoint")
        crafted = [MemGymIRInstance(**json.loads(line)) for line in open(crafted_path)]
        logger.info(f"  Loaded {len(crafted)} crafted instances")
    else:
        logger.info(f"\nStage 2 [Craft]: Building instances with 4-tier distractors ({concurrency} workers)")

        # For the initial craft, use a moderate token target.
        # Stage 3 will scale up to the final target.
        craft_target = min(target_tokens, 50_000)

        def _craft_one(idx_growth):
            idx, growth = idx_growth
            logger.info(f"  [{idx+1}/{len(growths)}] Crafting '{growth.question[:60]}...'")
            instance = craft_from_growth(
                growth=growth,
                worker_client=worker_client,
                target_tokens=craft_target,
                near_miss_per_hop=near_miss_per_hop,
                expand_paragraphs=expand_paragraphs,
                verifier_client=verifier_client,  # Invariant C: Craft-end answerability gate
            )
            if instance:
                logger.info(
                    f"    [{idx+1}] → {len(instance.turns)} turns, "
                    f"{len(instance.grounding_facts)} facts, "
                    f"{instance.total_tokens} tokens"
                )
            else:
                logger.info(f"    [{idx+1}] → Skipped (no memory-required facts)")
            return idx, instance

        results = [None] * len(growths)
        with ThreadPoolExecutor(max_workers=min(concurrency, len(growths))) as pool:
            for idx, instance in pool.map(_craft_one, enumerate(growths)):
                results[idx] = instance

        crafted = [inst for inst in results if inst is not None]
        logger.info(f"Stage 2 complete: {len(crafted)}/{len(growths)} instances")
        with open(crafted_path, "w") as f:
            for inst in crafted:
                f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    # ── Stage 3: Scale to target token count (BEFORE fictionalize) ──
    if resume and scaled_path.exists():
        logger.info(f"\nStage 3 [Scale]: Resuming from checkpoint")
        scaled = [MemGymIRInstance(**json.loads(line)) for line in open(scaled_path)]
        logger.info(f"  Loaded {len(scaled)} scaled instances")
    else:
        logger.info(f"\nStage 3 [Scale]: Scaling {len(crafted)} instances to {target_tokens:,} tokens")
        logger.info(f"  Search filler ratio: {search_filler_ratio:.0%}")

        def _scale_one(idx_inst):
            idx, inst = idx_inst
            logger.info(f"  [{idx+1}/{len(crafted)}] Scaling '{inst.question[:50]}...'")
            return idx, scale_to_target(
                instance=inst,
                target_tokens=target_tokens,
                topic=topic,
                client=worker_client,
                search_backend=search_backend,
                search_filler_ratio=search_filler_ratio,
                legacy_llm_filler=legacy_llm_filler,
            )

        scaled = [None] * len(crafted)
        with ThreadPoolExecutor(max_workers=min(concurrency, len(crafted))) as pool:
            for idx, result in pool.map(_scale_one, enumerate(crafted)):
                scaled[idx] = result

        logger.info(f"Stage 3 complete: {len(scaled)} instances scaled")
        avg_tokens = sum(s.total_tokens for s in scaled) // max(len(scaled), 1)
        logger.info(f"  Average tokens: {avg_tokens:,}")

        with open(scaled_path, "w") as f:
            for inst in scaled:
                f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    # ── Stage 3.5: Post-sanitize — catch any real entities that leaked ──
    # With Invariant A (fictional-first backend), most text is already
    # fictional. This thin sanitizer catches LLM-generated text that
    # reintroduced real entities from pretraining knowledge.
    fictionalized_path = output_path / "fictionalized.jsonl"
    if not skip_fictionalize:
        if resume and fictionalized_path.exists():
            logger.info(f"\nStage 3.5 [Sanitize]: Resuming from checkpoint")
            scaled = [MemGymIRInstance(**json.loads(line)) for line in open(fictionalized_path)]
            logger.info(f"  Loaded {len(scaled)} sanitized instances")
        else:
            subs = fictional_backend.substitutions
            n_entities = len(fictional_backend.real_entities)
            logger.info(
                f"\nStage 3.5 [Sanitize]: Applying {len(subs)} cached substitutions "
                f"({n_entities} real entities tracked)"
            )
            if subs:
                for idx, inst in enumerate(scaled):
                    sanitize_instance(inst, subs)
                    logger.info(f"  [{idx+1}/{len(scaled)}] Sanitized")
            else:
                logger.info("  No substitutions in wrapper cache — skipping sanitize")

            scaled_out = scaled
            logger.info(f"Stage 3.5 complete: {len(scaled_out)} instances sanitized")
            with open(fictionalized_path, "w") as f:
                for inst in scaled_out:
                    f.write(json.dumps(inst.to_jsonl_dict()) + "\n")
    else:
        logger.info(f"\nStage 3.5 [Sanitize]: Skipped")

    # ── Stage 4: Verify (memory ablation + hack resistance) ──
    if skip_verification:
        verified = scaled
        logger.info(f"\nStage 4 [Verify]: Skipped")
    elif resume and verified_path.exists():
        logger.info(f"\nStage 4 [Verify]: Resuming from checkpoint")
        verified = [MemGymIRInstance(**json.loads(line)) for line in open(verified_path)]
        logger.info(f"  Loaded {len(verified)} verified instances")
    else:
        logger.info(f"\nStage 4 [Verify]: Verifying {len(scaled)} instances")

        def _verify_one(idx_inst):
            idx, inst = idx_inst
            logger.info(f"  [{idx+1}/{len(scaled)}] Verifying '{inst.question[:50]}...'")
            return idx, verify_instance(
                inst, verifier_client, worker_client,
                min_memory_gap=min_memory_gap,
                max_hack_score=max_hack_score,
            )

        results = [None] * len(scaled)
        with ThreadPoolExecutor(max_workers=min(concurrency, len(scaled))) as pool:
            for idx, result in pool.map(_verify_one, enumerate(scaled)):
                results[idx] = result

        verified = []
        for idx, result in enumerate(results):
            if result:
                verified.append(result)
                v = result.verification or {}
                logger.info(
                    f"  [{idx+1}] PASS: gap={v.get('memory_gap', 0):.2f}, "
                    f"hack={v.get('hack_score', 0):.2f}"
                )
            else:
                logger.info(f"  [{idx+1}] FAIL")

            # Incremental checkpoint
            with open(verified_path, "w") as f:
                for v_inst in verified:
                    f.write(json.dumps(v_inst.to_jsonl_dict()) + "\n")

        logger.info(f"Stage 4 complete: {len(verified)}/{len(scaled)} passed verification")

    # ── Save final output ──
    final_path = output_path / "memgym_ir_instances.jsonl"
    with open(final_path, "w") as f:
        for inst in verified:
            f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("Deep Research Pipeline Complete!")
    print("=" * 60)
    print(f"Time: {elapsed:.0f}s")
    print(f"  {len(growths)} grown → {len(crafted)} crafted → {len(scaled)} scaled → {len(verified)} verified")
    if verified:
        avg_tokens = sum(v.total_tokens for v in verified) // len(verified)
        avg_facts = sum(len(v.grounding_facts) for v in verified) // len(verified)
        avg_mem = sum(len(v.memory_required_facts) for v in verified) // len(verified)
        print(f"  Avg tokens: {avg_tokens:,}")
        print(f"  Avg facts: {avg_facts}, Avg memory-required: {avg_mem}")
    print(f"  Output: {final_path}")

    return verified


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MemGym-IR Deep Research Pipeline"
    )
    parser.add_argument("--topic", type=str, default="machine learning",
                        help="Research topic to generate questions about")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--target-hops", type=int, default=4)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--hop-choices", type=str, default=None,
                        help="Comma-separated hop counts to sample per question, "
                             "e.g. '3,4,5,6'. Overrides --target-hops when set.")
    parser.add_argument("--target-tokens", type=int, default=100_000,
                        help="Target tokens per instance (10K to 1M)")
    parser.add_argument("--search-backend", type=str, default="",
                        help="Search backend spec (empty=auto-detect local DB)")
    parser.add_argument("--worker-model", type=str,
                        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--verifier-model", type=str,
                        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--near-miss-per-hop", type=int, default=3,
                        help="Sibling-seeded near-miss distractors generated per hop "
                             "(Invariant B). Replaces --adversarial-count / "
                             "--contradiction-count / --enhance-count from the "
                             "pre-Invariant-B pipeline.")
    parser.add_argument("--no-expand", dest="expand_paragraphs",
                        action="store_false", default=True)
    parser.add_argument("--min-memory-gap", type=float, default=0.3)
    parser.add_argument("--max-hack-score", type=float, default=0.45)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--skip-fictionalize", action="store_true",
                        help="Skip entity fictionalization (use real-world names)")
    parser.add_argument("--search-filler-ratio", type=float, default=1.0,
                        help="Ratio of filler from search APIs vs LLM "
                             "(1.0=pure search [default], 0.0=pure LLM)")
    parser.add_argument("--legacy-llm-filler", action="store_true",
                        help="Re-enable the baseline-train phase LLM-generated filler (off by default; "
                             "LLM filler leaks topic vocabulary into score_long_context)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of parallel workers for grow/craft/scale/verify (default 4, use 8-10 for large runs)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output_ir/deep_research")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Suppress LiteLLM verbose output that floods logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    hop_choices = (
        [int(x) for x in args.hop_choices.split(",") if x.strip()]
        if args.hop_choices else None
    )

    run_deep_research_pipeline(
        topic=args.topic,
        num_questions=args.num_questions,
        target_hops=args.target_hops,
        min_hops=args.min_hops,
        hop_choices=hop_choices,
        target_tokens=args.target_tokens,
        search_backend_spec=args.search_backend,
        worker_model=args.worker_model,
        verifier_model=args.verifier_model,
        near_miss_per_hop=args.near_miss_per_hop,
        expand_paragraphs=args.expand_paragraphs,
        min_memory_gap=args.min_memory_gap,
        max_hack_score=args.max_hack_score,
        skip_verification=args.skip_verification,
        skip_fictionalize=args.skip_fictionalize,
        search_filler_ratio=args.search_filler_ratio,
        legacy_llm_filler=args.legacy_llm_filler,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
