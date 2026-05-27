"""Stage 3: Craft -- Build search turns, inject distractors, assemble memory files.

Distractor sources:
1. Cross-instance: Supporting paragraphs from other MuSiQue questions
2. Same-topic: Dataset distractor paragraphs about the same entities
3. Adversarial near-miss: LLM-generated plausible but wrong passages
4. Same-entity contradictions: LLM-generated contradictions
"""

import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import tiktoken
from tqdm import tqdm

from ..types.schemas import (
    FilteredIRInstance,
    GroundingFactIR,
    DistractorFactIR,
    SearchTurn,
    ContextTokensIR,
    RubricIR,
    MemGymIRInstance,
    EvictionPolicy,
)
from ..llm.client import LLMClient
from .calibrate import _ask_and_judge  # Reused by Invariant C Craft-end answerability gate.
from ..llm.prompts import (
    ADVERSARIAL_DISTRACTOR_PROMPT,
    SAME_ENTITY_CONTRADICTION_PROMPT,
    DISTRACTOR_SCHEMA,
    RED_HERRING_QUERY_PROMPT,
    RED_HERRING_QUERY_SCHEMA,
    PARAGRAPH_EXPANSION_PROMPT,
    PARAGRAPH_EXPANSION_SCHEMA,
    OVERLAP_PARAPHRASE_PROMPT,
    OVERLAP_PARAPHRASE_SCHEMA,
    DEAD_END_DOCUMENT_PROMPT,
    DEAD_END_DOCUMENT_SCHEMA,
    CONFLICTING_SOURCES_PROMPT,
    CONFLICTING_SOURCES_SCHEMA,
    FILLER_NOISE_PROMPT,
    FILLER_NOISE_SCHEMA,
    DISTRACTOR_ENHANCE_PROMPT,
    DISTRACTOR_ENHANCE_SCHEMA,
    NEAR_MISS_FACT_PROMPT,
    NEAR_MISS_FACT_SCHEMA,
    BULK_CONFUSED_CONTENT_PROMPT,
    BULK_CONFUSED_CONTENT_SCHEMA,
)


_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _build_search_turns(
    instance: FilteredIRInstance,
    extractions: Dict,
) -> List[SearchTurn]:
    """Map decomposition to search turns, placing supporting paragraphs."""
    turns = []
    for i, hop in enumerate(instance.decomposition):
        docs = []
        # Add the supporting paragraph for this hop
        para_title = hop.get("paragraph_title", "")
        para_text = hop.get("paragraph_text", "")
        if para_title and para_text:
            docs.append({
                "title": para_title,
                "text": para_text,
                "is_supporting": True,
            })

        turns.append(SearchTurn(
            turn_index=i,
            sub_query=hop.get("sub_question", ""),
            sub_answer=hop.get("sub_answer", ""),
            documents=docs,
        ))

    return turns


def _build_red_herring_turns(
    instance: FilteredIRInstance,
    all_instances: List[FilteredIRInstance],
    num_red_herrings: int,
    client: Optional[LLMClient] = None,
    dry_run: bool = False,
) -> List[SearchTurn]:
    """Build red herring turns with plausible but irrelevant search results.

    Each red herring turn has a topically related sub-query and documents
    from other instances (no supporting docs for the actual question).
    """
    if num_red_herrings <= 0:
        return []

    red_herring_turns = []
    # Get topic from question for generating plausible queries
    topic_words = [
        w for w in instance.question.split()
        if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "that", "this", "with", "from", "about"}
    ]
    topic = " ".join(topic_words[:5]) if topic_words else "general knowledge"

    for i in range(num_red_herrings):
        # Generate a plausible but irrelevant sub-query
        if client and not dry_run:
            prompt = RED_HERRING_QUERY_PROMPT.format(
                question=instance.question,
                topic=topic,
            )
            result = client.get_json_completion(
                prompt=prompt,
                response_format=RED_HERRING_QUERY_SCHEMA,
                temperature=0.8,
            )
            sub_query = result.get("query", f"Related information about {topic}")
        else:
            sub_query = f"Related information about {topic}"

        # Fill with cross-instance distractors (no supporting docs)
        candidates = [
            inst for inst in all_instances
            if inst.instance_id != instance.instance_id
        ]
        random.shuffle(candidates)
        docs = []
        for cand in candidates[:3]:
            for p in cand.supporting_paragraphs[:1]:
                docs.append({
                    "title": p["title"],
                    "text": p["text"],
                    "is_supporting": False,
                })

        red_herring_turns.append(SearchTurn(
            turn_index=-1,  # Will be reassigned after insertion
            sub_query=sub_query,
            sub_answer="",
            documents=docs,
            is_red_herring=True,
        ))

    return red_herring_turns


def _insert_red_herring_turns(
    turns: List[SearchTurn],
    red_herrings: List[SearchTurn],
) -> List[SearchTurn]:
    """Insert red herring turns at random positions between real turns.

    Avoids placing red herrings at the very beginning (before any real content)
    or at the very end (where they would be the last thing seen).
    """
    if not red_herrings:
        return turns

    combined = list(turns)
    for rh in red_herrings:
        # Insert between existing turns (not at position 0, not at end)
        if len(combined) >= 2:
            pos = random.randint(1, len(combined) - 1)
        else:
            pos = len(combined)
        combined.insert(pos, rh)

    # Reassign turn indices
    for i, turn in enumerate(combined):
        turn.turn_index = i

    return combined


# =========================================================================
# Deep Research Noise: Paragraph Expansion
# =========================================================================

def _expand_supporting_paragraph(
    title: str,
    text: str,
    key_fact: str,
    client: LLMClient,
    target_tokens: int = 600,
) -> str:
    """Expand a short supporting paragraph into a longer article-style document.

    The key fact is preserved but buried in surrounding context.
    """
    prompt = PARAGRAPH_EXPANSION_PROMPT.format(
        title=title,
        text=text,
        key_fact=key_fact,
        target_tokens=target_tokens,
    )
    result = client.get_json_completion(
        prompt=prompt,
        response_format=PARAGRAPH_EXPANSION_SCHEMA,
        temperature=0.5,
    )
    expanded = result.get("expanded_text", "")
    return expanded if expanded else text


async def _expand_supporting_paragraph_async(
    title: str,
    text: str,
    key_fact: str,
    client: LLMClient,
    target_tokens: int = 600,
) -> str:
    """Async version of paragraph expansion."""
    prompt = PARAGRAPH_EXPANSION_PROMPT.format(
        title=title,
        text=text,
        key_fact=key_fact,
        target_tokens=target_tokens,
    )
    result = await client.get_json_completion_async(
        prompt=prompt,
        response_format=PARAGRAPH_EXPANSION_SCHEMA,
        temperature=0.5,
    )
    expanded = result.get("expanded_text", "")
    return expanded if expanded else text


# =========================================================================
# Deep Research Noise: Overlapping Documents
# =========================================================================

def _generate_overlapping_documents(
    turns: List[SearchTurn],
    client: LLMClient,
    overlap_count: int = 2,
) -> List[Dict]:
    """Generate paraphrased versions of supporting paragraphs for other turns.

    Creates redundant information that forces the agent to deduplicate.
    """
    overlaps = []
    supporting_docs = []
    for turn in turns:
        for doc in turn.documents:
            if doc.get("is_supporting"):
                supporting_docs.append((turn.turn_index, doc))

    for turn_idx, doc in supporting_docs:
        for _ in range(overlap_count):
            word_count = len(doc.get("text", "").split())
            prompt = OVERLAP_PARAPHRASE_PROMPT.format(
                text=doc.get("text", ""),
                target_length=word_count,
                title=doc.get("title", ""),
            )
            result = client.get_json_completion(
                prompt=prompt,
                response_format=OVERLAP_PARAPHRASE_SCHEMA,
                temperature=0.7,
            )
            text = result.get("paraphrased_text", "")
            if text:
                overlaps.append({
                    "title": result.get("title", f"{doc.get('title', '')} (alt)"),
                    "text": text,
                    "source": "overlap",
                    "original_turn": turn_idx,
                })
    return overlaps


async def _generate_overlapping_documents_async(
    turns: List[SearchTurn],
    client: LLMClient,
    overlap_count: int = 2,
) -> List[Dict]:
    """Async version of overlapping document generation."""
    supporting_docs = []
    for turn in turns:
        for doc in turn.documents:
            if doc.get("is_supporting"):
                supporting_docs.append((turn.turn_index, doc))

    tasks = []
    for turn_idx, doc in supporting_docs:
        for _ in range(overlap_count):
            word_count = len(doc.get("text", "").split())
            prompt = OVERLAP_PARAPHRASE_PROMPT.format(
                text=doc.get("text", ""),
                target_length=word_count,
                title=doc.get("title", ""),
            )
            tasks.append((turn_idx, doc, prompt))

    overlaps = []
    for turn_idx, doc, prompt in tasks:
        result = await client.get_json_completion_async(
            prompt=prompt,
            response_format=OVERLAP_PARAPHRASE_SCHEMA,
            temperature=0.7,
        )
        text = result.get("paraphrased_text", "")
        if text:
            overlaps.append({
                "title": result.get("title", f"{doc.get('title', '')} (alt)"),
                "text": text,
                "source": "overlap",
                "original_turn": turn_idx,
            })
    return overlaps


# =========================================================================
# Deep Research Noise: Conflicting Authority Sources
# =========================================================================

def _generate_conflicting_sources(
    instance: "FilteredIRInstance",
    facts: List[Dict],
    client: LLMClient,
    count: int = 2,
    target_tokens: int = 300,
) -> List[Dict]:
    """Generate pairs of authoritative docs with different wrong answers.

    Creates 3-way conflicts: doc A says X, doc B says Y, truth is Z.
    """
    sources = []
    if not facts:
        return sources
    for i in range(count):
        fact = facts[i % len(facts)]
        prompt = CONFLICTING_SOURCES_PROMPT.format(
            question=instance.question,
            correct_answer=instance.answer,
            supporting_fact=fact.get("content", ""),
            target_tokens=target_tokens,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=CONFLICTING_SOURCES_SCHEMA,
            temperature=0.7,
        )
        for key in ("doc_a", "doc_b"):
            doc_data = result.get(key, {})
            text = doc_data.get("text", "")
            if text:
                sources.append({
                    "title": doc_data.get("title", f"Conflicting source {key}"),
                    "text": text,
                    "source": "conflicting_authority",
                    "wrong_answer": doc_data.get("wrong_answer", ""),
                    "interferes_with": fact.get("fact_id"),
                })
    return sources


async def _generate_conflicting_sources_async(
    instance: "FilteredIRInstance",
    facts: List[Dict],
    client: LLMClient,
    count: int = 2,
    target_tokens: int = 300,
) -> List[Dict]:
    """Async version of conflicting source generation."""
    sources = []
    if not facts:
        return sources
    for i in range(count):
        fact = facts[i % len(facts)]
        prompt = CONFLICTING_SOURCES_PROMPT.format(
            question=instance.question,
            correct_answer=instance.answer,
            supporting_fact=fact.get("content", ""),
            target_tokens=target_tokens,
        )
        result = await client.get_json_completion_async(
            prompt=prompt,
            response_format=CONFLICTING_SOURCES_SCHEMA,
            temperature=0.7,
        )
        for key in ("doc_a", "doc_b"):
            doc_data = result.get(key, {})
            text = doc_data.get("text", "")
            if text:
                sources.append({
                    "title": doc_data.get("title", f"Conflicting source {key}"),
                    "text": text,
                    "source": "conflicting_authority",
                    "wrong_answer": doc_data.get("wrong_answer", ""),
                    "interferes_with": fact.get("fact_id"),
                })
    return sources


# =========================================================================
# Deep Research Noise: Enhanced Dead-End Documents
# =========================================================================

def _generate_dead_end_documents(
    instance: "FilteredIRInstance",
    client: LLMClient,
    count: int = 3,
    target_tokens: int = 400,
) -> List[Dict]:
    """Generate topically related but factually useless documents."""
    entities = [
        hop.get("paragraph_title", "")
        for hop in instance.decomposition
        if hop.get("paragraph_title")
    ]
    entity_str = ", ".join(entities[:5]) if entities else "general topic"

    docs = []
    for _ in range(count):
        prompt = DEAD_END_DOCUMENT_PROMPT.format(
            question=instance.question,
            entities=entity_str,
            target_tokens=target_tokens,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=DEAD_END_DOCUMENT_SCHEMA,
            temperature=0.8,
        )
        text = result.get("text", "")
        if text:
            docs.append({
                "title": result.get("title", "Related information"),
                "text": text,
                "source": "dead_end",
            })
    return docs


async def _generate_dead_end_documents_async(
    instance: "FilteredIRInstance",
    client: LLMClient,
    count: int = 3,
    target_tokens: int = 400,
) -> List[Dict]:
    """Async version of dead-end document generation."""
    entities = [
        hop.get("paragraph_title", "")
        for hop in instance.decomposition
        if hop.get("paragraph_title")
    ]
    entity_str = ", ".join(entities[:5]) if entities else "general topic"

    docs = []
    for _ in range(count):
        prompt = DEAD_END_DOCUMENT_PROMPT.format(
            question=instance.question,
            entities=entity_str,
            target_tokens=target_tokens,
        )
        result = await client.get_json_completion_async(
            prompt=prompt,
            response_format=DEAD_END_DOCUMENT_SCHEMA,
            temperature=0.8,
        )
        text = result.get("text", "")
        if text:
            docs.append({
                "title": result.get("title", "Related information"),
                "text": text,
                "source": "dead_end",
            })
    return docs


# =========================================================================
# Filler Noise: Topic-related but irrelevant paragraphs to fill token budget
# =========================================================================

def _generate_filler_noise(
    instance: "FilteredIRInstance",
    client: LLMClient,
    target_tokens: int = 500,
) -> List[Dict]:
    """Generate topic-related but irrelevant filler paragraphs.

    Fills remaining token budget with plausible-sounding but useless content.
    Each filler paragraph is ~100 tokens.
    """
    # Extract topic from question
    topic_words = [
        w for w in instance.question.split()
        if len(w) > 3 and w.lower() not in {
            "what", "which", "where", "when", "that", "this",
            "with", "from", "about", "does", "many", "much",
        }
    ]
    topic = " ".join(topic_words[:6]) if topic_words else "general knowledge"

    # Entities to exclude (so filler doesn't accidentally contain answers)
    excluded = [
        hop.get("paragraph_title", "")
        for hop in instance.decomposition
        if hop.get("paragraph_title")
    ]
    excluded_str = ", ".join(excluded[:5]) if excluded else "none"

    num_fillers = min(20, max(1, target_tokens // 100))
    fillers = []
    for _ in range(num_fillers):
        prompt = FILLER_NOISE_PROMPT.format(
            topic=topic,
            question=instance.question,
            excluded_entities=excluded_str,
        )
        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=FILLER_NOISE_SCHEMA,
                temperature=0.9,
            )
        except Exception:
            continue
        text = result.get("filler_text", "")
        if text and len(text.split()) > 20:
            fillers.append({
                "title": result.get("topic", "Background Information"),
                "text": text,
                "source": "filler",
            })
    return fillers


async def _generate_filler_noise_async(
    instance: "FilteredIRInstance",
    client: LLMClient,
    target_tokens: int = 500,
) -> List[Dict]:
    """Async version of filler noise generation."""
    topic_words = [
        w for w in instance.question.split()
        if len(w) > 3 and w.lower() not in {
            "what", "which", "where", "when", "that", "this",
            "with", "from", "about", "does", "many", "much",
        }
    ]
    topic = " ".join(topic_words[:6]) if topic_words else "general knowledge"
    excluded = [
        hop.get("paragraph_title", "")
        for hop in instance.decomposition
        if hop.get("paragraph_title")
    ]
    excluded_str = ", ".join(excluded[:5]) if excluded else "none"

    num_fillers = min(20, max(1, target_tokens // 100))
    fillers = []
    for _ in range(num_fillers):
        prompt = FILLER_NOISE_PROMPT.format(
            topic=topic,
            question=instance.question,
            excluded_entities=excluded_str,
        )
        try:
            result = await client.get_json_completion_async(
                prompt=prompt,
                response_format=FILLER_NOISE_SCHEMA,
                temperature=0.9,
            )
        except Exception:
            continue
        text = result.get("filler_text", "")
        if text and len(text.split()) > 20:
            fillers.append({
                "title": result.get("topic", "Background Information"),
                "text": text,
                "source": "filler",
            })
    return fillers


_STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "for",
    "and", "or", "on", "at", "by", "with", "from", "that", "which", "who",
    "what", "where", "when", "how", "did", "do", "does", "has", "have", "had",
    "this", "it", "its", "be", "been", "being", "not", "no", "as", "but",
}


def _get_cross_instance_distractors(
    instance: FilteredIRInstance,
    all_instances: List[FilteredIRInstance],
    max_count: int = 3,
) -> List[Dict]:
    """Get supporting paragraphs from OTHER instances as cross-instance noise.

    Prefers topically related instances (word overlap with question) to produce
    more challenging distractors instead of obviously irrelevant noise.
    """
    candidates = [
        inst for inst in all_instances
        if inst.instance_id != instance.instance_id
    ]

    # Score candidates by word overlap with current question
    q_words = {
        w.lower().strip("?.,!\"'")
        for w in instance.question.split()
        if w.lower().strip("?.,!\"'") not in _STOPWORDS and len(w) > 2
    }

    scored = []
    for cand in candidates:
        cand_words = {
            w.lower().strip("?.,!\"'")
            for w in cand.question.split()
            if w.lower().strip("?.,!\"'") not in _STOPWORDS and len(w) > 2
        }
        # Also check paragraph titles
        for p in cand.supporting_paragraphs:
            for w in p.get("title", "").split():
                cand_words.add(w.lower().strip("?.,!\"'"))
        overlap = len(q_words & cand_words)
        scored.append((overlap, cand))

    # Sort by overlap (descending), then shuffle within same-score groups
    scored.sort(key=lambda x: -x[0])
    # Take top candidates but add some randomness within score tiers
    top_candidates = [c for _, c in scored[:max_count * 3]]
    if not top_candidates:
        random.shuffle(candidates)
        top_candidates = candidates

    distractors = []
    for cand in top_candidates:
        for p in cand.supporting_paragraphs:
            if len(distractors) >= max_count:
                break
            distractors.append({
                "title": p["title"],
                "text": p["text"],
                "source": "cross_instance",
            })
        if len(distractors) >= max_count:
            break

    return distractors[:max_count]


def _get_same_topic_distractors(
    instance: FilteredIRInstance,
    max_count: int = 3,
) -> List[Dict]:
    """Use dataset's own distractor paragraphs as same-topic noise."""
    distractors = []
    for p in instance.distractor_paragraphs[:max_count]:
        distractors.append({
            "title": p["title"],
            "text": p["text"],
            "source": "same_topic",
        })
    return distractors


def _generate_adversarial_distractors(
    instance: FilteredIRInstance,
    facts: List[Dict],
    client: LLMClient,
    max_count: int = 2,
) -> List[Dict]:
    """Generate adversarial near-miss passages via LLM.

    Cycles through facts to generate max_count distractors even when
    there are fewer facts than requested count.
    """
    distractors = []
    if not facts:
        return distractors
    for i in range(max_count):
        fact_dict = facts[i % len(facts)]
        entity = fact_dict.get("source_paragraph_title", "unknown entity")
        correct_fact = fact_dict.get("content", "")

        prompt = ADVERSARIAL_DISTRACTOR_PROMPT.format(
            entity=entity,
            correct_fact=correct_fact,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=DISTRACTOR_SCHEMA,
            temperature=0.7,
        )
        text = result.get("distractor_text", "")
        if text:
            distractors.append({
                "title": f"Adversarial: {entity}",
                "text": text,
                "source": "adversarial",
                "wrong_claim": result.get("wrong_claim", ""),
                "interferes_with": fact_dict.get("fact_id"),
            })

    return distractors


def _generate_contradiction_distractors(
    instance: FilteredIRInstance,
    facts: List[Dict],
    client: LLMClient,
    max_count: int = 2,
) -> List[Dict]:
    """Generate same-entity contradictions via LLM.

    Cycles through facts to generate max_count distractors even when
    there are fewer facts than requested count.
    """
    distractors = []
    if not facts:
        return distractors
    for i in range(max_count):
        fact_dict = facts[i % len(facts)]
        entity = fact_dict.get("source_paragraph_title", "unknown")
        fact_content = fact_dict.get("content", "")

        prompt = SAME_ENTITY_CONTRADICTION_PROMPT.format(
            entity=entity,
            fact=fact_content,
        )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=DISTRACTOR_SCHEMA,
            temperature=0.7,
        )
        text = result.get("distractor_text", result.get("contradiction_text", ""))
        if text:
            distractors.append({
                "title": f"Contradiction: {entity}",
                "text": text,
                "source": "same_entity",
                "contradicted_claim": result.get("wrong_claim", result.get("contradicted_claim", "")),
                "interferes_with": fact_dict.get("fact_id"),
            })

    return distractors


def _inject_distractors_into_turns(
    turns: List[SearchTurn],
    distractors: List[Dict],
) -> List[DistractorFactIR]:
    """Place distractors into turns based on their type.

    Precedence: an explicit ``_target_turn`` hint (set by the craft stage for
    sibling-seeded near-miss distractors under Invariant B) wins over the
    source-based heuristics, so those distractors land in the same turn as
    the fact they're meant to confuse.
    """
    distractor_facts = []
    num_turns = len(turns)

    for i, d in enumerate(distractors):
        source = d.get("source", "cross_instance")

        # Invariant B: honor explicit turn hint first. Sibling near-misses
        # share the full chain context with their source hop; injecting them
        # into the same turn co-locates them with the fact they confuse.
        hint = d.pop("_target_turn", None)
        if hint is not None and 0 <= hint < num_turns:
            target = hint
        elif source == "sibling_near_miss":
            # Fall-through (shouldn't happen — hint is set upstream) but be
            # safe: scatter across turns deterministically.
            target = i % num_turns
        elif source == "cross_instance":
            target = i % num_turns  # Spread evenly
        elif source == "same_topic":
            # Place in turn with supporting paragraph it could confuse
            target = min(i, num_turns - 1)
        elif source == "adversarial":
            # Place 1-2 turns after the supporting paragraph
            target = min(i + 1, num_turns - 1)
        elif source == "same_entity":
            # Place BEFORE the correct info
            target = max(0, min(i, num_turns - 1) - 1)
        elif source == "overlap":
            # Place in a DIFFERENT turn from the original
            orig_turn = d.get("original_turn", 0)
            candidates = [t for t in range(num_turns) if t != orig_turn]
            target = random.choice(candidates) if candidates else orig_turn
        elif source == "conflicting_authority":
            # Place BEFORE the supporting paragraph
            target = max(0, min(i, num_turns - 1) - 1)
        elif source == "dead_end":
            # Spread across turns
            target = i % num_turns
        elif source == "filler":
            # Spread evenly across turns
            target = i % num_turns
        else:
            target = i % num_turns

        # Add to turn's documents
        turns[target].documents.append({
            "title": d.get("title", ""),
            "text": d.get("text", ""),
            "is_supporting": False,
        })

        distractor_facts.append(DistractorFactIR(
            id=f"D{i+1}",
            content=d.get("text", ""),
            source=source,
            target_turn=target,
            interferes_with=d.get("interferes_with"),
        ))

    return distractor_facts


def _compute_token_counts(
    turns: List[SearchTurn],
    distractors: List[DistractorFactIR],
    supporting_paragraphs: List[Dict],
) -> ContextTokensIR:
    """Compute token counts for each component.

    Counts supporting tokens from actual turn documents (which may be expanded),
    not from the original supporting_paragraphs list.
    """
    # Count supporting tokens from actual turn documents (captures expansion)
    supporting_tokens = sum(
        _count_tokens(doc.get("text", ""))
        for turn in turns
        for doc in turn.documents
        if doc.get("is_supporting")
    )

    cross_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "cross_instance"
    )
    same_topic_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "same_topic"
    )
    adversarial_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "adversarial"
    )
    contradiction_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "same_entity"
    )
    overlap_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "overlap"
    )
    dead_end_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "dead_end"
    )
    conflicting_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "conflicting_authority"
    )
    filler_tokens = sum(
        _count_tokens(d.content) for d in distractors if d.source == "filler"
    )

    total = (
        supporting_tokens + cross_tokens + same_topic_tokens
        + adversarial_tokens + contradiction_tokens
        + overlap_tokens + dead_end_tokens + conflicting_tokens
        + filler_tokens
    )

    return ContextTokensIR(
        supporting_paragraphs=supporting_tokens,
        cross_instance_noise=cross_tokens,
        same_topic_noise=same_topic_tokens,
        adversarial_noise=adversarial_tokens,
        same_entity_contradictions=contradiction_tokens,
        redundant_overlap=overlap_tokens,
        dead_end_noise=dead_end_tokens,
        conflicting_authority=conflicting_tokens,
        filler_noise=filler_tokens,
        total=total,
    )


def _build_memory_files(turns: List[SearchTurn]) -> Dict[str, str]:
    """Assemble search results into memory files (one per turn)."""
    files = {}
    for turn in turns:
        filename = f"search_results_turn_{turn.turn_index}.txt"
        content = f"Search Query: {turn.sub_query}\n"
        content += "=" * 60 + "\n\n"
        for j, doc in enumerate(turn.documents):
            content += f"Result {j+1}: {doc.get('title', 'Untitled')}\n"
            content += "-" * 40 + "\n"
            content += doc.get("text", "") + "\n\n"
        files[filename] = content
    return files


def _build_rubric(
    instance: FilteredIRInstance,
    grounding_facts: List[GroundingFactIR],
    contradictions: List[Dict],
) -> RubricIR:
    """Build evaluation rubric."""
    retention_facts = [f.content for f in grounding_facts if not f.is_discoverable]
    noise_contradictions = [
        c.get("contradicted_claim", c.get("wrong_claim", ""))
        for c in contradictions
        if c.get("contradicted_claim") or c.get("wrong_claim")
    ]
    per_hop_answers = [
        {"hop": f.needed_at_hop, "answer": f.intermediate_answer}
        for f in grounding_facts
        if f.intermediate_answer
    ]
    bridge_deps = [
        {
            "fact_id": f.fact_id,
            "from_hop": f.first_available_at_hop,
            "to_hop": f.needed_at_hop,
        }
        for f in grounding_facts
        if f.fact_type == "bridge_fact" and f.retention_span > 0
    ]
    temporal_facts = [
        {"fact_id": f.fact_id, "retention_span": f.retention_span}
        for f in grounding_facts
        if f.retention_span > 0
    ]

    return RubricIR(
        expected_answer=instance.answer,
        retention_facts=retention_facts,
        noise_contradictions=noise_contradictions,
        temporal_facts=temporal_facts,
        per_hop_answers=per_hop_answers,
        bridge_dependencies=bridge_deps,
    )


def _compute_memory_requirements(
    turns: List[SearchTurn],
    grounding_facts: List[GroundingFactIR],
    eviction_mode: str,
    window_size: int = 1,
) -> tuple:
    """Determine which facts require memory under a given eviction policy.

    For each grounding fact, checks whether the turn containing it will be
    evicted from context by the time it's needed. Returns:
      - memory_required_fact_ids: list of fact_id strings
      - per_turn_retention: list of dicts for the rubric

    Logic:
      Facts use hop indices (first_available_at_hop, needed_at_hop) but after
      red herring turn insertion, hop indices != turn indices. We build a
      hop_to_turn mapping to translate correctly.

      Under eviction, the visible window at turn T is [T - window_size + 1, T]
      for sliding_window, or just [T] for full_eviction. If a fact's source
      turn falls outside the visible window when it's needed, the fact is
      "memory-required" — the agent must have noted it.
    """
    num_turns = len(turns)
    memory_required = []
    per_turn_retention = []

    # Build hop_to_turn mapping: hop indices -> actual turn indices
    # Non-red-herring turns keep their decomposition order
    hop_to_turn = {}
    hop_idx = 0
    for t_idx, turn in enumerate(turns):
        if not turn.is_red_herring:
            hop_to_turn[hop_idx] = t_idx
            hop_idx += 1

    # Map: actual_turn_index -> list of facts available in that turn
    turn_facts = {}
    for fact in grounding_facts:
        # Remap hop index to actual turn index
        src_turn = hop_to_turn.get(fact.first_available_at_hop, fact.first_available_at_hop)
        turn_facts.setdefault(src_turn, []).append(fact)

    for t in range(num_turns):
        # Determine visible window at turn t
        if eviction_mode == "full_eviction":
            visible_start = t
        else:  # sliding_window
            visible_start = max(0, t - window_size + 1)
        visible_range = set(range(visible_start, t + 1))

        # Facts available in visible turns
        available = []
        for vt in visible_range:
            available.extend(turn_facts.get(vt, []))
        available_ids = {f.fact_id for f in available}

        # Facts that were evicted (appeared in earlier turns now outside window)
        evicted = []
        for et in range(0, visible_start):
            evicted.extend(turn_facts.get(et, []))
        evicted_ids = {f.fact_id for f in evicted}

        # Facts needed at this turn (remap hop index to turn index)
        needed_here = [
            f for f in grounding_facts
            if hop_to_turn.get(f.needed_at_hop, f.needed_at_hop) == t
        ]
        critical = []
        for f in needed_here:
            src_turn = hop_to_turn.get(f.first_available_at_hop, f.first_available_at_hop)
            if src_turn < visible_start:
                # This fact's source turn is evicted — memory required
                critical.append(f.fact_id)
                if f.fact_id not in memory_required:
                    memory_required.append(f.fact_id)

        per_turn_retention.append({
            "turn": t,
            "available_facts": sorted(available_ids),
            "evicted_facts": sorted(evicted_ids),
            "critical": critical,
        })

    return memory_required, per_turn_retention


def craft_instance(
    instance: FilteredIRInstance,
    extraction: Dict,
    all_instances: List[FilteredIRInstance],
    worker_client: LLMClient,
    target_tokens: int = 10000,
    adversarial_count: int = 2,
    contradiction_count: int = 2,
    dry_run: bool = False,
    red_herring_count: int = 0,
    reveal_mode: str = "immediate",
    expand_paragraphs: bool = False,
    expansion_target_tokens: int = 600,
    overlap_count: int = 0,
    conflicting_sources_count: int = 0,
    eviction_mode: str = "none",
    eviction_window_size: int = 1,
    allow_agent_notes: bool = True,
    agent_notes_budget: int = 500,
    min_memory_facts: int = 1,
) -> Optional[MemGymIRInstance]:
    """Craft a single MemGymIRInstance from a filtered instance + extraction."""
    if not extraction.get("quality_gate_passed", False):
        return None

    facts_data = extraction.get("facts", [])
    grounding_facts = [GroundingFactIR(**f) for f in facts_data]

    # Build search turns
    turns = _build_search_turns(instance, extraction)

    # Phase 1: Expand supporting paragraphs into longer article-style documents
    if expand_paragraphs and not dry_run:
        for turn in turns:
            for doc in turn.documents:
                if doc.get("is_supporting"):
                    # Find the key fact for this paragraph
                    key_fact = ""
                    for f in facts_data:
                        if f.get("source_paragraph_title") == doc.get("title"):
                            key_fact = f.get("content", "")
                            break
                    if not key_fact:
                        key_fact = doc.get("text", "")[:200]
                    expanded = _expand_supporting_paragraph(
                        title=doc.get("title", ""),
                        text=doc.get("text", ""),
                        key_fact=key_fact,
                        client=worker_client,
                        target_tokens=expansion_target_tokens,
                    )
                    doc["text"] = expanded
                    doc["expanded"] = True

    # Collect distractors from all sources
    all_distractors = []

    # Compute how much token budget we have for distractors
    supporting_tokens = sum(
        _count_tokens(p.get("text", "")) for p in instance.supporting_paragraphs
    )
    # If paragraphs were expanded, re-count from actual turn docs
    if expand_paragraphs:
        supporting_tokens = sum(
            _count_tokens(doc.get("text", ""))
            for turn in turns
            for doc in turn.documents
            if doc.get("is_supporting")
        )
    distractor_budget = max(0, target_tokens - supporting_tokens)

    if not dry_run:
        # Reserve budget for LLM-generated distractors
        llm_budget = int(distractor_budget * 0.30)
        adv_budget = llm_budget // 2
        contra_budget = llm_budget - adv_budget

        # 1. Adversarial (LLM) -- generate FIRST to ensure budget
        adv_count = max(adversarial_count, adv_budget // 250)
        adversarial = _generate_adversarial_distractors(
            instance, facts_data, worker_client, max_count=adv_count
        )
        all_distractors.extend(adversarial)

        # 2. Same-entity contradictions (LLM)
        contra_count = max(contradiction_count, contra_budget // 250)
        contradictions = _generate_contradiction_distractors(
            instance, facts_data, worker_client, max_count=contra_count
        )
        all_distractors.extend(contradictions)

        # Phase 2: Overlapping documents
        if overlap_count > 0:
            overlaps = _generate_overlapping_documents(
                turns, worker_client, overlap_count=overlap_count,
            )
            all_distractors.extend(overlaps)

        # Phase 3: Conflicting authority sources
        if conflicting_sources_count > 0:
            conflicts = _generate_conflicting_sources(
                instance, facts_data, worker_client,
                count=conflicting_sources_count,
            )
            all_distractors.extend(conflicts)
    else:
        contradictions = []

    # 3. Same-topic (no LLM) -- use all available from dataset
    same_topic_count = max(3, len(instance.distractor_paragraphs))
    same_topic = _get_same_topic_distractors(instance, max_count=same_topic_count)
    all_distractors.extend(same_topic)

    # 4. Cross-instance (no LLM) -- fill remaining budget
    current_tokens = sum(_count_tokens(d.get("text", "")) for d in all_distractors)
    remaining_budget = max(0, distractor_budget - current_tokens)
    cross_count = max(3, remaining_budget // 150)
    cross = _get_cross_instance_distractors(instance, all_instances, max_count=cross_count)
    all_distractors.extend(cross)

    # Top up if still under budget
    current_tokens = sum(_count_tokens(d.get("text", "")) for d in all_distractors)
    remaining = distractor_budget - current_tokens
    if remaining > 500:
        extra_cross = _get_cross_instance_distractors(
            instance, all_instances, max_count=remaining // 150
        )
        existing_hashes = {hash(d.get("text", "")[:100]) for d in all_distractors}
        for d in extra_cross:
            h = hash(d.get("text", "")[:100])
            if h not in existing_hashes:
                all_distractors.append(d)
                existing_hashes.add(h)

    # 5. Filler noise (LLM) -- fill remaining token budget with irrelevant content
    if not dry_run:
        current_tokens = sum(_count_tokens(d.get("text", "")) for d in all_distractors)
        filler_remaining = distractor_budget - current_tokens
        if filler_remaining > 300:
            fillers = _generate_filler_noise(
                instance, worker_client, target_tokens=filler_remaining,
            )
            all_distractors.extend(fillers)

    # Inject distractors into turns
    distractor_facts = _inject_distractors_into_turns(turns, all_distractors)

    # Token budget control: truncate documents if over budget
    total_tokens = sum(
        _count_tokens(doc.get("text", ""))
        for turn in turns
        for doc in turn.documents
    )
    if total_tokens > target_tokens:
        # Truncate non-supporting documents proportionally
        ratio = target_tokens / max(total_tokens, 1)
        enc = _get_encoder()
        for turn in turns:
            for doc in turn.documents:
                if not doc.get("is_supporting", False):
                    tokens = enc.encode(doc["text"])
                    max_len = max(50, int(len(tokens) * ratio))
                    if len(tokens) > max_len:
                        doc["text"] = enc.decode(tokens[:max_len]) + "..."

    # Insert red herring turns if requested
    # For deep research, use enhanced dead-end documents instead of random cross-instance
    if red_herring_count > 0:
        if expand_paragraphs and not dry_run:
            # Enhanced: use LLM-generated dead-end documents for red herring turns
            for rh_idx in range(red_herring_count):
                dead_ends = _generate_dead_end_documents(
                    instance, worker_client, count=3, target_tokens=400,
                )
                rh_docs = []
                for de in dead_ends:
                    rh_docs.append({
                        "title": de["title"],
                        "text": de["text"],
                        "is_supporting": False,
                    })
                    all_distractors.append(de)
                # Generate a plausible sub-query for the dead-end turn
                topic_words = [
                    w for w in instance.question.split()
                    if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "that", "this", "with", "from", "about"}
                ]
                topic = " ".join(topic_words[:5]) if topic_words else "general knowledge"
                if worker_client:
                    prompt = RED_HERRING_QUERY_PROMPT.format(
                        question=instance.question, topic=topic,
                    )
                    result = worker_client.get_json_completion(
                        prompt=prompt, response_format=RED_HERRING_QUERY_SCHEMA, temperature=0.8,
                    )
                    sub_query = result.get("query", f"Related information about {topic}")
                else:
                    sub_query = f"Related information about {topic}"
                rh_turn = SearchTurn(
                    turn_index=-1, sub_query=sub_query, sub_answer="",
                    documents=rh_docs, is_red_herring=True,
                )
                # Insert at random position between real turns
                if len(turns) >= 2:
                    pos = random.randint(1, len(turns) - 1)
                else:
                    pos = len(turns)
                turns.insert(pos, rh_turn)
            # Reassign turn indices
            for i, turn in enumerate(turns):
                turn.turn_index = i
        else:
            red_herrings = _build_red_herring_turns(
                instance, all_instances, red_herring_count,
                client=worker_client, dry_run=dry_run,
            )
            turns = _insert_red_herring_turns(turns, red_herrings)

    # Build memory files
    memory_files = _build_memory_files(turns)

    # Compute token counts
    context_tokens = _compute_token_counts(
        turns, distractor_facts, instance.supporting_paragraphs
    )

    # Build rubric
    contradiction_list = [d for d in all_distractors if d.get("source") == "same_entity"]
    rubric = _build_rubric(instance, grounding_facts, contradiction_list)

    # Compute memory requirements if eviction is enabled
    eviction_policy = None
    memory_required_facts = []
    if eviction_mode != "none":
        memory_required_facts, per_turn_retention = _compute_memory_requirements(
            turns, grounding_facts, eviction_mode, eviction_window_size,
        )
        rubric.per_turn_retention = per_turn_retention
        eviction_policy = EvictionPolicy(
            mode=eviction_mode,
            window_size=eviction_window_size,
            allow_notes=allow_agent_notes,
            notes_budget_tokens=agent_notes_budget,
        )
        # Quality gate: skip instances without enough memory challenge
        if len(memory_required_facts) < min_memory_facts:
            return None

    return MemGymIRInstance(
        instance_id=f"memgym_ir__{instance.instance_id}",
        base_instance=instance.instance_id,
        source_dataset=instance.source_dataset,
        question=instance.question,
        original_question=instance.question,
        answer=instance.answer,
        num_hops=instance.num_hops,
        decomposition=instance.decomposition,
        turns=turns,
        grounding_facts=grounding_facts,
        distractors=distractor_facts,
        memory_files=memory_files,
        gold_supporting_paragraphs=instance.supporting_paragraphs,
        contradictions=contradiction_list,
        transforms_applied=[],
        difficulty_preset="easy",
        context_tokens=context_tokens,
        total_tokens=context_tokens.total,
        rubric=rubric,
        reveal_mode=reveal_mode,
        eviction_policy=eviction_policy,
        memory_required_facts=memory_required_facts,
    )


def craft_from_growth(
    growth: "GrowthResult",
    worker_client: LLMClient,
    target_tokens: int = 100000,
    near_miss_per_hop: int = 3,
    expand_paragraphs: bool = True,
    expansion_target_tokens: int = 800,
    filler_count: int = 20,
    eviction_mode: str = "full_eviction",
    eviction_window_size: int = 1,
    allow_agent_notes: bool = True,
    agent_notes_budget: int = 200,
    min_memory_facts: int = 1,
    verifier_client: Optional[LLMClient] = None,
    answerability_threshold: float = 0.5,
) -> Optional[MemGymIRInstance]:
    """Craft a MemGymIRInstance from a GrowthResult (grow-by-search output).

    Distractor strategy (post-Invariant B, 2026-04-08):
      Tier 1: Natural distractors — non-supporting search results already in
              ``turns``. Free, and — crucially under Invariant B — they are
              *siblings* of the answer branch: results whose first K-1 hops
              are identical to the supporting chain and which only diverge at
              the current hop. Globally indistinguishable up to that point.
      Tier 2: Sibling-seeded near-miss — for each hop, feed the same-hop
              siblings into ``NEAR_MISS_FACT_PROMPT`` with the hop's grounding
              fact as the target. Produces distractors that share the answer
              branch's full local vocabulary and then minimally diverge from
              the correct fact. Co-located with the supporting doc in the
              same turn by the injector.
      Tier 3: Bulk filler — retrieved or synthesized to fill the token budget
              (Enabler E will make this pure retrieval; today it's a mix).

    The old Tier 3a (ADVERSARIAL_DISTRACTOR_PROMPT) and Tier 3b
    (SAME_ENTITY_CONTRADICTION_PROMPT) synthesized distractors from thin air
    with no local-vocabulary guarantee, which made them trivially rejectable
    without reading memory. Replaced entirely by sibling-seeded near-miss.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    from ..types.schemas import GrowthResult  # avoid circular import

    grounding_facts = list(growth.facts)
    facts_data = [f.model_dump() for f in grounding_facts]

    # Build SearchTurns from growth hops
    turns = []
    for hop in growth.hops:
        docs = []
        for i, r in enumerate(hop.results):
            is_supporting = i in hop.supporting_doc_indices
            docs.append({
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "is_supporting": is_supporting,
            })
        turns.append(SearchTurn(
            turn_index=hop.hop_index,
            sub_query=hop.query,
            sub_answer=hop.fact.content,
            documents=docs,
        ))

    # Phase 1: Expand documents to article length (parallel)
    # For large target_tokens, expand ALL documents (not just supporting)
    # to create a realistic long-context scenario
    if expand_paragraphs:
        from concurrent.futures import ThreadPoolExecutor

        expand_all = target_tokens >= 50000
        expand_tasks = []  # (turn_idx, doc_idx, key_fact)
        for ti, turn in enumerate(turns):
            for di, doc in enumerate(turn.documents):
                is_short = len(doc.get("text", "").split()) < 150
                should_expand = doc.get("is_supporting") or (expand_all and is_short)
                if not should_expand:
                    continue
                key_fact = doc.get("text", "")[:200]
                if doc.get("is_supporting"):
                    for f in facts_data:
                        if f.get("source_paragraph_title") == doc.get("title"):
                            key_fact = f.get("content", "")
                            break
                expand_tasks.append((ti, di, key_fact))

        def _expand_one(task):
            ti, di, key_fact = task
            doc = turns[ti].documents[di]
            try:
                expanded = _expand_supporting_paragraph(
                    title=doc.get("title", ""),
                    text=doc.get("text", ""),
                    key_fact=key_fact,
                    client=worker_client,
                    target_tokens=expansion_target_tokens,
                )
                return ti, di, expanded
            except Exception as e:
                _log.debug(f"Expansion failed for '{doc.get('title', '')[:40]}': {e}")
                return ti, di, None

        if expand_tasks:
            with ThreadPoolExecutor(max_workers=min(8, len(expand_tasks))) as pool:
                for ti, di, expanded in pool.map(_expand_one, expand_tasks):
                    if expanded:
                        turns[ti].documents[di]["text"] = expanded
                        turns[ti].documents[di]["expanded"] = True

    # Natural distractors (Tier 1) already sit in the turns as the
    # non-supporting docs from Grow — no extra collection needed. Under
    # Invariant B, they're also the raw material for Tier 2 via HopRecord's
    # ``siblings()`` helper (which reads from growth.hops directly).
    all_extra_distractors = []

    # Create a minimal FilteredIRInstance for helper functions
    pseudo_instance = FilteredIRInstance(
        instance_id=f"growth__{hash(growth.question) % 100000:05d}",
        source_dataset="deep_research",
        question=growth.question,
        answer=growth.answer,
        num_hops=growth.num_hops,
        decomposition=[
            {"sub_question": h.query, "sub_answer": h.fact.content,
             "paragraph_title": h.fact.source_paragraph_title}
            for h in growth.hops
        ],
        supporting_paragraphs=[
            {"title": doc["title"], "text": doc["text"], "is_supporting": True}
            for turn in turns for doc in turn.documents if doc.get("is_supporting")
        ],
        distractor_paragraphs=[],
    )

    # ── Tier 2: Sibling-seeded near-miss (Invariant B) ──
    # For each hop, pair its grounding fact with siblings from THAT hop and
    # rewrite each sibling into a near-miss distractor. Parallelize across
    # (hop, sibling) pairs so the craft stage doesn't bloat in wall-clock.
    from concurrent.futures import ThreadPoolExecutor

    def _sibling_near_miss_task(hop_idx: int, hop_fact, sib: Dict) -> Optional[Dict]:
        sib_text = sib.get("text", "")
        if not sib_text or len(sib_text.split()) < 20:
            return None
        try:
            result = worker_client.get_json_completion(
                prompt=NEAR_MISS_FACT_PROMPT.format(
                    question=growth.question,
                    answer=growth.answer,
                    real_fact=hop_fact.content,
                    source_title=hop_fact.source_paragraph_title or "",
                    distractor_text=sib_text,
                ),
                response_format=NEAR_MISS_FACT_SCHEMA,
                temperature=0.7,
            )
        except Exception:
            return None
        near_miss_text = result.get("near_miss_text", "")
        if not near_miss_text or len(near_miss_text.split()) <= 20:
            return None
        # Anti-leak guard: if the answer string appears verbatim in the
        # generated text (case-insensitive substring), drop the distractor.
        # The LLM occasionally ignores the no-leakage rule; this is a cheap
        # backstop against the most obvious failures.
        if growth.answer and growth.answer.strip():
            ans_lower = growth.answer.strip().lower()
            if len(ans_lower) >= 6 and ans_lower in near_miss_text.lower():
                return None
        return {
            "title": sib.get("title") or hop_fact.source_paragraph_title or "",
            "text": near_miss_text,
            "source": "sibling_near_miss",
            "targets_fact": result.get("targets_fact", hop_fact.fact_id),
            "changed_detail": result.get("changed_detail", ""),
            "preferred_turn": hop_idx,
            "interferes_with": hop_fact.fact_id,
        }

    sibling_tasks = []
    if near_miss_per_hop > 0:
        for hop in growth.hops:
            siblings = hop.siblings()[:near_miss_per_hop]
            for sib in siblings:
                sibling_tasks.append((hop.hop_index, hop.fact, sib))

    if sibling_tasks:
        max_workers = min(8, len(sibling_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for nm in pool.map(lambda t: _sibling_near_miss_task(*t), sibling_tasks):
                if nm is not None:
                    all_extra_distractors.append(nm)

    # ── Tier 4: Bulk fill to reach target token budget ──
    current_tokens = sum(
        _count_tokens(doc.get("text", "")) for turn in turns for doc in turn.documents
    ) + sum(_count_tokens(d.get("text", "")) for d in all_extra_distractors)
    remaining = max(0, target_tokens - current_tokens)

    if remaining > 1000:
        # Extract entities for bulk content generation
        entities = list({
            f.get("source_paragraph_title", "")
            for f in facts_data if f.get("source_paragraph_title")
        })
        entities_str = ", ".join(entities[:8]) if entities else growth.topic

        # Generate bulk confused content in batches
        batch_size = 5  # paragraphs per LLM call
        target_words_per_para = 300
        max_batches = min(10, remaining // (batch_size * target_words_per_para // 2))

        for _ in range(max(1, max_batches)):
            current_tokens = sum(
                _count_tokens(doc.get("text", "")) for turn in turns for doc in turn.documents
            ) + sum(_count_tokens(d.get("text", "")) for d in all_extra_distractors)
            if current_tokens >= target_tokens:
                break
            try:
                result = worker_client.get_json_completion(
                    prompt=BULK_CONFUSED_CONTENT_PROMPT.format(
                        topic=growth.topic,
                        entities=entities_str,
                        num_paragraphs=batch_size,
                        target_words=target_words_per_para,
                    ),
                    response_format=BULK_CONFUSED_CONTENT_SCHEMA,
                    temperature=0.8,
                )
                for para in result.get("paragraphs", []):
                    text = para.get("text", "")
                    if text and len(text.split()) > 30:
                        all_extra_distractors.append({
                            "title": para.get("title", "Background"),
                            "text": text,
                            "source": "filler",
                        })
            except Exception:
                continue

    # ── Inject distractors into turns + shuffle ──
    # Concentrate enhanced distractors near their target supporting turns
    for d in all_extra_distractors:
        preferred = d.pop("preferred_turn", None)
        if preferred is not None and 0 <= preferred < len(turns):
            d["_target_turn"] = preferred  # hint for injection

    distractor_facts = _inject_distractors_into_turns(turns, all_extra_distractors)

    # Shuffle document order within each turn (realistic search result ordering)
    for turn in turns:
        random.shuffle(turn.documents)

    # Token budget control
    total_tokens = sum(
        _count_tokens(doc.get("text", "")) for turn in turns for doc in turn.documents
    )
    if total_tokens > target_tokens:
        ratio = target_tokens / max(total_tokens, 1)
        enc = _get_encoder()
        for turn in turns:
            for doc in turn.documents:
                if not doc.get("is_supporting", False):
                    tokens = enc.encode(doc["text"])
                    max_len = max(50, int(len(tokens) * ratio))
                    if len(tokens) > max_len:
                        doc["text"] = enc.decode(tokens[:max_len]) + "..."

    # Build memory files, token counts, rubric
    memory_files = _build_memory_files(turns)
    context_tokens = _compute_token_counts(
        turns, distractor_facts, pseudo_instance.supporting_paragraphs,
    )
    contradiction_list = [d for d in all_extra_distractors if d.get("source") == "same_entity"]
    rubric = _build_rubric(pseudo_instance, grounding_facts, contradiction_list)

    # Compute memory requirements
    memory_required_facts, per_turn_retention = _compute_memory_requirements(
        turns, grounding_facts, eviction_mode, eviction_window_size,
    )
    rubric.per_turn_retention = per_turn_retention

    if len(memory_required_facts) < min_memory_facts:
        return None

    # ── Invariant C: Craft-end answerability gate ──
    # Probe whether the question is answerable from the grounding facts ALONE
    # (no visible documents). This catches "too hard" instances — those whose
    # answer is not fully contained in the facts the agent will be asked to
    # remember — before they consume Scale and Verify budget.
    #
    # Reuses the same _ask_and_judge primitive that calibrate.memory_fact_ablation
    # uses, so the decision is consistent across stages. Threshold 0.5 is the
    # genuine "the facts are sufficient" bar (the ablation's 0.75 no-memory
    # threshold is loosened only because last-turn docs leak partial answers,
    # which doesn't apply here since visible_documents="").
    if verifier_client is not None and grounding_facts:
        bridge_facts = [f for f in grounding_facts if f.retention_span > 0]
        gate_facts = bridge_facts or grounding_facts
        notes = "\n".join(f"- {f.content}" for f in gate_facts)
        try:
            probe_score = _ask_and_judge(
                question=growth.question,
                notes=notes,
                visible_documents="(no documents available)",
                expected_answer=growth.answer,
                client=verifier_client,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                f"Craft-end answerability gate errored, allowing through: {e}"
            )
            probe_score = 1.0
        if probe_score < answerability_threshold:
            import logging as _logging
            _logging.getLogger(__name__).info(
                f"  ✗ Craft-end answerability gate rejected "
                f"(probe={probe_score:.2f} < {answerability_threshold:.2f}): "
                f"'{growth.question[:60]}...'"
            )
            return None

    eviction_policy = EvictionPolicy(
        mode=eviction_mode,
        window_size=eviction_window_size,
        allow_notes=allow_agent_notes,
        notes_budget_tokens=agent_notes_budget,
    )

    return MemGymIRInstance(
        instance_id=f"memgym_ir__research__{hash(growth.question) % 100000:05d}",
        base_instance=growth.seed_question,
        source_dataset="deep_research",
        question=growth.question,
        original_question=growth.seed_question,
        answer=growth.answer,
        num_hops=growth.num_hops,
        decomposition=[
            {"sub_question": h.query, "sub_answer": h.fact.content}
            for h in growth.hops
        ],
        turns=turns,
        grounding_facts=grounding_facts,
        distractors=distractor_facts,
        memory_files=memory_files,
        gold_supporting_paragraphs=pseudo_instance.supporting_paragraphs,
        contradictions=contradiction_list,
        transforms_applied=["grow_by_search"],
        difficulty_preset="research_memory",
        reveal_mode="immediate",
        eviction_policy=eviction_policy,
        memory_required_facts=memory_required_facts,
        context_tokens=context_tokens,
        total_tokens=context_tokens.total,
        rubric=rubric,
    )


async def _generate_adversarial_async(instance, facts_data, client, max_count):
    distractors = []
    if not facts_data:
        return distractors
    for i in range(max_count):
        fact_dict = facts_data[i % len(facts_data)]
        entity = fact_dict.get("source_paragraph_title", "unknown entity")
        correct_fact = fact_dict.get("content", "")
        prompt = ADVERSARIAL_DISTRACTOR_PROMPT.format(entity=entity, correct_fact=correct_fact)
        result = await client.get_json_completion_async(
            prompt=prompt, response_format=DISTRACTOR_SCHEMA, temperature=0.7,
        )
        text = result.get("distractor_text", "")
        if text:
            distractors.append({
                "title": f"Adversarial: {entity}", "text": text, "source": "adversarial",
                "wrong_claim": result.get("wrong_claim", ""),
                "interferes_with": fact_dict.get("fact_id"),
            })
    return distractors


async def _generate_contradiction_async(instance, facts_data, client, max_count):
    distractors = []
    if not facts_data:
        return distractors
    for i in range(max_count):
        fact_dict = facts_data[i % len(facts_data)]
        entity = fact_dict.get("source_paragraph_title", "unknown")
        fact_content = fact_dict.get("content", "")
        prompt = SAME_ENTITY_CONTRADICTION_PROMPT.format(entity=entity, fact=fact_content)
        result = await client.get_json_completion_async(
            prompt=prompt, response_format=DISTRACTOR_SCHEMA, temperature=0.7,
        )
        text = result.get("distractor_text", result.get("contradiction_text", ""))
        if text:
            distractors.append({
                "title": f"Contradiction: {entity}", "text": text, "source": "same_entity",
                "contradicted_claim": result.get("wrong_claim", result.get("contradicted_claim", "")),
                "interferes_with": fact_dict.get("fact_id"),
            })
    return distractors


async def craft_instance_async(
    instance, extraction, all_instances, worker_client,
    target_tokens=10000, adversarial_count=2, contradiction_count=2,
    red_herring_count=0, reveal_mode="immediate",
    expand_paragraphs=False, expansion_target_tokens=600,
    overlap_count=0, conflicting_sources_count=0,
    eviction_mode="none", eviction_window_size=1,
    allow_agent_notes=True, agent_notes_budget=500,
    min_memory_facts=1,
):
    """Async version of craft_instance — parallelizes LLM distractor generation."""
    if not extraction.get("quality_gate_passed", False):
        return None

    facts_data = extraction.get("facts", [])
    grounding_facts = [GroundingFactIR(**f) for f in facts_data]
    turns = _build_search_turns(instance, extraction)

    # Phase 1: Expand supporting paragraphs
    if expand_paragraphs:
        expansion_tasks = []
        for turn in turns:
            for doc in turn.documents:
                if doc.get("is_supporting"):
                    key_fact = ""
                    for f in facts_data:
                        if f.get("source_paragraph_title") == doc.get("title"):
                            key_fact = f.get("content", "")
                            break
                    if not key_fact:
                        key_fact = doc.get("text", "")[:200]
                    expansion_tasks.append((turn, doc, key_fact))

        for turn, doc, key_fact in expansion_tasks:
            expanded = await _expand_supporting_paragraph_async(
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                key_fact=key_fact,
                client=worker_client,
                target_tokens=expansion_target_tokens,
            )
            doc["text"] = expanded
            doc["expanded"] = True

    all_distractors = []
    # Re-count supporting tokens after possible expansion
    supporting_tokens = sum(
        _count_tokens(doc.get("text", ""))
        for turn in turns
        for doc in turn.documents
        if doc.get("is_supporting")
    )
    distractor_budget = max(0, target_tokens - supporting_tokens)

    # Reserve budget for LLM-generated distractors
    llm_budget = int(distractor_budget * 0.30)
    adv_budget = llm_budget // 2
    contra_budget = llm_budget - adv_budget

    # 1. LLM distractors — run in parallel
    adv_count = max(adversarial_count, adv_budget // 250)
    contra_count = max(contradiction_count, contra_budget // 250)

    gather_tasks = [
        _generate_adversarial_async(instance, facts_data, worker_client, adv_count),
        _generate_contradiction_async(instance, facts_data, worker_client, contra_count),
    ]
    if overlap_count > 0:
        gather_tasks.append(
            _generate_overlapping_documents_async(turns, worker_client, overlap_count)
        )
    if conflicting_sources_count > 0:
        gather_tasks.append(
            _generate_conflicting_sources_async(
                instance, facts_data, worker_client, conflicting_sources_count,
            )
        )

    results = await asyncio.gather(*gather_tasks)
    # Unpack results
    adversarial = results[0]
    contradictions = results[1]
    all_distractors.extend(adversarial)
    all_distractors.extend(contradictions)
    idx = 2
    if overlap_count > 0:
        all_distractors.extend(results[idx])
        idx += 1
    if conflicting_sources_count > 0:
        all_distractors.extend(results[idx])
        idx += 1

    # 2. Same-topic (no LLM)
    same_topic_count = max(3, len(instance.distractor_paragraphs))
    same_topic = _get_same_topic_distractors(instance, max_count=same_topic_count)
    all_distractors.extend(same_topic)

    # 3. Cross-instance fills remaining budget
    current_tokens = sum(_count_tokens(d.get("text", "")) for d in all_distractors)
    remaining_budget = max(0, distractor_budget - current_tokens)
    cross_count = max(3, remaining_budget // 150)
    cross = _get_cross_instance_distractors(instance, all_instances, max_count=cross_count)
    all_distractors.extend(cross)

    # Top up if still under budget
    current_tokens = sum(_count_tokens(d.get("text", "")) for d in all_distractors)
    remaining = distractor_budget - current_tokens
    if remaining > 500:
        extra_cross = _get_cross_instance_distractors(instance, all_instances, max_count=remaining // 150)
        existing_hashes = {hash(d.get("text", "")[:100]) for d in all_distractors}
        for d in extra_cross:
            h = hash(d.get("text", "")[:100])
            if h not in existing_hashes:
                all_distractors.append(d)
                existing_hashes.add(h)

    distractor_facts = _inject_distractors_into_turns(turns, all_distractors)

    # Token budget truncation
    total_tokens = sum(_count_tokens(doc.get("text", "")) for turn in turns for doc in turn.documents)
    if total_tokens > target_tokens:
        ratio = target_tokens / max(total_tokens, 1)
        enc = _get_encoder()
        for turn in turns:
            for doc in turn.documents:
                if not doc.get("is_supporting", False):
                    tokens = enc.encode(doc["text"])
                    max_len = max(50, int(len(tokens) * ratio))
                    if len(tokens) > max_len:
                        doc["text"] = enc.decode(tokens[:max_len]) + "..."

    # Insert red herring turns if requested
    if red_herring_count > 0:
        if expand_paragraphs:
            # Enhanced dead-end documents for deep research mode
            for rh_idx in range(red_herring_count):
                dead_ends = await _generate_dead_end_documents_async(
                    instance, worker_client, count=3, target_tokens=400,
                )
                rh_docs = []
                for de in dead_ends:
                    rh_docs.append({
                        "title": de["title"],
                        "text": de["text"],
                        "is_supporting": False,
                    })
                    all_distractors.append(de)
                topic_words = [
                    w for w in instance.question.split()
                    if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "that", "this", "with", "from", "about"}
                ]
                topic = " ".join(topic_words[:5]) if topic_words else "general knowledge"
                prompt = RED_HERRING_QUERY_PROMPT.format(
                    question=instance.question, topic=topic,
                )
                result = await worker_client.get_json_completion_async(
                    prompt=prompt, response_format=RED_HERRING_QUERY_SCHEMA, temperature=0.8,
                )
                sub_query = result.get("query", f"Related information about {topic}")
                rh_turn = SearchTurn(
                    turn_index=-1, sub_query=sub_query, sub_answer="",
                    documents=rh_docs, is_red_herring=True,
                )
                if len(turns) >= 2:
                    pos = random.randint(1, len(turns) - 1)
                else:
                    pos = len(turns)
                turns.insert(pos, rh_turn)
            for i, turn in enumerate(turns):
                turn.turn_index = i
        else:
            red_herrings = _build_red_herring_turns(
                instance, all_instances, red_herring_count,
                client=worker_client, dry_run=False,
            )
            turns = _insert_red_herring_turns(turns, red_herrings)

    memory_files = _build_memory_files(turns)
    context_tokens = _compute_token_counts(turns, distractor_facts, instance.supporting_paragraphs)
    contradiction_list = [d for d in all_distractors if d.get("source") == "same_entity"]
    rubric = _build_rubric(instance, grounding_facts, contradiction_list)

    # Compute memory requirements if eviction is enabled
    eviction_policy = None
    memory_required_facts = []
    if eviction_mode != "none":
        memory_required_facts, per_turn_retention = _compute_memory_requirements(
            turns, grounding_facts, eviction_mode, eviction_window_size,
        )
        rubric.per_turn_retention = per_turn_retention
        eviction_policy = EvictionPolicy(
            mode=eviction_mode,
            window_size=eviction_window_size,
            allow_notes=allow_agent_notes,
            notes_budget_tokens=agent_notes_budget,
        )
        # Quality gate: skip instances without enough memory challenge
        if len(memory_required_facts) < min_memory_facts:
            return None

    return MemGymIRInstance(
        instance_id=f"memgym_ir__{instance.instance_id}",
        base_instance=instance.instance_id,
        source_dataset=instance.source_dataset,
        question=instance.question,
        original_question=instance.question,
        answer=instance.answer,
        num_hops=instance.num_hops,
        decomposition=instance.decomposition,
        turns=turns,
        grounding_facts=grounding_facts,
        distractors=distractor_facts,
        memory_files=memory_files,
        gold_supporting_paragraphs=instance.supporting_paragraphs,
        contradictions=contradiction_list,
        transforms_applied=[],
        difficulty_preset="easy",
        context_tokens=context_tokens,
        total_tokens=context_tokens.total,
        rubric=rubric,
        reveal_mode=reveal_mode,
        eviction_policy=eviction_policy,
        memory_required_facts=memory_required_facts,
    )


def craft_all(
    instances: List[FilteredIRInstance],
    extractions: List[Dict],
    worker_client: LLMClient,
    target_tokens: int = 10000,
    adversarial_count: int = 2,
    contradiction_count: int = 2,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 10,
    red_herring_count: int = 0,
    reveal_mode: str = "immediate",
    expand_paragraphs: bool = False,
    expansion_target_tokens: int = 600,
    overlap_count: int = 0,
    conflicting_sources_count: int = 0,
    eviction_mode: str = "none",
    eviction_window_size: int = 1,
    allow_agent_notes: bool = True,
    agent_notes_budget: int = 500,
    min_memory_facts: int = 1,
) -> List[MemGymIRInstance]:
    """Craft all instances."""
    extraction_map = {e["instance_id"]: e for e in extractions}

    if dry_run or max_concurrent <= 1:
        crafted = []
        iterator = tqdm(instances, desc="Crafting instances") if show_progress else instances
        for inst in iterator:
            extraction = extraction_map.get(inst.instance_id, {})
            result = craft_instance(
                inst, extraction, instances, worker_client,
                target_tokens=target_tokens,
                adversarial_count=adversarial_count,
                contradiction_count=contradiction_count,
                dry_run=dry_run,
                red_herring_count=red_herring_count,
                reveal_mode=reveal_mode,
                expand_paragraphs=expand_paragraphs,
                expansion_target_tokens=expansion_target_tokens,
                overlap_count=overlap_count,
                conflicting_sources_count=conflicting_sources_count,
                eviction_mode=eviction_mode,
                eviction_window_size=eviction_window_size,
                allow_agent_notes=allow_agent_notes,
                agent_notes_budget=agent_notes_budget,
                min_memory_facts=min_memory_facts,
            )
            if result:
                crafted.append(result)
                if output_path:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result.to_jsonl_dict()) + "\n")
    else:
        crafted = _craft_all_async(
            instances, extraction_map, worker_client,
            target_tokens, adversarial_count, contradiction_count,
            output_path, max_concurrent,
            red_herring_count, reveal_mode,
            expand_paragraphs, expansion_target_tokens,
            overlap_count, conflicting_sources_count,
            eviction_mode, eviction_window_size,
            allow_agent_notes, agent_notes_budget,
            min_memory_facts,
        )

    print(f"  Crafted {len(crafted)}/{len(instances)} instances")
    return crafted


def _craft_all_async(instances, extraction_map, worker_client, target_tokens,
                     adversarial_count, contradiction_count, output_path, max_concurrent,
                     red_herring_count=0, reveal_mode="immediate",
                     expand_paragraphs=False, expansion_target_tokens=600,
                     overlap_count=0, conflicting_sources_count=0,
                     eviction_mode="none", eviction_window_size=1,
                     allow_agent_notes=True, agent_notes_budget=500,
                     min_memory_facts=1):
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc="Crafting instances (async)")

        async def _one(inst):
            async with sem:
                extraction = extraction_map.get(inst.instance_id, {})
                result = await craft_instance_async(
                    inst, extraction, instances, worker_client,
                    target_tokens=target_tokens,
                    adversarial_count=adversarial_count,
                    contradiction_count=contradiction_count,
                    red_herring_count=red_herring_count,
                    reveal_mode=reveal_mode,
                    expand_paragraphs=expand_paragraphs,
                    expansion_target_tokens=expansion_target_tokens,
                    overlap_count=overlap_count,
                    conflicting_sources_count=conflicting_sources_count,
                    eviction_mode=eviction_mode,
                    eviction_window_size=eviction_window_size,
                    allow_agent_notes=allow_agent_notes,
                    agent_notes_budget=agent_notes_budget,
                    min_memory_facts=min_memory_facts,
                )
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()
        crafted = [r for r in results if r is not None]
        if output_path:
            with open(output_path, "a") as f:
                for r in crafted:
                    f.write(json.dumps(r.to_jsonl_dict()) + "\n")
        return crafted

    return asyncio.run(_run())
