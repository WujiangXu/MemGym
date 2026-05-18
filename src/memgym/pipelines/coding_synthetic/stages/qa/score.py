"""Per-fact scoring engine for Coding QA evaluation.

Two scoring modes:
- Keyword-based (fast, free): fuzzy term matching, ≥40% overlap = recalled
- LLM-based (accurate, costs money): LLM judge per fact

Also scores noise rejection (distractor detection) and source attribution.
"""

from typing import Dict, List, Optional

from ...types.schemas import (
    GroundingFact,
    DistractorFact,
    FactScore,
    QAEvalResult,
    CodingQAInstance,
    FACT_JUDGE_QA_SCHEMA,
    NOISE_JUDGE_QA_SCHEMA,
)
from ...llm.client import LLMClient
from ...llm.prompts import FACT_JUDGE_PROMPT, NOISE_JUDGE_PROMPT


# ---------------------------------------------------------------------------
# Keyword-based scoring (fast, no LLM)
# ---------------------------------------------------------------------------

def score_fact_recall_keyword(
    answer: str,
    facts: List[GroundingFact],
    threshold: float = 0.4,
) -> List[FactScore]:
    """Fast keyword-based fact recall.

    For each fact, extract key terms (>4 chars), check what fraction
    appear in the answer. If ≥threshold, mark as recalled.
    """
    answer_lower = answer.lower()
    results = []

    for fact in facts:
        content_lower = fact.content.lower()
        key_terms = [
            w.strip(".,!?\"'()[]{}:;")
            for w in content_lower.split()
            if len(w.strip(".,!?\"'()[]{}:;")) > 4
        ]
        if not key_terms:
            key_terms = content_lower.split()

        if not key_terms:
            results.append(FactScore(fact_id=fact.id, recalled=False, confidence=0.0))
            continue

        matched = sum(1 for term in key_terms if term in answer_lower)
        ratio = matched / len(key_terms)
        results.append(FactScore(
            fact_id=fact.id,
            recalled=ratio >= threshold,
            confidence=ratio,
        ))

    return results


# ---------------------------------------------------------------------------
# Gold-answer similarity scoring (for distractor/leakage checks)
# ---------------------------------------------------------------------------

# Common programming terms that shouldn't count as signal
_STOPWORDS = frozenset({
    "function", "method", "class", "return", "value", "default",
    "should", "would", "called", "instead", "because", "which",
    "about", "given", "expected", "behavior", "result", "parameter",
    "object", "instance", "error", "raises", "returns", "raises",
    "implementation", "abstract", "concrete", "based", "using",
    "check", "verify", "validate", "where", "there", "their",
    "these", "those", "other", "after", "before", "first",
    "second", "however", "therefore", "specific", "actual",
})


def score_answer_similarity(
    candidate: str,
    gold_answer: str,
    threshold: float = 0.4,
) -> float:
    """Score how similar a candidate answer is to the gold answer.

    Extracts key terms (>4 chars, not stopwords) from gold_answer,
    checks what fraction appear in the candidate. Returns the ratio.

    Used for distractor/leakage checks where we want to compare
    against the specific gold answer rather than broad fact content.
    """
    gold_lower = gold_answer.lower()
    candidate_lower = candidate.lower()

    key_terms = [
        w.strip(".,!?\"'()[]{}:;")
        for w in gold_lower.split()
        if len(w.strip(".,!?\"'()[]{}:;")) > 4
    ]
    # Filter out common programming stopwords
    key_terms = [t for t in key_terms if t not in _STOPWORDS]

    if not key_terms:
        return 0.0

    matched = sum(1 for term in key_terms if term in candidate_lower)
    return matched / len(key_terms)


# ---------------------------------------------------------------------------
# LLM-based scoring (accurate)
# ---------------------------------------------------------------------------

def score_fact_recall_llm(
    answer: str,
    facts: List[GroundingFact],
    question: str,
    client: LLMClient,
) -> List[FactScore]:
    """LLM-based fact recall — judge per fact.

    For each fact, asks LLM: "Does this answer contain this fact's info?"
    """
    results = []

    for fact in facts:
        prompt = FACT_JUDGE_PROMPT.format(
            fact_content=fact.content,
            answer=answer[:3000],
            question=question,
        )
        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=FACT_JUDGE_QA_SCHEMA,
                temperature=0.0,
            )
            results.append(FactScore(
                fact_id=fact.id,
                recalled=result.get("contains_fact", False),
                confidence=result.get("confidence", 0.0),
            ))
        except Exception:
            results.append(FactScore(
                fact_id=fact.id, recalled=False, confidence=0.0,
            ))

    return results


# ---------------------------------------------------------------------------
# Noise rejection scoring
# ---------------------------------------------------------------------------

def score_noise_rejection(
    answer: str,
    distractors: List[DistractorFact],
    question: str,
    client: LLMClient,
) -> float:
    """Check if agent was misled by distractors.

    Returns 0.0 (all misled) to 1.0 (none misled).
    """
    if not distractors:
        return 1.0

    misled_count = 0
    for d in distractors:
        prompt = NOISE_JUDGE_PROMPT.format(
            distractor_content=d.content[:1000],
            answer=answer[:3000],
            question=question,
        )
        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=NOISE_JUDGE_QA_SCHEMA,
                temperature=0.0,
            )
            if result.get("misled", False):
                misled_count += 1
        except Exception:
            pass

    return 1.0 - (misled_count / len(distractors))


def score_noise_rejection_keyword(
    answer: str,
    distractors: List[DistractorFact],
    grounding_facts: Optional[List[GroundingFact]] = None,
    threshold: float = 0.3,
) -> float:
    """Fast keyword-based noise rejection check.

    Only counts distractor-UNIQUE terms — terms that appear in a distractor
    but NOT in any grounding fact. This avoids false "misled" signals from
    shared vocabulary (class names, module names, etc.).
    """
    if not distractors:
        return 1.0

    # Build set of terms from grounding facts to exclude
    fact_terms: set = set()
    if grounding_facts:
        for f in grounding_facts:
            for w in f.content.lower().split():
                t = w.strip(".,!?\"'()[]{}:;")
                if len(t) > 4:
                    fact_terms.add(t)

    answer_lower = answer.lower()
    misled_count = 0

    for d in distractors:
        content_lower = d.content.lower()
        key_terms = [
            w.strip(".,!?\"'()[]{}:;")
            for w in content_lower.split()
            if len(w.strip(".,!?\"'()[]{}:;")) > 4
        ]
        # Only keep distractor-unique terms (not in grounding facts)
        unique_terms = [t for t in key_terms if t not in fact_terms]
        if not unique_terms:
            continue
        matched = sum(1 for term in unique_terms if term in answer_lower)
        if matched / len(unique_terms) >= threshold:
            misled_count += 1

    return 1.0 - (misled_count / len(distractors))


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

def aggregate_scores(
    all_fact_scores: List[FactScore],
    noise_score: float,
    qa_scores: List[Dict],
    instance: CodingQAInstance,
) -> QAEvalResult:
    """Combine per-fact, noise, and QA scores into QAEvalResult."""
    facts = instance.grounding_facts
    fact_map = {f.id: f for f in facts}

    # Overall fact recall
    recalled = sum(1 for fs in all_fact_scores if fs.recalled)
    total = len(all_fact_scores)
    fact_recall = recalled / total if total else 0.0

    # By relevance
    critical = [fs for fs in all_fact_scores if fact_map.get(fs.fact_id, None)
                and fact_map[fs.fact_id].relevance == "critical"]
    critical_recall = (
        sum(1 for fs in critical if fs.recalled) / len(critical)
        if critical else 0.0
    )

    # By discoverable
    external = [fs for fs in all_fact_scores if fact_map.get(fs.fact_id, None)
                and not fact_map[fs.fact_id].discoverable]
    discoverable = [fs for fs in all_fact_scores if fact_map.get(fs.fact_id, None)
                    and fact_map[fs.fact_id].discoverable]
    external_recall = (
        sum(1 for fs in external if fs.recalled) / len(external)
        if external else 0.0
    )
    discoverable_recall = (
        sum(1 for fs in discoverable if fs.recalled) / len(discoverable)
        if discoverable else 0.0
    )

    # QA accuracy
    correct_qs = sum(1 for qs in qa_scores if qs.get("score", 0) >= 0.5)
    qa_accuracy = correct_qs / len(qa_scores) if qa_scores else 0.0

    # Category breakdown
    category_breakdown: Dict[str, float] = {}
    categories = set(f.category for f in facts)
    for cat in categories:
        cat_scores = [
            fs for fs in all_fact_scores
            if fact_map.get(fs.fact_id, None) and fact_map[fs.fact_id].category == cat
        ]
        if cat_scores:
            category_breakdown[cat] = (
                sum(1 for fs in cat_scores if fs.recalled) / len(cat_scores)
            )

    return QAEvalResult(
        instance_id=instance.instance_id,
        strategy="",  # Filled by caller
        qa_scores=qa_scores,
        fact_scores=all_fact_scores,
        fact_recall=fact_recall,
        critical_fact_recall=critical_recall,
        external_fact_recall=external_recall,
        discoverable_fact_recall=discoverable_recall,
        noise_rejection=noise_score,
        qa_accuracy=qa_accuracy,
        category_breakdown=category_breakdown,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results: List[QAEvalResult]) -> str:
    """Format evaluation results as a human-readable report."""
    if not results:
        return "No results to report."

    n = len(results)
    lines = [
        f"Coding QA Evaluation Report ({n} instances)",
        "=" * 60,
        "",
        f"{'Metric':<35} {'Avg':>8} {'Min':>8} {'Max':>8}",
        "-" * 60,
    ]

    metrics = [
        ("Fact Recall (overall)", [r.fact_recall for r in results]),
        ("Critical Fact Recall", [r.critical_fact_recall for r in results]),
        ("External Fact Recall", [r.external_fact_recall for r in results]),
        ("Discoverable Fact Recall", [r.discoverable_fact_recall for r in results]),
        ("Noise Rejection", [r.noise_rejection for r in results]),
        ("QA Accuracy", [r.qa_accuracy for r in results]),
    ]

    for name, vals in metrics:
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"{name:<35} {avg:>7.1%} {min(vals):>7.1%} {max(vals):>7.1%}")

    # Category breakdown (averaged across instances)
    all_cats: Dict[str, List[float]] = {}
    for r in results:
        for cat, score in r.category_breakdown.items():
            all_cats.setdefault(cat, []).append(score)

    if all_cats:
        lines.append("")
        lines.append("Category Breakdown (avg recall)")
        lines.append("-" * 40)
        for cat in sorted(all_cats.keys()):
            avg = sum(all_cats[cat]) / len(all_cats[cat])
            lines.append(f"  {cat:<30} {avg:>7.1%}")

    return "\n".join(lines)
