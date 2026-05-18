"""QA Verification stage: validate solvability, distractor quality, and question quality.

Runs three ablation checks per QA pair:
  Check A (solvability):          question + linked_facts → answer → score HIGH
  Check B (distractor confusion): question + distractors only → answer → score LOW
  Check C (question leakage):     question alone → answer → score LOW

A QA pair is "valid" if all three checks pass.

Usage:
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa verify-qa \
        --input results/coding_qa_v3/coding_qa_1000_with_repo.jsonl \
        --output results/coding_qa_v3/coding_qa_1000_verified.jsonl \
        --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --num-workers 8
"""

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ...types.schemas import (
    CodingQAInstance,
    DistractorFact,
    GroundingFact,
    QAPair,
    QAVerificationResult,
    InstanceVerificationResult,
    VERIFY_QA_ANSWER_SCHEMA,
)
from ...llm.client import LLMClient
from ...llm.prompts import (
    VERIFY_POSITIVE_PROMPT,
    VERIFY_DISTRACTOR_PROMPT,
    VERIFY_LEAKAGE_PROMPT,
)
from .score import score_fact_recall_keyword, score_fact_recall_llm, score_answer_similarity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESCAPE_POSITIVE = "INSUFFICIENT_FACTS"
ESCAPE_DISTRACTOR = "INSUFFICIENT_FACTS"
ESCAPE_LEAKAGE = "NEED_PROJECT_CONTEXT"


# ---------------------------------------------------------------------------
# QA Deduplication
# ---------------------------------------------------------------------------

def _question_word_set(question: str) -> set:
    """Extract meaningful words from a question for overlap comparison."""
    return {
        w.strip(".,!?\"'()[]{}:;").lower()
        for w in question.split()
        if len(w.strip(".,!?\"'()[]{}:;")) > 3
    }


def dedup_qa_pairs(
    qa_pairs: List[QAPair],
    qa_results: Optional[List[QAVerificationResult]] = None,
    overlap_threshold: float = 0.7,
    max_per_instance: int = 5,
) -> List[QAPair]:
    """Remove near-duplicate QA pairs and cap per instance.

    Dedup strategy:
      1. Score each QA pair by verification quality (positive_score if available)
      2. Greedily keep best-scoring pairs that don't overlap with already-kept ones
      3. Cap at max_per_instance

    Two questions overlap if >overlap_threshold of the shorter question's
    words appear in the longer one.

    Args:
        qa_pairs: Valid QA pairs to dedup.
        qa_results: Optional verification results (for quality-based ordering).
        overlap_threshold: Word overlap fraction to consider duplicate (default 0.7).
        max_per_instance: Max QA pairs to keep per instance (default 5).

    Returns:
        Deduplicated, capped list of QA pairs.
    """
    if not qa_pairs:
        return []

    # Build score map for ordering: prefer higher positive_score
    score_map: Dict[str, float] = {}
    if qa_results:
        for vr in qa_results:
            score_map[vr.qa_id] = vr.positive_score

    # Sort: requires_memory first, then by positive_score descending
    scored = sorted(
        qa_pairs,
        key=lambda qp: (qp.requires_memory, score_map.get(qp.qa_id, 0.0)),
        reverse=True,
    )

    kept: List[QAPair] = []
    kept_words: List[set] = []

    for qp in scored:
        if len(kept) >= max_per_instance:
            break

        words = _question_word_set(qp.question)
        if not words:
            continue

        # Check overlap with all kept questions
        is_dup = False
        for kw in kept_words:
            if not kw:
                continue
            overlap = len(words & kw) / min(len(words), len(kw))
            if overlap > overlap_threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(qp)
            kept_words.append(words)

    return kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_facts_for_verification(facts: List[GroundingFact]) -> str:
    """Format facts with IDs and metadata for the positive-check prompt."""
    parts = []
    for f in facts:
        parts.append(f"[{f.id}] (category: {f.category}, relevance: {f.relevance})\n  {f.content}")
    return "\n\n".join(parts)


def format_distractors_as_facts(distractors: List[DistractorFact]) -> str:
    """Format distractors identically to how real facts are formatted.

    Uses D-prefixed IDs and plausible categories so the prompt structure
    matches Check A exactly — we're testing the facts, not the framing.
    """
    parts = []
    for d in distractors:
        cat = "bug_description" if d.source in ("adversarial", "adversarial_qa") else "debug_finding"
        parts.append(f"[{d.id}] (category: {cat}, relevance: helpful)\n  {d.content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Check A: Positive (solvability)
# ---------------------------------------------------------------------------

def verify_qa_pair_positive(
    qp: QAPair,
    fact_map: Dict[str, GroundingFact],
    client: LLMClient,
    use_llm_scoring: bool = True,
) -> Tuple[float, str, List[str]]:
    """Check A: Can LLM answer from facts alone?

    Returns (score, answer, facts_used).
    """
    linked_facts = [fact_map[fid] for fid in qp.linked_fact_ids if fid in fact_map]
    if not linked_facts:
        return 0.0, "NO_LINKED_FACTS", []

    facts_text = format_facts_for_verification(linked_facts)
    prompt = VERIFY_POSITIVE_PROMPT.format(
        facts_text=facts_text,
        question=qp.question,
    )

    try:
        result = client.get_json_completion(
            prompt=prompt,
            response_format=VERIFY_QA_ANSWER_SCHEMA,
            temperature=0.0,
        )
    except Exception as e:
        return 0.0, f"LLM_ERROR: {e}", []

    answer = result.get("answer", "")
    facts_used = result.get("facts_used", [])

    if ESCAPE_POSITIVE in answer:
        return 0.0, answer, facts_used

    if use_llm_scoring:
        fact_scores = score_fact_recall_llm(answer, linked_facts, qp.question, client)
    else:
        fact_scores = score_fact_recall_keyword(answer, linked_facts)

    score = sum(1 for fs in fact_scores if fs.recalled) / max(1, len(fact_scores))
    return score, answer, facts_used


# ---------------------------------------------------------------------------
# Check B: Distractor confusion
# ---------------------------------------------------------------------------

def verify_qa_pair_distractor(
    qp: QAPair,
    distractors: List[DistractorFact],
    client: LLMClient,
    use_llm_scoring: bool = True,
) -> Tuple[float, str]:
    """Check B: Do distractors accidentally provide the correct answer?

    Returns (score, answer). Low score = PASS (distractors didn't help).
    Scores against the gold_answer (concise, specific) rather than broad
    fact content to avoid false positives from shared vocabulary.
    """
    if not distractors:
        return 0.0, "NO_DISTRACTORS"

    distractor_text = format_distractors_as_facts(distractors)
    prompt = VERIFY_DISTRACTOR_PROMPT.format(
        distractor_facts_text=distractor_text,
        question=qp.question,
    )

    try:
        result = client.get_json_completion(
            prompt=prompt,
            response_format=VERIFY_QA_ANSWER_SCHEMA,
            temperature=0.0,
        )
    except Exception as e:
        return 0.0, f"LLM_ERROR: {e}"

    answer = result.get("answer", "")

    if ESCAPE_DISTRACTOR in answer:
        return 0.0, answer

    # Score against gold_answer — more precise than broad fact content
    score = score_answer_similarity(answer, qp.gold_answer)
    return score, answer


# ---------------------------------------------------------------------------
# Check C: Question leakage
# ---------------------------------------------------------------------------

def verify_qa_pair_leakage(
    qp: QAPair,
    client: LLMClient,
    use_llm_scoring: bool = True,
) -> Tuple[float, str]:
    """Check C: Does the question alone leak the answer?

    Returns (score, answer). Low score = PASS (question doesn't self-answer).
    Scores against gold_answer for consistency with Check B.
    """
    prompt = VERIFY_LEAKAGE_PROMPT.format(question=qp.question)

    try:
        result = client.get_json_completion(
            prompt=prompt,
            response_format=VERIFY_QA_ANSWER_SCHEMA,
            temperature=0.0,
        )
    except Exception as e:
        return 0.0, f"LLM_ERROR: {e}"

    answer = result.get("answer", "")

    if ESCAPE_LEAKAGE in answer:
        return 0.0, answer

    score = score_answer_similarity(answer, qp.gold_answer)
    return score, answer


# ---------------------------------------------------------------------------
# Per-QA-pair verification
# ---------------------------------------------------------------------------

def verify_qa_pair(
    qp: QAPair,
    instance: CodingQAInstance,
    fact_map: Dict[str, GroundingFact],
    client: LLMClient,
    positive_threshold: float = 0.4,
    distractor_threshold: float = 0.5,
    leakage_threshold: float = 0.3,
    use_llm_scoring: bool = True,
) -> QAVerificationResult:
    """Run all three checks and return verdict."""

    # Check A: solvability
    pos_score, pos_answer, facts_used = verify_qa_pair_positive(
        qp, fact_map, client, use_llm_scoring=use_llm_scoring,
    )
    pos_passed = pos_score >= positive_threshold

    # Check B: distractor confusion
    dist_score, dist_answer = verify_qa_pair_distractor(
        qp, instance.distractors, client,
        use_llm_scoring=use_llm_scoring,
    )
    dist_passed = dist_score < distractor_threshold

    # Check C: question leakage
    leak_score, leak_answer = verify_qa_pair_leakage(
        qp, client, use_llm_scoring=use_llm_scoring,
    )
    leak_passed = leak_score < leakage_threshold

    # Verdict
    if not pos_passed:
        verdict = "broken"
    elif not dist_passed and not leak_passed:
        verdict = "confusable_and_leaky"
    elif not dist_passed:
        verdict = "confusable"
    elif not leak_passed:
        verdict = "leaky"
    else:
        verdict = "valid"

    return QAVerificationResult(
        qa_id=qp.qa_id,
        question=qp.question,
        requires_memory=qp.requires_memory,
        positive_score=pos_score,
        positive_answer=pos_answer,
        positive_facts_used=facts_used,
        positive_passed=pos_passed,
        distractor_score=dist_score,
        distractor_answer=dist_answer,
        distractor_passed=dist_passed,
        leakage_score=leak_score,
        leakage_answer=leak_answer,
        leakage_passed=leak_passed,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Per-instance verification
# ---------------------------------------------------------------------------

def verify_instance(
    instance: CodingQAInstance,
    client: LLMClient,
    positive_threshold: float = 0.4,
    distractor_threshold: float = 0.5,
    leakage_threshold: float = 0.3,
    use_llm_scoring: bool = True,
    max_qa_per_instance: int = 5,
    dedup_overlap: float = 0.7,
) -> Tuple[Optional[CodingQAInstance], InstanceVerificationResult]:
    """Verify all QA pairs in an instance, dedup, and cap.

    Returns:
        (filtered_instance_or_None, verification_metadata)
    """
    fact_map = {f.id: f for f in instance.grounding_facts}

    qa_results = []
    for qp in instance.qa_pairs:
        vr = verify_qa_pair(
            qp, instance, fact_map, client,
            positive_threshold=positive_threshold,
            distractor_threshold=distractor_threshold,
            leakage_threshold=leakage_threshold,
            use_llm_scoring=use_llm_scoring,
        )
        qa_results.append(vr)

    # Count verdicts (before dedup — reflects verification quality)
    counts = {
        "valid": 0, "broken": 0, "confusable": 0,
        "leaky": 0, "confusable_and_leaky": 0,
    }
    for vr in qa_results:
        counts[vr.verdict] += 1

    # Filter: keep valid QA pairs
    valid_ids = {vr.qa_id for vr in qa_results if vr.verdict == "valid"}
    valid_pairs = [qp for qp in instance.qa_pairs if qp.qa_id in valid_ids]
    valid_results = [vr for vr in qa_results if vr.verdict == "valid"]

    # Dedup + cap: remove near-duplicate questions, keep best ones
    deduped_pairs = dedup_qa_pairs(
        valid_pairs,
        qa_results=valid_results,
        overlap_threshold=dedup_overlap,
        max_per_instance=max_qa_per_instance,
    )
    deduped_ids = {qp.qa_id for qp in deduped_pairs}
    num_removed_by_dedup = len(valid_pairs) - len(deduped_pairs)

    has_memory_required = any(
        qp.requires_memory for qp in deduped_pairs
    )

    # Instance gate: >= 2 deduped QA pairs AND >= 1 with requires_memory
    kept = len(deduped_pairs) >= 2 and has_memory_required

    meta = InstanceVerificationResult(
        instance_id=instance.instance_id,
        image_name=instance.image_name or "",
        num_qa_total=len(instance.qa_pairs),
        num_valid=counts["valid"],
        num_broken=counts["broken"],
        num_confusable=counts["confusable"],
        num_leaky=counts["leaky"],
        num_confusable_and_leaky=counts["confusable_and_leaky"],
        num_after_dedup=len(deduped_pairs),
        num_removed_by_dedup=num_removed_by_dedup,
        qa_results=qa_results,
        kept=kept,
        kept_qa_ids=list(deduped_ids),
    )

    if not kept:
        return None, meta

    # Build filtered instance with only deduped QA pairs
    filtered = instance.model_copy(update={"qa_pairs": deduped_pairs})
    return filtered, meta


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def _verify_one(args: Tuple) -> Tuple[Optional[dict], dict]:
    """Worker function for parallel verification."""
    (instance_dict, model, pos_thresh, dist_thresh, leak_thresh, use_llm,
     max_qa, dedup_overlap) = args
    instance = CodingQAInstance(**instance_dict)
    client = LLMClient(model=model)

    filtered, meta = verify_instance(
        instance, client,
        positive_threshold=pos_thresh,
        distractor_threshold=dist_thresh,
        leakage_threshold=leak_thresh,
        use_llm_scoring=use_llm,
        max_qa_per_instance=max_qa,
        dedup_overlap=dedup_overlap,
    )

    filtered_dict = filtered.to_jsonl_dict() if filtered else None
    return filtered_dict, meta.to_jsonl_dict()


def verify_all(
    input_path: str,
    output_path: str,
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    positive_threshold: float = 0.4,
    distractor_threshold: float = 0.5,
    leakage_threshold: float = 0.3,
    num_workers: int = 8,
    limit: Optional[int] = None,
    metadata_path: Optional[str] = None,
    use_llm_scoring: bool = True,
    show_progress: bool = True,
    max_qa_per_instance: int = 5,
    dedup_overlap: float = 0.7,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry: load, verify in parallel, write filtered + metadata."""

    # Load instances
    instances = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
            if limit and len(instances) >= limit:
                break

    print(f"Loaded {len(instances)} instances for verification")
    total_qa = sum(len(d.get("qa_pairs", [])) for d in instances)
    print(f"Total QA pairs: {total_qa}")

    # Checkpoint: resolve path, load done IDs
    if checkpoint_path is None:
        checkpoint_path = str(Path(output_path).with_suffix("")) + ".checkpoint.jsonl"
    elif checkpoint_path == "":
        checkpoint_path = None  # explicitly disabled

    done_ids: dict[str, dict] = {}  # instance_id -> checkpoint entry
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                done_ids[entry["instance_id"]] = entry
        print(f"Checkpoint: {len(done_ids)} already verified, "
              f"{len(instances) - len(done_ids)} remaining")

    # Filter to unfinished instances (preserve original indices for ordering)
    indexed_instances = [
        (i, inst) for i, inst in enumerate(instances)
        if inst.get("instance_id", "") not in done_ids
    ]

    # Build worker args (only for unfinished)
    worker_args = [
        (inst, model, positive_threshold, distractor_threshold, leakage_threshold,
         use_llm_scoring, max_qa_per_instance, dedup_overlap)
        for _, inst in indexed_instances
    ]

    # Checkpoint writer (thread-safe)
    ckpt_lock = threading.Lock()
    ckpt_file = None
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        ckpt_file = open(checkpoint_path, "a")

    def _save_checkpoint(inst_dict, filtered_dict, meta_dict):
        if ckpt_file is None:
            return
        entry = {
            "instance_id": inst_dict.get("instance_id", ""),
            "filtered": filtered_dict,
            "meta": meta_dict,
        }
        line = json.dumps(entry) + "\n"
        with ckpt_lock:
            ckpt_file.write(line)
            ckpt_file.flush()

    # Run verification
    new_results: dict[int, tuple] = {}  # original index -> (filtered, meta)
    if show_progress:
        from tqdm import tqdm
        if num_workers <= 1:
            for (orig_idx, inst), args in zip(indexed_instances, worker_args):
                result = _verify_one(args)
                new_results[orig_idx] = result
                _save_checkpoint(inst, result[0], result[1])
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            futures = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for (orig_idx, inst), args in zip(indexed_instances, worker_args):
                    fut = executor.submit(_verify_one, args)
                    futures[fut] = (orig_idx, inst)
                pbar = tqdm(total=len(worker_args), desc="Verifying")
                for future in as_completed(futures):
                    orig_idx, inst = futures[future]
                    result = future.result()
                    new_results[orig_idx] = result
                    _save_checkpoint(inst, result[0], result[1])
                    pbar.update(1)
                pbar.close()
    else:
        for (orig_idx, inst), args in zip(indexed_instances, worker_args):
            result = _verify_one(args)
            new_results[orig_idx] = result
            _save_checkpoint(inst, result[0], result[1])

    if ckpt_file is not None:
        ckpt_file.close()

    # Merge checkpoint + new results in original input order
    kept_instances = []
    all_meta = []
    for i, inst in enumerate(instances):
        iid = inst.get("instance_id", "")
        if i in new_results:
            filtered_dict, meta_dict = new_results[i]
        elif iid in done_ids:
            entry = done_ids[iid]
            filtered_dict = entry.get("filtered")
            meta_dict = entry.get("meta", {})
        else:
            continue
        all_meta.append(meta_dict)
        if filtered_dict is not None:
            kept_instances.append(filtered_dict)

    # Write filtered output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for inst in kept_instances:
            f.write(json.dumps(inst) + "\n")

    # Write metadata
    if metadata_path is None:
        metadata_path = str(out).replace(".jsonl", "_verification_meta.jsonl")
    meta_out = Path(metadata_path)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")

    # Summary stats
    n_total = len(instances)
    n_kept = len(kept_instances)
    qa_valid = sum(m["num_valid"] for m in all_meta)
    qa_broken = sum(m["num_broken"] for m in all_meta)
    qa_confusable = sum(m["num_confusable"] for m in all_meta)
    qa_leaky = sum(m["num_leaky"] for m in all_meta)
    qa_confusable_leaky = sum(m["num_confusable_and_leaky"] for m in all_meta)
    qa_after_dedup = sum(m.get("num_after_dedup", 0) for m in all_meta)
    qa_removed_dedup = sum(m.get("num_removed_by_dedup", 0) for m in all_meta)

    summary = {
        "total_instances": n_total,
        "kept_instances": n_kept,
        "dropped_instances": n_total - n_kept,
        "total_qa_pairs": total_qa,
        "valid_qa": qa_valid,
        "broken_qa": qa_broken,
        "confusable_qa": qa_confusable,
        "leaky_qa": qa_leaky,
        "confusable_and_leaky_qa": qa_confusable_leaky,
        "qa_after_dedup": qa_after_dedup,
        "qa_removed_by_dedup": qa_removed_dedup,
        "positive_threshold": positive_threshold,
        "distractor_threshold": distractor_threshold,
        "leakage_threshold": leakage_threshold,
        "max_qa_per_instance": max_qa_per_instance,
        "dedup_overlap": dedup_overlap,
    }

    print(f"\n{'='*50}")
    print("QA-Level Results")
    print(f"{'='*50}")
    print(f"  valid:               {qa_valid:5d} ({100*qa_valid/max(1,total_qa):.1f}%)")
    print(f"  broken:              {qa_broken:5d} ({100*qa_broken/max(1,total_qa):.1f}%)")
    print(f"  confusable:          {qa_confusable:5d} ({100*qa_confusable/max(1,total_qa):.1f}%)")
    print(f"  leaky:               {qa_leaky:5d} ({100*qa_leaky/max(1,total_qa):.1f}%)")
    print(f"  confusable_and_leaky:{qa_confusable_leaky:5d} ({100*qa_confusable_leaky/max(1,total_qa):.1f}%)")
    print()
    print(f"Dedup Results")
    print(f"{'='*50}")
    print(f"  after dedup+cap:     {qa_after_dedup:5d} (from {qa_valid} valid)")
    print(f"  removed by dedup:    {qa_removed_dedup:5d}")
    print(f"  max per instance:    {max_qa_per_instance}")
    print(f"  overlap threshold:   {dedup_overlap}")
    print()
    print(f"Instance-Level Results")
    print(f"{'='*50}")
    print(f"  kept:    {n_kept}/{n_total} ({100*n_kept/max(1,n_total):.1f}%)")
    print(f"  dropped: {n_total - n_kept}/{n_total} ({100*(n_total-n_kept)/max(1,n_total):.1f}%)")
    print(f"\nOutput: {output_path}")
    print(f"Metadata: {metadata_path}")

    return summary
