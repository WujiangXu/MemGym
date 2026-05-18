"""Stage 4: Verify -- 3-condition ablation testing.

Condition A: question + ALL documents (perfect memory) -> should be correct
Condition B: question + ONLY non-supporting docs from last turn (no memory) -> should be wrong
Condition C: question + ALL docs MINUS gold supporting -> should be wrong

Accept if A >> B and A >> C.

Includes:
- Haiku pre-check: skip expensive Sonnet verification if Haiku can't answer with all docs
- Retry loop: on failure, simplify distractors (too hard) or reshuffle turns (too easy)
"""

import asyncio
import copy
import json
import random
from typing import List, Dict, Optional

from tqdm import tqdm

from ..types.schemas import MemGymIRInstance
from ..llm.client import LLMClient
from ..llm.prompts import (
    ABLATION_QA_PROMPT,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)


def _format_documents(docs: List[Dict]) -> str:
    """Format a list of documents for the LLM prompt."""
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"Document {i+1}: {doc.get('title', 'Untitled')}")
        parts.append(doc.get("text", ""))
        parts.append("")
    return "\n".join(parts)


def _ask_and_judge(
    client: LLMClient,
    question: str,
    gold_answer: str,
    documents_str: str,
) -> float:
    """Ask the question with given documents and judge the answer."""
    prompt = ABLATION_QA_PROMPT.format(
        documents=documents_str,
        question=question,
    )
    response = client.get_completion(prompt=prompt, temperature=0.0)

    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    if not predicted:
        return 0.0

    judge_prompt = ANSWER_JUDGE_PROMPT.format(
        gold_answer=gold_answer,
        predicted_answer=predicted,
        question=question,
    )
    result = client.get_json_completion(
        prompt=judge_prompt,
        response_format=ANSWER_JUDGE_SCHEMA,
        temperature=0.0,
    )
    return result.get("score", 0.0)


def _get_all_docs(instance: MemGymIRInstance) -> List[Dict]:
    """Get all documents from all turns."""
    all_docs = []
    for turn in instance.turns:
        all_docs.extend(turn.documents)
    return all_docs


def pre_verify(
    instance: MemGymIRInstance,
    worker_client: LLMClient,
    threshold: float = 0.3,
) -> bool:
    """Cheap pre-check with worker (Haiku): can the model answer with supporting docs?

    Uses ONLY gold supporting paragraphs (not distractors) to test whether
    the question is answerable given perfect memory. If Haiku can't answer
    with just the supporting docs, the question itself is flawed.
    """
    supporting_docs = instance.gold_supporting_paragraphs
    if not supporting_docs:
        return False
    docs_str = _format_documents(supporting_docs)
    score = _ask_and_judge(worker_client, instance.question, instance.answer, docs_str)
    return score >= threshold


def _ask_and_judge_delayed(
    client: LLMClient,
    question: str,
    gold_answer: str,
    turns_docs: List[List[Dict]],
) -> float:
    """Ask the question in delayed-reveal mode: documents presented sequentially,
    then the question is asked. Uses multi-turn conversation so the model must
    retain information from earlier turns.
    """
    messages = []
    for i, docs in enumerate(turns_docs):
        docs_str = _format_documents(docs)
        messages.append({
            "role": "user",
            "content": f"Here are search results from turn {i+1}. Read and remember the key information.\n\n{docs_str}",
        })
        messages.append({
            "role": "assistant",
            "content": f"I've read the documents from turn {i+1}. I'll remember the key information.",
        })

    # Final turn: ask the question
    messages.append({
        "role": "user",
        "content": (
            "Now answer the following question using ONLY the information from the documents "
            "you read in the previous turns. If you don't have enough information, say "
            "\"insufficient information\".\n\n"
            f"Question: {question}\n\n"
            "Provide your answer in JSON:\n"
            "{\"answer\": \"<your answer>\", \"confidence\": <0.0-1.0>}"
        ),
    })

    response = client.get_completion_messages(messages=messages, temperature=0.0)

    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    if not predicted:
        return 0.0

    judge_prompt = ANSWER_JUDGE_PROMPT.format(
        gold_answer=gold_answer,
        predicted_answer=predicted,
        question=question,
    )
    result = client.get_json_completion(
        prompt=judge_prompt,
        response_format=ANSWER_JUDGE_SCHEMA,
        temperature=0.0,
    )
    return result.get("score", 0.0)


def verify_instance(
    instance: MemGymIRInstance,
    client: LLMClient,
    min_a_score: float = 0.5,
    min_gap_ab: float = 0.30,
    max_c_score: float = 0.4,
    verify_delayed: bool = False,
    max_docs_a: int = 15,
) -> Dict:
    """Run 3-condition ablation on a single instance.

    If verify_delayed=True, also runs Condition D: delayed-reveal verification
    where documents are presented turn-by-turn before the question.

    Returns verification result dict.
    """
    # Condition A: Supporting docs + capped distractors (avoid overwhelming verifier)
    all_docs = _get_all_docs(instance)
    supporting = [d for d in all_docs if d.get("is_supporting", False)]
    non_supporting = [d for d in all_docs if not d.get("is_supporting", False)]
    random.shuffle(non_supporting)
    docs_a_list = supporting + non_supporting[:max_docs_a]
    random.shuffle(docs_a_list)
    docs_a = _format_documents(docs_a_list)
    score_a = _ask_and_judge(client, instance.question, instance.answer, docs_a)

    # Condition B: ONLY non-supporting docs from last turn (simulates no memory)
    # The agent sees the current turn's search results but has no memory of
    # supporting info from previous turns. Only distractors remain.
    if instance.turns:
        last_docs = [
            d for d in instance.turns[-1].documents
            if not d.get("is_supporting", False)
        ]
    else:
        last_docs = []
    docs_b = _format_documents(last_docs)
    score_b = _ask_and_judge(client, instance.question, instance.answer, docs_b)

    # Condition C: ALL docs MINUS gold supporting paragraphs
    gold_titles = {p["title"] for p in instance.gold_supporting_paragraphs}
    non_gold_docs = [
        d for d in all_docs if d.get("title", "") not in gold_titles
    ]
    docs_c = _format_documents(non_gold_docs)
    score_c = _ask_and_judge(client, instance.question, instance.answer, docs_c)

    gap_ab = score_a - score_b
    gap_ac = score_a - score_c

    accepted = (
        score_a >= min_a_score
        and (score_b < 0.3 or gap_ab >= min_gap_ab)
        and score_c < max_c_score
    )

    result = {
        "instance_id": instance.instance_id,
        "score_a": score_a,
        "score_b": score_b,
        "score_c": score_c,
        "gap_ab": gap_ab,
        "gap_ac": gap_ac,
        "accepted": accepted,
    }

    # Condition D: Delayed-reveal verification (optional)
    if verify_delayed and accepted:
        turns_docs = [turn.documents for turn in instance.turns]
        score_d = _ask_and_judge_delayed(
            client, instance.question, instance.answer, turns_docs,
        )
        result["score_d"] = score_d
        result["delayed_reveal_answerable"] = score_d >= min_a_score

    return result


def _simplify_for_retry(instance: MemGymIRInstance) -> MemGymIRInstance:
    """Create a simplified copy with fewer distractors (for too-hard cases)."""
    inst = copy.deepcopy(instance)
    # Remove half the distractors from each turn
    for turn in inst.turns:
        non_supporting = [d for d in turn.documents if not d.get("is_supporting", False)]
        supporting = [d for d in turn.documents if d.get("is_supporting", False)]
        # Keep only half the non-supporting docs
        keep_count = max(1, len(non_supporting) // 2)
        random.shuffle(non_supporting)
        turn.documents = supporting + non_supporting[:keep_count]
    return inst


def verify_with_retry(
    instance: MemGymIRInstance,
    client: LLMClient,
    max_retries: int = 1,
    min_a_score: float = 0.5,
    min_gap_ab: float = 0.30,
    max_c_score: float = 0.4,
    max_docs_a: int = 15,
) -> Dict:
    """Verify with retry: simplify distractors on too-hard failure."""
    result = verify_instance(
        instance, client,
        min_a_score=min_a_score,
        min_gap_ab=min_gap_ab,
        max_c_score=max_c_score,
        max_docs_a=max_docs_a,
    )
    if result["accepted"]:
        return result

    # Too hard (score_a < min_a_score): try with fewer distractors
    if result["score_a"] < min_a_score and max_retries > 0:
        simplified = _simplify_for_retry(instance)
        retry_result = verify_instance(
            simplified, client,
            min_a_score=min_a_score,
            min_gap_ab=min_gap_ab,
            max_c_score=max_c_score,
            max_docs_a=max_docs_a,
        )
        retry_result["retry_reason"] = "simplified_distractors"
        if retry_result["accepted"]:
            instance.turns = simplified.turns
            return retry_result

    return result


def verify_all(
    instances: List[MemGymIRInstance],
    client: LLMClient,
    worker_client: Optional[LLMClient] = None,
    min_a_score: float = 0.5,
    min_gap_ab: float = 0.30,
    max_c_score: float = 0.4,
    pre_verify_threshold: float = 0.3,
    show_progress: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 10,
    max_docs_a: int = 15,
) -> List[Dict]:
    """Verify all instances via ablation.

    Args:
        client: Verifier (Sonnet) for ablation tests.
        worker_client: Worker (Haiku) for cheap pre-check. If None, skip pre-check.
        min_a_score: Minimum score with all docs (Condition A).
        min_gap_ab: Minimum gap between A and B scores.
        max_c_score: Maximum score without gold docs (Condition C).
        pre_verify_threshold: Minimum score for Haiku pre-check.
        max_concurrent: Max parallel LLM calls.
    """
    if dry_run:
        print("  [dry-run] Skipping verification LLM calls")
        return [
            {
                "instance_id": inst.instance_id,
                "score_a": 1.0,
                "score_b": 0.0,
                "score_c": 0.0,
                "gap_ab": 1.0,
                "gap_ac": 1.0,
                "accepted": True,
            }
            for inst in instances
        ]

    if max_concurrent > 1:
        results = _verify_all_async(
            instances, client, worker_client,
            min_a_score, min_gap_ab, max_c_score, pre_verify_threshold,
            max_concurrent, max_docs_a,
        )
        # Attach verification to instances
        result_map = {r["instance_id"]: r for r in results}
        for inst in instances:
            if inst.instance_id in result_map:
                inst.verification = result_map[inst.instance_id]
    else:
        results = []
        skipped = 0
        iterator = tqdm(instances, desc="Verifying") if show_progress else instances

        for inst in iterator:
            if worker_client is not None:
                if not pre_verify(inst, worker_client, threshold=pre_verify_threshold):
                    results.append({
                        "instance_id": inst.instance_id,
                        "score_a": 0.0,
                        "score_b": 0.0,
                        "score_c": 0.0,
                        "gap_ab": 0.0,
                        "gap_ac": 0.0,
                        "accepted": False,
                        "skipped_reason": "pre_verify_failed",
                    })
                    skipped += 1
                    continue

            result = verify_with_retry(
                inst, client,
                min_a_score=min_a_score,
                min_gap_ab=min_gap_ab,
                max_c_score=max_c_score,
                max_docs_a=max_docs_a,
            )
            results.append(result)
            inst.verification = result

    accepted = sum(1 for r in results if r["accepted"])
    skipped = sum(1 for r in results if r.get("skipped_reason"))
    print(f"  Verification: {accepted}/{len(results)} accepted"
          f" ({skipped} skipped by pre-check)")

    return results


# -------------------------------------------------------------------------
# Async API
# -------------------------------------------------------------------------


async def _ask_and_judge_async(client, question, gold_answer, documents_str):
    """Async version of _ask_and_judge."""
    prompt = ABLATION_QA_PROMPT.format(documents=documents_str, question=question)
    response = await client.get_completion_async(prompt=prompt, temperature=0.0)
    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()
    if not predicted:
        return 0.0
    judge_prompt = ANSWER_JUDGE_PROMPT.format(
        gold_answer=gold_answer, predicted_answer=predicted, question=question,
    )
    result = await client.get_json_completion_async(
        prompt=judge_prompt, response_format=ANSWER_JUDGE_SCHEMA, temperature=0.0,
    )
    return result.get("score", 0.0)


async def pre_verify_async(instance, worker_client, threshold=0.3):
    """Async pre-check with worker (Haiku)."""
    supporting_docs = instance.gold_supporting_paragraphs
    if not supporting_docs:
        return False
    docs_str = _format_documents(supporting_docs)
    score = await _ask_and_judge_async(worker_client, instance.question, instance.answer, docs_str)
    return score >= threshold


async def verify_instance_async(instance, client, min_a_score=0.5, min_gap_ab=0.30,
                                max_c_score=0.4, max_docs_a=15):
    """Async 3-condition ablation — runs A, B, C in parallel."""
    all_docs = _get_all_docs(instance)
    supporting = [d for d in all_docs if d.get("is_supporting", False)]
    non_supporting = [d for d in all_docs if not d.get("is_supporting", False)]
    random.shuffle(non_supporting)
    docs_a_list = supporting + non_supporting[:max_docs_a]
    random.shuffle(docs_a_list)
    docs_a = _format_documents(docs_a_list)

    if instance.turns:
        last_docs = [d for d in instance.turns[-1].documents if not d.get("is_supporting", False)]
    else:
        last_docs = []
    docs_b = _format_documents(last_docs)

    gold_titles = {p["title"] for p in instance.gold_supporting_paragraphs}
    non_gold_docs = [d for d in all_docs if d.get("title", "") not in gold_titles]
    docs_c = _format_documents(non_gold_docs)

    # Run A, B, C in parallel
    score_a, score_b, score_c = await asyncio.gather(
        _ask_and_judge_async(client, instance.question, instance.answer, docs_a),
        _ask_and_judge_async(client, instance.question, instance.answer, docs_b),
        _ask_and_judge_async(client, instance.question, instance.answer, docs_c),
    )

    gap_ab = score_a - score_b
    gap_ac = score_a - score_c
    accepted = (
        score_a >= min_a_score
        and (score_b < 0.3 or gap_ab >= min_gap_ab)
        and score_c < max_c_score
    )
    return {
        "instance_id": instance.instance_id,
        "score_a": score_a, "score_b": score_b, "score_c": score_c,
        "gap_ab": gap_ab, "gap_ac": gap_ac, "accepted": accepted,
    }


async def verify_with_retry_async(instance, client, max_retries=1,
                                   min_a_score=0.5, min_gap_ab=0.30, max_c_score=0.4,
                                   max_docs_a=15):
    """Async verify with retry on too-hard failure."""
    result = await verify_instance_async(instance, client, min_a_score, min_gap_ab, max_c_score, max_docs_a)
    if result["accepted"]:
        return result
    if result["score_a"] < min_a_score and max_retries > 0:
        simplified = _simplify_for_retry(instance)
        retry_result = await verify_instance_async(simplified, client, min_a_score, min_gap_ab, max_c_score, max_docs_a)
        retry_result["retry_reason"] = "simplified_distractors"
        if retry_result["accepted"]:
            instance.turns = simplified.turns
            return retry_result
    return result


def _verify_all_async(instances, client, worker_client,
                      min_a_score, min_gap_ab, max_c_score, pre_verify_threshold,
                      max_concurrent, max_docs_a=15):
    """Run verification in parallel using asyncio."""
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc="Verifying (async)")

        async def _one(inst):
            async with sem:
                # Cheap pre-check
                if worker_client is not None:
                    passed = await pre_verify_async(inst, worker_client, threshold=pre_verify_threshold)
                    if not passed:
                        pbar.update(1)
                        return {
                            "instance_id": inst.instance_id,
                            "score_a": 0.0, "score_b": 0.0, "score_c": 0.0,
                            "gap_ab": 0.0, "gap_ac": 0.0,
                            "accepted": False, "skipped_reason": "pre_verify_failed",
                        }
                result = await verify_with_retry_async(
                    inst, client, min_a_score=min_a_score,
                    min_gap_ab=min_gap_ab, max_c_score=max_c_score,
                    max_docs_a=max_docs_a,
                )
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()
        return list(results)

    return asyncio.run(_run())
