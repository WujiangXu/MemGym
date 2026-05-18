"""Stage 1: Pre-screen -- Reject questions answerable without context.

Two tests:
1. Parametric: Ask LLM the question with no context. Reject if correct.
2. Partial-context: Give LLM only the last hop's paragraph. Reject if correct.
"""

import asyncio
import json
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm

from ..types.schemas import FilteredIRInstance
from ..llm.client import LLMClient
from ..llm.prompts import (
    PARAMETRIC_TEST_PROMPT,
    PARTIAL_CONTEXT_TEST_PROMPT,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)


def _judge_answer(
    client: LLMClient,
    question: str,
    gold_answer: str,
    predicted_answer: str,
) -> float:
    """Score predicted answer against gold using LLM judge."""
    prompt = ANSWER_JUDGE_PROMPT.format(
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        question=question,
    )
    result = client.get_json_completion(
        prompt=prompt,
        response_format=ANSWER_JUDGE_SCHEMA,
        temperature=0.0,
    )
    return result.get("score", 0.0)


def _test_parametric(
    client: LLMClient,
    instance: FilteredIRInstance,
) -> float:
    """Test if question is answerable from parametric knowledge."""
    prompt = PARAMETRIC_TEST_PROMPT.format(question=instance.question)
    response = client.get_completion(prompt=prompt, temperature=0.0)

    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    if not predicted or predicted.lower() in ("i don't know", "unknown", ""):
        return 0.0

    return _judge_answer(client, instance.question, instance.answer, predicted)


def _test_partial_context(
    client: LLMClient,
    instance: FilteredIRInstance,
) -> float:
    """Test if question is answerable with only the last hop's paragraph."""
    # Get the last supporting paragraph
    if not instance.supporting_paragraphs:
        return 0.0

    last_para = instance.supporting_paragraphs[-1]
    context = f"Title: {last_para['title']}\n{last_para['text']}"

    prompt = PARTIAL_CONTEXT_TEST_PROMPT.format(
        context=context,
        question=instance.question,
    )
    response = client.get_completion(prompt=prompt, temperature=0.0)

    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    if not predicted:
        return 0.0

    return _judge_answer(client, instance.question, instance.answer, predicted)


def prescreen_instance(
    instance: FilteredIRInstance,
    client: LLMClient,
    parametric_threshold: float = 0.5,
    partial_threshold: float = 0.6,
) -> Dict:
    """Pre-screen a single instance.

    Returns:
        Dict with instance_id, parametric_score, partial_score, rejected, reason.
    """
    parametric_score = _test_parametric(client, instance)

    if parametric_score >= parametric_threshold:
        return {
            "instance_id": instance.instance_id,
            "parametric_score": parametric_score,
            "partial_score": 0.0,
            "rejected": True,
            "reason": f"parametric_score={parametric_score:.2f} >= {parametric_threshold}",
        }

    partial_score = _test_partial_context(client, instance)

    rejected = partial_score >= partial_threshold
    reason = ""
    if rejected:
        reason = f"partial_score={partial_score:.2f} >= {partial_threshold}"

    return {
        "instance_id": instance.instance_id,
        "parametric_score": parametric_score,
        "partial_score": partial_score,
        "rejected": rejected,
        "reason": reason,
    }


async def _judge_answer_async(client, question, gold_answer, predicted_answer):
    prompt = ANSWER_JUDGE_PROMPT.format(
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        question=question,
    )
    result = await client.get_json_completion_async(
        prompt=prompt, response_format=ANSWER_JUDGE_SCHEMA, temperature=0.0,
    )
    return result.get("score", 0.0)


async def _test_parametric_async(client, instance):
    prompt = PARAMETRIC_TEST_PROMPT.format(question=instance.question)
    response = await client.get_completion_async(prompt=prompt, temperature=0.0)
    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()
    if not predicted or predicted.lower() in ("i don't know", "unknown", ""):
        return 0.0
    return await _judge_answer_async(client, instance.question, instance.answer, predicted)


async def _test_partial_context_async(client, instance):
    if not instance.supporting_paragraphs:
        return 0.0
    last_para = instance.supporting_paragraphs[-1]
    context = f"Title: {last_para['title']}\n{last_para['text']}"
    prompt = PARTIAL_CONTEXT_TEST_PROMPT.format(context=context, question=instance.question)
    response = await client.get_completion_async(prompt=prompt, temperature=0.0)
    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()
    if not predicted:
        return 0.0
    return await _judge_answer_async(client, instance.question, instance.answer, predicted)


async def prescreen_instance_async(instance, client, parametric_threshold=0.5, partial_threshold=0.6):
    parametric_score = await _test_parametric_async(client, instance)
    if parametric_score >= parametric_threshold:
        return {
            "instance_id": instance.instance_id,
            "parametric_score": parametric_score,
            "partial_score": 0.0,
            "rejected": True,
            "reason": f"parametric_score={parametric_score:.2f} >= {parametric_threshold}",
        }
    partial_score = await _test_partial_context_async(client, instance)
    rejected = partial_score >= partial_threshold
    reason = f"partial_score={partial_score:.2f} >= {partial_threshold}" if rejected else ""
    return {
        "instance_id": instance.instance_id,
        "parametric_score": parametric_score,
        "partial_score": partial_score,
        "rejected": rejected,
        "reason": reason,
    }


def prescreen_all(
    instances: List[FilteredIRInstance],
    client: LLMClient,
    parametric_threshold: float = 0.5,
    partial_threshold: float = 0.6,
    show_progress: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 10,
) -> Tuple[List[FilteredIRInstance], List[Dict]]:
    """Pre-screen all instances, rejecting parametric questions.

    Returns:
        (kept_instances, all_results)
    """
    if dry_run:
        print("  [dry-run] Skipping pre-screen LLM calls")
        results = [
            {"instance_id": i.instance_id, "parametric_score": 0.0,
             "partial_score": 0.0, "rejected": False, "reason": "dry_run"}
            for i in instances
        ]
        return instances, results

    if max_concurrent > 1:
        results = _prescreen_all_async(
            instances, client, parametric_threshold, partial_threshold, max_concurrent,
        )
    else:
        results = []
        iterator = tqdm(instances, desc="Pre-screening") if show_progress else instances
        for inst in iterator:
            result = prescreen_instance(
                inst, client, parametric_threshold, partial_threshold
            )
            results.append(result)

    kept = [inst for inst, r in zip(instances, results) if not r["rejected"]]
    rejected_count = len(instances) - len(kept)
    print(f"  Pre-screen: {len(kept)} kept, {rejected_count} rejected "
          f"({rejected_count/max(len(instances),1)*100:.1f}%)")

    return kept, results


def _prescreen_all_async(instances, client, parametric_threshold, partial_threshold, max_concurrent):
    """Run prescreen in parallel using asyncio."""
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc="Pre-screening (async)")

        async def _one(inst):
            async with sem:
                result = await prescreen_instance_async(
                    inst, client, parametric_threshold, partial_threshold,
                )
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()
        return list(results)

    return asyncio.run(_run())
