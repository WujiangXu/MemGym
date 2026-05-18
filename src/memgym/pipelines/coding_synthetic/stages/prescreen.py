"""
Pre-screening stage: reject instances that are solvable without memory.

Runs BEFORE fact extraction to save compute. Tests if an LLM can produce
a reasonable fix from just the vague task prompt + repo context alone.
If it can, the instance is fundamentally too easy and should be rejected.
"""

import json
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from ..types.schemas import FilteredInstance, LLM_JUDGE_SCHEMA
from ..llm.client import LLMClient
from ..llm.prompts import PATCH_GENERATION_SYSTEM, LLM_JUDGE_PROMPT
from ..utils.patch_utils import reverse_patch


def _craft_vague_prompt(instance: FilteredInstance) -> str:
    """Create a minimal vague prompt from the problem statement.

    This is intentionally using just the first 2 sentences — we're trying
    to check if even a vague description is enough to solve the bug.
    """
    ps = instance.problem_statement
    sentences = ps.replace("\n", " ").split(".")
    return ". ".join(sentences[:2]).strip() + "."


def _generate_patch_for_prescreen(
    task_prompt: str,
    repo_context: Optional[Dict[str, str]],
    model: str,
) -> str:
    """Generate a fix patch using only task prompt + repo (no memory)."""
    import litellm

    messages = [
        {"role": "system", "content": PATCH_GENERATION_SYSTEM},
    ]

    if repo_context:
        repo_parts = []
        total = 0
        for fp, content in repo_context.items():
            truncated = content[:15000] if len(content) > 15000 else content
            repo_parts.append(f"=== {fp} ===\n{truncated}")
            total += len(truncated)
            if total > 50000:
                break
        repo_block = "\n\n".join(repo_parts)
        messages.append({
            "role": "user",
            "content": f"Here are the repository source files:\n\n{repo_block}",
        })
        messages.append({
            "role": "assistant",
            "content": "I've read the repository files. What's the issue?",
        })

    messages.append({
        "role": "user",
        "content": (
            f"{task_prompt}\n\n"
            "Generate the unified diff patch that fixes this problem. "
            "Output ONLY the diff:"
        ),
    })

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            max_completion_tokens=16000,
            temperature=0.7,
        )
        patch = response.choices[0].message.content.strip()
        # Strip markdown fences
        if patch.startswith("```"):
            lines = patch.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            patch = "\n".join(lines)
        return patch
    except Exception:
        return ""


def _score_patch(
    generated_patch: str,
    gold_fix: str,
    model: str,
) -> float:
    """Score generated patch against gold fix using LLM judge."""
    if not generated_patch.strip():
        return 0.0

    import litellm

    judge_prompt = LLM_JUDGE_PROMPT.format(
        generated=generated_patch[:3000],
        gold_fix=gold_fix[:3000],
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a code evaluation judge. Output only valid JSON.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            max_completion_tokens=4000,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        result = json.loads(text)
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def prescreen_instance(
    instance: FilteredInstance,
    repo_context: Dict[str, str],
    worker_model: str,
    threshold: float = 0.4,
) -> Tuple[bool, float]:
    """Pre-screen an instance: check if it's solvable without memory.

    Args:
        instance: FilteredInstance to check
        repo_context: Dict mapping file_path -> content
        worker_model: LLM model for patch generation + judging
        threshold: Score threshold above which instance is "too easy"

    Returns:
        (too_easy: bool, score: float)
    """
    vague_prompt = _craft_vague_prompt(instance)
    gold_fix = reverse_patch(instance.patch)

    # Generate a fix attempt from vague prompt + repo only
    patch = _generate_patch_for_prescreen(vague_prompt, repo_context, worker_model)

    # Score against gold fix
    score = _score_patch(patch, gold_fix, worker_model)

    return score >= threshold, score


def prescreen_all(
    instances: List[FilteredInstance],
    repo_contexts: Dict[str, Dict[str, str]],
    worker_model: str,
    threshold: float = 0.4,
    show_progress: bool = True,
    num_workers: int = 1,
) -> Tuple[List[FilteredInstance], List[Dict[str, Any]]]:
    """Pre-screen all instances, rejecting those solvable without memory.

    Args:
        instances: List of FilteredInstance objects
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        worker_model: Worker LLM model name
        threshold: Score threshold for rejection
        show_progress: Show progress bar
        num_workers: Number of parallel workers

    Returns:
        (kept_instances, prescreen_results)
    """
    from ..utils.parallel import parallel_map

    def _process(instance: FilteredInstance) -> Dict[str, Any]:
        repo_ctx = repo_contexts.get(instance.instance_id, {})
        too_easy, score = prescreen_instance(
            instance, repo_ctx, worker_model, threshold,
        )
        if too_easy and show_progress:
            tqdm.write(
                f"  REJECTED {instance.instance_id}: "
                f"solvable without memory (score={score:.2f})"
            )
        return {
            "instance_id": instance.instance_id,
            "too_easy": too_easy,
            "score": score,
        }

    results = parallel_map(
        _process, instances,
        num_workers=num_workers,
        desc="Pre-screening",
        show_progress=show_progress,
        filter_none=False,
    )

    kept = []
    result_ids_easy = {r["instance_id"] for r in results if r["too_easy"]}
    for instance in instances:
        if instance.instance_id not in result_ids_easy:
            kept.append(instance)

    rejected = len(instances) - len(kept)
    print(
        f"Pre-screen: {len(kept)} kept, {rejected} rejected "
        f"({rejected / max(1, len(instances)) * 100:.0f}% too easy)"
    )

    return kept, results
