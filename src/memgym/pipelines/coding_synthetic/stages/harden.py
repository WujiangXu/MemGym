"""
Stage 4 — Difficulty: Apply composable transforms to increase instance difficulty.

Replaces the v1 monolithic hardening loop with:
  1. Preset-based difficulty application (easy/medium/hard)
  2. Adaptive difficulty based on verification feedback
"""

from typing import Dict, List, Optional, Any

from tqdm import tqdm

from ..types.schemas import MemGymInstance
from ..llm.client import LLMClient
from ..transforms import (
    PRESETS,
    PromptFuzzTransform,
    DistractorScaleTransform,
    IndirectionTransform,
    FactFragmentTransform,
)


# Registry of available transforms
_TRANSFORM_REGISTRY = {
    "prompt_fuzz": PromptFuzzTransform(),
    "distractor_scale": DistractorScaleTransform(),
    "indirection": IndirectionTransform(),
    "fact_fragment": FactFragmentTransform(),
}


def apply_difficulty(
    instance: MemGymInstance,
    preset: str = "medium",
    worker_client: Optional[LLMClient] = None,
    repo_context: Optional[Dict[str, str]] = None,
) -> MemGymInstance:
    """Apply difficulty preset to instance.

    Args:
        instance: MemGymInstance to transform
        preset: Difficulty preset name (easy/medium/hard)
        worker_client: Worker LLM client (needed for LLM transforms)
        repo_context: Dict of repo files (needed for indirection)

    Returns:
        Transformed MemGymInstance
    """
    transform_specs = PRESETS.get(preset, PRESETS["medium"])

    result = instance
    for transform_name, kwargs in transform_specs:
        transform = _TRANSFORM_REGISTRY.get(transform_name)
        if transform is None:
            continue

        if transform.requires_llm and not worker_client:
            continue

        if not transform.can_apply(result):
            continue

        result = transform.apply(
            result,
            worker_client=worker_client,
            repo_context=repo_context,
            **kwargs,
        )

    # Update difficulty preset
    result = result.model_copy(update={"difficulty_preset": preset})
    return result


def apply_adaptive_difficulty(
    instance: MemGymInstance,
    verification_result: Dict[str, Any],
    worker_client: Optional[LLMClient] = None,
    repo_context: Optional[Dict[str, str]] = None,
) -> MemGymInstance:
    """Apply targeted transforms based on what failed in verification.

    Args:
        instance: MemGymInstance to transform
        verification_result: Result dict from verify stage
        worker_client: Worker LLM client
        repo_context: Dict of repo files

    Returns:
        Transformed MemGymInstance with targeted difficulty adjustments
    """
    cond_a_passes = verification_result.get("cond_a_docker_passes", 0)
    cond_b_passes = verification_result.get("cond_b_docker_passes", 0)
    cond_a_llm = verification_result.get("cond_a_llm_score", 0.0)
    cond_b_llm = verification_result.get("cond_b_llm_score", 0.0)

    result = instance

    # Agent solved without memory → prompt leaks too much, apply heavier fuzz
    if cond_b_passes > 0 or cond_b_llm > 0.5:
        fuzz = PromptFuzzTransform()
        result = fuzz.apply(
            result,
            worker_client=worker_client,
            repo_context=repo_context,
            intensity="heavy",
        )

    # Gap too small → need more indirection + distractors
    gap_docker = cond_a_passes - cond_b_passes
    gap_llm = cond_a_llm - cond_b_llm
    if gap_docker <= 0 and gap_llm < 0.15:
        if repo_context:
            indirection = IndirectionTransform()
            result = indirection.apply(
                result,
                worker_client=worker_client,
                repo_context=repo_context,
            )
        distractor = DistractorScaleTransform()
        result = distractor.apply(
            result,
            worker_client=worker_client,
            repo_context=repo_context,
            target_count=max(8, len(instance.distractors) + 3),
        )

    return result


def apply_difficulty_all(
    instances: List[MemGymInstance],
    preset: str = "medium",
    worker_client: Optional[LLMClient] = None,
    repo_contexts: Optional[Dict[str, Dict[str, str]]] = None,
    show_progress: bool = True,
    num_workers: int = 1,
) -> List[MemGymInstance]:
    """Apply difficulty transforms to all instances.

    Args:
        instances: List of MemGymInstance objects
        preset: Difficulty preset (easy/medium/hard)
        worker_client: Worker LLM client
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        show_progress: Show progress bar
        num_workers: Number of parallel workers

    Returns:
        List of transformed MemGymInstance objects
    """
    from ..utils.parallel import parallel_map

    def _process(instance: MemGymInstance) -> MemGymInstance:
        repo_ctx = {}
        if repo_contexts:
            repo_ctx = repo_contexts.get(instance.instance_id, {})
            if not repo_ctx:
                repo_ctx = repo_contexts.get(instance.base_instance, {})
        return apply_difficulty(
            instance, preset, worker_client, repo_ctx
        )

    results = parallel_map(
        _process, instances,
        num_workers=num_workers,
        desc="Difficulty",
        show_progress=show_progress,
        filter_none=False,
    )

    # Summary
    transforms_applied = {}
    for inst in results:
        for t in inst.transforms_applied:
            transforms_applied[t] = transforms_applied.get(t, 0) + 1
    print(f"Applied difficulty (preset={preset}) to {len(results)} instances")
    if transforms_applied:
        print(f"  Transforms: {transforms_applied}")

    return results
