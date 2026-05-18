"""Stage 0.5: Upgrade -- Fictionalize entities and extend hop chains.

Takes filtered instances (typically 2-hop from MuSiQue) and:
1. Replaces all real entity names with fictional alternatives (anti-contamination)
2. Extends the reasoning chain to N hops by inserting intermediate bridge entities

This prevents LLMs from using parametric knowledge to bypass the documents,
and creates more memory-required facts under eviction.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ..types.schemas import FilteredIRInstance
from ..llm.client import LLMClient
from ..llm.prompts import (
    FICTIONALIZE_ENTITIES_PROMPT,
    FICTIONALIZE_ENTITIES_SCHEMA,
    HOP_UPGRADE_CHAIN_PROMPT,
    HOP_UPGRADE_CHAIN_SCHEMA,
)


def _fictionalize_instance(
    instance: FilteredIRInstance,
    client: LLMClient,
) -> Optional[FilteredIRInstance]:
    """Replace all real entities with fictional names and perturb facts."""
    entities_json = json.dumps(
        [
            {"title": p["title"], "text": p["text"]}
            for p in instance.supporting_paragraphs
        ],
        indent=2,
    )
    decomposition_json = json.dumps(instance.decomposition, indent=2)

    prompt = FICTIONALIZE_ENTITIES_PROMPT.format(
        question=instance.question,
        answer=instance.answer,
        entities_json=entities_json,
        decomposition_json=decomposition_json,
    )

    result = client.get_json_completion(
        prompt=prompt,
        response_format=FICTIONALIZE_ENTITIES_SCHEMA,
        temperature=0.7,
    )

    if not result or not result.get("question"):
        return None

    # Validate: must have same number of decomposition steps
    new_decomp = result.get("decomposition", [])
    if len(new_decomp) != len(instance.decomposition):
        return None

    # Validate: must have same number of supporting paragraphs
    new_supporting = result.get("supporting_paragraphs", [])
    if len(new_supporting) != len(instance.supporting_paragraphs):
        return None

    # Also fictionalize distractor paragraphs using the entity mapping
    # Convert array format [{real_name, fictional_name}] to dict
    raw_mapping = result.get("entity_mapping", [])
    if isinstance(raw_mapping, list):
        entity_mapping = {
            entry["real_name"]: entry["fictional_name"]
            for entry in raw_mapping
            if "real_name" in entry and "fictional_name" in entry
        }
    else:
        entity_mapping = raw_mapping
    new_distractors = []
    for dist in instance.distractor_paragraphs:
        new_title = dist["title"]
        new_text = dist.get("text", "")
        for real_name, fictional_name in entity_mapping.items():
            new_title = new_title.replace(real_name, fictional_name)
            new_text = new_text.replace(real_name, fictional_name)
        new_distractors.append({
            "title": new_title,
            "text": new_text,
            "is_supporting": False,
        })

    return FilteredIRInstance(
        instance_id=instance.instance_id,
        source_dataset=instance.source_dataset,
        question=result["question"],
        answer=result["answer"],
        num_hops=instance.num_hops,
        decomposition=new_decomp,
        supporting_paragraphs=[
            {"title": p["title"], "text": p["text"], "is_supporting": True}
            for p in new_supporting
        ],
        distractor_paragraphs=new_distractors,
        question_type=instance.question_type,
        entity_mapping=entity_mapping,
    )


async def _fictionalize_instance_async(
    instance: FilteredIRInstance,
    client: LLMClient,
) -> Optional[FilteredIRInstance]:
    """Async version of _fictionalize_instance."""
    entities_json = json.dumps(
        [
            {"title": p["title"], "text": p["text"]}
            for p in instance.supporting_paragraphs
        ],
        indent=2,
    )
    decomposition_json = json.dumps(instance.decomposition, indent=2)

    prompt = FICTIONALIZE_ENTITIES_PROMPT.format(
        question=instance.question,
        answer=instance.answer,
        entities_json=entities_json,
        decomposition_json=decomposition_json,
    )

    result = await client.get_json_completion_async(
        prompt=prompt,
        response_format=FICTIONALIZE_ENTITIES_SCHEMA,
        temperature=0.7,
    )

    if not result or not result.get("question"):
        return None

    new_decomp = result.get("decomposition", [])
    if len(new_decomp) != len(instance.decomposition):
        return None

    new_supporting = result.get("supporting_paragraphs", [])
    if len(new_supporting) != len(instance.supporting_paragraphs):
        return None

    # Convert array format [{real_name, fictional_name}] to dict
    raw_mapping = result.get("entity_mapping", [])
    if isinstance(raw_mapping, list):
        entity_mapping = {
            entry["real_name"]: entry["fictional_name"]
            for entry in raw_mapping
            if "real_name" in entry and "fictional_name" in entry
        }
    else:
        entity_mapping = raw_mapping
    new_distractors = []
    for dist in instance.distractor_paragraphs:
        new_title = dist["title"]
        new_text = dist.get("text", "")
        for real_name, fictional_name in entity_mapping.items():
            new_title = new_title.replace(real_name, fictional_name)
            new_text = new_text.replace(real_name, fictional_name)
        new_distractors.append({
            "title": new_title,
            "text": new_text,
            "is_supporting": False,
        })

    return FilteredIRInstance(
        instance_id=instance.instance_id,
        source_dataset=instance.source_dataset,
        question=result["question"],
        answer=result["answer"],
        num_hops=instance.num_hops,
        decomposition=new_decomp,
        supporting_paragraphs=[
            {"title": p["title"], "text": p["text"], "is_supporting": True}
            for p in new_supporting
        ],
        distractor_paragraphs=new_distractors,
        question_type=instance.question_type,
        entity_mapping=entity_mapping,
    )


def _upgrade_hops(
    instance: FilteredIRInstance,
    target_hops: int,
    client: LLMClient,
    max_retries: int = 2,
) -> Optional[FilteredIRInstance]:
    """Extend the hop chain from current_hops to target_hops.

    Includes retry logic with increasing temperature to improve yield.
    Relaxed validation: allows ±1 hop deviation and 20-word minimum paragraphs.
    """
    current_hops = instance.num_hops
    if current_hops >= target_hops:
        return instance  # Already at or above target

    new_hops = target_hops - current_hops
    chain_json = json.dumps(instance.decomposition, indent=2)

    for attempt in range(1 + max_retries):
        temperature = 0.7 + (attempt * 0.1)  # Increase randomness on retries

        prompt = HOP_UPGRADE_CHAIN_PROMPT.format(
            current_hops=current_hops,
            target_hops=target_hops,
            new_hops=new_hops,
            chain_json=chain_json,
            question=instance.question,
            answer=instance.answer,
            last_hop_index=target_hops - 1,
        )

        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=HOP_UPGRADE_CHAIN_SCHEMA,
                temperature=temperature,
            )
        except Exception:
            continue

        if not result or not result.get("question"):
            continue

        new_decomp = result.get("decomposition", [])
        new_supporting = result.get("supporting_paragraphs", [])

        # Relaxed validation: allow ±1 hop deviation
        if abs(len(new_decomp) - target_hops) > 1:
            continue

        # Relaxed validation: 20-word minimum (down from 30)
        all_valid = True
        for p in new_supporting:
            if len(p.get("text", "").split()) < 20:
                all_valid = False
                break
        if not all_valid:
            continue

        # If we got fewer/more hops than target, adjust
        actual_hops = len(new_decomp)

        return FilteredIRInstance(
            instance_id=instance.instance_id,
            source_dataset=instance.source_dataset,
            question=result["question"],
            answer=result["answer"],
            num_hops=actual_hops,
            decomposition=new_decomp,
            supporting_paragraphs=[
                {"title": p["title"], "text": p["text"], "is_supporting": True}
                for p in new_supporting
            ],
            distractor_paragraphs=instance.distractor_paragraphs,
            question_type=instance.question_type,
            entity_mapping=getattr(instance, "entity_mapping", None),
        )

    return None  # All retries exhausted


async def _upgrade_hops_async(
    instance: FilteredIRInstance,
    target_hops: int,
    client: LLMClient,
    max_retries: int = 2,
) -> Optional[FilteredIRInstance]:
    """Async version of _upgrade_hops with retry logic."""
    current_hops = instance.num_hops
    if current_hops >= target_hops:
        return instance

    new_hops = target_hops - current_hops
    chain_json = json.dumps(instance.decomposition, indent=2)

    for attempt in range(1 + max_retries):
        temperature = 0.7 + (attempt * 0.1)

        prompt = HOP_UPGRADE_CHAIN_PROMPT.format(
            current_hops=current_hops,
            target_hops=target_hops,
            new_hops=new_hops,
            chain_json=chain_json,
            question=instance.question,
            answer=instance.answer,
            last_hop_index=target_hops - 1,
        )

        try:
            result = await client.get_json_completion_async(
                prompt=prompt,
                response_format=HOP_UPGRADE_CHAIN_SCHEMA,
                temperature=temperature,
            )
        except Exception:
            continue

        if not result or not result.get("question"):
            continue

        new_decomp = result.get("decomposition", [])
        new_supporting = result.get("supporting_paragraphs", [])

        if abs(len(new_decomp) - target_hops) > 1:
            continue

        all_valid = True
        for p in new_supporting:
            if len(p.get("text", "").split()) < 20:
                all_valid = False
                break
        if not all_valid:
            continue

        actual_hops = len(new_decomp)

        return FilteredIRInstance(
            instance_id=instance.instance_id,
            source_dataset=instance.source_dataset,
            question=result["question"],
            answer=result["answer"],
            num_hops=actual_hops,
            decomposition=new_decomp,
            supporting_paragraphs=[
                {"title": p["title"], "text": p["text"], "is_supporting": True}
                for p in new_supporting
            ],
            distractor_paragraphs=instance.distractor_paragraphs,
            question_type=instance.question_type,
            entity_mapping=getattr(instance, "entity_mapping", None),
        )

    return None


def upgrade_instance(
    instance: FilteredIRInstance,
    target_hops: int,
    client: LLMClient,
) -> Optional[FilteredIRInstance]:
    """Fictionalize entities and extend hop chain for a single instance."""
    # Step 1: Fictionalize
    fictionalized = _fictionalize_instance(instance, client)
    if fictionalized is None:
        return None

    # Step 2: Extend hops
    if target_hops > fictionalized.num_hops:
        upgraded = _upgrade_hops(fictionalized, target_hops, client)
        return upgraded

    return fictionalized


async def _upgrade_instance_async(
    instance: FilteredIRInstance,
    target_hops: int,
    client: LLMClient,
) -> Optional[FilteredIRInstance]:
    """Async: fictionalize + extend hops."""
    fictionalized = await _fictionalize_instance_async(instance, client)
    if fictionalized is None:
        return None

    if target_hops > fictionalized.num_hops:
        upgraded = await _upgrade_hops_async(fictionalized, target_hops, client)
        return upgraded

    return fictionalized


def upgrade_all(
    instances: List[FilteredIRInstance],
    target_hops: int,
    client: LLMClient,
    output_path: Optional[Path] = None,
    max_concurrent: int = 10,
    dry_run: bool = False,
) -> List[FilteredIRInstance]:
    """Fictionalize and upgrade all instances to target_hops.

    Args:
        instances: Filtered instances to upgrade.
        target_hops: Target number of hops per instance.
        client: LLM client for generation.
        output_path: Optional checkpoint path.
        max_concurrent: Max parallel LLM calls.
        dry_run: Skip LLM calls (return originals unchanged).
    """
    if dry_run:
        print(f"  [dry-run] Skipping upgrade (would target {target_hops} hops)")
        return instances

    if max_concurrent <= 1:
        # Sequential
        upgraded = []
        for inst in tqdm(instances, desc=f"Upgrading to {target_hops} hops"):
            result = upgrade_instance(inst, target_hops, client)
            if result is not None:
                upgraded.append(result)
                if output_path:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result.model_dump()) + "\n")
        print(f"  Upgraded {len(upgraded)}/{len(instances)} instances")
        return upgraded

    # Async parallel
    return _upgrade_all_async(
        instances, target_hops, client, output_path, max_concurrent,
    )


def _upgrade_all_async(instances, target_hops, client, output_path, max_concurrent):
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc=f"Upgrading to {target_hops} hops (async)")

        async def _one(inst):
            async with sem:
                result = await _upgrade_instance_async(inst, target_hops, client)
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()
        upgraded = [r for r in results if r is not None]
        if output_path:
            with open(output_path, "a") as f:
                for r in upgraded:
                    f.write(json.dumps(r.model_dump()) + "\n")
        print(f"  Upgraded {len(upgraded)}/{len(instances)} instances")
        return upgraded

    return asyncio.run(_run())


def load_upgraded_instances(path: Path) -> List[FilteredIRInstance]:
    """Load upgraded instances from checkpoint."""
    instances = []
    with open(path) as f:
        for line in f:
            if line.strip():
                instances.append(FilteredIRInstance(**json.loads(line)))
    print(f"  Loaded {len(instances)} upgraded instances from checkpoint")
    return instances
