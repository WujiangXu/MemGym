"""Stage 5: Harden -- Apply composable difficulty transforms.

Transforms:
- QueryFuzz: Rephrase question to be vaguer
- DistractorScale: Multiply distractor count
- TemporalShift: Add time-shifted versions of supporting paragraphs
- ContraPlacement: Move contradictions after correct information

Presets: easy, medium, hard.
"""

import asyncio
import copy
import random
from typing import List, Dict, Optional

from tqdm import tqdm

from ..types.schemas import MemGymIRInstance, DistractorFactIR
from ..llm.client import LLMClient
from ..llm.prompts import QUERY_FUZZ_PROMPT, QUERY_FUZZ_SCHEMA


PRESETS = {
    "easy": [
        ("contra_placement", {}),
        ("distractor_scale", {"factor": 1.0}),
    ],
    "medium": [
        ("query_fuzz", {"intensity": "light"}),
        ("contra_placement", {}),
        ("temporal_shift", {}),
        ("distractor_scale", {"factor": 1.5}),
    ],
    "hard": [
        ("query_fuzz", {"intensity": "heavy"}),
        ("distractor_scale", {"factor": 2.5}),
        ("temporal_shift", {}),
        ("contra_placement", {}),
    ],
}


def _apply_query_fuzz(
    instance: MemGymIRInstance,
    client: LLMClient,
    intensity: str = "light",
    dry_run: bool = False,
) -> MemGymIRInstance:
    """Rephrase question to be vaguer."""
    if dry_run:
        instance.transforms_applied.append("query_fuzz")
        return instance

    prompt = QUERY_FUZZ_PROMPT.format(
        question=instance.question,
        answer=instance.answer,
        intensity=intensity,
    )
    result = client.get_json_completion(
        prompt=prompt,
        response_format=QUERY_FUZZ_SCHEMA,
        temperature=0.7,
    )
    fuzzed = result.get("fuzzed_question", "")
    if fuzzed and len(fuzzed) > 10:
        instance.question = fuzzed
    instance.transforms_applied.append("query_fuzz")
    return instance


def _apply_distractor_scale(
    instance: MemGymIRInstance,
    factor: float = 1.5,
) -> MemGymIRInstance:
    """Scale distractor count by duplicating with slight variation."""
    if factor <= 1.0:
        return instance

    existing = list(instance.distractors)
    num_to_add = int(len(existing) * (factor - 1.0))

    for i in range(min(num_to_add, len(existing))):
        src = existing[i % len(existing)]
        new_d = DistractorFactIR(
            id=f"D_scaled_{i}",
            content=src.content,
            source=src.source,
            target_turn=src.target_turn,
            interferes_with=src.interferes_with,
        )
        instance.distractors.append(new_d)

        # Also add to the turn's documents
        if src.target_turn < len(instance.turns):
            instance.turns[src.target_turn].documents.append({
                "title": f"Scaled distractor {i}",
                "text": src.content,
                "is_supporting": False,
            })

    instance.transforms_applied.append(f"distractor_scale_{factor}")
    return instance


def _apply_temporal_shift(
    instance: MemGymIRInstance,
    client: Optional[LLMClient] = None,
    dry_run: bool = False,
) -> MemGymIRInstance:
    """Add shifted versions of supporting paragraphs as distractors.

    Generates variants with altered facts for ALL grounding fact types:
    - temporal_fact/numerical_fact: different dates/numbers
    - bridge_fact: altered bridge entity or relationship
    - terminal_fact: changed final answer detail

    Injects shifted versions BEFORE the turn where the fact is needed,
    creating confusion about which version is correct.
    """
    if dry_run or client is None:
        instance.transforms_applied.append("temporal_shift")
        return instance

    from ..llm.prompts import TEMPORAL_SHIFT_PROMPT, BRIDGE_SHIFT_PROMPT, DISTRACTOR_SCHEMA

    if not instance.grounding_facts:
        instance.transforms_applied.append("temporal_shift")
        return instance

    num_turns = len(instance.turns)
    generated = 0

    for fact in instance.grounding_facts:
        # Choose prompt based on fact type
        if fact.fact_type in ("temporal_fact", "numerical_fact"):
            prompt = TEMPORAL_SHIFT_PROMPT.format(
                entity=fact.source_paragraph_title,
                fact=fact.content,
            )
        else:
            # bridge_fact or terminal_fact — use the bridge shift prompt
            prompt = BRIDGE_SHIFT_PROMPT.format(
                entity=fact.source_paragraph_title,
                fact=fact.content,
                fact_type=fact.fact_type,
            )

        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=DISTRACTOR_SCHEMA,
                temperature=0.7,
            )
        except Exception:
            continue

        text = result.get("distractor_text", "")
        if not text:
            continue

        # Place BEFORE the turn where the fact is needed (not after)
        # This creates confusion: agent sees wrong version first
        needed_turn = min(fact.needed_at_hop, num_turns - 1)
        target_turn = max(0, needed_turn - 1) if needed_turn > 0 else 0

        instance.turns[target_turn].documents.append({
            "title": f"Shifted: {fact.source_paragraph_title}",
            "text": text,
            "is_supporting": False,
        })
        instance.distractors.append(DistractorFactIR(
            id=f"D_shifted_{fact.fact_id}",
            content=text,
            source="same_entity",
            target_turn=target_turn,
            interferes_with=fact.fact_id,
        ))
        generated += 1

    instance.transforms_applied.append(f"temporal_shift({generated})")
    return instance


def _apply_contra_placement(
    instance: MemGymIRInstance,
) -> MemGymIRInstance:
    """Move contradictions to appear AFTER the correct information.

    For each same_entity distractor, move it to a later turn than its
    current placement.
    """
    num_turns = len(instance.turns)
    if num_turns < 2:
        instance.transforms_applied.append("contra_placement")
        return instance

    for d in instance.distractors:
        if d.source == "same_entity" and d.target_turn < num_turns - 1:
            # Move to a later turn
            old_turn = d.target_turn
            new_turn = min(old_turn + 1, num_turns - 1)
            d.target_turn = new_turn

            # Move document from old turn to new turn
            old_docs = instance.turns[old_turn].documents
            for doc in old_docs:
                if doc.get("text", "") == d.content:
                    old_docs.remove(doc)
                    instance.turns[new_turn].documents.append(doc)
                    break

    instance.transforms_applied.append("contra_placement")
    return instance


def harden_instance(
    instance: MemGymIRInstance,
    preset: str = "easy",
    client: Optional[LLMClient] = None,
    dry_run: bool = False,
    difficulty: Optional[object] = None,
) -> MemGymIRInstance:
    """Apply difficulty transforms based on preset or DifficultyConfig.

    If difficulty (DifficultyConfig) is provided, builds transforms from its fields.
    Otherwise falls back to preset-based PRESETS dict.
    """
    if difficulty is not None:
        # Build transforms from DifficultyConfig fields
        instance.difficulty_preset = preset
        transforms = []

        if difficulty.query_fuzz_intensity != "none":
            transforms.append(("query_fuzz", {"intensity": difficulty.query_fuzz_intensity}))
        if difficulty.enable_contra_placement:
            transforms.append(("contra_placement", {}))
        if difficulty.enable_temporal_shift:
            transforms.append(("temporal_shift", {}))
        if difficulty.distractor_scale_factor > 1.0:
            transforms.append(("distractor_scale", {"factor": difficulty.distractor_scale_factor}))
    else:
        transforms = PRESETS.get(preset, PRESETS["easy"])
        instance.difficulty_preset = preset

    for name, params in transforms:
        if name == "query_fuzz":
            instance = _apply_query_fuzz(
                instance, client,
                intensity=params.get("intensity", "light"),
                dry_run=dry_run,
            )
        elif name == "distractor_scale":
            instance = _apply_distractor_scale(
                instance, factor=params.get("factor", 1.0)
            )
        elif name == "temporal_shift":
            instance = _apply_temporal_shift(instance, client=client, dry_run=dry_run)
        elif name == "contra_placement":
            instance = _apply_contra_placement(instance)

    return instance


def harden_all(
    instances: List[MemGymIRInstance],
    preset: str = "easy",
    client: Optional[LLMClient] = None,
    show_progress: bool = True,
    dry_run: bool = False,
    difficulty: Optional[object] = None,
    max_concurrent: int = 10,
) -> List[MemGymIRInstance]:
    """Apply difficulty transforms to all instances."""
    label = preset if difficulty is None else "custom"

    if not dry_run and max_concurrent > 1:
        hardened = _harden_all_async(
            instances, preset, client, difficulty, max_concurrent,
        )
    else:
        hardened = []
        iterator = tqdm(instances, desc=f"Hardening ({label})") if show_progress else instances
        for inst in iterator:
            result = harden_instance(
                inst, preset=preset, client=client, dry_run=dry_run,
                difficulty=difficulty,
            )
            hardened.append(result)

    print(f"  Hardened {len(hardened)} instances with preset={label}")
    return hardened


# -------------------------------------------------------------------------
# Async API
# -------------------------------------------------------------------------


async def _apply_query_fuzz_async(instance, client, intensity="light"):
    """Async query fuzz."""
    prompt = QUERY_FUZZ_PROMPT.format(
        question=instance.question, answer=instance.answer, intensity=intensity,
    )
    result = await client.get_json_completion_async(
        prompt=prompt, response_format=QUERY_FUZZ_SCHEMA, temperature=0.7,
    )
    fuzzed = result.get("fuzzed_question", "")
    if fuzzed and len(fuzzed) > 10:
        instance.question = fuzzed
    instance.transforms_applied.append("query_fuzz")
    return instance


async def _apply_temporal_shift_async(instance, client):
    """Async temporal shift — generates shifted versions for ALL fact types."""
    from ..llm.prompts import TEMPORAL_SHIFT_PROMPT, BRIDGE_SHIFT_PROMPT, DISTRACTOR_SCHEMA

    if not instance.grounding_facts:
        instance.transforms_applied.append("temporal_shift")
        return instance

    num_turns = len(instance.turns)
    generated = 0

    for fact in instance.grounding_facts:
        if fact.fact_type in ("temporal_fact", "numerical_fact"):
            prompt = TEMPORAL_SHIFT_PROMPT.format(
                entity=fact.source_paragraph_title, fact=fact.content,
            )
        else:
            prompt = BRIDGE_SHIFT_PROMPT.format(
                entity=fact.source_paragraph_title, fact=fact.content,
                fact_type=fact.fact_type,
            )

        try:
            result = await client.get_json_completion_async(
                prompt=prompt, response_format=DISTRACTOR_SCHEMA, temperature=0.7,
            )
        except Exception:
            continue

        text = result.get("distractor_text", "")
        if not text:
            continue

        needed_turn = min(fact.needed_at_hop, num_turns - 1)
        target_turn = max(0, needed_turn - 1) if needed_turn > 0 else 0

        instance.turns[target_turn].documents.append({
            "title": f"Shifted: {fact.source_paragraph_title}",
            "text": text, "is_supporting": False,
        })
        instance.distractors.append(DistractorFactIR(
            id=f"D_shifted_{fact.fact_id}", content=text,
            source="same_entity", target_turn=target_turn,
            interferes_with=fact.fact_id,
        ))
        generated += 1

    instance.transforms_applied.append(f"temporal_shift({generated})")
    return instance


async def harden_instance_async(instance, preset="easy", client=None, difficulty=None):
    """Async version of harden_instance."""
    if difficulty is not None:
        instance.difficulty_preset = preset
        transforms = []
        if difficulty.query_fuzz_intensity != "none":
            transforms.append(("query_fuzz", {"intensity": difficulty.query_fuzz_intensity}))
        if difficulty.enable_contra_placement:
            transforms.append(("contra_placement", {}))
        if difficulty.enable_temporal_shift:
            transforms.append(("temporal_shift", {}))
        if difficulty.distractor_scale_factor > 1.0:
            transforms.append(("distractor_scale", {"factor": difficulty.distractor_scale_factor}))
    else:
        transforms = PRESETS.get(preset, PRESETS["easy"])
        instance.difficulty_preset = preset

    for name, params in transforms:
        if name == "query_fuzz" and client:
            instance = await _apply_query_fuzz_async(
                instance, client, intensity=params.get("intensity", "light"),
            )
        elif name == "distractor_scale":
            instance = _apply_distractor_scale(instance, factor=params.get("factor", 1.0))
        elif name == "temporal_shift" and client:
            instance = await _apply_temporal_shift_async(instance, client)
        elif name == "contra_placement":
            instance = _apply_contra_placement(instance)
        elif name == "query_fuzz":
            instance.transforms_applied.append("query_fuzz")
        elif name == "temporal_shift":
            instance.transforms_applied.append("temporal_shift")

    return instance


def _harden_all_async(instances, preset, client, difficulty, max_concurrent):
    """Run hardening in parallel using asyncio."""
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc="Hardening (async)")

        async def _one(inst):
            async with sem:
                result = await harden_instance_async(
                    inst, preset=preset, client=client, difficulty=difficulty,
                )
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()
        return list(results)

    return asyncio.run(_run())
