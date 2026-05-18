"""Stage 2: Split -- Extract grounding facts from supporting paragraphs.

Pass 1: Structural analysis (no LLM) -- parse decomposition, identify dependencies.
Pass 2: Fact extraction (Worker LLM) -- extract facts, classify types.
Pass 3: Verifier cross-check -- verify facts are supported by paragraphs.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm

from ..types.schemas import FilteredIRInstance, GroundingFactIR
from ..llm.client import LLMClient, VerifierClient
from ..llm.prompts import (
    FACT_EXTRACTION_PROMPT,
    FACT_EXTRACTION_SCHEMA,
    FACT_VERIFICATION_PROMPT,
)


def _structural_analysis(instance: FilteredIRInstance) -> Dict:
    """Pass 1: Parse decomposition to identify hop dependencies (no LLM)."""
    decomposition = instance.decomposition
    num_hops = len(decomposition)

    # Build dependency graph: which hops feed into later hops
    dependencies = []
    for i, hop in enumerate(decomposition):
        sub_q = hop.get("sub_question", "")
        # Check if this sub-question references a previous answer via #N pattern
        refs = []
        for j in range(i):
            ref_marker = f"#{j+1}"
            if ref_marker in sub_q:
                refs.append(j)
        dependencies.append({
            "hop_index": i,
            "depends_on": refs,
            "sub_question": sub_q,
            "sub_answer": hop.get("sub_answer", ""),
            "paragraph_title": hop.get("paragraph_title", ""),
        })

    # Determine fact availability: each hop's paragraph is first available at that hop
    availability = {}
    for i, hop in enumerate(decomposition):
        title = hop.get("paragraph_title", "")
        if title and title not in availability:
            availability[title] = i

    return {
        "num_hops": num_hops,
        "dependencies": dependencies,
        "paragraph_availability": availability,
    }


def _extract_facts_llm(
    instance: FilteredIRInstance,
    structure: Dict,
    worker_client: LLMClient,
) -> List[Dict]:
    """Pass 2: Use Worker LLM to extract grounding facts."""
    decomposition_str = ""
    for i, hop in enumerate(instance.decomposition):
        para_title = hop.get("paragraph_title", "N/A")
        para_text = hop.get("paragraph_text", "")[:500]
        decomposition_str += (
            f"Hop {i}: Q: {hop.get('sub_question', '')}\n"
            f"  A: {hop.get('sub_answer', '')}\n"
            f"  Supporting paragraph: [{para_title}] {para_text}\n\n"
        )

    prompt = FACT_EXTRACTION_PROMPT.format(
        question=instance.question,
        answer=instance.answer,
        decomposition=decomposition_str,
    )

    result = worker_client.get_json_completion(
        prompt=prompt,
        response_format=FACT_EXTRACTION_SCHEMA,
        temperature=0.3,
    )

    return result.get("facts", [])


def _verify_facts(
    facts: List[Dict],
    instance: FilteredIRInstance,
    verifier_client: VerifierClient,
) -> List[Dict]:
    """Pass 3: Verifier cross-checks facts against paragraphs."""
    if not facts:
        return facts

    facts_str = json.dumps(facts, indent=2)
    paragraphs_str = ""
    for p in instance.supporting_paragraphs:
        paragraphs_str += f"[{p['title']}]\n{p['text']}\n\n"

    prompt = FACT_VERIFICATION_PROMPT.format(
        facts=facts_str,
        paragraphs=paragraphs_str,
    )

    result = verifier_client.get_json_completion(prompt=prompt, temperature=0.0)
    verifications = result.get("verifications", [])

    # Update facts with verification results
    verification_map = {v["fact_id"]: v for v in verifications}
    for fact in facts:
        v = verification_map.get(fact["fact_id"], {})
        fact["is_discoverable"] = v.get("is_discoverable", False)
        if not v.get("supported_by_paragraph", True):
            fact["_unsupported"] = True

    # Remove unsupported facts
    facts = [f for f in facts if not f.get("_unsupported", False)]
    return facts


def extract_grounding_facts(
    instance: FilteredIRInstance,
    worker_client: LLMClient,
    verifier_client: VerifierClient,
    dry_run: bool = False,
) -> Dict:
    """Extract grounding facts for a single instance.

    Returns dict with instance_id, facts, structure, quality_gate_passed.
    """
    structure = _structural_analysis(instance)

    if dry_run:
        # Build facts from decomposition without LLM
        # Bridge facts are available at their hop but needed at a later hop
        facts = []
        num_hops = len(instance.decomposition)
        for i, hop in enumerate(instance.decomposition):
            is_last = i == num_hops - 1
            fact_type = "terminal_fact" if is_last else "bridge_fact"
            # Bridge facts are first available at hop i, needed at the final hop
            needed_at = num_hops - 1 if not is_last else i
            facts.append({
                "fact_id": f"F{i+1}",
                "content": f"{hop.get('sub_question', '')} -> {hop.get('sub_answer', '')}",
                "fact_type": fact_type,
                "source_paragraph_title": hop.get("paragraph_title", ""),
                "needed_at_hop": needed_at,
                "first_available_at_hop": i,
                "intermediate_answer": hop.get("sub_answer", ""),
                "is_discoverable": False,
            })
    else:
        raw_facts = _extract_facts_llm(instance, structure, worker_client)
        facts = _verify_facts(raw_facts, instance, verifier_client)

    # Fix needed_at_hop using structural dependency graph:
    # A bridge fact at hop i is needed by the latest hop that depends on hop i.
    # The LLM often sets needed_at_hop = first_available_at_hop (span=0), which is wrong.
    dep_graph = structure.get("dependencies", [])
    # Build reverse map: hop i -> latest hop that depends on i
    needed_by = {}
    for dep in dep_graph:
        for ref in dep.get("depends_on", []):
            # hop dep["hop_index"] depends on hop ref
            needed_by[ref] = max(needed_by.get(ref, 0), dep["hop_index"])

    num_hops = structure.get("num_hops", len(instance.decomposition))

    # Compute retention_span and build GroundingFactIR objects
    grounding_facts = []
    for f in facts:
        available = f.get("first_available_at_hop", 0)
        fact_type = f.get("fact_type", "bridge_fact")

        if fact_type in ("bridge_fact",) and available in needed_by:
            # Use structural dependency: bridge fact is needed at the downstream hop
            needed = needed_by[available]
        elif fact_type == "terminal_fact":
            # Terminal facts are needed at the final hop
            needed = num_hops - 1
        else:
            # Fallback: bridge facts without explicit dependency are needed at final hop
            needed = max(f.get("needed_at_hop", 0), min(available + 1, num_hops - 1))

        retention_span = max(0, needed - available)
        grounding_facts.append(GroundingFactIR(
            fact_id=f["fact_id"],
            content=f["content"],
            fact_type=fact_type,
            source_paragraph_title=f.get("source_paragraph_title", ""),
            needed_at_hop=needed,
            first_available_at_hop=available,
            retention_span=retention_span,
            is_discoverable=f.get("is_discoverable", False),
            intermediate_answer=f.get("intermediate_answer", ""),
        ))

    # Quality gate
    facts_with_span = [f for f in grounding_facts if f.retention_span >= 1]
    bridge_facts = [f for f in grounding_facts if f.fact_type == "bridge_fact"]
    non_discoverable = [f for f in grounding_facts if not f.is_discoverable]

    quality_gate_passed = (
        len(facts_with_span) >= 1
        and len(bridge_facts) >= 1
        and len(non_discoverable) >= 1
    )

    return {
        "instance_id": instance.instance_id,
        "facts": [f.model_dump() for f in grounding_facts],
        "structure": structure,
        "quality_gate_passed": quality_gate_passed,
        "num_facts": len(grounding_facts),
        "num_facts_with_span": len(facts_with_span),
        "num_bridge_facts": len(bridge_facts),
    }


async def _extract_facts_llm_async(instance, structure, worker_client):
    decomposition_str = ""
    for i, hop in enumerate(instance.decomposition):
        para_title = hop.get("paragraph_title", "N/A")
        para_text = hop.get("paragraph_text", "")[:500]
        decomposition_str += (
            f"Hop {i}: Q: {hop.get('sub_question', '')}\n"
            f"  A: {hop.get('sub_answer', '')}\n"
            f"  Supporting paragraph: [{para_title}] {para_text}\n\n"
        )
    prompt = FACT_EXTRACTION_PROMPT.format(
        question=instance.question, answer=instance.answer, decomposition=decomposition_str,
    )
    result = await worker_client.get_json_completion_async(
        prompt=prompt, response_format=FACT_EXTRACTION_SCHEMA, temperature=0.3,
    )
    return result.get("facts", [])


async def _verify_facts_async(facts, instance, verifier_client):
    if not facts:
        return facts
    facts_str = json.dumps(facts, indent=2)
    paragraphs_str = ""
    for p in instance.supporting_paragraphs:
        paragraphs_str += f"[{p['title']}]\n{p['text']}\n\n"
    prompt = FACT_VERIFICATION_PROMPT.format(facts=facts_str, paragraphs=paragraphs_str)
    result = await verifier_client.get_json_completion_async(prompt=prompt, temperature=0.0)
    verifications = result.get("verifications", [])
    verification_map = {v["fact_id"]: v for v in verifications}
    for fact in facts:
        v = verification_map.get(fact["fact_id"], {})
        fact["is_discoverable"] = v.get("is_discoverable", False)
        if not v.get("supported_by_paragraph", True):
            fact["_unsupported"] = True
    return [f for f in facts if not f.get("_unsupported", False)]


async def extract_grounding_facts_async(instance, worker_client, verifier_client):
    """Async version of extract_grounding_facts."""
    structure = _structural_analysis(instance)
    raw_facts = await _extract_facts_llm_async(instance, structure, worker_client)
    facts = await _verify_facts_async(raw_facts, instance, verifier_client)

    # Same post-processing as sync version
    dep_graph = structure.get("dependencies", [])
    needed_by = {}
    for dep in dep_graph:
        for ref in dep.get("depends_on", []):
            needed_by[ref] = max(needed_by.get(ref, 0), dep["hop_index"])

    num_hops = structure.get("num_hops", len(instance.decomposition))
    grounding_facts = []
    for f in facts:
        available = f.get("first_available_at_hop", 0)
        fact_type = f.get("fact_type", "bridge_fact")
        if fact_type in ("bridge_fact",) and available in needed_by:
            needed = needed_by[available]
        elif fact_type == "terminal_fact":
            needed = num_hops - 1
        else:
            needed = max(f.get("needed_at_hop", 0), min(available + 1, num_hops - 1))
        retention_span = max(0, needed - available)
        grounding_facts.append(GroundingFactIR(
            fact_id=f["fact_id"], content=f["content"], fact_type=fact_type,
            source_paragraph_title=f.get("source_paragraph_title", ""),
            needed_at_hop=needed, first_available_at_hop=available,
            retention_span=retention_span, is_discoverable=f.get("is_discoverable", False),
            intermediate_answer=f.get("intermediate_answer", ""),
        ))

    facts_with_span = [f for f in grounding_facts if f.retention_span >= 1]
    bridge_facts = [f for f in grounding_facts if f.fact_type == "bridge_fact"]
    non_discoverable = [f for f in grounding_facts if not f.is_discoverable]
    quality_gate_passed = (
        len(facts_with_span) >= 1 and len(bridge_facts) >= 1 and len(non_discoverable) >= 1
    )
    return {
        "instance_id": instance.instance_id,
        "facts": [f.model_dump() for f in grounding_facts],
        "structure": structure,
        "quality_gate_passed": quality_gate_passed,
        "num_facts": len(grounding_facts),
        "num_facts_with_span": len(facts_with_span),
        "num_bridge_facts": len(bridge_facts),
    }


def extract_all(
    instances: List[FilteredIRInstance],
    worker_client: LLMClient,
    verifier_client: VerifierClient,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    dry_run: bool = False,
    max_concurrent: int = 10,
) -> List[Dict]:
    """Extract grounding facts for all instances."""
    if dry_run or max_concurrent <= 1:
        results = []
        iterator = tqdm(instances, desc="Extracting facts") if show_progress else instances
        for inst in iterator:
            result = extract_grounding_facts(
                inst, worker_client, verifier_client, dry_run=dry_run
            )
            results.append(result)
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
    else:
        results = _extract_all_async(
            instances, worker_client, verifier_client, output_path, max_concurrent,
        )

    passed = sum(1 for r in results if r["quality_gate_passed"])
    print(f"  Extraction: {passed}/{len(results)} passed quality gate")
    return results


def _extract_all_async(instances, worker_client, verifier_client, output_path, max_concurrent):
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        pbar = tqdm(total=len(instances), desc="Extracting facts (async)")
        all_results = []

        async def _one(inst):
            async with sem:
                result = await extract_grounding_facts_async(inst, worker_client, verifier_client)
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        pbar.close()

        # Write checkpoint sequentially
        if output_path:
            with open(output_path, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        return list(results)

    return asyncio.run(_run())


def load_extractions(path: Path) -> List[Dict]:
    """Load extractions from JSONL checkpoint."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results
