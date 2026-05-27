"""Token expansion stage: scale instance context length with distractors and filler.

Expands existing MemGymInstance objects to a target token budget by:
1. Generating additional adversarial distractors and injecting into memory files
2. Expanding memory file prose with filler content (LLM or repo-file sampling)
3. Adding unrelated code context from the same repo

This is a post-hoc operation on already-crafted instances (no re-generation needed).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from ..llm.client import LLMClient
from ..llm.prompts import ADVERSARIAL_DISTRACTOR_PROMPT
from ..types.schemas import (
    MemGymInstance,
    ContextTokens,
    DistractorFact,
    ADVERSARIAL_DISTRACTOR_SCHEMA,
)
from ..utils.token_counter import count_tokens
from ..utils.parallel import parallel_map
from ..utils.expansion import inject_distractors_into_files, count_base_tokens


# Filler-mode names accepted by expand_qa_instance and helpers.
FILLER_MODES = ("local", "llm", "hybrid")


FILLER_EXPANSION_PROMPT = """\
You are expanding a technical debugging document with realistic filler content.
The document is from a developer's notes about investigating a software bug.

Here is the current document content:

---
{content}
---

Add more realistic content to this document to make it longer. Include things like:
- Additional investigation steps that didn't lead anywhere
- Red herring observations about unrelated code behavior
- Verbose logging output excerpts
- Tangential discussions about code architecture
- Notes about environment setup, dependencies, or tooling
- Historical context about past similar issues that turned out to be unrelated

IMPORTANT: Do NOT change or remove any existing content. Only ADD new paragraphs,
sections, or details between or after existing content. Keep the same document style
and tone. Target approximately {target_words} additional words.

Output the full expanded document."""


def _current_tokens(instance: MemGymInstance) -> int:
    """Count total tokens in an instance."""
    return count_base_tokens(
        instance.task_prompt, instance.memory_files,
        instance.repo_context or "", instance.grounding_facts,
        instance.distractors,
    )


def _generate_more_distractors(
    instance: MemGymInstance,
    worker_client: LLMClient,
    num_extra: int,
) -> List[DistractorFact]:
    """Generate additional adversarial distractors."""
    facts_str = "\n".join(f"- {f.content}" for f in instance.grounding_facts)
    # Use existing distractors as negative examples to avoid duplicates
    existing_str = "\n".join(f"- {d.content}" for d in instance.distractors[:5])

    prompt = ADVERSARIAL_DISTRACTOR_PROMPT.format(
        num_distractors=num_extra,
        real_facts=facts_str,
        code_context=f"Existing distractors (do NOT repeat these):\n{existing_str}",
    )

    try:
        result = worker_client.get_json_completion(
            prompt=prompt,
            response_format=ADVERSARIAL_DISTRACTOR_SCHEMA,
            temperature=0.8,
        )
        base_id = len(instance.distractors)
        return [
            DistractorFact(
                id=f"D{base_id + i + 1}",
                content=d.get("content", ""),
                source="adversarial_expansion",
            )
            for i, d in enumerate(result.get("distractors", [])[:num_extra])
        ]
    except Exception:
        return []


def _inject_distractors_into_files(
    memory_files: Dict[str, str],
    distractors: List[DistractorFact],
) -> Dict[str, str]:
    """Inject distractor content into memory files at paragraph boundaries."""
    return inject_distractors_into_files(memory_files, distractors)

    return updated


def _expand_file_prose(
    content: str,
    worker_client: LLMClient,
    target_extra_words: int,
    max_iterations: int = 10,
) -> str:
    """Expand a single memory file with LLM-generated filler.

    Iterates until the target word count is approximately reached,
    up to max_iterations rounds. Each round requests the remaining
    deficit. This enables scaling to 500K+ token targets.
    """
    if target_extra_words < 100:
        return content

    current_content = content
    original_words = len(content.split())
    target_total_words = original_words + target_extra_words

    for iteration in range(max_iterations):
        current_words = len(current_content.split())
        remaining = target_total_words - current_words
        if remaining < 100:
            break

        # Cap per-iteration request to avoid LLM output limits
        request_words = min(remaining, 3000)

        prompt = FILLER_EXPANSION_PROMPT.format(
            content=current_content[-12000:],  # Use tail for context in later iterations
            target_words=request_words,
        )

        try:
            result = worker_client.get_completion(
                prompt=prompt,
                temperature=0.7,
            )
            if result and len(result) > len(current_content):
                current_content = result
            else:
                break  # LLM didn't produce longer output, stop
        except Exception:
            break

    return current_content


def _expand_file_prose_local(
    content: str,
    target_extra_tokens: int,
    filler_pool: List[tuple],
    rng: random.Random,
    exclude_paths: Optional[set] = None,
) -> str:
    """Pad a memory file with real source excerpts instead of LLM filler.

    `filler_pool` is a list of (path, source) tuples — typically built from
    `instance.repo_files` minus the referenced files. Each chosen excerpt is
    framed as a "context excerpt" block so a reviewer can audit what was
    inserted and a memory method can't trivially identify it as machine-written.

    Stops when the running token budget is met or the pool is exhausted.
    """
    if target_extra_tokens < 100 or not filler_pool:
        return content

    exclude_paths = exclude_paths or set()
    candidates = [(p, s) for p, s in filler_pool if p not in exclude_paths and s]
    if not candidates:
        return content

    # Shuffle once so the same instance gets a deterministic-ish layout
    # without re-reading the same file twice in one pass.
    rng.shuffle(candidates)

    chunks = [content]
    used_tokens = 0
    for path, source in candidates:
        if used_tokens >= target_extra_tokens:
            break
        # Optionally split very long files into ~2K-token chunks so we don't
        # overshoot the budget by a single huge file.
        body = source.strip()
        body_tokens = count_tokens(body)
        if body_tokens > 4000:
            # Take a window of ~2K tokens (~8K chars) at a random offset
            max_chars = min(len(body), 8000)
            start = rng.randint(0, max(0, len(body) - max_chars))
            body = body[start:start + max_chars]
            body_tokens = count_tokens(body)
        snippet = (
            f"\n\n# --- context excerpt: {path} ---\n"
            f"{body}\n"
            f"# --- end excerpt ---\n"
        )
        chunks.append(snippet)
        used_tokens += body_tokens

    return "".join(chunks)


def expand_instance(
    instance: MemGymInstance,
    target_tokens: int,
    worker_client: LLMClient,
) -> MemGymInstance:
    """Expand an instance to reach target token count.

    Strategy:
    1. Add more distractors (cheap, effective noise)
    2. Expand memory file prose with filler
    3. Update context_tokens

    Args:
        instance: MemGymInstance to expand
        target_tokens: Target total token count
        worker_client: LLM client for generation

    Returns:
        Expanded MemGymInstance
    """
    current = _current_tokens(instance)
    if current >= target_tokens:
        return instance  # Already at or above target

    deficit = target_tokens - current

    # Phase 1: Add distractors (roughly 50 tokens each, cheap)
    max_new_distractors = min(deficit // 50, 25)  # Cap at 25 extra
    new_distractors = []
    if max_new_distractors >= 3:
        new_distractors = _generate_more_distractors(
            instance, worker_client, max_new_distractors
        )

    # Inject distractors into memory files
    memory_files = dict(instance.memory_files)
    if new_distractors:
        memory_files = _inject_distractors_into_files(memory_files, new_distractors)

    all_distractors = list(instance.distractors) + new_distractors

    # Recount after distractors
    current = count_tokens(instance.task_prompt)
    for content in memory_files.values():
        current += count_tokens(content)
    current += count_tokens(instance.repo_context or "")
    for f in instance.grounding_facts:
        current += count_tokens(f.content)
    for d in all_distractors:
        current += count_tokens(d.content)

    # Phase 2: Expand memory file prose if still below target
    if current < target_tokens:
        remaining_deficit = target_tokens - current
        # Distribute extra words across files
        n_files = len(memory_files)
        if n_files > 0:
            extra_words_per_file = (remaining_deficit // n_files) // 1  # ~1 token per word
            expanded_files = {}
            for fn, content in memory_files.items():
                expanded_files[fn] = _expand_file_prose(
                    content, worker_client, extra_words_per_file
                )
            memory_files = expanded_files

    # Recount final tokens
    tok_task = count_tokens(instance.task_prompt)
    tok_memory = sum(count_tokens(c) for c in memory_files.values())
    tok_repo = count_tokens(instance.repo_context or "")
    tok_facts = sum(count_tokens(f.content) for f in instance.grounding_facts)
    tok_dist = sum(count_tokens(d.content) for d in all_distractors)

    context_tokens = ContextTokens(
        task_prompt=tok_task,
        memory_files=tok_memory,
        repo_context=tok_repo,
        grounding_facts=tok_facts,
        distractors=tok_dist,
        total=tok_task + tok_memory + tok_repo + tok_facts + tok_dist,
    )

    transforms = list(instance.transforms_applied)
    transforms.append(f"token_expand_{target_tokens}")

    return instance.model_copy(
        update={
            "memory_files": memory_files,
            "distractors": all_distractors,
            "context_tokens": context_tokens,
            "transforms_applied": transforms,
        }
    )


def expand_all_instances(
    instances: List[MemGymInstance],
    target_tokens: int,
    worker_client: LLMClient,
    output_path: Optional[Path] = None,
    num_workers: int = 4,
    show_progress: bool = True,
) -> List[MemGymInstance]:
    """Expand all instances to target token count.

    Args:
        instances: List of MemGymInstance objects
        target_tokens: Target total token count per instance
        worker_client: LLM client
        output_path: Optional path to write expanded instances
        num_workers: Parallel workers
        show_progress: Show progress bar

    Returns:
        List of expanded instances
    """
    # Filter to only instances below target
    to_expand = [inst for inst in instances if _current_tokens(inst) < target_tokens]
    already_ok = [inst for inst in instances if _current_tokens(inst) >= target_tokens]

    if not to_expand:
        print(f"  All {len(instances)} instances already at or above {target_tokens} tokens")
        return instances

    print(f"  Expanding {len(to_expand)} instances to {target_tokens} tokens "
          f"({len(already_ok)} already at target)")

    def _expand(inst):
        return expand_instance(inst, target_tokens, worker_client)

    expanded = parallel_map(
        _expand,
        to_expand,
        num_workers=num_workers,
        desc=f"Expanding to {target_tokens} tokens",
        output_path=output_path,
        serialize_fn=lambda r: json.dumps(r.to_jsonl_dict()),
        show_progress=show_progress,
        item_id_fn=lambda inst: inst.instance_id,
    )

    return already_ok + expanded
