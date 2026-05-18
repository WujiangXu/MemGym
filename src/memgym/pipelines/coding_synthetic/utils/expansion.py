"""Shared expansion primitives for distractor injection and token counting.

Consolidated from expand.py and distractor_scale.py to avoid duplication.
"""

import random
from typing import Dict, List

from ..types.schemas import DistractorFact
from .token_counter import count_tokens


def inject_distractors_into_files(
    memory_files: Dict[str, str],
    distractors: List[DistractorFact],
) -> Dict[str, str]:
    """Inject distractor content into memory files at paragraph boundaries.

    Round-robins across files. No LLM call needed — distractor content is
    already well-formed prose from adversarial generation.
    """
    updated = dict(memory_files)
    filenames = list(updated.keys())
    if not filenames:
        return updated

    for i, d in enumerate(distractors):
        target_file = filenames[i % len(filenames)]
        content = updated[target_file]
        paragraphs = content.split("\n\n")
        insert_pos = random.randint(1, max(1, len(paragraphs) - 1))
        snippet = f"Additionally, {d.content}"
        paragraphs.insert(insert_pos, snippet)
        updated[target_file] = "\n\n".join(paragraphs)

    return updated


def count_base_tokens(
    task_prompt: str,
    memory_files: Dict[str, str],
    repo_context: str,
    grounding_facts: list,
    distractors: list,
) -> int:
    """Count total tokens across all base instance fields (excluding repo_files).

    Works for both MemGymInstance and CodingQAInstance by accepting raw fields.
    """
    total = count_tokens(task_prompt)
    for content in memory_files.values():
        total += count_tokens(content)
    total += count_tokens(repo_context or "")
    for f in grounding_facts:
        content = f.content if hasattr(f, "content") else f.get("content", "")
        total += count_tokens(content)
    for d in distractors:
        content = d.content if hasattr(d, "content") else d.get("content", "")
        total += count_tokens(content)
    return total
