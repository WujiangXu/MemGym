"""Pipeline stages for Coding-Synthetic generation v2."""

from .filter import filter_instances, load_filtered_instances
from .extract import (
    split_instance, split_all_instances, load_extractions,
    extract_repo_context, extract_all_repo_contexts, load_repo_contexts,
    analyze_repo_context, extract_grounding_facts,
    verify_fact_classifications, passes_quality_gate,
)
from .generate import (
    craft_instance, craft_all_instances,
    craft_and_verify_instance, craft_and_verify_all,
)
from .harden import apply_difficulty, apply_difficulty_all, apply_adaptive_difficulty
from .validate import verify_instance, verify_all_instances, is_accepted

__all__ = [
    # Stage 1: Mine
    "filter_instances",
    "load_filtered_instances",
    # Stage 2: Split
    "split_instance",
    "split_all_instances",
    "load_extractions",
    "extract_repo_context",
    "extract_all_repo_contexts",
    "load_repo_contexts",
    "analyze_repo_context",
    "extract_grounding_facts",
    "verify_fact_classifications",
    "passes_quality_gate",
    # Stage 3: Craft
    "craft_instance",
    "craft_all_instances",
    # Stage 3+4+5: Integrated Craft+Verify
    "craft_and_verify_instance",
    "craft_and_verify_all",
    # Stage 4: Difficulty
    "apply_difficulty",
    "apply_difficulty_all",
    "apply_adaptive_difficulty",
    # Stage 5: Verify
    "verify_instance",
    "verify_all_instances",
    "is_accepted",
]
