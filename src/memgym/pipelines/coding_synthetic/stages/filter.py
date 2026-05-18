"""
Step A: Filter SWE-smith instances based on complexity criteria.

Filters instances to keep only those that are suitable for multi-turn
memory benchmarks - complex enough to require information across turns.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from datasets import load_dataset
from tqdm import tqdm

from ..utils.diff_parser import parse_patch, meets_filter_criteria
from ..types.schemas import FilteredInstance, PatchMetrics


def load_swe_smith(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load SWE-smith dataset from HuggingFace.

    Args:
        limit: Optional limit on number of instances to load

    Returns:
        List of instance dictionaries
    """
    dataset = load_dataset("SWE-bench/SWE-smith", split="train")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return list(dataset)


def parse_test_list(test_str: str) -> List[str]:
    """Parse test string into list of test names."""
    if not test_str:
        return []
    if isinstance(test_str, list):
        return test_str
    # Handle JSON-encoded lists
    if test_str.startswith('['):
        try:
            return json.loads(test_str)
        except json.JSONDecodeError:
            pass
    # Handle space or newline separated
    return [t.strip() for t in test_str.replace('\n', ' ').split() if t.strip()]


def filter_instance(
    instance: Dict[str, Any],
    min_lines_changed: int = 8,
    min_fail_to_pass: int = 2,
    min_hunks: int = 2,
    min_problem_length: int = 200,
    max_lines_changed: int = 500,
) -> Optional[FilteredInstance]:
    """
    Filter a single instance based on complexity criteria.

    Args:
        instance: Raw SWE-smith instance
        min_lines_changed: Minimum total lines changed in patch
        min_fail_to_pass: Minimum number of FAIL_TO_PASS tests
        min_hunks: Minimum number of distinct hunks in patch
        min_problem_length: Minimum problem statement length
        max_lines_changed: Maximum total lines changed (0 = no limit)

    Returns:
        FilteredInstance if criteria met, None otherwise
    """
    # Extract fields
    instance_id = instance.get("instance_id", "")
    problem_statement = instance.get("problem_statement", "")
    patch = instance.get("patch", "")
    fail_to_pass = parse_test_list(instance.get("FAIL_TO_PASS", ""))
    pass_to_pass = parse_test_list(instance.get("PASS_TO_PASS", ""))

    # Parse patch metrics
    metrics = parse_patch(patch)

    # Check filter criteria
    if not meets_filter_criteria(
        metrics=metrics,
        problem_statement=problem_statement,
        fail_to_pass_tests=fail_to_pass,
        min_lines_changed=min_lines_changed,
        min_fail_to_pass=min_fail_to_pass,
        min_hunks=min_hunks,
        min_problem_length=min_problem_length,
        max_lines_changed=max_lines_changed,
    ):
        return None

    # Build image name if not present
    image_name = instance.get("image_name", "")
    if not image_name:
        # Docker doesn't allow double underscore
        id_docker = instance_id.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()

    # Create FilteredInstance
    return FilteredInstance(
        instance_id=instance_id,
        problem_statement=problem_statement,
        patch=patch,
        patch_metrics=PatchMetrics(
            lines_added=metrics.lines_added,
            lines_removed=metrics.lines_removed,
            lines_changed=metrics.lines_changed,
            num_hunks=metrics.num_hunks,
            files_changed=metrics.files_changed
        ),
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        image_name=image_name,
        repo=instance.get("repo", ""),
        base_commit=instance.get("base_commit", "")
    )


def filter_instances(
    limit: Optional[int] = None,
    min_lines_changed: int = 8,
    min_fail_to_pass: int = 2,
    min_hunks: int = 2,
    min_problem_length: int = 200,
    max_lines_changed: int = 500,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
) -> List[FilteredInstance]:
    """
    Filter SWE-smith instances based on complexity criteria.

    Args:
        limit: Optional limit on instances to process
        min_lines_changed: Minimum total lines changed
        min_fail_to_pass: Minimum FAIL_TO_PASS tests
        min_hunks: Minimum distinct hunks
        min_problem_length: Minimum problem statement length
        output_path: Optional path to save filtered instances
        show_progress: Whether to show progress bar

    Returns:
        List of FilteredInstance objects that meet criteria
    """
    # Load dataset
    print(f"Loading SWE-smith dataset...")
    instances = load_swe_smith(limit=limit)
    print(f"Loaded {len(instances)} instances")

    # Filter instances
    filtered = []
    iterator = tqdm(instances, desc="Filtering") if show_progress else instances

    for instance in iterator:
        result = filter_instance(
            instance,
            min_lines_changed=min_lines_changed,
            min_fail_to_pass=min_fail_to_pass,
            min_hunks=min_hunks,
            min_problem_length=min_problem_length,
            max_lines_changed=max_lines_changed,
        )
        if result:
            filtered.append(result)

    print(f"Filtered to {len(filtered)} instances ({len(filtered)/len(instances)*100:.1f}%)")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for instance in filtered:
                f.write(json.dumps(instance.model_dump()) + "\n")
        print(f"Saved to {output_path}")

    return filtered


def load_filtered_instances(path: Path) -> List[FilteredInstance]:
    """Load filtered instances from JSONL file."""
    instances = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            instances.append(FilteredInstance(**data))
    return instances
