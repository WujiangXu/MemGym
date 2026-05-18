"""Parse unified diff patches to extract metrics."""

import re
from typing import List
from dataclasses import dataclass
from unidiff import PatchSet


@dataclass
class PatchMetrics:
    """Metrics extracted from a unified diff patch."""
    lines_added: int
    lines_removed: int
    lines_changed: int
    num_hunks: int
    files_changed: List[str]


def _parse_patch_simple(patch_text: str) -> PatchMetrics:
    """
    Simple fallback parser that counts lines by prefix.
    Used when unidiff fails to parse.
    """
    lines = patch_text.split('\n')
    lines_added = 0
    lines_removed = 0
    num_hunks = 0
    files_changed = []

    for line in lines:
        if line.startswith('@@'):
            num_hunks += 1
        elif line.startswith('+') and not line.startswith('+++'):
            lines_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_removed += 1
        elif line.startswith('--- a/') or line.startswith('--- '):
            # Extract file path from --- line
            match = re.match(r'^--- (?:a/)?(.+)$', line)
            if match:
                path = match.group(1).strip()
                if path and path not in files_changed and path != '/dev/null':
                    files_changed.append(path)
        elif line.startswith('+++ b/') or line.startswith('+++ '):
            # Extract file path from +++ line (for new files)
            match = re.match(r'^\+\+\+ (?:b/)?(.+)$', line)
            if match:
                path = match.group(1).strip()
                if path and path not in files_changed and path != '/dev/null':
                    files_changed.append(path)

    return PatchMetrics(
        lines_added=lines_added,
        lines_removed=lines_removed,
        lines_changed=lines_added + lines_removed,
        num_hunks=num_hunks,
        files_changed=files_changed
    )


def parse_patch(patch_text: str) -> PatchMetrics:
    """
    Parse a unified diff patch and extract metrics.

    Args:
        patch_text: The unified diff patch as a string

    Returns:
        PatchMetrics with counts of lines added, removed, hunks, and files changed
    """
    if not patch_text or not patch_text.strip():
        return PatchMetrics(
            lines_added=0,
            lines_removed=0,
            lines_changed=0,
            num_hunks=0,
            files_changed=[]
        )

    try:
        patch_set = PatchSet(patch_text)

        lines_added = 0
        lines_removed = 0
        num_hunks = 0
        files_changed = []

        for patched_file in patch_set:
            # Track file path
            file_path = patched_file.path
            if file_path and file_path not in files_changed:
                files_changed.append(file_path)

            # Count hunks and lines
            for hunk in patched_file:
                num_hunks += 1
                lines_added += hunk.added
                lines_removed += hunk.removed

        return PatchMetrics(
            lines_added=lines_added,
            lines_removed=lines_removed,
            lines_changed=lines_added + lines_removed,
            num_hunks=num_hunks,
            files_changed=files_changed
        )
    except Exception:
        # Fall back to simple parser if unidiff fails
        return _parse_patch_simple(patch_text)


def meets_filter_criteria(
    metrics: PatchMetrics,
    problem_statement: str,
    fail_to_pass_tests: List[str],
    min_lines_changed: int = 8,
    min_fail_to_pass: int = 2,
    min_hunks: int = 2,
    min_problem_length: int = 200,
    max_lines_changed: int = 0,
) -> bool:
    """
    Check if an instance meets the filtering criteria.

    Args:
        metrics: PatchMetrics from the patch
        problem_statement: The problem description
        fail_to_pass_tests: List of tests that should pass after fix
        min_lines_changed: Minimum total lines changed
        min_fail_to_pass: Minimum number of FAIL_TO_PASS tests
        min_hunks: Minimum number of distinct hunks
        min_problem_length: Minimum problem statement length
        max_lines_changed: Maximum total lines changed (0 = no limit)

    Returns:
        True if all criteria are met
    """
    if metrics.lines_changed < min_lines_changed:
        return False
    if max_lines_changed > 0 and metrics.lines_changed > max_lines_changed:
        return False
    if len(fail_to_pass_tests) < min_fail_to_pass:
        return False
    if metrics.num_hunks < min_hunks:
        return False
    if len(problem_statement) < min_problem_length:
        return False
    return True
