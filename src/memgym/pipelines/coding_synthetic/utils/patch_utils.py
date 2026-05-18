"""Utilities for manipulating unified diff patches."""

import re
from typing import List


def reverse_patch(patch_text: str) -> str:
    """
    Reverse a unified diff patch (swap added/removed lines).

    In SWE-smith, the patch field is the bug-introducing diff.
    Reversing it produces the fix (buggy -> correct).

    Args:
        patch_text: Unified diff patch string

    Returns:
        Reversed patch string
    """
    if not patch_text or not patch_text.strip():
        return ""

    lines = patch_text.split('\n')
    reversed_lines = []

    for line in lines:
        if line.startswith('--- a/'):
            reversed_lines.append(line.replace('--- a/', '--- b/', 1))
        elif line.startswith('+++ b/'):
            reversed_lines.append(line.replace('+++ b/', '+++ a/', 1))
        elif line.startswith('@@'):
            # Swap the hunk header ranges: @@ -old,count +new,count @@
            match = re.match(r'^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)$', line)
            if match:
                old_range, new_range, rest = match.groups()
                reversed_lines.append(f'@@ -{new_range} +{old_range} @@{rest}')
            else:
                reversed_lines.append(line)
        elif line.startswith('+') and not line.startswith('+++'):
            reversed_lines.append('-' + line[1:])
        elif line.startswith('-') and not line.startswith('---'):
            reversed_lines.append('+' + line[1:])
        else:
            reversed_lines.append(line)

    return '\n'.join(reversed_lines)


def extract_files_from_patch(patch_text: str) -> List[str]:
    """
    Extract file paths modified in a patch.

    Args:
        patch_text: Unified diff patch string

    Returns:
        List of file paths
    """
    files = []
    for line in patch_text.split('\n'):
        if line.startswith('+++ b/'):
            path = line[6:].strip()
            if path and path != '/dev/null' and path not in files:
                files.append(path)
        elif line.startswith('--- a/'):
            path = line[6:].strip()
            if path and path != '/dev/null' and path not in files:
                files.append(path)
    return files
