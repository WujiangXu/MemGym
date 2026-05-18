"""
Stage 3 — Craft: Generate task prompts, distractors, and memory files.

Replaces the v1 conversation generation with:
  1. Task prompt crafting (vague bug report)
  2. Distractor generation (cross-instance, same-repo, adversarial)
  3. Memory file assembly (natural documents)
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from tqdm import tqdm

from ..types.schemas import (
    FilteredInstance,
    GroundingFact,
    DistractorFact,
    ContextTokens,
    MemGymInstance,
    TASK_PROMPT_SCHEMA,
    ADVERSARIAL_DISTRACTOR_SCHEMA,
    SAME_FUNCTION_DISTRACTOR_SCHEMA,
    MEMORY_FILE_ASSEMBLY_SCHEMA,
)
from ..llm.client import LLMClient
from ..llm.prompts import (
    TASK_PROMPT_CRAFT_PROMPT,
    ADVERSARIAL_DISTRACTOR_PROMPT,
    SAME_FUNCTION_DISTRACTOR_PROMPT,
    MEMORY_FILE_ASSEMBLY_PROMPT,
)
from ..utils.patch_utils import reverse_patch, extract_files_from_patch
from ..utils.token_counter import count_tokens
from .extract import synthesize_repo_context


# ---------------------------------------------------------------------------
# Task prompt generation
# ---------------------------------------------------------------------------

def _check_task_prompt_leaks(
    task_prompt: str,
    facts: List[Dict],
    patch: str,
) -> List[str]:
    """Check if task prompt leaks specific code identifiers from facts or patch.

    Only flags identifiers that look like code symbols (contain underscore,
    dots, or CamelCase) — NOT generic domain vocabulary like "oauth", "token".

    Returns list of leaked keywords found.
    """
    leaked = []
    prompt_lower = task_prompt.lower()

    # Check critical external facts for code-specific identifier leaks
    critical_external = [
        f for f in facts
        if not f.get("discoverable", True) and f.get("relevance") == "critical"
    ]

    for fact in critical_external:
        content = fact.get("content", "")
        # Only match identifiers that are clearly code symbols:
        # 1. snake_case or dotted.names (contain _ or .)
        code_idents = re.findall(r'\b[a-zA-Z_]\w*[_.]\w+\b', content)
        # 2. CamelCase class/function names (e.g., DeviceAuthorizationEndpoint)
        camel_idents = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+){1,}\b', content)

        all_idents = set()
        for ident in code_idents + camel_idents:
            all_idents.add(ident.lower())

        for ident in all_idents:
            if ident in prompt_lower:
                leaked.append(ident)

    # Check for file basenames from the patch (e.g., "device_authorization.py")
    patch_files = extract_files_from_patch(patch)
    for fp in patch_files:
        basename = fp.split("/")[-1].lower()
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        if len(stem) > 6 and stem in prompt_lower:
            leaked.append(basename)

    return list(set(leaked))


def craft_task_prompt(
    instance: FilteredInstance,
    facts: List[Dict],
    worker_client: LLMClient,
) -> Optional[str]:
    """Rewrite problem_statement into a vague task prompt.

    Rules enforced in prompt:
    - Describe symptom, not root cause
    - Reference general code area, not exact functions
    - Omit all critical external facts
    - Sound like a real user request

    Args:
        instance: FilteredInstance
        facts: Grounding facts for this instance
        worker_client: Worker LLM client

    Returns:
        Vague task prompt string, or None if crafting fails
    """
    # Collect critical external facts that must be omitted from the prompt
    critical_external = [
        f
        for f in facts
        if not f.get("discoverable", True) and f.get("relevance") == "critical"
    ]
    critical_str = "\n".join(
        f"- {f['content']}" for f in critical_external
    ) or "- (none identified)"

    base_prompt = TASK_PROMPT_CRAFT_PROMPT.format(
        problem_statement=instance.problem_statement[:2000],
        critical_external_facts=critical_str,
    )

    # Try up to 3 times with leak checking
    for attempt in range(3):
        prompt = base_prompt
        temperature = 0.7 + (attempt * 0.1)  # increase randomness on retries

        try:
            result = worker_client.get_json_completion(
                prompt=prompt,
                response_format=TASK_PROMPT_SCHEMA,
                temperature=temperature,
            )
            task_prompt = result.get("task_prompt", "")
            if not task_prompt or len(task_prompt) <= 20:
                continue

            # Check for keyword leaks
            leaks = _check_task_prompt_leaks(
                task_prompt, facts, instance.patch
            )
            if not leaks:
                return task_prompt

            # Leaks found — retry with explicit removal instruction
            print(f"  Attempt {attempt + 1}: leaked keywords {leaks}, retrying...")
            base_prompt = (
                TASK_PROMPT_CRAFT_PROMPT.format(
                    problem_statement=instance.problem_statement[:2000],
                    critical_external_facts=critical_str,
                )
                + f"\n\nIMPORTANT: Your previous output leaked these terms: "
                f"{', '.join(leaks)}. You MUST NOT include these or similar "
                f"technical identifiers in the task prompt."
            )
        except Exception as e:
            print(f"  Task prompt crafting attempt {attempt + 1} failed: {e}")

    # All attempts failed — reject the instance
    print(f"  Task prompt crafting failed after 3 attempts for {instance.instance_id}")
    return None


# ---------------------------------------------------------------------------
# Distractor generation
# ---------------------------------------------------------------------------

def _generate_cross_instance_distractors(
    facts: List[Dict],
    all_facts_pool: List[Dict],
    num: int = 2,
) -> List[Dict]:
    """Sample real facts from other instances as distractors (no LLM).

    Args:
        facts: Current instance's facts (to exclude)
        all_facts_pool: Facts from all other instances in the batch
        num: Number to sample

    Returns:
        List of distractor dicts
    """
    # Filter out facts that are too similar to current instance
    current_contents = {f.get("content", "").lower()[:50] for f in facts}
    candidates = [
        f
        for f in all_facts_pool
        if f.get("content", "").lower()[:50] not in current_contents
    ]

    sampled = random.sample(candidates, min(num, len(candidates)))
    return [
        {
            "id": f"D_cross_{i + 1}",
            "content": s.get("content", ""),
            "source": "cross_instance",
        }
        for i, s in enumerate(sampled)
    ]


def _generate_same_repo_distractors(
    facts: List[Dict],
    repo_context: Dict[str, str],
    num: int = 2,
) -> List[Dict]:
    """Extract facts about the same repo but different modules (no LLM).

    Extracts actual docstrings from unrelated functions to create more
    realistic distractors (not just "function X may be relevant").

    Args:
        facts: Current instance's facts
        repo_context: Dict mapping file_path -> content
        num: Number to generate

    Returns:
        List of distractor dicts
    """
    # Get files already involved in the fix
    involved_files = set()
    for f in facts:
        content = f.get("content", "")
        for fp in repo_context:
            if fp.split("/")[-1] in content or fp in content:
                involved_files.add(fp)

    distractors = []
    for fp, content in repo_context.items():
        if fp in involved_files:
            continue
        # Extract functions with docstrings for richer distractors
        func_blocks = re.finditer(
            r'def\s+(\w+)\s*\([^)]*\).*?:\s*\n(\s+""".*?"""|\s+\'\'\'.*?\'\'\')',
            content, re.DOTALL,
        )
        for match in func_blocks:
            func_name = match.group(1)
            docstring = (
                match.group(2).strip().strip('"""').strip("'''").strip()
            )
            if len(docstring) > 20:
                distractors.append({
                    "id": f"D_repo_{len(distractors) + 1}",
                    "content": (
                        f"While investigating, found that `{func_name}` in "
                        f"`{fp.split('/')[-1]}` {docstring[:150]}. "
                        f"This may be related to the observed behavior."
                    ),
                    "source": "same_repo",
                })
                if len(distractors) >= num:
                    break
        if len(distractors) >= num:
            break

    # Fallback to generic if no docstrings found
    if len(distractors) < num:
        for fp, content in repo_context.items():
            if fp in involved_files:
                continue
            funcs = re.findall(r"def\s+(\w+)\s*\([^)]*\)", content)
            for func in funcs[:1]:
                distractors.append({
                    "id": f"D_repo_{len(distractors) + 1}",
                    "content": (
                        f"The function `{func}` in `{fp.split('/')[-1]}` "
                        f"may be relevant to this issue."
                    ),
                    "source": "same_repo",
                })
                if len(distractors) >= num:
                    break
            if len(distractors) >= num:
                break

    return distractors[:num]


def _generate_adversarial_distractors(
    facts: List[Dict],
    code_context: str,
    worker_client: LLMClient,
    num: int = 2,
) -> List[Dict]:
    """Generate adversarial near-miss distractors via Worker LLM.

    Args:
        facts: Current instance's real facts
        code_context: Summary of code context
        worker_client: Worker LLM client
        num: Number to generate

    Returns:
        List of distractor dicts
    """
    facts_str = "\n".join(f"- {f['content']}" for f in facts)

    prompt = ADVERSARIAL_DISTRACTOR_PROMPT.format(
        num_distractors=num,
        real_facts=facts_str,
        code_context=code_context[:3000],
    )

    try:
        result = worker_client.get_json_completion(
            prompt=prompt,
            response_format=ADVERSARIAL_DISTRACTOR_SCHEMA,
            temperature=0.7,
        )
        raw = result.get("distractors", [])
        return [
            {
                "id": d.get("id", f"D_adv_{i + 1}"),
                "content": d.get("content", ""),
                "source": "adversarial",
            }
            for i, d in enumerate(raw[:num])
        ]
    except Exception:
        return []


def _generate_same_function_distractors(
    facts: List[Dict],
    code_context: str,
    worker_client: LLMClient,
    num: int = 2,
) -> List[Dict]:
    """Generate distractors that contradict real facts about the SAME functions.

    These are the hardest distractors to filter because they reference the
    same code areas as real facts but describe incorrect behavior.

    Args:
        facts: Current instance's real facts
        code_context: Summary of code context
        worker_client: Worker LLM client
        num: Number to generate

    Returns:
        List of distractor dicts
    """
    # Only use critical facts as targets for contradiction
    critical_facts = [
        f for f in facts
        if f.get("relevance") in ("critical", "helpful")
    ]
    if not critical_facts:
        critical_facts = facts[:3]

    facts_str = "\n".join(f"- {f['content']}" for f in critical_facts)

    prompt = SAME_FUNCTION_DISTRACTOR_PROMPT.format(
        num_distractors=num,
        real_facts=facts_str,
        code_context=code_context[:3000],
    )

    try:
        result = worker_client.get_json_completion(
            prompt=prompt,
            response_format=SAME_FUNCTION_DISTRACTOR_SCHEMA,
            temperature=0.7,
        )
        raw = result.get("distractors", [])
        return [
            {
                "id": d.get("id", f"D_sf_{i + 1}"),
                "content": d.get("content", ""),
                "source": "same_function",
            }
            for i, d in enumerate(raw[:num])
        ]
    except Exception:
        return []


def generate_distractors(
    instance: FilteredInstance,
    facts: List[Dict],
    all_facts_pool: List[Dict],
    repo_context: Dict[str, str],
    worker_client: LLMClient,
    num_distractors: int = 5,
) -> List[Dict]:
    """Generate distractors from four sources.

    Sources:
      1. Cross-instance: real facts from other instances (no LLM)
      2. Same-repo: functions/docstrings from unrelated files (no LLM)
      3. Adversarial: plausible near-miss facts (Worker LLM)
      4. Same-function: contradicting facts about the SAME code (Worker LLM)

    Args:
        instance: FilteredInstance
        facts: Current instance's grounding facts
        all_facts_pool: Facts from other instances in the batch
        repo_context: Dict of repo files
        worker_client: Worker LLM client
        num_distractors: Target total distractors

    Returns:
        List of distractor dicts
    """
    # Split budget across 4 sources
    n_cross = max(1, num_distractors // 4)
    n_repo = max(1, num_distractors // 4)
    n_adv = max(1, num_distractors // 4)
    n_sf = num_distractors - n_cross - n_repo - n_adv

    distractors = []

    # 1. Cross-instance (no LLM)
    if all_facts_pool:
        distractors.extend(
            _generate_cross_instance_distractors(facts, all_facts_pool, n_cross)
        )

    # 2. Same-repo (no LLM)
    if repo_context:
        distractors.extend(
            _generate_same_repo_distractors(facts, repo_context, n_repo)
        )

    code_summary = "\n".join(
        f"=== {fp} (first 50 lines) ===\n"
        + "\n".join(content.split("\n")[:50])
        for fp, content in list(repo_context.items())[:3]
    )

    # 3. Adversarial near-misses (Worker LLM)
    distractors.extend(
        _generate_adversarial_distractors(facts, code_summary, worker_client, n_adv)
    )

    # 4. Same-function contradictions (Worker LLM) — hardest to filter
    distractors.extend(
        _generate_same_function_distractors(facts, code_summary, worker_client, n_sf)
    )

    # Re-number IDs
    for i, d in enumerate(distractors):
        d["id"] = f"D{i + 1}"

    return distractors[:num_distractors]


# ---------------------------------------------------------------------------
# Patch context extraction
# ---------------------------------------------------------------------------

def _extract_patch_context(patch: str) -> str:
    """Extract code context from a unified diff patch.

    Extracts function signatures from hunk headers and context lines
    (lines without +/- prefix) to provide real source code snippets.

    Args:
        patch: Unified diff string

    Returns:
        Formatted string of code snippets from the patch
    """
    if not patch:
        return "(no code snippets available)"

    snippets = []
    current_file = None

    for line in patch.split("\n"):
        # Track current file
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fname = line.split("/", 1)[-1] if "/" in line else line[6:]
            if line.startswith("+++ b/"):
                current_file = fname

        # Extract function signatures from hunk headers
        elif line.startswith("@@") and current_file:
            # Hunk header like: @@ -10,5 +10,7 @@ def some_function(args):
            match = re.search(r'@@.*@@\s*(.+)', line)
            if match:
                func_sig = match.group(1).strip()
                if func_sig:
                    snippets.append(f"# {current_file}\n{func_sig}")

        # Collect context lines (unchanged code — real source)
        elif line and not line.startswith("+") and not line.startswith("-") and not line.startswith("\\"):
            # Context lines start with a space in unified diffs
            if line.startswith(" "):
                stripped = line[1:]  # Remove leading space
                if stripped.strip():  # Skip blank lines
                    snippets.append(stripped)

    if not snippets:
        return "(no code snippets available)"

    # Deduplicate and format
    seen = set()
    unique = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return "\n".join(unique[:60])  # Cap at 60 lines


# ---------------------------------------------------------------------------
# Memory file expansion (filler/noise content)
# ---------------------------------------------------------------------------

_FILLER_PROMPT = '''\
You are expanding a developer document with realistic additional content.
The document currently has {current_words} words. Add MORE content to reach
approximately {target_words} words total.

## Current document ({filename})
{current_content}

## Instructions
Add realistic off-topic content that a real developer document would contain:
- Team discussions about OTHER features, deployments, or sprint planning
- Code review comments on unrelated PRs or refactoring efforts
- Mentions of other bugs, tech debt, performance concerns
- Testing strategies, CI/CD pipeline issues, code style debates
- Meeting notes, deadlines, team processes, standup updates
- Code snippets from unrelated parts of the codebase
- References to other team members (use realistic names)

IMPORTANT: Keep the existing content EXACTLY as-is. Only ADD new sections,
paragraphs, or entries AROUND the existing content. The new content should
blend naturally with the document's format/style.

Return ONLY the expanded document content (no JSON wrapping, no explanation).'''


def _expand_single_file(
    filename: str,
    content: str,
    worker_client: LLMClient,
    target_words_per_file: int,
) -> Tuple[str, str]:
    """Expand a single memory file with filler content. Returns (filename, content)."""
    current_words = len(content.split())
    if current_words >= target_words_per_file * 0.8:
        return filename, content

    try:
        prompt = _FILLER_PROMPT.format(
            current_words=current_words,
            target_words=target_words_per_file,
            filename=filename,
            current_content=content,
        )
        result = worker_client.get_completion(
            prompt=prompt,
            temperature=0.8,
            max_tokens=8000,
        )
        if result and len(result.split()) > current_words:
            return filename, result.strip()
    except Exception:
        pass
    return filename, content


def _expand_memory_files(
    files: Dict[str, str],
    worker_client: LLMClient,
    target_words_per_file: int = 1200,
) -> Dict[str, str]:
    """Expand memory files with filler content to reach target word count.

    Uses parallel threads to expand multiple files concurrently.

    Args:
        files: Dict of filename -> content
        worker_client: Worker LLM client
        target_words_per_file: Target words per file

    Returns:
        Dict of filename -> expanded content
    """
    if not files:
        return files

    max_workers = min(len(files), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _expand_single_file, fn, ct, worker_client, target_words_per_file
            ): fn
            for fn, ct in files.items()
        }
        expanded = {}
        for future in as_completed(futures):
            fn, content = future.result()
            expanded[fn] = content

    return expanded


# ---------------------------------------------------------------------------
# Memory file assembly
# ---------------------------------------------------------------------------

def assemble_memory_files(
    facts: List[Dict],
    distractors: List[Dict],
    worker_client: LLMClient,
    patch: str = "",
    target_words_per_file: int = 1200,
    total_words_range: str = "3000-6000",
) -> Dict[str, str]:
    """Package facts + distractors into natural-looking document files.

    Args:
        facts: Grounding facts
        distractors: Distractor facts
        worker_client: Worker LLM client
        patch: Optional unified diff to extract code snippets from
        target_words_per_file: Target words per file for filler expansion
        total_words_range: Word count range for the assembly prompt

    Returns:
        Dict mapping filename -> content
    """
    # Build combined fact list with UNIFORM IDs (no F/D leak)
    combined = list(facts) + list(distractors)
    random.shuffle(combined)
    all_items = []
    for i, item in enumerate(combined, 1):
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        all_items.append(f"[I{i}] {content}")
    all_facts_str = "\n".join(f"- {item}" for item in all_items)

    # Extract code snippets from the patch
    code_snippets = _extract_patch_context(patch) if patch else "(no code snippets available)"

    prompt = MEMORY_FILE_ASSEMBLY_PROMPT.format(
        all_facts=all_facts_str,
        code_snippets=code_snippets,
        total_words_range=total_words_range,
    )

    # Try up to 2 attempts before fallback
    for attempt in range(2):
        try:
            result = worker_client.get_json_completion(
                prompt=prompt,
                response_format=MEMORY_FILE_ASSEMBLY_SCHEMA,
                temperature=0.7,
                max_tokens=16000,
            )
            files = {}
            for f in result.get("files", []):
                filename = f.get("filename", "")
                content = f.get("content", "")
                if filename and content:
                    files[filename] = content
            if files:
                files = _expand_memory_files(
                    files, worker_client, target_words_per_file
                )
                return files
        except Exception as e:
            if attempt == 0:
                print(f"  Memory file assembly attempt 1 failed: {e}, retrying...")
                continue
            print(f"  Memory file assembly failed after 2 attempts: {e}")

    # Fallback: single file with all facts as bullet points
    # Still run filler expansion on fallback
    content_lines = ["# Debug Notes\n"]
    content_lines.append("## Relevant findings\n")
    for item in all_items:
        content_lines.append(f"- {item}")
    fallback = {"debug_notes.md": "\n".join(content_lines)}
    return _expand_memory_files(fallback, worker_client, target_words_per_file)


# ---------------------------------------------------------------------------
# Main craft function
# ---------------------------------------------------------------------------

def craft_instance(
    instance: FilteredInstance,
    extraction: Dict[str, Any],
    repo_context: Dict[str, str],
    worker_client: LLMClient,
    all_facts_pool: List[Dict],
    num_distractors: int = 5,
    target_context_tokens: int = 12000,
) -> Optional[MemGymInstance]:
    """Run the full Craft stage on a single instance.

    Args:
        instance: FilteredInstance
        extraction: Split result dict (with 'facts', 'referenced_files')
        repo_context: Dict of repo files
        worker_client: Worker LLM client
        all_facts_pool: Facts from other instances for cross-instance distractors
        num_distractors: Target number of distractors
        target_context_tokens: Target total context tokens for the instance

    Returns:
        MemGymInstance or None if crafting fails
    """
    facts = extraction.get("facts", [])
    referenced_files = extraction.get("referenced_files", [])

    if len(facts) < 3:
        return None

    # Also get referenced files from patch if extraction didn't provide them
    if not referenced_files:
        referenced_files = extract_files_from_patch(instance.patch)

    # 1. Craft task prompt + generate distractors concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        task_future = pool.submit(craft_task_prompt, instance, facts, worker_client)
        dist_future = pool.submit(
            generate_distractors, instance, facts, all_facts_pool,
            repo_context, worker_client, num_distractors,
        )
        task_prompt = task_future.result()
        distractors = dist_future.result()

    if task_prompt is None:
        return None  # Instance rejected: task prompt crafting failed

    # 3. Assemble memory files (pass patch for code snippet extraction)
    # Compute dynamic word targets from target_context_tokens
    total_target_words = int(target_context_tokens / 1.3)
    low_words = max(1000, total_target_words // 2)
    total_words_range = f"{low_words}-{total_target_words}"
    # Estimate per-file target (assume 6 files average, refine after assembly)
    est_files = 6
    target_words_per_file = max(400, min(3000, total_target_words // est_files))

    memory_files = assemble_memory_files(
        facts, distractors, worker_client, patch=instance.patch,
        target_words_per_file=target_words_per_file,
        total_words_range=total_words_range,
    )

    # 4. Parse into models
    grounding_facts = []
    for f in facts:
        try:
            grounding_facts.append(GroundingFact(**f))
        except Exception:
            grounding_facts.append(
                GroundingFact(
                    id=f.get("id", "F?"),
                    content=f.get("content", ""),
                    category=f.get("category", "bug_description"),
                    discoverable=f.get("discoverable", True),
                    relevance=f.get("relevance", "helpful"),
                    hunks_enabled=f.get("hunks_enabled", []),
                    evidence=f.get("evidence", ""),
                )
            )

    distractor_facts = []
    for d in distractors:
        try:
            distractor_facts.append(DistractorFact(**d))
        except Exception:
            distractor_facts.append(
                DistractorFact(
                    id=d.get("id", "D?"),
                    content=d.get("content", ""),
                    source=d.get("source", "adversarial"),
                )
            )

    # 5. Compute gold fix
    gold_fix = reverse_patch(instance.patch)

    # 5b. Synthesize repo context from patch + tests (no Docker)
    synth_repo_ctx = synthesize_repo_context(instance)

    # 6. Compute token counts
    tok_task = count_tokens(task_prompt)
    tok_memory = sum(count_tokens(v) for v in memory_files.values())
    tok_repo = count_tokens(synth_repo_ctx) + sum(
        count_tokens(v) for v in repo_context.values()
    )
    tok_facts = sum(count_tokens(f.get("content", "")) for f in facts)
    tok_distractors = sum(count_tokens(d.get("content", "")) for d in distractors)
    context_tokens = ContextTokens(
        task_prompt=tok_task,
        memory_files=tok_memory,
        repo_context=tok_repo,
        grounding_facts=tok_facts,
        distractors=tok_distractors,
        total=tok_task + tok_memory + tok_repo + tok_facts + tok_distractors,
    )

    return MemGymInstance(
        instance_id=f"memgym__{instance.instance_id}",
        base_instance=instance.instance_id,
        task_prompt=task_prompt,
        memory_files=memory_files,
        grounding_facts=grounding_facts,
        distractors=distractor_facts,
        gold_fix=gold_fix,
        transforms_applied=[],
        difficulty_preset="medium",
        patch=instance.patch,
        FAIL_TO_PASS=instance.FAIL_TO_PASS,
        PASS_TO_PASS=instance.PASS_TO_PASS,
        image_name=instance.image_name,
        referenced_files=referenced_files,
        repo_context=synth_repo_ctx,
        context_tokens=context_tokens,
    )


def craft_all_instances(
    instances: List[FilteredInstance],
    extractions: List[Dict[str, Any]],
    repo_contexts: Dict[str, Dict[str, str]],
    worker_client: LLMClient,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    num_distractors: int = 5,
    target_context_tokens: int = 12000,
    num_workers: int = 1,
) -> List[MemGymInstance]:
    """Run Craft stage on all instances.

    Args:
        instances: List of FilteredInstance objects
        extractions: List of Split results
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        worker_client: Worker LLM client
        output_path: Path to save crafted.jsonl
        show_progress: Show progress bar
        num_distractors: Target distractors per instance
        target_context_tokens: Target total context tokens per instance
        num_workers: Number of parallel workers

    Returns:
        List of MemGymInstance objects
    """
    from ..utils.parallel import parallel_map

    extraction_map = {e["instance_id"]: e for e in extractions}

    # Build pool of all facts from all instances (for cross-instance distractors)
    all_facts_pool = []
    for ext in extractions:
        for f in ext.get("facts", []):
            all_facts_pool.append(f)

    # Filter to eligible instances
    eligible = []
    for instance in instances:
        extraction = extraction_map.get(instance.instance_id, {})
        if extraction.get("quality_gate_passed", False):
            eligible.append(instance)

    def _process(instance: FilteredInstance) -> Optional[MemGymInstance]:
        extraction = extraction_map.get(instance.instance_id, {})
        repo_ctx = repo_contexts.get(instance.instance_id, {})

        current_fact_ids = {
            f.get("id") for f in extraction.get("facts", [])
        }
        other_facts = [
            f for f in all_facts_pool if f.get("id") not in current_fact_ids
        ]

        return craft_instance(
            instance=instance,
            extraction=extraction,
            repo_context=repo_ctx,
            worker_client=worker_client,
            all_facts_pool=other_facts,
            num_distractors=num_distractors,
            target_context_tokens=target_context_tokens,
        )

    results = parallel_map(
        _process, eligible,
        num_workers=num_workers,
        desc="Crafting",
        output_path=output_path,
        serialize_fn=lambda r: json.dumps(r.to_jsonl_dict()),
        show_progress=show_progress,
        item_id_fn=lambda inst: inst.instance_id,
    )

    print(f"Crafted {len(results)} instances from {len(instances)} inputs")
    return results


# ---------------------------------------------------------------------------
# Integrated craft + verify loop (per-instance)
# ---------------------------------------------------------------------------

def craft_and_verify_instance(
    instance: FilteredInstance,
    extraction: Dict[str, Any],
    repo_context: Dict[str, str],
    worker_client: LLMClient,
    all_facts_pool: List[Dict],
    verifier_model: str,
    docker_available: bool = False,
    difficulty_preset: str = "medium",
    adaptive_rounds: int = 2,
    num_distractors: int = 5,
    target_context_tokens: int = 12000,
) -> Tuple[Optional[MemGymInstance], Optional[Dict[str, Any]]]:
    """Craft, apply difficulty, verify, and adaptively retry per-instance.

    Flow:
        1. craft_instance() → MemGymInstance
        2. apply_difficulty() with preset
        3. verify_instance() → ablation result
        4. If not accepted and adaptive_rounds > 0:
           apply_adaptive_difficulty() → re-verify
        5. Track best attempt by ablation gap

    Args:
        instance: FilteredInstance
        extraction: Split result dict
        repo_context: Dict of repo files
        worker_client: Worker LLM client
        all_facts_pool: Facts from other instances
        verifier_model: Model name for ablation verification
        docker_available: Whether Docker is available
        difficulty_preset: Difficulty preset to apply
        adaptive_rounds: Max adaptive retry rounds
        num_distractors: Target distractor count
        target_context_tokens: Target total context tokens per instance

    Returns:
        (MemGymInstance, verification_result) or (None, None) if crafting fails
    """
    from .harden import apply_difficulty, apply_adaptive_difficulty
    from .validate import verify_instance

    # 1. Craft
    crafted = craft_instance(
        instance=instance,
        extraction=extraction,
        repo_context=repo_context,
        worker_client=worker_client,
        all_facts_pool=all_facts_pool,
        num_distractors=num_distractors,
        target_context_tokens=target_context_tokens,
    )
    if crafted is None:
        return None, None

    # 2. Apply difficulty preset
    current = apply_difficulty(
        crafted, difficulty_preset, worker_client, repo_context
    )

    # 3. Verify
    vresult = verify_instance(
        current,
        verifier_model=verifier_model,
        repo_context=repo_context,
        docker_available=docker_available,
    )

    best_instance = current
    best_vresult = vresult
    best_gap = _ablation_gap(vresult)

    # 4. Adaptive rounds
    if not vresult["accepted"] and adaptive_rounds > 0:
        for round_num in range(adaptive_rounds):
            current = apply_adaptive_difficulty(
                current, vresult, worker_client, repo_context,
            )
            vresult = verify_instance(
                current,
                verifier_model=verifier_model,
                repo_context=repo_context,
                docker_available=docker_available,
            )
            gap = _ablation_gap(vresult)
            if gap > best_gap:
                best_instance = current
                best_vresult = vresult
                best_gap = gap

            if vresult["accepted"]:
                best_instance = current
                best_vresult = vresult
                break

    # Attach verification to instance
    best_instance = best_instance.model_copy(
        update={"verification": best_vresult}
    )
    return best_instance, best_vresult


def _ablation_gap(vresult: Dict[str, Any]) -> float:
    """Compute ablation gap from verification result (higher = better)."""
    docker_gap = (
        vresult.get("cond_a_docker_passes", 0)
        - vresult.get("cond_b_docker_passes", 0)
    )
    llm_gap = (
        vresult.get("cond_a_llm_score", 0.0)
        - vresult.get("cond_b_llm_score", 0.0)
    )
    # Docker gap dominates when available; LLM gap as fallback
    if docker_gap != 0:
        return float(docker_gap)
    return llm_gap


def craft_and_verify_all(
    instances: List[FilteredInstance],
    extractions: List[Dict[str, Any]],
    repo_contexts: Dict[str, Dict[str, str]],
    worker_client: LLMClient,
    verifier_model: str = "claude-opus-4-20250514",
    docker_available: bool = False,
    difficulty_preset: str = "medium",
    adaptive_rounds: int = 2,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    num_distractors: int = 5,
    target_context_tokens: int = 12000,
    num_workers: int = 1,
) -> Tuple[List[MemGymInstance], List[Dict[str, Any]]]:
    """Run Craft + Difficulty + Verify on all instances (per-instance loop).

    Args:
        instances: List of FilteredInstance objects
        extractions: List of Split results
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        worker_client: Worker LLM client
        verifier_model: Model for verification ablation
        docker_available: Whether Docker is available
        difficulty_preset: Difficulty preset
        adaptive_rounds: Max adaptive retries per instance
        output_path: Path to save crafted.jsonl
        show_progress: Show progress bar
        num_distractors: Target distractors per instance
        target_context_tokens: Target total context tokens per instance
        num_workers: Number of parallel workers

    Returns:
        (List[MemGymInstance], List[verification_results])
    """
    from ..utils.parallel import parallel_map

    extraction_map = {e["instance_id"]: e for e in extractions}

    # Build pool of all facts for cross-instance distractors
    all_facts_pool = []
    for ext in extractions:
        for f in ext.get("facts", []):
            all_facts_pool.append(f)

    # Filter to eligible instances
    eligible = []
    for instance in instances:
        extraction = extraction_map.get(instance.instance_id, {})
        if extraction.get("quality_gate_passed", False):
            eligible.append(instance)

    # Shared list for verifications (parallel_map only returns one value)
    import threading
    verifications_lock = threading.Lock()
    verifications: List[Tuple[int, Dict[str, Any]]] = []

    def _process(item: Tuple[int, FilteredInstance]) -> Optional[MemGymInstance]:
        idx, instance = item
        extraction = extraction_map.get(instance.instance_id, {})
        repo_ctx = repo_contexts.get(instance.instance_id, {})

        current_fact_ids = {
            f.get("id") for f in extraction.get("facts", [])
        }
        other_facts = [
            f for f in all_facts_pool if f.get("id") not in current_fact_ids
        ]

        inst, vresult = craft_and_verify_instance(
            instance=instance,
            extraction=extraction,
            repo_context=repo_ctx,
            worker_client=worker_client,
            all_facts_pool=other_facts,
            verifier_model=verifier_model,
            docker_available=docker_available,
            difficulty_preset=difficulty_preset,
            adaptive_rounds=adaptive_rounds,
            num_distractors=num_distractors,
            target_context_tokens=target_context_tokens,
        )

        if inst is not None and vresult is not None:
            with verifications_lock:
                verifications.append((idx, vresult))
            return inst
        return None

    indexed_eligible = list(enumerate(eligible))
    results = parallel_map(
        _process, indexed_eligible,
        num_workers=num_workers,
        desc="Craft+Verify",
        output_path=output_path,
        serialize_fn=lambda r: json.dumps(r.to_jsonl_dict()),
        show_progress=show_progress,
        item_id_fn=lambda pair: pair[1].instance_id,
    )

    # Sort verifications by original index to maintain order
    verifications.sort(key=lambda x: x[0])
    verification_results = [v for _, v in verifications]

    # Summary
    accepted = sum(1 for v in verification_results if v and v.get("accepted"))
    avg_gap = (
        sum(_ablation_gap(v) for v in verification_results) / len(verification_results)
        if verification_results
        else 0
    )
    print(
        f"Craft+Verify: {len(results)} instances crafted, "
        f"{accepted} accepted ({accepted / max(1, len(results)) * 100:.0f}%), "
        f"avg gap={avg_gap:.2f}"
    )
    return results, verification_results
