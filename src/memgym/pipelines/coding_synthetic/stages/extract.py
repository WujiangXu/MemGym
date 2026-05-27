"""
Stage 2 — Split: Extract grounding facts from filtered instances.

Three passes:
  1. Static analysis of repo context (no LLM)
  2. Fact extraction via Worker LLM
  3. Classification verification via Verifier LLM

Quality gate: instance must have >=2 facts that are both
discoverable=False AND relevance=critical.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from ..types.schemas import (
    FilteredInstance,
    GroundingFact,
    GROUNDING_FACT_EXTRACTION_SCHEMA,
    FACT_VERIFICATION_SCHEMA,
    FACT_SEED_SCHEMA,
)
from ..llm.client import LLMClient, VerifierClient
from ..llm.prompts import (
    FACT_EXTRACTION_PROMPT,
    FACT_SEED_PROMPT,
    FACT_VERIFICATION_PROMPT,
)
from ..utils.patch_utils import extract_files_from_patch

# Patterns that indicate a seed question leaks code-change information
_SEED_LEAK_PATTERNS = [
    r'\b(added|removed|deleted|inserted|changed)\b',
    r'\bline\s+\d+\b',
    r'\b(the patch|the diff|the change)\b',
    r'[+\-]{3}\s',  # diff markers
    r'@@\s',  # hunk headers
]


# ---------------------------------------------------------------------------
# Pass 1 — Repo static analysis (no LLM)
# ---------------------------------------------------------------------------

def analyze_repo_context(
    patch: str,
    repo_context: Dict[str, str],
) -> Dict[str, Any]:
    """Analyze what the repo already tells us about the patch.

    For each changed file/function:
    - Extract function signatures, docstrings, type hints
    - Find callers (grep for function name in repo)
    - Find existing tests that reference the function
    - Extract surrounding code comments
    - Identify imports and dependencies

    Args:
        patch: The unified diff patch
        repo_context: Dict mapping file_path -> full file content

    Returns:
        Structured dict of repo-discoverable information per file/function.
    """
    analysis: Dict[str, Any] = {
        "functions": {},
        "callers": {},
        "test_references": {},
        "imports": {},
        "comments": {},
    }

    # Extract function names from patch (both modified and context lines)
    func_names: Dict[str, List[str]] = {}  # file -> [func_name]
    current_file = None
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            in_hunk = False
        elif line.startswith("+++ "):
            current_file = line[4:]
            in_hunk = False
        elif line.startswith("---"):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
            # Extract function from hunk header
            if current_file:
                func_match = re.search(r"@@.*@@\s*(?:def|class)\s+(\w+)", line)
                if func_match:
                    func_names.setdefault(current_file, [])
                    name = func_match.group(1)
                    if name not in func_names[current_file]:
                        func_names[current_file].append(name)
        elif in_hunk and current_file:
            content = line[1:] if line and line[0] in ("+", "-", " ") else line
            func_match = re.search(r"def\s+(\w+)\s*\(", content)
            if func_match:
                func_names.setdefault(current_file, [])
                name = func_match.group(1)
                if name not in func_names[current_file]:
                    func_names[current_file].append(name)

    # For each function, extract info from repo context
    for file_path, names in func_names.items():
        # Find matching file in repo context
        source = None
        for ctx_path, ctx_content in repo_context.items():
            if ctx_path == file_path or ctx_path.endswith(file_path):
                source = ctx_content
                break
            if file_path.split("/")[-1] == ctx_path.split("/")[-1]:
                source = ctx_content
                break

        if source is None:
            continue

        source_lines = source.split("\n")

        for fname in names:
            func_info: Dict[str, Any] = {"file": file_path, "name": fname}

            # Find function definition and extract signature + docstring
            for i, sline in enumerate(source_lines):
                match = re.match(
                    r"^(\s*)def\s+" + re.escape(fname) + r"\s*\((.*)$", sline
                )
                if match:
                    # Collect full signature (may span multiple lines)
                    sig_lines = [sline.rstrip()]
                    j = i + 1
                    while j < len(source_lines) and ")" not in sig_lines[-1]:
                        sig_lines.append(source_lines[j].rstrip())
                        j += 1
                    func_info["signature"] = "\n".join(sig_lines)

                    # Check for docstring
                    body_start = j if j < len(source_lines) else i + 1
                    if body_start < len(source_lines):
                        stripped = source_lines[body_start].strip()
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            doc_lines = [stripped]
                            quote = stripped[:3]
                            if stripped.count(quote) >= 2 and len(stripped) > 6:
                                pass  # single-line docstring
                            else:
                                k = body_start + 1
                                while k < len(source_lines):
                                    doc_lines.append(source_lines[k].strip())
                                    if quote in source_lines[k]:
                                        break
                                    k += 1
                            func_info["docstring"] = "\n".join(doc_lines)[:500]

                    # Extract type hints from signature
                    full_sig = " ".join(sig_lines)
                    type_hints = re.findall(r":\s*([A-Z]\w+(?:\[.*?\])?)", full_sig)
                    return_hint = re.search(r"->\s*(\S+)", full_sig)
                    if type_hints:
                        func_info["type_hints"] = type_hints
                    if return_hint:
                        func_info["return_type"] = return_hint.group(1)
                    break

            analysis["functions"][fname] = func_info

            # Find callers (grep for function name in all repo files)
            callers = []
            for ctx_path, ctx_content in repo_context.items():
                if ctx_path == file_path:
                    continue  # skip self
                for line_num, cline in enumerate(ctx_content.split("\n"), 1):
                    if re.search(r"\b" + re.escape(fname) + r"\b", cline):
                        callers.append(
                            f"{ctx_path}:{line_num}: {cline.strip()[:120]}"
                        )
                        if len(callers) >= 5:
                            break
                if len(callers) >= 5:
                    break
            if callers:
                analysis["callers"][fname] = callers

            # Find test references
            test_refs = []
            for ctx_path, ctx_content in repo_context.items():
                if "test" in ctx_path.lower():
                    for line_num, tline in enumerate(ctx_content.split("\n"), 1):
                        if re.search(r"\b" + re.escape(fname) + r"\b", tline):
                            test_refs.append(
                                f"{ctx_path}:{line_num}: {tline.strip()[:120]}"
                            )
                            if len(test_refs) >= 5:
                                break
                if len(test_refs) >= 5:
                    break
            if test_refs:
                analysis["test_references"][fname] = test_refs

    # Extract imports from changed files
    for file_path in func_names:
        for ctx_path, ctx_content in repo_context.items():
            if ctx_path == file_path or ctx_path.endswith(file_path):
                imports = [
                    line.strip()
                    for line in ctx_content.split("\n")
                    if line.strip().startswith(("import ", "from "))
                ]
                if imports:
                    analysis["imports"][file_path] = imports[:20]
                break

    return analysis


def _format_repo_analysis(analysis: Dict[str, Any]) -> str:
    """Format repo analysis dict into a readable string for prompts."""
    parts = []

    if analysis.get("functions"):
        parts.append("### Functions modified")
        for fname, info in analysis["functions"].items():
            sig = info.get("signature", f"def {fname}(...)")
            parts.append(f"\n**{fname}** in `{info.get('file', '?')}`")
            parts.append(f"```python\n{sig}\n```")
            if "docstring" in info:
                parts.append(f"Docstring: {info['docstring']}")
            if "return_type" in info:
                parts.append(f"Return type: {info['return_type']}")
            if "type_hints" in info:
                parts.append(f"Type hints: {', '.join(info['type_hints'])}")

    if analysis.get("callers"):
        parts.append("\n### Callers")
        for fname, callers in analysis["callers"].items():
            parts.append(f"\n**{fname}** is called by:")
            for c in callers:
                parts.append(f"  - {c}")

    if analysis.get("test_references"):
        parts.append("\n### Test references")
        for fname, refs in analysis["test_references"].items():
            parts.append(f"\n**{fname}** referenced in tests:")
            for r in refs:
                parts.append(f"  - {r}")

    if analysis.get("imports"):
        parts.append("\n### Imports")
        for fp, imps in analysis["imports"].items():
            parts.append(f"\n`{fp}`:")
            for imp in imps[:10]:
                parts.append(f"  - {imp}")

    return "\n".join(parts) if parts else "(no static analysis results)"


# ---------------------------------------------------------------------------
# Pass 2a — Fact seed generation (Worker LLM sees patch, outputs questions)
# ---------------------------------------------------------------------------

def _filter_leaky_seeds(seeds: list) -> list:
    """Remove seed questions that leak code-change information."""
    import re as _re
    filtered = []
    for seed in seeds:
        question = seed.get("question", "").lower()
        is_leaky = any(
            _re.search(pattern, question, _re.IGNORECASE)
            for pattern in _SEED_LEAK_PATTERNS
        )
        if not is_leaky:
            filtered.append(seed)
    return filtered


def _generate_fact_seeds(
    patch: str,
    patch_files_str: str,
    analysis_str: str,
    worker_client: LLMClient,
) -> str:
    """Generate behavioral seed questions from the patch.

    This is the ONLY place the raw patch is sent to an LLM. The output
    is behavioral questions (not answers) that guide fact extraction.

    Args:
        patch: The raw unified diff patch
        patch_files_str: Formatted file paths
        analysis_str: Formatted repo analysis
        worker_client: Worker LLM client

    Returns:
        Formatted string of seed questions for the extraction prompt
    """
    prompt = FACT_SEED_PROMPT.format(
        patch=patch[:4000],
        patch_files=patch_files_str,
        repo_analysis=analysis_str,
    )

    try:
        result = worker_client.get_json_completion(
            prompt=prompt,
            response_format=FACT_SEED_SCHEMA,
            temperature=0.3,
        )
        seeds = result.get("seeds", [])

        # Post-filter: remove seeds that leak code-change language
        seeds = _filter_leaky_seeds(seeds)

        if seeds:
            lines = []
            for i, seed in enumerate(seeds, 1):
                cat = seed.get("category", "unknown")
                hunks = seed.get("hunk_indices", [])
                lines.append(
                    f"{i}. [{cat}] (hunks {hunks}) {seed['question']}"
                )
            return "\n".join(lines)
    except Exception as e:
        print(f"  Fact seed generation failed: {e}")

    # Fallback: generic seed questions
    return (
        "1. [bug_description] What is the expected behavior vs actual behavior?\n"
        "2. [repro_condition] Under what conditions does this bug manifest?\n"
        "3. [error_detail] What error messages or test failures are observed?\n"
        "4. [user_requirement] What user-facing requirement is violated?\n"
        "5. [constraint] What constraints must the fix satisfy?\n"
        "6. [design_decision] What design context is relevant to the fix?"
    )


# ---------------------------------------------------------------------------
# Pass 2b — Fact extraction (Worker LLM, NO patch)
# ---------------------------------------------------------------------------

def extract_grounding_facts(
    instance: FilteredInstance,
    repo_analysis: Dict[str, Any],
    worker_client: LLMClient,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Extract grounding facts using Worker LLM.

    Args:
        instance: FilteredInstance to process
        repo_analysis: Output of analyze_repo_context()
        worker_client: Worker LLM client (cheap model)
        max_retries: Max retries on failure

    Returns:
        Dict with 'facts' list and 'referenced_files' list
    """
    fail_to_pass_str = "\n".join(f"- {t}" for t in instance.FAIL_TO_PASS[:10])
    if len(instance.FAIL_TO_PASS) > 10:
        fail_to_pass_str += f"\n... and {len(instance.FAIL_TO_PASS) - 10} more"

    patch_files = extract_files_from_patch(instance.patch)
    patch_files_str = (
        "\n".join(f"- {f}" for f in patch_files) if patch_files else "- (none)"
    )

    analysis_str = _format_repo_analysis(repo_analysis)

    # Phase 1: Generate behavioral seed questions (only place patch is seen)
    fact_seeds = _generate_fact_seeds(
        instance.patch, patch_files_str, analysis_str, worker_client,
    )

    # Phase 2: Extract facts WITHOUT the raw patch
    prompt = FACT_EXTRACTION_PROMPT.format(
        problem_statement=instance.problem_statement[:8000],
        fact_seeds=fact_seeds,
        fail_to_pass_tests=fail_to_pass_str,
        patch_files=patch_files_str,
        repo_analysis=analysis_str,
    )

    for attempt in range(max_retries + 1):
        try:
            result = worker_client.get_json_completion(
                prompt=prompt,
                response_format=GROUNDING_FACT_EXTRACTION_SCHEMA,
                temperature=0.3,
            )

            facts = result.get("facts", [])
            referenced_files = result.get("referenced_files", patch_files)

            if len(facts) >= 3:
                return {
                    "facts": facts,
                    "referenced_files": referenced_files,
                }
        except Exception as e:
            if attempt == max_retries:
                print(f"  Fact extraction failed for {instance.instance_id}: {e}")

    return {"facts": [], "referenced_files": patch_files}


# ---------------------------------------------------------------------------
# Pass 3 — Classification verification (Verifier LLM)
# ---------------------------------------------------------------------------

def verify_fact_classifications(
    facts: List[Dict],
    repo_context: Dict[str, str],
    verifier_client: LLMClient,
) -> List[Dict]:
    """Verifier checks each 'external' fact against full repo context.

    Args:
        facts: List of fact dicts from extraction
        repo_context: Dict mapping file_path -> content
        verifier_client: Verifier LLM client (expensive model)

    Returns:
        Updated facts list with reclassifications applied
    """
    # Collect non-discoverable facts for verification
    to_verify = [f for f in facts if not f.get("discoverable", True)]

    if not to_verify:
        return facts

    facts_str = "\n".join(
        f"- **{f['id']}** ({f.get('category', '?')}): {f['content']}\n"
        f"  Evidence: {f.get('evidence', 'none')}"
        for f in to_verify
    )

    # Build truncated repo context
    repo_parts = []
    total_chars = 0
    for fp, content in repo_context.items():
        truncated = content[:8000] if len(content) > 8000 else content
        repo_parts.append(f"=== {fp} ===\n{truncated}")
        total_chars += len(truncated)
        if total_chars > 60000:
            break
    repo_str = "\n\n".join(repo_parts)

    prompt = FACT_VERIFICATION_PROMPT.format(
        facts_to_verify=facts_str,
        repo_context=repo_str,
    )

    try:
        result = verifier_client.get_json_completion(
            prompt=prompt,
            response_format=FACT_VERIFICATION_SCHEMA,
            temperature=0.2,
        )

        # Apply reclassifications
        reclassifications = {
            r["fact_id"]: r
            for r in result.get("reclassifications", [])
        }

        for fact in facts:
            if fact["id"] in reclassifications:
                reclass = reclassifications[fact["id"]]
                fact["discoverable"] = reclass["new_discoverable"]
                fact["evidence"] = (
                    f"[Verifier] {reclass['evidence']}"
                )

    except Exception as e:
        print(f"  Fact verification failed: {e}")

    return facts


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def passes_quality_gate(facts: List[Dict], min_external_critical: int = 2) -> bool:
    """Check if instance has enough external critical facts.

    Args:
        facts: List of fact dicts
        min_external_critical: Minimum number of non-discoverable critical facts

    Returns:
        True if quality gate passes
    """
    external_critical = sum(
        1
        for f in facts
        if not f.get("discoverable", True) and f.get("relevance") == "critical"
    )
    return external_critical >= min_external_critical


# ---------------------------------------------------------------------------
# Patch coverage check (reused from v1, adapted for facts)
# ---------------------------------------------------------------------------

def compute_patch_coverage(facts: List[Dict], patch: str) -> Dict[str, Any]:
    """Check how well grounding facts cover all hunks in the patch.

    Args:
        facts: List of fact dicts with hunks_enabled
        patch: Unified diff patch

    Returns:
        Dict with coverage_ratio, covered_hunks, uncovered_hunks
    """
    # Count hunks
    num_hunks = sum(1 for line in patch.split("\n") if line.startswith("@@"))
    if num_hunks == 0:
        return {"coverage_ratio": 1.0, "covered_hunks": [], "uncovered_hunks": []}

    covered = set()
    for f in facts:
        for h in f.get("hunks_enabled", []):
            covered.add(h)

    all_hunks = set(range(num_hunks))
    uncovered = all_hunks - covered

    return {
        "coverage_ratio": len(covered) / num_hunks if num_hunks > 0 else 1.0,
        "covered_hunks": sorted(covered),
        "uncovered_hunks": sorted(uncovered),
    }


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def split_instance(
    instance: FilteredInstance,
    repo_context: Dict[str, str],
    worker_client: LLMClient,
    verifier_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Run the full Split stage on a single instance.

    Args:
        instance: FilteredInstance to process
        repo_context: Dict mapping file_path -> content
        worker_client: Worker LLM (cheap)
        verifier_client: Verifier LLM (expensive, optional)

    Returns:
        Dict with instance_id, facts, repo_analysis, quality_gate_passed, etc.
    """
    # Pass 1: Static analysis
    repo_analysis = analyze_repo_context(instance.patch, repo_context)

    # Pass 2: Fact extraction via Worker
    extraction = extract_grounding_facts(instance, repo_analysis, worker_client)
    facts = extraction["facts"]
    referenced_files = extraction["referenced_files"]

    # Pass 3: Verification via Verifier (if available and we have repo context)
    if verifier_client and repo_context and facts:
        facts = verify_fact_classifications(facts, repo_context, verifier_client)

    # Quality gate
    qg_passed = passes_quality_gate(facts)

    # Patch coverage
    coverage = compute_patch_coverage(facts, instance.patch)

    return {
        "instance_id": instance.instance_id,
        "facts": facts,
        "referenced_files": referenced_files,
        "repo_analysis": repo_analysis,
        "quality_gate_passed": qg_passed,
        "patch_coverage": coverage["coverage_ratio"],
        "num_facts": len(facts),
        "num_external_critical": sum(
            1
            for f in facts
            if not f.get("discoverable", True) and f.get("relevance") == "critical"
        ),
    }


def split_all_instances(
    instances: List[FilteredInstance],
    repo_contexts: Dict[str, Dict[str, str]],
    worker_client: LLMClient,
    verifier_client: Optional[LLMClient] = None,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run Split stage on all instances.

    Args:
        instances: List of FilteredInstance objects
        repo_contexts: Dict mapping instance_id -> {file_path: content}
        worker_client: Worker LLM client
        verifier_client: Verifier LLM client (optional)
        output_path: Path to save extracted.jsonl
        show_progress: Show progress bar
        num_workers: Number of parallel workers

    Returns:
        List of extraction result dicts
    """
    from ..utils.parallel import parallel_map

    def _process(instance: FilteredInstance) -> Dict[str, Any]:
        repo_ctx = repo_contexts.get(instance.instance_id, {})
        return split_instance(
            instance, repo_ctx, worker_client, verifier_client
        )

    results = parallel_map(
        _process, instances,
        num_workers=num_workers,
        desc="Splitting",
        output_path=output_path,
        serialize_fn=lambda r: json.dumps(r, default=str),
        show_progress=show_progress,
        filter_none=False,
        item_id_fn=lambda inst: inst.instance_id,
    )

    # Summary
    passed = sum(1 for r in results if r["quality_gate_passed"])
    avg_facts = (
        sum(r["num_facts"] for r in results) / len(results) if results else 0
    )
    avg_coverage = (
        sum(r["patch_coverage"] for r in results) / len(results) if results else 0
    )
    print(
        f"Split {len(results)} instances: "
        f"{passed} passed quality gate ({passed / len(results) * 100:.0f}%), "
        f"avg {avg_facts:.1f} facts, "
        f"avg {avg_coverage:.0%} patch coverage"
    )

    return results


def load_extractions(path: Path) -> List[Dict[str, Any]]:
    """Load extractions from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Synthetic repo context (no Docker needed)
# ---------------------------------------------------------------------------

def synthesize_repo_context(instance: FilteredInstance) -> str:
    """Build a synthetic repo skeleton from patch + test info (no Docker).

    Extracts file paths, function/class signatures, and import statements
    from the unified diff and FAIL_TO_PASS test names to reconstruct a
    partial view of the codebase structure.

    Args:
        instance: FilteredInstance with patch, FAIL_TO_PASS

    Returns:
        String containing a synthetic repo skeleton
    """
    patch = instance.patch or ""
    sections = []

    # 1. Extract file structure from patch
    files_info: Dict[str, List[str]] = {}  # file -> list of signatures/content
    current_file = None

    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file not in files_info:
                files_info[current_file] = []

        elif line.startswith("@@") and current_file:
            # Extract function/class context from hunk header
            match = re.search(r'@@.*@@\s*(.+)', line)
            if match:
                sig = match.group(1).strip()
                if sig and sig not in files_info[current_file]:
                    files_info[current_file].append(sig)

        elif current_file and line.startswith(" "):
            # Context lines — capture import statements and class/def lines
            stripped = line[1:].strip()
            if stripped.startswith(("import ", "from ", "class ", "def ")):
                if stripped not in files_info.get(current_file, []):
                    files_info[current_file].append(stripped)

    # Format file skeletons
    if files_info:
        sections.append("## Repository Structure (files involved in the fix)\n")
        for fp, sigs in files_info.items():
            sections.append(f"### {fp}")
            if sigs:
                sections.append("```python")
                for sig in sigs[:20]:  # Cap per file
                    sections.append(sig)
                sections.append("```")
            sections.append("")

    # 2. Derive module paths from FAIL_TO_PASS test names
    if instance.FAIL_TO_PASS:
        sections.append("## Relevant Test Modules\n")
        seen_modules = set()
        for test_name in instance.FAIL_TO_PASS[:10]:
            # Test names like "tests.test_oauth.TestScope::test_method"
            # or "tests/test_oauth.py::TestScope::test_method"
            parts = test_name.replace("::", ".").split(".")
            if len(parts) >= 2:
                module = ".".join(parts[:2])
                if module not in seen_modules:
                    seen_modules.add(module)
                    sections.append(f"- `{module}`")
            # Also extract class names
            for part in parts:
                if part and part[0].isupper() and not part.startswith("Test"):
                    sections.append(f"- Class reference: `{part}`")
        sections.append("")

    # 3. Extract all changed function names as an index
    func_names = set()
    for line in patch.split("\n"):
        if line.startswith("@@"):
            match = re.search(r'def\s+(\w+)', line)
            if match:
                func_names.add(match.group(1))
        # Also from added/removed lines
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            for m in re.finditer(r'\bdef\s+(\w+)', line):
                func_names.add(m.group(1))

    if func_names:
        sections.append("## Functions Modified or Referenced\n")
        for fn in sorted(func_names):
            sections.append(f"- `{fn}()`")
        sections.append("")

    if not sections:
        return ""

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Repo context extraction (full files from Docker) — kept from v1
# ---------------------------------------------------------------------------

def extract_repo_context(
    instance: FilteredInstance,
    referenced_files: List[str],
    output_dir: Path,
    timeout: int = 120,
) -> Dict[str, str]:
    """Extract full file contents from a Docker image for repo context.

    Args:
        instance: FilteredInstance with image_name
        referenced_files: List of file paths to extract
        output_dir: Directory for caching
        timeout: Docker command timeout in seconds

    Returns:
        Dict mapping file_path -> full file content
    """
    cache_dir = output_dir / "repo_context" / instance.instance_id
    cache_file = cache_dir / "files.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    if not referenced_files:
        return {}

    patch_files = extract_files_from_patch(instance.patch)
    all_files = list(set(referenced_files + patch_files))

    result: Dict[str, str] = {}
    container_name = None

    try:
        container_name = f"memgym_extract_{instance.instance_id[:20].replace('/', '_')}"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=10,
        )
        start_result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                instance.image_name,
                "sleep", "300",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if start_result.returncode != 0:
            print(f"    Failed to start container: {start_result.stderr[:200]}")
            return {}

        for file_path in all_files:
            try:
                cat_result = subprocess.run(
                    ["docker", "exec", container_name, "cat", f"/testbed/{file_path}"],
                    capture_output=True, text=True, timeout=timeout,
                )
                if cat_result.returncode == 0 and cat_result.stdout:
                    result[file_path] = cat_result.stdout
            except (subprocess.TimeoutExpired, Exception):
                continue

    except Exception as e:
        print(f"    Docker extraction error: {e}")
    finally:
        if container_name:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=10,
            )

    if result:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f)

    return result


def extract_all_repo_contexts(
    instances: List[FilteredInstance],
    extractions: List[Dict[str, Any]],
    output_dir: Path,
    show_progress: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Extract repo context for all instances.

    Args:
        instances: List of FilteredInstance objects
        extractions: List of extraction results (with referenced_files)
        output_dir: Directory for caching
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping instance_id -> {file_path: content}
    """
    extraction_by_id = {e["instance_id"]: e for e in extractions}

    results: Dict[str, Dict[str, str]] = {}
    iterator = (
        tqdm(instances, desc="Extracting repo context")
        if show_progress
        else instances
    )

    for instance in iterator:
        ext = extraction_by_id.get(instance.instance_id, {})
        referenced_files = ext.get("referenced_files", [])

        context = extract_repo_context(
            instance=instance,
            referenced_files=referenced_files,
            output_dir=output_dir,
        )
        results[instance.instance_id] = context

        if context and show_progress:
            total_chars = sum(len(c) for c in context.values())
            tqdm.write(
                f"  {instance.instance_id}: {len(context)} files, "
                f"~{total_chars // 4} tokens"
            )

    extracted_count = sum(1 for ctx in results.values() if ctx)
    avg_tokens = 0
    if extracted_count:
        avg_tokens = sum(
            sum(len(c) for c in ctx.values()) // 4
            for ctx in results.values()
            if ctx
        ) // extracted_count
    print(f"Extracted repo context for {extracted_count}/{len(instances)} instances")
    print(f"Average repo context: ~{avg_tokens} tokens")

    return results


def load_repo_contexts(output_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load cached repo contexts from disk."""
    results: Dict[str, Dict[str, str]] = {}
    repo_dir = output_dir / "repo_context"
    if not repo_dir.exists():
        return results

    for instance_dir in repo_dir.iterdir():
        if instance_dir.is_dir():
            cache_file = instance_dir / "files.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    results[instance_dir.name] = json.load(f)

    return results
