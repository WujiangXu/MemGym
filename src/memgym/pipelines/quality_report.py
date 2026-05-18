#!/usr/bin/env python3
"""
Quality analysis script for MemGym pipeline output (v2 schema).

Usage:
    python quality_report.py <instances.jsonl>
    python quality_report.py <instances.jsonl> --verbose
    python quality_report.py <instances.jsonl> --flag-only
    python quality_report.py <instances.jsonl> --check keyword_leak,patch_coverage
    python quality_report.py <instances.jsonl> --json-output results.json
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from memgym.pipelines.coding_synthetic.types.schemas import MemGymInstance
from memgym.pipelines.coding_synthetic.utils.patch_utils import reverse_patch, extract_files_from_patch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single quality check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_identifiers(text: str) -> List[str]:
    """Extract plausible code identifiers from fact content.

    Returns snake_case, camelCase, and dotted names (>=4 chars) that
    look like function names, variable names, class names, etc.
    """
    # Match identifiers: word chars with underscores/dots, at least 4 chars
    raw = re.findall(r'[A-Za-z_][A-Za-z0-9_.]{3,}', text)
    # Filter out common English words that happen to be long
    stopwords = {
        "this", "that", "with", "from", "have", "will", "should",
        "could", "would", "been", "being", "does", "than", "then",
        "when", "where", "which", "while", "before", "after",
        "about", "each", "some", "their", "there", "these", "those",
        "other", "such", "only", "also", "more", "most", "into",
        "over", "just", "like", "make", "made", "well", "back",
        "even", "still", "here", "every", "both", "many", "much",
        "must", "very", "need", "take", "come", "know", "want",
        "look", "find", "give", "tell", "call", "true", "false",
        "none", "null", "return", "function", "class", "method",
        "variable", "parameter", "argument", "value", "result",
        "error", "exception", "test", "file", "line", "code",
        "patch", "diff", "hunk", "change", "added", "removed",
    }
    return [w for w in raw if w.lower() not in stopwords]


def _count_patch_hunks(patch_text: str) -> int:
    """Count the number of hunks in a unified diff."""
    return len(re.findall(r'^@@\s', patch_text, re.MULTILINE))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_instances(jsonl_path: str) -> List[MemGymInstance]:
    """Load MemGymInstance objects from a JSONL file."""
    instances = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                instances.append(MemGymInstance(**data))
            except Exception as e:
                print(f"WARNING: Failed to parse line {line_num}: {e}")
    return instances


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_keyword_leak(instance: MemGymInstance) -> CheckResult:
    """Check A: task_prompt doesn't leak keywords from critical external
    grounding facts (discoverable=False AND relevance='critical')."""
    # Identify critical external facts
    critical_external = [
        f for f in instance.grounding_facts
        if not f.discoverable and f.relevance == "critical"
    ]

    if not critical_external:
        return CheckResult(
            name="keyword_leak",
            passed=True,
            message="No critical external grounding facts to check",
            details={"num_critical_external": 0, "leaked_keywords": []},
        )

    prompt_lower = instance.task_prompt.lower()
    leaked: List[str] = []

    for fact in critical_external:
        identifiers = _extract_identifiers(fact.content)
        for ident in identifiers:
            if ident.lower() in prompt_lower:
                leaked.append(f"{fact.id}:{ident}")

    total_identifiers = sum(
        len(_extract_identifiers(f.content)) for f in critical_external
    )
    leak_ratio = len(leaked) / total_identifiers if total_identifiers > 0 else 0.0

    passed = len(leaked) == 0
    if passed:
        message = (
            f"No keyword leaks from {len(critical_external)} "
            f"critical external facts ({total_identifiers} identifiers checked)"
        )
    else:
        message = (
            f"{len(leaked)} keywords leaked in task_prompt "
            f"(leak ratio: {leak_ratio:.0%}): {', '.join(leaked[:5])}"
        )

    return CheckResult(
        name="keyword_leak",
        passed=passed,
        message=message,
        details={
            "num_critical_external": len(critical_external),
            "leaked_keywords": leaked,
            "total_identifiers": total_identifiers,
            "leak_ratio": leak_ratio,
        },
    )


def check_memory_distribution(instance: MemGymInstance) -> CheckResult:
    """Check B: grounding facts are distributed across multiple memory files.

    Fail if all facts appear in a single file.
    """
    if not instance.grounding_facts:
        return CheckResult(
            name="memory_distribution",
            passed=False,
            message="No grounding facts",
            details={"total_facts": 0, "files_with_facts": 0},
        )

    if not instance.memory_files:
        return CheckResult(
            name="memory_distribution",
            passed=False,
            message="No memory files",
            details={"total_facts": len(instance.grounding_facts), "files_with_facts": 0},
        )

    # For each file, check which facts appear in its content
    facts_per_file: Dict[str, List[str]] = {}
    for filename, content in instance.memory_files.items():
        content_lower = content.lower()
        found = []
        for fact in instance.grounding_facts:
            # Check if a meaningful portion of the fact content appears in the file
            # Use first 40 chars (lowered) as a fingerprint to avoid partial matches
            snippet = fact.content[:40].lower().strip()
            if len(snippet) >= 10 and snippet in content_lower:
                found.append(fact.id)
        if found:
            facts_per_file[filename] = found

    files_with_facts = len(facts_per_file)
    max_in_one = max(len(v) for v in facts_per_file.values()) if facts_per_file else 0
    total_facts = len(instance.grounding_facts)
    concentration = max_in_one / total_facts if total_facts > 0 else 1.0

    passed = files_with_facts >= 2
    if not passed:
        if files_with_facts == 0:
            message = "No grounding facts found in any memory file"
        else:
            message = f"All facts concentrated in 1 file (concentration: {concentration:.0%})"
    else:
        message = (
            f"Facts distributed across {files_with_facts} files, "
            f"max concentration: {concentration:.0%}"
        )

    return CheckResult(
        name="memory_distribution",
        passed=passed,
        message=message,
        details={
            "files_with_facts": files_with_facts,
            "total_facts": total_facts,
            "max_concentration": concentration,
            "facts_per_file": {k: len(v) for k, v in facts_per_file.items()},
        },
    )


def check_patch_coverage(instance: MemGymInstance) -> CheckResult:
    """Check C: grounding facts' hunks_enabled cover >=50% of patch hunks."""
    num_hunks = _count_patch_hunks(instance.patch)

    if num_hunks == 0:
        return CheckResult(
            name="patch_coverage",
            passed=False,
            message="Patch has no hunks",
            details={"num_hunks": 0, "coverage_ratio": 0.0},
        )

    # Collect all hunk indices covered by grounding facts
    covered_indices: set = set()
    for fact in instance.grounding_facts:
        for idx in fact.hunks_enabled:
            covered_indices.add(idx)

    all_indices = set(range(num_hunks))
    covered = covered_indices & all_indices
    uncovered = all_indices - covered_indices
    ratio = len(covered) / num_hunks

    passed = ratio >= 0.5
    message = (
        f"Patch coverage: {ratio:.0%} "
        f"({len(covered)}/{num_hunks} hunks covered, "
        f"{len(uncovered)} uncovered)"
    )

    return CheckResult(
        name="patch_coverage",
        passed=passed,
        message=message,
        details={
            "num_hunks": num_hunks,
            "covered_hunks": sorted(covered),
            "uncovered_hunks": sorted(uncovered),
            "coverage_ratio": ratio,
        },
    )


def check_referenced_files(instance: MemGymInstance) -> CheckResult:
    """Check D: overlap between referenced_files and files in patch."""
    ref_files = instance.referenced_files
    patch_files = extract_files_from_patch(instance.patch)

    num_ref = len(ref_files)
    overlap = set(patch_files) & set(ref_files)

    if not patch_files:
        return CheckResult(
            name="referenced_files",
            passed=num_ref >= 1,
            message=f"{num_ref} referenced files, no files extracted from patch",
            details={
                "num_referenced": num_ref,
                "patch_files": [],
                "overlap_with_patch": [],
            },
        )

    overlap_ratio = len(overlap) / len(patch_files)

    passed = num_ref >= 1 and len(overlap) > 0
    message = (
        f"{num_ref} referenced files, "
        f"{len(overlap)}/{len(patch_files)} patch files covered "
        f"({overlap_ratio:.0%})"
    )

    return CheckResult(
        name="referenced_files",
        passed=passed,
        message=message,
        details={
            "num_referenced": num_ref,
            "patch_files": patch_files,
            "overlap_with_patch": sorted(overlap),
            "overlap_ratio": overlap_ratio,
        },
    )


def check_gold_fix_validity(instance: MemGymInstance) -> CheckResult:
    """Check E: gold_fix non-empty, contains diff markers, matches
    reverse_patch(patch)."""
    if not instance.gold_fix or not instance.gold_fix.strip():
        return CheckResult(
            name="gold_fix_validity",
            passed=False,
            message="gold_fix is empty",
            details={"has_gold_fix": False},
        )

    expected_fix = reverse_patch(instance.patch)
    matches = instance.gold_fix.strip() == expected_fix.strip()

    has_diff_markers = any(
        line.startswith(("---", "+++", "@@"))
        for line in instance.gold_fix.split("\n")
    )

    issues = []
    if not matches:
        issues.append("gold_fix does not match reverse_patch(patch)")
    if not has_diff_markers:
        issues.append("gold_fix lacks diff markers")

    passed = len(issues) == 0
    message = "Gold fix is valid" if passed else "; ".join(issues)

    return CheckResult(
        name="gold_fix_validity",
        passed=passed,
        message=message,
        details={
            "has_gold_fix": True,
            "matches_reversed_patch": matches,
            "has_diff_markers": has_diff_markers,
        },
    )


def check_fact_id_uniformity(instance: MemGymInstance) -> CheckResult:
    """Check F: memory file content doesn't contain [F or [D prefixed IDs
    that would leak whether an item is a real fact vs distractor."""
    leaked_ids: List[str] = []

    # Pattern: [F1], [D3], [F12], etc. -- the bracketed ID format
    id_pattern = re.compile(r'\[([FD]\d+)\]')

    for filename, content in instance.memory_files.items():
        matches = id_pattern.findall(content)
        for m in matches:
            leaked_ids.append(f"{filename}:{m}")

    passed = len(leaked_ids) == 0
    if passed:
        message = "No fact/distractor ID prefixes leaked in memory files"
    else:
        message = (
            f"{len(leaked_ids)} leaked IDs found in memory files: "
            f"{', '.join(leaked_ids[:5])}"
        )

    return CheckResult(
        name="fact_id_uniformity",
        passed=passed,
        message=message,
        details={"leaked_ids": leaked_ids},
    )


def check_distractor_ratio(instance: MemGymInstance) -> CheckResult:
    """Check G: >=2 distractors and ratio >=30% of total
    (grounding_facts + distractors)."""
    num_facts = len(instance.grounding_facts)
    num_distractors = len(instance.distractors)
    total = num_facts + num_distractors
    ratio = num_distractors / total if total > 0 else 0.0

    passed = num_distractors >= 2 and ratio >= 0.3
    message = (
        f"{num_distractors} distractors / {total} total = {ratio:.0%} ratio"
    )
    if not passed:
        if num_distractors < 2:
            message += " (fewer than 2 distractors)"
        elif ratio < 0.3:
            message += " (ratio below 30%)"

    return CheckResult(
        name="distractor_ratio",
        passed=passed,
        message=message,
        details={
            "num_grounding_facts": num_facts,
            "num_distractors": num_distractors,
            "total": total,
            "ratio": ratio,
        },
    )


def check_ablation_gap(instance: MemGymInstance) -> CheckResult:
    """Check H: if verification exists, check cond_a_llm_score >= 0.4
    and gap (cond_a - cond_b) >= 0.30."""
    if instance.verification is None:
        return CheckResult(
            name="ablation_gap",
            passed=True,
            message="No verification data (skipped)",
            details={"has_verification": False},
        )

    cond_a = instance.verification.get("cond_a_llm_score")
    cond_b = instance.verification.get("cond_b_llm_score")

    if cond_a is None or cond_b is None:
        return CheckResult(
            name="ablation_gap",
            passed=False,
            message=(
                f"Verification missing scores: "
                f"cond_a={cond_a}, cond_b={cond_b}"
            ),
            details={
                "has_verification": True,
                "cond_a_llm_score": cond_a,
                "cond_b_llm_score": cond_b,
            },
        )

    gap = cond_a - cond_b
    issues = []
    if cond_a < 0.4:
        issues.append(f"cond_a_llm_score={cond_a:.2f} < 0.40")
    if gap < 0.30:
        issues.append(f"gap={gap:.2f} < 0.30")

    passed = len(issues) == 0
    if passed:
        message = (
            f"Ablation OK: cond_a={cond_a:.2f}, "
            f"cond_b={cond_b:.2f}, gap={gap:.2f}"
        )
    else:
        message = (
            f"Ablation FAIL: cond_a={cond_a:.2f}, "
            f"cond_b={cond_b:.2f}, gap={gap:.2f} -- "
            + "; ".join(issues)
        )

    return CheckResult(
        name="ablation_gap",
        passed=passed,
        message=message,
        details={
            "has_verification": True,
            "cond_a_llm_score": cond_a,
            "cond_b_llm_score": cond_b,
            "gap": gap,
        },
    )


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

ALL_CHECKS: List[Callable] = [
    check_keyword_leak,           # A
    check_memory_distribution,    # B
    check_patch_coverage,         # C
    check_referenced_files,       # D
    check_gold_fix_validity,      # E
    check_fact_id_uniformity,     # F (new)
    check_distractor_ratio,       # G
    check_ablation_gap,           # H (new)
]

CHECK_NAMES = {fn.__name__.replace("check_", ""): fn for fn in ALL_CHECKS}


def run_checks(
    instance: MemGymInstance,
    selected_checks: Optional[List[str]] = None,
) -> Dict[str, CheckResult]:
    """Run all or selected checks on one instance."""
    checks = ALL_CHECKS
    if selected_checks:
        checks = [CHECK_NAMES[n] for n in selected_checks if n in CHECK_NAMES]

    results = {}
    for check_fn in checks:
        name = check_fn.__name__.replace("check_", "")
        try:
            results[name] = check_fn(instance)
        except Exception as e:
            results[name] = CheckResult(
                name=name, passed=False,
                message=f"Exception: {e}",
                details={"error": str(e)},
            )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_instance_detail(
    instance: MemGymInstance, results: Dict[str, CheckResult]
) -> None:
    """Print detailed per-instance report."""
    status = "PASS" if all(r.passed for r in results.values()) else "FAIL"
    print(f"\n{'='*70}")
    print(f"[{status}] {instance.instance_id}")
    print(
        f"  Difficulty: {instance.difficulty_preset} | "
        f"Facts: {len(instance.grounding_facts)} | "
        f"Distractors: {len(instance.distractors)} | "
        f"Memory files: {len(instance.memory_files)} | "
        f"Ref files: {len(instance.referenced_files)}"
    )
    print(f"  Base: {instance.base_instance}")
    if instance.transforms_applied:
        print(f"  Transforms: {', '.join(instance.transforms_applied)}")
    print()

    for name, result in results.items():
        icon = "PASS" if result.passed else "FAIL"
        print(f"  [{icon}] {name}: {result.message}")

    # Show task prompt preview
    if instance.task_prompt:
        preview = instance.task_prompt[:200].replace("\n", " ")
        print(f"\n  Task prompt preview: {preview}...")


def aggregate_report(
    all_results: Dict[str, Dict[str, CheckResult]],
    instances: List[MemGymInstance],
) -> None:
    """Print aggregate quality statistics."""
    n = len(all_results)
    if n == 0:
        print("No instances to report on.")
        return

    print("\n" + "=" * 70)
    print(f"QUALITY REPORT: {n} instances")
    print("=" * 70)

    # --- Pass rates ---
    check_names = list(dict.fromkeys(
        name for results in all_results.values() for name in results
    ))

    print("\n--- Pass Rates ---")
    for check in check_names:
        passed = sum(
            1 for r in all_results.values()
            if r.get(check) and r[check].passed
        )
        rate = passed / n
        bar_len = 40
        filled = int(rate * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"  {check:<25s} {passed:>3d}/{n:<3d} ({rate:>6.1%}) [{bar}]")

    fully_clean = sum(
        1 for r in all_results.values() if all(cr.passed for cr in r.values())
    )
    print(f"\n  Fully clean instances:    {fully_clean}/{n} ({fully_clean/n:.1%})")

    # --- Distributions ---
    print("\n--- Distributions ---")

    # Difficulty presets
    presets = defaultdict(int)
    for inst in instances:
        presets[inst.difficulty_preset] += 1
    print(f"  Difficulty presets: {dict(presets)}")

    # Patch coverage
    coverages = []
    for r in all_results.values():
        if "patch_coverage" in r:
            ratio = r["patch_coverage"].details.get("coverage_ratio", 0)
            coverages.append(ratio)
    if coverages:
        print(
            f"  Patch coverage:  min={min(coverages):.0%}  "
            f"median={statistics.median(coverages):.0%}  "
            f"mean={statistics.mean(coverages):.0%}  "
            f"max={max(coverages):.0%}"
        )

    # Grounding facts per instance
    fact_counts = [len(inst.grounding_facts) for inst in instances]
    if fact_counts:
        print(
            f"  Grounding facts: min={min(fact_counts)}  "
            f"median={statistics.median(fact_counts):.0f}  "
            f"mean={statistics.mean(fact_counts):.1f}  "
            f"max={max(fact_counts)}"
        )

    # Distractor counts
    dist_counts = [len(inst.distractors) for inst in instances]
    if dist_counts:
        print(
            f"  Distractors:     min={min(dist_counts)}  "
            f"median={statistics.median(dist_counts):.0f}  "
            f"mean={statistics.mean(dist_counts):.1f}  "
            f"max={max(dist_counts)}"
        )

    # Memory file counts
    file_counts = [len(inst.memory_files) for inst in instances]
    if file_counts:
        print(
            f"  Memory files:    min={min(file_counts)}  "
            f"median={statistics.median(file_counts):.0f}  "
            f"mean={statistics.mean(file_counts):.1f}  "
            f"max={max(file_counts)}"
        )

    # Referenced files
    ref_counts = [len(inst.referenced_files) for inst in instances]
    if ref_counts:
        print(
            f"  Referenced files: min={min(ref_counts)}  "
            f"median={statistics.median(ref_counts):.0f}  "
            f"mean={statistics.mean(ref_counts):.1f}  "
            f"max={max(ref_counts)}"
        )

    # Ablation gap (for instances with verification)
    gaps = []
    for r in all_results.values():
        if "ablation_gap" in r:
            gap_val = r["ablation_gap"].details.get("gap")
            if gap_val is not None:
                gaps.append(gap_val)
    if gaps:
        print(
            f"  Ablation gap:    min={min(gaps):.2f}  "
            f"median={statistics.median(gaps):.2f}  "
            f"mean={statistics.mean(gaps):.2f}  "
            f"max={max(gaps):.2f}"
        )

    # --- Flagged instances ---
    flagged = {
        iid: [name for name, cr in results.items() if not cr.passed]
        for iid, results in all_results.items()
        if any(not cr.passed for cr in results.values())
    }

    if flagged:
        print(f"\n--- Flagged Instances ({len(flagged)}/{n}) ---")
        for iid, failures in flagged.items():
            print(f"  {iid}: [{', '.join(failures)}]")
    else:
        print("\n--- All instances passed all checks ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quality analysis for Coding-Synthetic pipeline output (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quality_report.py output/coding_synthetic_instances.jsonl
  python quality_report.py output/coding_synthetic_instances.jsonl --verbose
  python quality_report.py output/coding_synthetic_instances.jsonl --flag-only
  python quality_report.py output/coding_synthetic_instances.jsonl --check keyword_leak,patch_coverage
        """,
    )
    parser.add_argument(
        "input", help="Path to JSONL file with MemGymInstance objects"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-instance reports",
    )
    parser.add_argument(
        "--flag-only", action="store_true",
        help="Only print flagged (failing) instances",
    )
    parser.add_argument(
        "--check", type=str, default=None,
        help="Comma-separated list of checks to run (default: all)",
    )
    parser.add_argument(
        "--json-output", type=str, default=None,
        help="Write full results as JSON to this path",
    )

    args = parser.parse_args()
    selected_checks = args.check.split(",") if args.check else None

    # Validate selected check names
    if selected_checks:
        unknown = [c for c in selected_checks if c not in CHECK_NAMES]
        if unknown:
            print(
                f"ERROR: Unknown check(s): {unknown}. "
                f"Available: {sorted(CHECK_NAMES.keys())}"
            )
            sys.exit(1)

    # Load
    instances = load_instances(args.input)
    print(f"Loaded {len(instances)} instances from {args.input}")
    if not instances:
        sys.exit(1)

    # Run checks
    all_results: Dict[str, Dict[str, CheckResult]] = {}
    for instance in instances:
        results = run_checks(instance, selected_checks)
        all_results[instance.instance_id] = results

        if args.verbose:
            print_instance_detail(instance, results)
        elif args.flag_only:
            failures = [n for n, cr in results.items() if not cr.passed]
            if failures:
                print(f"FLAGGED {instance.instance_id}: {failures}")

    # Aggregate
    aggregate_report(all_results, instances)

    # JSON output
    if args.json_output:
        json_data = {}
        for iid, results in all_results.items():
            json_data[iid] = {
                name: {
                    "passed": cr.passed,
                    "message": cr.message,
                    "details": cr.details,
                }
                for name, cr in results.items()
            }
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"\nJSON results written to {args.json_output}")


if __name__ == "__main__":
    main()
