"""Quality report generation for MemGym-IR pipeline output."""

import json
from pathlib import Path
from typing import Any


def generate_quality_report(
    output_dir: Path,
    instances: list,
    pipeline_stats: dict[str, Any] | None = None,
    verification_results: list[dict] | None = None,
) -> dict:
    """Generate quality report and save to output directory.

    Args:
        output_dir: Directory to save reports.
        instances: Final MemGymIRInstance objects.
        pipeline_stats: Stage-by-stage counts (filtered, prescreened, etc.)
        verification_results: Raw verification results from Stage 4.

    Returns:
        The full report dict.
    """
    report: dict[str, Any] = {}

    # 1. Pipeline summary
    stats = pipeline_stats or {}
    report["pipeline_summary"] = {
        "raw_loaded": stats.get("raw_loaded", "N/A"),
        "after_filter": stats.get("after_filter", "N/A"),
        "after_prescreen": stats.get("after_prescreen", "N/A"),
        "after_extract": stats.get("after_extract", "N/A"),
        "after_craft": stats.get("after_craft", "N/A"),
        "after_verify": stats.get("after_verify", "N/A"),
        "after_harden": stats.get("after_harden", "N/A"),
        "final_count": len(instances),
    }

    if not instances:
        report["dataset_statistics"] = {}
        report["per_instance"] = []
        _save(output_dir, report)
        return report

    # 2. Dataset statistics
    n = len(instances)
    report["dataset_statistics"] = {
        "count": n,
        "avg_hops": _avg(i.num_hops for i in instances),
        "avg_grounding_facts": _avg(len(i.grounding_facts) for i in instances),
        "avg_distractors": _avg(len(i.distractors) for i in instances),
        "avg_turns": _avg(len(i.turns) for i in instances),
        "avg_total_tokens": _avg(i.total_tokens for i in instances),
        "difficulty_distribution": _count_values(i.difficulty_preset for i in instances),
    }

    # 3. Ablation gap analysis
    if verification_results:
        scores_a = [r.get("score_a", 0) for r in verification_results if "score_a" in r]
        scores_b = [r.get("score_b", 0) for r in verification_results if "score_b" in r]
        scores_c = [r.get("score_c", 0) for r in verification_results if "score_c" in r]
        gaps_ab = [r.get("gap_ab", 0) for r in verification_results if "gap_ab" in r]
        gaps_ac = [r.get("gap_ac", 0) for r in verification_results if "gap_ac" in r]
        accepted = sum(1 for r in verification_results if r.get("accepted"))
        rejected = sum(1 for r in verification_results if not r.get("accepted"))
        report["ablation_analysis"] = {
            "total_verified": len(verification_results),
            "accepted": accepted,
            "rejected": rejected,
            "avg_score_a": _safe_avg(scores_a),
            "avg_score_b": _safe_avg(scores_b),
            "avg_score_c": _safe_avg(scores_c),
            "avg_gap_ab": _safe_avg(gaps_ab),
            "avg_gap_ac": _safe_avg(gaps_ac),
            "min_gap_ab": min(gaps_ab) if gaps_ab else None,
            "max_gap_ab": max(gaps_ab) if gaps_ab else None,
        }
    else:
        report["ablation_analysis"] = {"status": "verification_skipped"}

    # 4. Quality gates
    report["quality_gates"] = {
        "filter_rate": _rate(stats.get("after_filter"), stats.get("raw_loaded")),
        "prescreen_rejection_rate": _rejection_rate(
            stats.get("after_filter"), stats.get("after_prescreen")
        ),
        "extract_quality_gate_rate": _rate(
            stats.get("after_extract"), stats.get("after_prescreen")
        ),
        "verification_acceptance_rate": _rate(
            stats.get("after_verify"), stats.get("after_craft")
        ),
    }

    # 5. Distractor composition
    total_ctx = {"supporting": 0, "cross_instance": 0, "same_topic": 0,
                 "adversarial": 0, "contradictions": 0, "filler": 0}
    for inst in instances:
        if inst.context_tokens:
            total_ctx["supporting"] += inst.context_tokens.supporting_paragraphs
            total_ctx["cross_instance"] += inst.context_tokens.cross_instance_noise
            total_ctx["same_topic"] += inst.context_tokens.same_topic_noise
            total_ctx["adversarial"] += inst.context_tokens.adversarial_noise
            total_ctx["contradictions"] += inst.context_tokens.same_entity_contradictions
            total_ctx["filler"] += inst.context_tokens.filler_noise
    report["distractor_composition"] = {
        "total_tokens": total_ctx,
        "avg_tokens": {k: round(v / n) for k, v in total_ctx.items()},
    }

    # 6. Per-instance breakdown
    report["per_instance"] = []
    for inst in instances:
        entry = {
            "instance_id": inst.instance_id,
            "question": inst.question[:100],
            "answer": inst.answer,
            "num_hops": inst.num_hops,
            "num_facts": len(inst.grounding_facts),
            "num_distractors": len(inst.distractors),
            "num_turns": len(inst.turns),
            "total_tokens": inst.total_tokens,
            "difficulty": inst.difficulty_preset,
            "transforms": inst.transforms_applied,
        }
        if inst.verification:
            entry["score_a"] = inst.verification.get("score_a")
            entry["score_b"] = inst.verification.get("score_b")
            entry["score_c"] = inst.verification.get("score_c")
            entry["gap_ab"] = inst.verification.get("gap_ab")
            entry["gap_ac"] = inst.verification.get("gap_ac")
        report["per_instance"].append(entry)

    _save(output_dir, report)
    return report


def _save(output_dir: Path, report: dict) -> None:
    """Save report as JSON and Markdown."""
    # JSON
    json_path = output_dir / "quality_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown
    md_path = output_dir / "quality_report.md"
    with open(md_path, "w") as f:
        f.write(_render_markdown(report))

    print(f"  Quality report saved to {json_path} and {md_path}")


def _render_markdown(report: dict) -> str:
    lines = ["# MemGym-IR Quality Report", ""]

    # Pipeline summary
    ps = report.get("pipeline_summary", {})
    lines += ["## Pipeline Summary", ""]
    lines += ["| Stage | Count |", "|-------|-------|"]
    for key in ["raw_loaded", "after_filter", "after_prescreen", "after_extract",
                "after_craft", "after_verify", "after_harden", "final_count"]:
        label = key.replace("_", " ").title()
        lines.append(f"| {label} | {ps.get(key, 'N/A')} |")
    lines.append("")

    # Dataset statistics
    ds = report.get("dataset_statistics", {})
    if ds:
        lines += ["## Dataset Statistics", ""]
        lines += ["| Metric | Value |", "|--------|-------|"]
        for k, v in ds.items():
            if k != "difficulty_distribution":
                lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
        if "difficulty_distribution" in ds:
            lines.append(f"| Difficulty Distribution | {ds['difficulty_distribution']} |")
        lines.append("")

    # Ablation
    ab = report.get("ablation_analysis", {})
    if ab and ab.get("status") != "verification_skipped":
        lines += ["## Ablation Gap Analysis", ""]
        lines += ["| Metric | Value |", "|--------|-------|"]
        for k, v in ab.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
        lines.append("")
    elif ab.get("status") == "verification_skipped":
        lines += ["## Ablation Gap Analysis", "", "Verification was skipped.", ""]

    # Quality gates
    qg = report.get("quality_gates", {})
    if qg:
        lines += ["## Quality Gates", ""]
        lines += ["| Gate | Rate |", "|------|------|"]
        for k, v in qg.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
        lines.append("")

    # Distractor composition
    dc = report.get("distractor_composition", {})
    if dc:
        lines += ["## Distractor Composition (Avg Tokens)", ""]
        avg = dc.get("avg_tokens", {})
        lines += ["| Source | Avg Tokens |", "|--------|-----------|"]
        for k, v in avg.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v:,} |")
        lines.append("")

    # Per-instance
    instances = report.get("per_instance", [])
    if instances:
        lines += ["## Per-Instance Breakdown", ""]
        lines += ["| ID | Hops | Facts | Distractors | Tokens | Difficulty |",
                   "|----|------|-------|-------------|--------|------------|"]
        for inst in instances:
            lines.append(
                f"| {inst['instance_id']} | {inst['num_hops']} | "
                f"{inst['num_facts']} | {inst['num_distractors']} | "
                f"{inst['total_tokens']:,} | {inst['difficulty']} |"
            )
        lines.append("")

    return "\n".join(lines)


def _avg(gen):
    vals = list(gen)
    return round(sum(vals) / len(vals), 1) if vals else 0


def _safe_avg(vals):
    return round(sum(vals) / len(vals), 3) if vals else None


def _rate(num, denom):
    if num is None or denom is None or denom == 0:
        return "N/A"
    return f"{num}/{denom} ({num/denom*100:.0f}%)"


def _rejection_rate(before, after):
    if before is None or after is None or before == 0:
        return "N/A"
    rejected = before - after
    return f"{rejected}/{before} ({rejected/before*100:.0f}%)"


def _count_values(gen):
    counts: dict[str, int] = {}
    for v in gen:
        counts[v] = counts.get(v, 0) + 1
    return counts
