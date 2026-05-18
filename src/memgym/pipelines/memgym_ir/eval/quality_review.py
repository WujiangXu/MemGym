"""Quality review script for MemGym-IR instances.

Evaluates generated instances using LLM-based criteria and produces
a human-readable markdown report for manual inspection.

Usage:
    python -m pipelines.memgym_ir.quality_review /tmp/ir_100_instances.jsonl
    python -m pipelines.memgym_ir.quality_review data.jsonl --skip-llm
    python -m pipelines.memgym_ir.quality_review data.jsonl --instances 0,1,2
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

from ..llm.client import LLMClient
from ..llm.prompts import (
    QUALITY_MULTIHOP_PROMPT,
    QUALITY_MULTIHOP_SCHEMA,
    QUALITY_DISTRACTOR_PROMPT,
    QUALITY_DISTRACTOR_SCHEMA,
    QUALITY_ANSWER_PROMPT,
    QUALITY_ANSWER_SCHEMA,
    QUALITY_OVERALL_PROMPT,
    QUALITY_OVERALL_SCHEMA,
)


# =========================================================================
# Data Loading
# =========================================================================


def load_instances(path: str) -> List[Dict]:
    """Load instances from JSONL file."""
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


# =========================================================================
# LLM Evaluation Functions
# =========================================================================


def _format_decomposition(inst: Dict) -> str:
    parts = []
    for i, hop in enumerate(inst.get("decomposition", [])):
        parts.append(
            f"Hop {i}: Q: {hop.get('sub_question', '')}\n"
            f"  A: {hop.get('sub_answer', '')}\n"
            f"  Source: {hop.get('paragraph_title', 'N/A')}"
        )
    return "\n".join(parts)


def _format_supporting(inst: Dict) -> str:
    parts = []
    for p in inst.get("gold_supporting_paragraphs", []):
        parts.append(f"[{p.get('title', '')}]\n{p.get('text', '')[:3000]}")
    return "\n\n".join(parts)


def _format_composition(inst: Dict) -> str:
    ct = inst.get("context_tokens") or {}
    if isinstance(ct, dict):
        return (
            f"supporting={ct.get('supporting_paragraphs', 0)}, "
            f"cross_instance={ct.get('cross_instance_noise', 0)}, "
            f"same_topic={ct.get('same_topic_noise', 0)}, "
            f"adversarial={ct.get('adversarial_noise', 0)}, "
            f"contradictions={ct.get('same_entity_contradictions', 0)}, "
            f"total={ct.get('total', 0)}"
        )
    return "N/A"


def _get_distractors_by_source(inst: Dict) -> Dict[str, List[Dict]]:
    """Group distractors by source type."""
    by_source = {"cross_instance": [], "same_topic": [], "adversarial": [],
                 "same_entity": [], "sibling_near_miss": [], "filler": []}
    for d in inst.get("distractors", []):
        source = d.get("source", "cross_instance")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(d)
    return by_source


def _sample_distractors(inst: Dict, source: str, n: int = 3) -> str:
    """Get sample distractor text for a source type."""
    by_source = _get_distractors_by_source(inst)
    distractors = by_source.get(source, [])
    samples = distractors[:n]
    if not samples:
        return "(none)"
    parts = []
    for d in samples:
        text = d.get("content", "")[:200]
        parts.append(f"- {text}...")
    return "\n".join(parts)


async def evaluate_multihop(client: LLMClient, inst: Dict) -> Dict:
    prompt = QUALITY_MULTIHOP_PROMPT.format(
        question=inst.get("question", ""),
        answer=inst.get("answer", ""),
        decomposition=_format_decomposition(inst),
        paragraphs=_format_supporting(inst),
    )
    return await client.get_json_completion_async(
        prompt=prompt, response_format=QUALITY_MULTIHOP_SCHEMA, temperature=0.0,
    )


async def evaluate_distractors(client: LLMClient, inst: Dict) -> Dict:
    by_source = _get_distractors_by_source(inst)

    # Deep-research instances use sibling_near_miss + filler; legacy uses cross_instance + adversarial
    siblings = by_source.get("sibling_near_miss", [])
    fillers = by_source.get("filler", [])
    is_deep_research = bool(siblings or fillers)

    if is_deep_research:
        # Show sibling near-miss distractors (the challenging ones)
        sib_text = "\n".join(
            f"- [sibling_near_miss] {d.get('content', '')[:200]}..." for d in siblings[:5]
        ) if siblings else "(none)"
        # Show sample filler (search-retrieved papers from Scale stage)
        filler_text = "\n".join(
            f"- [filler] {d.get('content', '')[:200]}..." for d in fillers[:5]
        ) if fillers else "(none)"
        cross_text = sib_text
        adv_text = filler_text
    else:
        # Legacy format
        cross_samples = by_source["cross_instance"][:5]
        cross_text = "\n".join(
            f"- {d.get('content', '')[:150]}..." for d in cross_samples
        ) if cross_samples else "(none)"
        adv = by_source["adversarial"] + by_source["same_entity"]
        adv_text = "\n".join(
            f"- [{d.get('source', '')}] {d.get('content', '')[:200]}..." for d in adv
        ) if adv else "(none)"

    prompt = QUALITY_DISTRACTOR_PROMPT.format(
        question=inst.get("question", ""),
        answer=inst.get("answer", ""),
        supporting=_format_supporting(inst),
        cross_instance_sample=cross_text,
        adversarial_sample=adv_text,
        composition=_format_composition(inst),
    )
    return await client.get_json_completion_async(
        prompt=prompt, response_format=QUALITY_DISTRACTOR_SCHEMA, temperature=0.0,
    )


async def evaluate_answer(client: LLMClient, inst: Dict) -> Dict:
    prompt = QUALITY_ANSWER_PROMPT.format(
        question=inst.get("question", ""),
        answer=inst.get("answer", ""),
        decomposition=_format_decomposition(inst),
        paragraphs=_format_supporting(inst),
    )
    return await client.get_json_completion_async(
        prompt=prompt, response_format=QUALITY_ANSWER_SCHEMA, temperature=0.0,
    )


async def evaluate_overall(client: LLMClient, inst: Dict, sub_scores: Dict) -> Dict:
    v = inst.get("verification") or {}
    # Support both schemas: deep_research (score_all_memory/...) and legacy (score_a/...).
    score_a = v.get("score_all_memory", v.get("score_a", "N/A"))
    score_b = v.get("score_no_memory", v.get("score_b", "N/A"))
    score_c = v.get("score_long_context", v.get("score_c", "N/A"))
    gap_ab = v.get("memory_gap", v.get("gap_ab", "N/A"))
    prompt = QUALITY_OVERALL_PROMPT.format(
        question=inst.get("question", ""),
        answer=inst.get("answer", ""),
        num_hops=inst.get("num_hops", 0),
        num_facts=len(inst.get("grounding_facts", [])),
        score_a=score_a,
        score_b=score_b,
        score_c=score_c,
        gap_ab=gap_ab,
        composition=_format_composition(inst),
        transforms=", ".join(inst.get("transforms_applied", [])),
        multihop_score=sub_scores.get("multihop", {}).get("score", "N/A"),
        distractor_score=sub_scores.get("distractor", {}).get("score", "N/A"),
        answer_score=sub_scores.get("answer", {}).get("score", "N/A"),
    )
    return await client.get_json_completion_async(
        prompt=prompt, response_format=QUALITY_OVERALL_SCHEMA, temperature=0.0,
    )


async def evaluate_instance(client: LLMClient, inst: Dict) -> Dict:
    """Run all 4 evaluation criteria for one instance."""
    # Run first 3 criteria in parallel
    multihop, distractor, answer = await asyncio.gather(
        evaluate_multihop(client, inst),
        evaluate_distractors(client, inst),
        evaluate_answer(client, inst),
    )
    sub_scores = {"multihop": multihop, "distractor": distractor, "answer": answer}
    # Overall depends on sub-scores
    overall = await evaluate_overall(client, inst, sub_scores)
    return {
        "instance_id": inst.get("instance_id", ""),
        "multihop": multihop,
        "distractor": distractor,
        "answer": answer,
        "overall": overall,
    }


def evaluate_all(
    client: LLMClient,
    instances: List[Dict],
    max_concurrent: int = 5,
) -> List[Dict]:
    """Evaluate all instances with async parallelism."""
    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        results = []

        async def _one(inst):
            async with sem:
                return await evaluate_instance(client, inst)

        results = await asyncio.gather(*[_one(inst) for inst in instances])
        return list(results)

    return asyncio.run(_run())


# =========================================================================
# Markdown Formatting
# =========================================================================


def format_instance_markdown(inst: Dict, eval_result: Optional[Dict] = None) -> str:
    """Format a single instance as a markdown section."""
    lines = []
    iid = inst.get("instance_id", "unknown")
    lines.append(f"## {iid}")
    lines.append("")

    # Basic info
    lines.append(f"**Question**: {inst.get('question', '')}")
    if inst.get("original_question") and inst["original_question"] != inst["question"]:
        lines.append(f"**Original Question**: {inst.get('original_question', '')}")
    lines.append(f"**Answer**: {inst.get('answer', '')}")

    v = inst.get("verification") or {}
    # Support both deep_research and legacy schemas.
    all_mem = v.get("score_all_memory", v.get("score_a", "?"))
    no_mem = v.get("score_no_memory", v.get("score_b", "?"))
    long_ctx = v.get("score_long_context", v.get("score_c", "?"))
    gap = v.get("memory_gap", v.get("gap_ab", "?"))
    hack = v.get("hack_score", "?")
    lines.append(
        f"**Hops**: {inst.get('num_hops', '?')} | "
        f"**Difficulty**: {inst.get('difficulty_preset', '?')} | "
        f"**Verification**: all_mem={all_mem}, no_mem={no_mem}, "
        f"long_ctx={long_ctx}, gap={gap}, hack={hack}"
    )
    lines.append(f"**Transforms**: {', '.join(inst.get('transforms_applied', []))}")
    lines.append("")

    # Reasoning chain
    decomp = inst.get("decomposition", [])
    if decomp:
        lines.append("### Reasoning Chain")
        lines.append("| Hop | Sub-question | Sub-answer | Source |")
        lines.append("|-----|-------------|------------|--------|")
        for i, hop in enumerate(decomp):
            sq = hop.get("sub_question", "")[:80]
            sa = hop.get("sub_answer", "")[:50]
            sp = hop.get("paragraph_title", "")[:40]
            lines.append(f"| {i} | {sq} | {sa} | {sp} |")
        lines.append("")

    # Grounding facts
    facts = inst.get("grounding_facts", [])
    if facts:
        lines.append("### Grounding Facts")
        for f in facts:
            span = f.get("retention_span", 0)
            avail = f.get("first_available_at_hop", 0)
            needed = f.get("needed_at_hop", 0)
            disc = f.get("is_discoverable", False)
            lines.append(
                f"- **{f.get('fact_id', '?')}** ({f.get('fact_type', '?')}): "
                f"{f.get('content', '')[:120]}"
            )
            lines.append(
                f"  - Span={span} (available@{avail} → needed@{needed}), "
                f"discoverable={disc}"
            )
        lines.append("")

    # Supporting paragraphs (collapsible)
    gold = inst.get("gold_supporting_paragraphs", [])
    if gold:
        lines.append("### Supporting Paragraphs")
        for p in gold:
            title = p.get("title", "Untitled")
            text = p.get("text", "")
            lines.append(f"<details><summary>{title} ({len(text.split())} words)</summary>")
            lines.append("")
            lines.append(text)
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # Distractor breakdown
    by_source = _get_distractors_by_source(inst)
    ct = inst.get("context_tokens") or {}
    total_d = len(inst.get("distractors", []))

    lines.append("### Distractor Breakdown")
    lines.append(f"- **Total**: {total_d} distractors ({inst.get('total_tokens', '?')} tokens)")
    if isinstance(ct, dict):
        lines.append(f"- Cross-instance: {len(by_source['cross_instance'])} ({ct.get('cross_instance_noise', 0)} tokens)")
        lines.append(f"- Same-topic: {len(by_source['same_topic'])} ({ct.get('same_topic_noise', 0)} tokens)")
        lines.append(f"- Adversarial: {len(by_source['adversarial'])} ({ct.get('adversarial_noise', 0)} tokens)")
        lines.append(f"- Same-entity: {len(by_source['same_entity'])} ({ct.get('same_entity_contradictions', 0)} tokens)")
    lines.append("")

    # Show all adversarial/same-entity distractors (full text, since few)
    adv_distractors = by_source.get("adversarial", []) + by_source.get("same_entity", [])
    if adv_distractors:
        lines.append("### Adversarial/Same-Entity Distractors (full text)")
        for d in adv_distractors:
            lines.append(f"**[{d.get('source', '')}]** interferes_with={d.get('interferes_with', 'N/A')}")
            lines.append(f"> {d.get('content', '')[:500]}")
            lines.append("")

    # Sample cross-instance distractors (titles only)
    cross = by_source.get("cross_instance", [])[:3]
    if cross:
        lines.append("### Sample Cross-Instance Distractors")
        for d in cross:
            text = d.get("content", "")[:100]
            lines.append(f"- {text}...")
        lines.append("")

    # Search turns (raw agent memory view)
    search_turns = inst.get("turns", [])
    if search_turns:
        lines.append("### Search Turns (Agent Memory View)")
        for turn in search_turns:
            tidx = turn.get("turn_index", "?")
            sq = turn.get("sub_query", "")
            lines.append(f"**Turn {tidx}**: {sq}")
            docs = turn.get("documents", [])
            lines.append(f"*{len(docs)} documents returned:*")
            lines.append("")
            for j, doc in enumerate(docs):
                title = doc.get("title", "Untitled")
                text = doc.get("text", "")[:100]
                is_gold = doc.get("is_supporting", False)
                prefix = "[GOLD] " if is_gold else ""
                lines.append(f"  {j+1}. {prefix}**{title}**: {text}...")
            lines.append("")

    # LLM evaluation results
    if eval_result:
        lines.append("### LLM Quality Assessment")
        lines.append("| Criterion | Score | Key Finding |")
        lines.append("|-----------|-------|-------------|")

        mh = eval_result.get("multihop", {})
        lines.append(f"| Multi-hop genuineness | {mh.get('score', '?')}/5 | {mh.get('reasoning', '')[:60]} |")

        di = eval_result.get("distractor", {})
        lines.append(f"| Distractor difficulty | {di.get('score', '?')}/5 | {di.get('reasoning', '')[:60]} |")

        an = eval_result.get("answer", {})
        lines.append(f"| Answer correctness | {an.get('score', '?')}/5 | {an.get('reasoning', '')[:60]} |")

        ov = eval_result.get("overall", {})
        lines.append(f"| **Overall** | **{ov.get('score', '?')}/5** | **{ov.get('recommendation', '?')}** |")
        lines.append("")

        issues = ov.get("issues", [])
        if issues:
            lines.append("**Issues**: " + "; ".join(issues))
            lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def format_aggregate_markdown(
    instances: List[Dict],
    eval_results: Optional[List[Dict]] = None,
) -> str:
    """Format aggregate quality summary."""
    lines = []
    lines.append("# MemGym-IR Quality Review Report")
    lines.append("")
    lines.append(f"**Instances reviewed**: {len(instances)}")
    lines.append("")

    # Hop distribution
    hop_counts = {}
    for inst in instances:
        h = inst.get("num_hops", 0)
        hop_counts[h] = hop_counts.get(h, 0) + 1
    lines.append("## Instance Distribution")
    lines.append(f"- Hop counts: {dict(sorted(hop_counts.items()))}")

    # Token stats
    tokens = [inst.get("total_tokens", 0) for inst in instances]
    if tokens:
        lines.append(f"- Avg tokens: {sum(tokens)/len(tokens):.0f} (min={min(tokens)}, max={max(tokens)})")

    # Distractor composition
    ct_totals = {"supporting": 0, "cross_instance": 0, "same_topic": 0,
                 "adversarial": 0, "contradictions": 0}
    for inst in instances:
        ct = inst.get("context_tokens") or {}
        if isinstance(ct, dict):
            ct_totals["supporting"] += ct.get("supporting_paragraphs", 0)
            ct_totals["cross_instance"] += ct.get("cross_instance_noise", 0)
            ct_totals["same_topic"] += ct.get("same_topic_noise", 0)
            ct_totals["adversarial"] += ct.get("adversarial_noise", 0)
            ct_totals["contradictions"] += ct.get("same_entity_contradictions", 0)

    total_ct = sum(ct_totals.values())
    if total_ct > 0:
        lines.append("")
        lines.append("## Distractor Composition (avg % of tokens)")
        for k, v in ct_totals.items():
            pct = v / total_ct * 100
            avg = v / len(instances)
            lines.append(f"- {k}: {avg:.0f} tokens ({pct:.1f}%)")

    # Verification score stats (deep-research schema:
    # score_no_memory / score_all_memory / score_long_context / memory_gap / hack_score)
    scores_no_mem = []
    scores_all_mem = []
    scores_long_ctx = []
    gaps = []
    hacks = []
    for inst in instances:
        v = inst.get("verification") or {}
        if "score_all_memory" in v:
            scores_all_mem.append(v["score_all_memory"])
            scores_no_mem.append(v.get("score_no_memory", 0))
            scores_long_ctx.append(v.get("score_long_context", 0))
            gaps.append(v.get("memory_gap", 0))
            hacks.append(v.get("hack_score", 0))
    if scores_all_mem:
        n = len(scores_all_mem)
        lines.append("")
        lines.append("## Verification Scores")
        lines.append(f"- score_all_memory:  mean={sum(scores_all_mem)/n:.2f}, min={min(scores_all_mem):.2f}, max={max(scores_all_mem):.2f}")
        lines.append(f"- score_no_memory:   mean={sum(scores_no_mem)/n:.2f}, min={min(scores_no_mem):.2f}, max={max(scores_no_mem):.2f}")
        lines.append(f"- score_long_context: mean={sum(scores_long_ctx)/n:.2f}, min={min(scores_long_ctx):.2f}, max={max(scores_long_ctx):.2f}")
        lines.append(f"- memory_gap:        mean={sum(gaps)/n:.2f}, min={min(gaps):.2f}, max={max(gaps):.2f}")
        lines.append(f"- hack_score:        mean={sum(hacks)/n:.2f}, max={max(hacks):.2f}")

    # LLM evaluation aggregates
    if eval_results:
        lines.append("")
        lines.append("## LLM Quality Scores")
        lines.append("| Criterion | Mean | Min | Max |")
        lines.append("|-----------|------|-----|-----|")

        for criterion in ["multihop", "distractor", "answer", "overall"]:
            scores = [
                r.get(criterion, {}).get("score", 0)
                for r in eval_results if r.get(criterion, {}).get("score") is not None
            ]
            if scores:
                lines.append(
                    f"| {criterion} | {sum(scores)/len(scores):.1f} | "
                    f"{min(scores)} | {max(scores)} |"
                )

        # Recommendation triage
        recs = {"keep": 0, "review": 0, "reject": 0}
        for r in eval_results:
            rec = r.get("overall", {}).get("recommendation", "review")
            recs[rec] = recs.get(rec, 0) + 1
        lines.append("")
        lines.append("## Instance Triage")
        lines.append(f"- **Keep**: {recs.get('keep', 0)}")
        lines.append(f"- **Review**: {recs.get('review', 0)}")
        lines.append(f"- **Reject**: {recs.get('reject', 0)}")

        # Common issues
        all_issues = []
        for r in eval_results:
            all_issues.extend(r.get("overall", {}).get("issues", []))
        if all_issues:
            # Count frequency
            issue_counts = {}
            for issue in all_issues:
                # Normalize by lowercasing and taking first 50 chars
                key = issue.lower()[:50]
                issue_counts[key] = issue_counts.get(key, 0) + 1
            sorted_issues = sorted(issue_counts.items(), key=lambda x: -x[1])
            lines.append("")
            lines.append("## Common Issues")
            for issue, count in sorted_issues[:10]:
                lines.append(f"- ({count}x) {issue}")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Quality review for MemGym-IR instances")
    parser.add_argument("input", help="Path to instances JSONL file")
    parser.add_argument("--output", "-o", default="quality_review_report.md",
                        help="Output markdown report path")
    parser.add_argument("--json-output", default=None,
                        help="Optional JSON output for raw evaluation results")
    parser.add_argument("--model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="LLM model for evaluation")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Max parallel LLM calls")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM evaluation, generate human-readable report only")
    parser.add_argument("--instances", default=None,
                        help="Comma-separated instance indices to evaluate (e.g., 0,1,2)")

    args = parser.parse_args()

    # Load instances
    print(f"Loading instances from {args.input}...")
    instances = load_instances(args.input)
    print(f"Loaded {len(instances)} instances")

    # Filter to specific indices if requested
    if args.instances:
        indices = [int(i) for i in args.instances.split(",")]
        instances = [instances[i] for i in indices if i < len(instances)]
        print(f"Filtered to {len(instances)} instances (indices: {indices})")

    # LLM evaluation
    eval_results = None
    if not args.skip_llm:
        import litellm
        litellm.modify_params = True

        print(f"Running LLM evaluation with {args.model}...")
        client = LLMClient(model=args.model)
        eval_results = evaluate_all(client, instances, max_concurrent=args.max_concurrent)
        print(f"Evaluation complete for {len(eval_results)} instances")

        # Print quick summary
        for r in eval_results:
            ov = r.get("overall", {})
            print(
                f"  {r.get('instance_id', '?')[:50]:50s} "
                f"overall={ov.get('score', '?')}/5 "
                f"rec={ov.get('recommendation', '?')}"
            )

        # Save JSON results
        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"JSON results saved to {args.json_output}")

    # Build eval result map for per-instance lookup
    eval_map = {}
    if eval_results:
        for r in eval_results:
            eval_map[r.get("instance_id", "")] = r

    # Generate markdown report
    print(f"Generating markdown report...")
    parts = []

    # Aggregate summary first
    parts.append(format_aggregate_markdown(instances, eval_results))

    # Per-instance sections
    for inst in instances:
        iid = inst.get("instance_id", "")
        eval_r = eval_map.get(iid)
        parts.append(format_instance_markdown(inst, eval_r))

    report = "\n".join(parts)

    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
