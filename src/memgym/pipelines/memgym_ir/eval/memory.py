"""Evaluate MemGym-IR instances under context eviction (Condition E).

Presents search turns sequentially, evicting earlier turns from context.
The agent writes notes between turns to retain key facts. Measures whether
memory (note-taking) is actually required to answer correctly.

Supports multiple memory strategies to demonstrate strategy differentiation:
- none:       No note-taking; agent relies only on visible window
- basic:      Free-form note-taking (original behavior)
- structured: Constrained structured format (key-value facts only)
- selective:  Agent must prioritize which facts to keep (tiny budget)

Usage:
    python -m pipelines.memgym_ir.eval_memory <instances.jsonl> [options]

Examples:
    # Default: full_eviction with basic notes
    python -m pipelines.memgym_ir.eval_memory output_ir/memgym_ir_instances.jsonl --limit 10

    # Compare all strategies side-by-side
    python -m pipelines.memgym_ir.eval_memory output_ir/memgym_ir_instances.jsonl \
        --compare-strategies --limit 10

    # Single strategy: no notes
    python -m pipelines.memgym_ir.eval_memory output_ir/memgym_ir_instances.jsonl \
        --strategy none --limit 10

    # Compare against Condition A (all docs at once)
    python -m pipelines.memgym_ir.eval_memory output_ir/memgym_ir_instances.jsonl \
        --compare-a --limit 10
"""

import argparse
import json
import sys

from ..types.schemas import MemGymIRInstance, EvictionPolicy
from ..stages.validate import _ask_and_judge, _format_documents, _get_all_docs
from ..llm.client import LLMClient
from ..llm.prompts import (
    NOTE_TAKING_PROMPT,
    NOTE_TAKING_SCHEMA,
    EVICTED_ANSWER_PROMPT,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)


# =========================================================================
# Strategy-specific note-taking prompts
# =========================================================================

STRUCTURED_NOTE_PROMPT = """\
You are an AI research assistant conducting a multi-step search. After each batch \
of search results, earlier results will be REMOVED from your context permanently.

Your task: Extract and record KEY FACTS in a strict structured format.

Question you are trying to answer: {question}

Current search results (Turn {turn_index}):
{documents}

{previous_notes_section}

Record your notes using ONLY this format — one fact per line:
  ENTITY: <entity name> | RELATION: <relationship> | VALUE: <specific value>
  BRIDGE: <entity A> → <entity B> via <relationship>

Rules:
- Maximum {budget} tokens
- ONLY record facts directly relevant to the question
- Each line must follow the format above exactly
- Do NOT write prose, summaries, or general observations
- Prioritize bridge facts (connections between entities)

Respond in JSON:
{{"notes": "<your structured notes>"}}
"""

SELECTIVE_NOTE_PROMPT = """\
You are an AI research assistant with VERY LIMITED memory. After each batch of \
search results, earlier results will be REMOVED. You can only retain {budget} tokens \
of notes — choose VERY carefully what to remember.

Question: {question}

Current search results (Turn {turn_index}):
{documents}

{previous_notes_section}

You have space for approximately {fact_slots} key facts. Write ONLY the most \
critical facts — those that connect entities across different documents and are \
essential for the final answer. Discard everything else.

Format: One fact per line, as brief as possible.

Respond in JSON:
{{"notes": "<your critical facts>"}}
"""


def _build_visible_docs(instance, turn_index, eviction_policy):
    """Get documents visible at a given turn under the eviction policy."""
    if eviction_policy.mode == "full_eviction":
        visible_start = turn_index
    else:  # sliding_window
        visible_start = max(0, turn_index - eviction_policy.window_size + 1)

    visible_docs = []
    for t in range(visible_start, turn_index + 1):
        if t < len(instance.turns):
            for doc in instance.turns[t].documents:
                visible_docs.append(doc)
    return visible_docs, visible_start


def _get_note_prompt(strategy, question, turn_index, documents, previous_notes_section, budget):
    """Select the note-taking prompt based on strategy."""
    if strategy == "structured":
        return STRUCTURED_NOTE_PROMPT.format(
            question=question,
            turn_index=turn_index,
            documents=documents,
            previous_notes_section=previous_notes_section,
            budget=budget,
        )
    elif strategy == "selective":
        fact_slots = max(1, budget // 30)  # ~30 tokens per fact
        return SELECTIVE_NOTE_PROMPT.format(
            question=question,
            turn_index=turn_index,
            documents=documents,
            previous_notes_section=previous_notes_section,
            budget=budget,
            fact_slots=fact_slots,
        )
    else:  # "basic" — original prompt
        return NOTE_TAKING_PROMPT.format(
            question=question,
            turn_index=turn_index,
            documents=documents,
            previous_notes_section=previous_notes_section,
            budget=budget,
        )


def evaluate_with_eviction(
    instance,
    client,
    eviction_policy=None,
    notes_budget=500,
    strategy="basic",
):
    """Run the eviction evaluation protocol on a single instance.

    Args:
        strategy: Memory strategy — "none", "basic", "structured", "selective"

    Returns dict with score_e, notes, note_fact_recall, etc.
    """
    if eviction_policy is None:
        eviction_policy = instance.eviction_policy or EvictionPolicy()

    notes_budget = eviction_policy.notes_budget_tokens
    accumulated_notes = ""
    turn_notes = []

    allow_notes = eviction_policy.allow_notes and strategy != "none"

    # Phase 1: Sequential turn processing with note-taking
    for t_idx, turn in enumerate(instance.turns):
        visible_docs, visible_start = _build_visible_docs(
            instance, t_idx, eviction_policy,
        )
        docs_str = _format_documents(visible_docs)

        # Build previous notes section
        if accumulated_notes:
            previous_notes_section = (
                f"Your notes from previous turns:\n{accumulated_notes}"
            )
        else:
            previous_notes_section = "This is the first turn — no previous notes."

        if allow_notes:
            prompt = _get_note_prompt(
                strategy=strategy,
                question=instance.question,
                turn_index=t_idx + 1,
                documents=docs_str,
                previous_notes_section=previous_notes_section,
                budget=notes_budget,
            )
            result = client.get_json_completion(
                prompt=prompt,
                response_format=NOTE_TAKING_SCHEMA,
                temperature=0.0,
            )
            new_notes = result.get("notes", "")
            accumulated_notes = new_notes
            turn_notes.append({
                "turn": t_idx,
                "notes": new_notes,
                "visible_window": f"[{visible_start}, {t_idx}]",
            })

    # Phase 2: Final answer with accumulated notes + visible docs
    visible_docs, _ = _build_visible_docs(
        instance, len(instance.turns) - 1, eviction_policy,
    )
    docs_str = _format_documents(visible_docs)

    if docs_str.strip():
        visible_section = f"Documents still visible from recent turns:\n{docs_str}"
    else:
        visible_section = "No documents are currently visible."

    answer_prompt = EVICTED_ANSWER_PROMPT.format(
        question=instance.question,
        notes=accumulated_notes if accumulated_notes else "(no notes taken)",
        visible_documents_section=visible_section,
    )

    response = client.get_completion(prompt=answer_prompt, temperature=0.0)
    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    # Phase 3: Judge the answer
    if not predicted:
        score_e = 0.0
    else:
        judge_prompt = ANSWER_JUDGE_PROMPT.format(
            gold_answer=instance.answer,
            predicted_answer=predicted,
            question=instance.question,
        )
        judge_result = client.get_json_completion(
            prompt=judge_prompt,
            response_format=ANSWER_JUDGE_SCHEMA,
            temperature=0.0,
        )
        score_e = judge_result.get("score", 0.0)

    # Phase 4: Compute note quality (fact recall)
    note_fact_recall = _compute_note_fact_recall(
        accumulated_notes, instance,
    )

    return {
        "instance_id": instance.instance_id,
        "score_e": score_e,
        "predicted_answer": predicted,
        "note_fact_recall": note_fact_recall,
        "num_turns": len(instance.turns),
        "num_memory_required_facts": len(instance.memory_required_facts),
        "eviction_mode": eviction_policy.mode,
        "window_size": eviction_policy.window_size,
        "notes_budget": notes_budget,
        "strategy": strategy,
        "final_notes": accumulated_notes,
        "turn_notes": turn_notes,
    }


def _compute_note_fact_recall(notes, instance):
    """Check what fraction of memory-required facts appear in the agent's notes."""
    if not instance.memory_required_facts or not notes:
        return 0.0

    notes_lower = notes.lower()
    recalled = 0

    fact_map = {f.fact_id: f for f in instance.grounding_facts}

    for fact_id in instance.memory_required_facts:
        fact = fact_map.get(fact_id)
        if not fact:
            continue

        content = fact.content.lower()
        key_terms = [
            w.strip(".,!?\"'()") for w in content.split()
            if len(w.strip(".,!?\"'()")) > 4
        ]
        if not key_terms:
            key_terms = content.split()

        matched = sum(1 for term in key_terms if term in notes_lower)
        if key_terms and matched / len(key_terms) >= 0.4:
            recalled += 1

        if fact.intermediate_answer:
            ia_lower = fact.intermediate_answer.lower()
            if ia_lower in notes_lower:
                recalled += 1
                recalled = min(recalled, len(instance.memory_required_facts))

    return recalled / len(instance.memory_required_facts)


# =========================================================================
# Multi-strategy comparison
# =========================================================================

STRATEGIES = ["none", "basic", "structured", "selective"]


def compare_strategies(
    instances,
    client,
    strategies=None,
    eviction_policy=None,
    include_condition_a=False,
):
    """Run all strategies on each instance and return comparison results.

    Returns list of dicts, one per instance, each with per-strategy scores.
    """
    if strategies is None:
        strategies = STRATEGIES

    all_results = []

    for inst in instances:
        inst_result = {
            "instance_id": inst.instance_id,
            "num_hops": inst.num_hops,
            "num_memory_required_facts": len(inst.memory_required_facts),
        }

        # Condition A (optional baseline)
        if include_condition_a:
            all_docs = _get_all_docs(inst)
            docs_a = _format_documents(all_docs)
            score_a = _ask_and_judge(client, inst.question, inst.answer, docs_a)
            inst_result["score_a"] = score_a

        # Run each strategy
        for strat in strategies:
            result = evaluate_with_eviction(
                inst, client,
                eviction_policy=eviction_policy,
                strategy=strat,
            )
            inst_result[f"score_{strat}"] = result["score_e"]
            inst_result[f"recall_{strat}"] = result["note_fact_recall"]

        all_results.append(inst_result)

    return all_results


def _print_comparison(results, strategies, include_a=False):
    """Pretty-print the strategy comparison table."""
    # Header
    cols = []
    if include_a:
        cols.append(("A(all)", 7))
    for s in strategies:
        cols.append((s[:8], 8))
    cols.append(("MemF", 5))

    header = f"{'ID':<35} "
    header += " ".join(f"{name:>{width}}" for name, width in cols)
    print(header)
    print("-" * len(header))

    # Per-instance rows
    agg = {f"score_{s}": [] for s in strategies}
    if include_a:
        agg["score_a"] = []

    for r in results:
        short_id = r["instance_id"].replace("memgym_ir__musique__", "").replace("memgym_ir__2wiki__", "")
        if len(short_id) > 33:
            short_id = short_id[:33] + ".."

        row = f"{short_id:<35} "
        if include_a:
            sa = r.get("score_a", 0)
            agg["score_a"].append(sa)
            row += f"{sa:>7.2f} "

        for s in strategies:
            se = r.get(f"score_{s}", 0)
            agg[f"score_{s}"].append(se)
            row += f"{se:>8.2f}"

        row += f" {r.get('num_memory_required_facts', 0):>5}"
        print(row)

    # Summary
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<35} "
    if include_a:
        vals = agg["score_a"]
        avg_row += f"{sum(vals)/len(vals) if vals else 0:>7.3f} "
    for s in strategies:
        vals = agg[f"score_{s}"]
        avg_row += f"{sum(vals)/len(vals) if vals else 0:>8.3f}"
    print(avg_row)

    # Strategy differentiation summary
    print()
    print("Strategy Differentiation:")
    for s in strategies:
        vals = agg[f"score_{s}"]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {s:>12}: {avg:.1%}")

    if len(strategies) >= 2:
        best_key = max(strategies, key=lambda s: sum(agg[f"score_{s}"]))
        worst_key = min(strategies, key=lambda s: sum(agg[f"score_{s}"]))
        best_avg = sum(agg[f"score_{best_key}"]) / len(agg[f"score_{best_key}"])
        worst_avg = sum(agg[f"score_{worst_key}"]) / len(agg[f"score_{worst_key}"])
        spread = best_avg - worst_avg
        print(f"\n  Best: {best_key} ({best_avg:.1%}), Worst: {worst_key} ({worst_avg:.1%})")
        print(f"  Strategy spread: {spread*100:.1f} pp")
        if spread < 0.10:
            print("  WARNING: Low spread — strategies are barely differentiated.")
            print("           Consider harder instances (more hops, smaller notes budget).")
        elif spread >= 0.25:
            print("  GOOD: Meaningful strategy differentiation achieved.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MemGym-IR instances under context eviction"
    )
    parser.add_argument("instances", help="Path to memgym_ir_instances.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--model",
        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    parser.add_argument(
        "--eviction-mode",
        choices=["full_eviction", "sliding_window"],
        default=None,
        help="Override eviction mode (default: use instance policy or full_eviction)",
    )
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--notes-budget", type=int, default=None)
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Disable note-taking (agent gets no memory aid)",
    )
    parser.add_argument(
        "--compare-a",
        action="store_true",
        help="Also run Condition A (all docs) for comparison",
    )
    # New: strategy options
    parser.add_argument(
        "--strategy",
        choices=["none", "basic", "structured", "selective"],
        default="basic",
        help="Memory strategy to use (default: basic)",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Run ALL strategies and show side-by-side comparison",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies to compare (default: all)",
    )
    args = parser.parse_args()

    # Load instances
    instances = []
    with open(args.instances) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            instances.append(MemGymIRInstance(**json.loads(line)))

    if not instances:
        print("No instances loaded.")
        sys.exit(1)

    client = LLMClient(model=args.model)

    # Build eviction policy override
    policy_override = None
    if args.eviction_mode or args.window or args.notes_budget is not None or args.no_notes:
        policy_override = EvictionPolicy(
            mode=args.eviction_mode or "full_eviction",
            window_size=args.window or 1,
            allow_notes=not args.no_notes,
            notes_budget_tokens=args.notes_budget or 500,
        )

    mode_desc = (policy_override or EvictionPolicy()).mode
    window_desc = (policy_override or EvictionPolicy()).window_size

    # =====================================================================
    # Multi-strategy comparison mode
    # =====================================================================
    if args.compare_strategies:
        strategies = STRATEGIES
        if args.strategies:
            strategies = [s.strip() for s in args.strategies.split(",")]

        print(f"Comparing {len(strategies)} strategies on {len(instances)} instances")
        print(f"Model: {args.model}")
        print(f"Eviction: {mode_desc} (window={window_desc})")
        print(f"Strategies: {', '.join(strategies)}")
        if args.compare_a:
            print("Including Condition A baseline")
        print()

        results = compare_strategies(
            instances, client,
            strategies=strategies,
            eviction_policy=policy_override,
            include_condition_a=args.compare_a,
        )

        _print_comparison(results, strategies, include_a=args.compare_a)

        # Save results
        out_path = args.instances.replace(".jsonl", "_strategy_comparison.json")
        summary = {
            "model": args.model,
            "eviction_mode": mode_desc,
            "window_size": window_desc,
            "strategies": strategies,
            "num_instances": len(instances),
            "per_instance": results,
        }
        # Compute per-strategy averages
        for s in strategies:
            vals = [r.get(f"score_{s}", 0) for r in results]
            summary[f"avg_score_{s}"] = sum(vals) / len(vals) if vals else 0

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")
        return

    # =====================================================================
    # Single-strategy mode (original behavior)
    # =====================================================================
    strategy = args.strategy
    if args.no_notes:
        strategy = "none"

    print(f"Evaluating {len(instances)} instances under context eviction")
    print(f"Model: {args.model}")
    print(f"Eviction: {mode_desc} (window={window_desc})")
    print(f"Strategy: {strategy}")
    if args.compare_a:
        print("Also running Condition A for comparison")
    print()

    # Header
    if args.compare_a:
        print(f"{'ID':<45} {'E(evict)':>9} {'A(all)':>7} {'Gap':>6} {'NoteR':>6} {'MemFacts':>8}")
        print("-" * 84)
    else:
        print(f"{'ID':<45} {'E(evict)':>9} {'NoteR':>6} {'MemFacts':>8}")
        print("-" * 71)

    all_results = []
    scores_e = []
    scores_a = []
    note_recalls = []

    for inst in instances:
        result = evaluate_with_eviction(
            inst, client,
            eviction_policy=policy_override,
            strategy=strategy,
        )
        se = result["score_e"]
        nr = result["note_fact_recall"]
        mf = result["num_memory_required_facts"]
        scores_e.append(se)
        note_recalls.append(nr)

        sa = None
        if args.compare_a:
            all_docs = _get_all_docs(inst)
            docs_a = _format_documents(all_docs)
            sa = _ask_and_judge(client, inst.question, inst.answer, docs_a)
            scores_a.append(sa)
            result["score_a"] = sa
            result["eviction_gap"] = sa - se

        all_results.append(result)
        short_id = inst.instance_id.replace("memgym_ir__musique__", "").replace("memgym_ir__2wiki__", "")

        if args.compare_a:
            gap = sa - se
            print(f"{short_id:<45} {se:>9.2f} {sa:>7.2f} {gap:>6.2f} {nr:>6.2f} {mf:>8}")
        else:
            print(f"{short_id:<45} {se:>9.2f} {nr:>6.2f} {mf:>8}")

    # Summary
    print("-" * (84 if args.compare_a else 71))
    avg_e = sum(scores_e) / len(scores_e) if scores_e else 0
    avg_nr = sum(note_recalls) / len(note_recalls) if note_recalls else 0

    if args.compare_a:
        avg_a = sum(scores_a) / len(scores_a) if scores_a else 0
        avg_gap = avg_a - avg_e
        print(f"{'AVERAGE':<45} {avg_e:>9.3f} {avg_a:>7.3f} {avg_gap:>6.3f} {avg_nr:>6.3f}")
    else:
        print(f"{'AVERAGE':<45} {avg_e:>9.3f} {avg_nr:>6.3f}")

    print()
    print(f"Strategy:                    {strategy}")
    print(f"Condition E (eviction):      {avg_e:.1%}")
    if args.compare_a:
        print(f"Condition A (all docs):      {avg_a:.1%}")
        print(f"Eviction gap (A - E):        {avg_gap*100:.1f} pp")
    print(f"Note fact recall:            {avg_nr:.1%}")

    mem_instances = sum(1 for r in all_results if r["num_memory_required_facts"] > 0)
    print(f"Instances with mem-required: {mem_instances}/{len(all_results)}")

    # Save results
    out_path = args.instances.replace(".jsonl", f"_eviction_eval_{strategy}.json")
    summary = {
        "model": args.model,
        "eviction_mode": mode_desc,
        "window_size": window_desc,
        "strategy": strategy,
        "num_instances": len(instances),
        "avg_score_e": avg_e,
        "avg_note_fact_recall": avg_nr,
        "instances_with_memory_required": mem_instances,
        "per_instance": all_results,
    }
    if args.compare_a:
        summary["avg_score_a"] = avg_a
        summary["avg_eviction_gap"] = avg_gap

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
