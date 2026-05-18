"""Shared evaluator for MemGym-IR v3 instances.

Works with instances from ANY generation path (deep research trajectories,
structural builder, or v2 pipeline). Uses IR-specific memory managers
via IRMemoryAdapter to bridge BaseMemoryManager into turn-by-turn processing.

Usage:
    # Single strategy
    python -m memgym.pipelines.memgym_ir.eval_agent instances.jsonl \
        --strategy ir_summarizing

    # Compare all IR memory managers
    python -m memgym.pipelines.memgym_ir.eval_agent instances.jsonl \
        --compare-managers ir_passthrough,ir_summarizing,ir_structured

    # With specific model
    python -m memgym.pipelines.memgym_ir.eval_agent instances.jsonl \
        --strategy ir_structured --model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from ..types.schemas import MemGymIRInstance, EvictionPolicy
from ..llm.client import LLMClient
from ..llm.prompts import (
    EVICTED_ANSWER_PROMPT,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)
from .memory import _compute_note_fact_recall

logger = logging.getLogger(__name__)

# IR memory manager registry name → class
IR_STRATEGIES = [
    "ir_passthrough", "ir_summarizing", "ir_structured",
    "ir_naive_rag", "ir_bm25", "ir_amem", "ir_lightmem",
    "ir_hipporag", "ir_simplemem",
]


def _get_ir_manager(strategy: str, question: str = "", max_size: int = 10, summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"):
    """Create an IR memory manager by strategy name."""
    from memgym.memory.base import get_memory_model

    if strategy == "ir_passthrough":
        return get_memory_model("ir_passthrough")
    elif strategy == "ir_summarizing":
        return get_memory_model(
            "ir_summarizing",
            max_size=max_size,
            summarization_model=summarization_model,
            question=question,
        )
    elif strategy == "ir_structured":
        return get_memory_model(
            "ir_structured",
            max_size=max_size,
            summarization_model=summarization_model,
            question=question,
        )
    elif strategy in ("ir_naive_rag", "ir_bm25", "ir_amem", "ir_lightmem"):
        return get_memory_model(strategy, question=question)
    elif strategy in ("ir_hipporag", "ir_simplemem"):
        return get_memory_model(strategy, question=question)
    else:
        raise ValueError(f"Unknown IR strategy: {strategy}. Available: {IR_STRATEGIES}")


class IRMemoryAdapter:
    """Bridges IR memory managers into the turn-by-turn evaluation protocol.

    For each search turn:
    1. Format the turn's documents as an observation string
    2. Feed it to the memory manager via manage_context()
    3. Extract the current memory state for the answer phase
    """

    def __init__(self, manager, question: str = ""):
        self.manager = manager
        self.question = question
        self._context: List[Any] = []

    def process_turn(self, turn, visible_docs: List[Dict]) -> str:
        """Process a single search turn through the memory manager.

        Returns the current notes/summary as text.
        """
        # Format the turn as an observation
        obs_parts = [f"=== Turn {turn.turn_index + 1}: {turn.sub_query} ==="]
        for doc in visible_docs:
            obs_parts.append(f"--- {doc.get('title', 'Untitled')} ---")
            obs_parts.append(doc.get("text", ""))
        observation = "\n\n".join(obs_parts)

        # Feed to memory manager
        filtered = self.manager.manage_context(
            original_context=self._context,
            current_observation=observation,
            metadata={"turn_index": turn.turn_index, "question": self.question},
        )
        self._context = filtered.content if isinstance(filtered.content, list) else [filtered.content]

        return self.get_notes_text()

    def get_notes_text(self) -> str:
        """Get current accumulated notes as text."""
        if hasattr(self.manager, "get_notes_text"):
            return self.manager.get_notes_text()
        # Fallback: join context
        return "\n\n".join(str(c) for c in self._context)


def _build_visible_docs(instance, turn_index, eviction_policy):
    """Get documents visible at a given turn under the eviction policy."""
    if eviction_policy.mode == "full_eviction":
        visible_start = turn_index
    else:
        visible_start = max(0, turn_index - eviction_policy.window_size + 1)

    visible_docs = []
    for t in range(visible_start, turn_index + 1):
        if t < len(instance.turns):
            for doc in instance.turns[t].documents:
                visible_docs.append(doc)
    return visible_docs, visible_start


def evaluate_with_manager(
    instance: MemGymIRInstance,
    eval_client: LLMClient,
    strategy: str = "ir_summarizing",
    max_size: int = 10,
    summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
) -> Dict:
    """Evaluate a single instance using an IR memory manager.

    The memory manager processes turns sequentially, accumulating a
    summary/notes. After all turns, the final memory state is used
    to answer the question.
    """
    eviction_policy = instance.eviction_policy or EvictionPolicy()

    # Auto-scale max_size so condensation triggers at least once for
    # non-passthrough strategies. With default max_size=10 and 4-turn
    # instances, condensation never fires → all strategies score identically.
    num_turns = len(instance.turns)
    if strategy != "ir_passthrough" and max_size >= num_turns:
        max_size = max(2, num_turns - 1)
        logger.info(f"  Auto-scaled max_size to {max_size} for {num_turns}-turn instance")

    # Create the memory manager
    manager = _get_ir_manager(
        strategy,
        question=instance.question,
        max_size=max_size,
        summarization_model=summarization_model,
    )
    adapter = IRMemoryAdapter(manager, question=instance.question)

    turn_notes = []

    # Process each turn through the memory manager
    for t_idx, turn in enumerate(instance.turns):
        visible_docs, visible_start = _build_visible_docs(
            instance, t_idx, eviction_policy,
        )
        notes = adapter.process_turn(turn, visible_docs)
        turn_notes.append({
            "turn": t_idx,
            "notes_length": len(notes),
            "visible_window": f"[{visible_start}, {t_idx}]",
        })

    # Get final memory state — retrieval-based methods use question-specific
    # retrieval instead of accumulated notes
    if hasattr(manager, "retrieve_for_question"):
        final_notes = manager.retrieve_for_question(instance.question)
    else:
        final_notes = adapter.get_notes_text()

    # Get visible docs from last turn
    visible_docs, _ = _build_visible_docs(
        instance, len(instance.turns) - 1, eviction_policy,
    )
    docs_str = _format_documents(visible_docs)

    if docs_str.strip():
        visible_section = f"Documents still visible from recent turns:\n{docs_str}"
    else:
        visible_section = "No documents are currently visible."

    # Get final answer
    answer_prompt = EVICTED_ANSWER_PROMPT.format(
        question=instance.question,
        notes=final_notes if final_notes else "(no notes taken)",
        visible_documents_section=visible_section,
    )

    response = eval_client.get_completion(prompt=answer_prompt, temperature=0.0)
    try:
        data = json.loads(response)
        predicted = data.get("answer", "")
    except json.JSONDecodeError:
        predicted = response.strip()

    # Judge the answer
    if not predicted:
        score = 0.0
    else:
        judge_prompt = ANSWER_JUDGE_PROMPT.format(
            gold_answer=instance.answer,
            predicted_answer=predicted,
            question=instance.question,
        )
        judge_result = eval_client.get_json_completion(
            prompt=judge_prompt,
            response_format=ANSWER_JUDGE_SCHEMA,
            temperature=0.0,
        )
        score = judge_result.get("score", 0.0)

    # Compute note fact recall
    note_recall = _compute_note_fact_recall(final_notes, instance)

    # Get manager stats
    stats = manager.get_stats() if hasattr(manager, "get_stats") else {}

    return {
        "instance_id": instance.instance_id,
        "score": score,
        "predicted_answer": predicted,
        "note_fact_recall": note_recall,
        "num_turns": len(instance.turns),
        "num_memory_required_facts": len(instance.memory_required_facts),
        "strategy": strategy,
        "final_notes_length": len(final_notes),
        "turn_notes": turn_notes,
        "manager_stats": stats,
    }


def compare_managers(
    instances: List[MemGymIRInstance],
    eval_client: LLMClient,
    strategies: Optional[List[str]] = None,
    max_size: int = 10,
    summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
) -> List[Dict]:
    """Compare multiple IR memory strategies on each instance."""
    if strategies is None:
        strategies = IR_STRATEGIES

    all_results = []

    for inst in instances:
        inst_result = {
            "instance_id": inst.instance_id,
            "num_hops": inst.num_hops,
            "num_memory_required_facts": len(inst.memory_required_facts),
            "source_dataset": inst.source_dataset,
        }

        for strat in strategies:
            result = evaluate_with_manager(
                inst, eval_client,
                strategy=strat,
                max_size=max_size,
                summarization_model=summarization_model,
            )
            inst_result[f"score_{strat}"] = result["score"]
            inst_result[f"recall_{strat}"] = result["note_fact_recall"]

        all_results.append(inst_result)

    return all_results


def _format_documents(docs: List[Dict]) -> str:
    """Format a list of document dicts into readable text."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.get('title', 'Untitled')}\n{doc.get('text', '')}")
    return "\n\n".join(parts)


def _print_comparison(results, strategies):
    """Pretty-print the strategy comparison table."""
    cols = [(s.replace("ir_", "")[:10], 10) for s in strategies]
    cols.append(("MemF", 5))

    header = f"{'ID':<40} "
    header += " ".join(f"{name:>{width}}" for name, width in cols)
    print(header)
    print("-" * len(header))

    agg = {f"score_{s}": [] for s in strategies}

    for r in results:
        short_id = r["instance_id"][-35:]
        row = f"{short_id:<40} "

        for s in strategies:
            se = r.get(f"score_{s}", 0)
            agg[f"score_{s}"].append(se)
            row += f"{se:>10.2f}"

        row += f" {r.get('num_memory_required_facts', 0):>5}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'AVERAGE':<40} "
    for s in strategies:
        vals = agg[f"score_{s}"]
        avg_row += f"{sum(vals)/len(vals) if vals else 0:>10.3f}"
    print(avg_row)

    # Strategy spread
    print()
    print("Strategy Differentiation:")
    for s in strategies:
        vals = agg[f"score_{s}"]
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {s:>20}: {avg:.1%}")

    if len(strategies) >= 2:
        best_key = max(strategies, key=lambda s: sum(agg[f"score_{s}"]))
        worst_key = min(strategies, key=lambda s: sum(agg[f"score_{s}"]))
        best_avg = sum(agg[f"score_{best_key}"]) / len(agg[f"score_{best_key}"])
        worst_avg = sum(agg[f"score_{worst_key}"]) / len(agg[f"score_{worst_key}"])
        spread = best_avg - worst_avg
        print(f"\n  Best: {best_key} ({best_avg:.1%}), Worst: {worst_key} ({worst_avg:.1%})")
        print(f"  Strategy spread: {spread*100:.1f} pp")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MemGym-IR instances with IR memory managers"
    )
    parser.add_argument("instances", help="Path to instances JSONL")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        help="Evaluation LLM model")
    parser.add_argument("--summarization-model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Model for memory summarization")
    parser.add_argument("--max-size", type=int, default=10,
                        help="Max context entries before condensation")

    # Strategy
    parser.add_argument("--strategy", default="ir_summarizing",
                        choices=IR_STRATEGIES,
                        help="Single IR memory strategy")
    parser.add_argument("--compare-managers", type=str, default=None,
                        help="Comma-separated strategies to compare")

    # Output
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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

    eval_client = LLMClient(model=args.model)

    # Comparison mode
    if args.compare_managers:
        strategies = [s.strip() for s in args.compare_managers.split(",")]
        print(f"Comparing {len(strategies)} IR memory managers on {len(instances)} instances")
        print(f"Eval model: {args.model}")
        print(f"Summarization model: {args.summarization_model}")
        print()

        results = compare_managers(
            instances, eval_client,
            strategies=strategies,
            max_size=args.max_size,
            summarization_model=args.summarization_model,
        )

        _print_comparison(results, strategies)

        # Save results
        out_path = args.output or args.instances.replace(".jsonl", "_manager_comparison.json")
        summary = {
            "eval_model": args.model,
            "summarization_model": args.summarization_model,
            "max_size": args.max_size,
            "strategies": strategies,
            "num_instances": len(instances),
            "per_instance": results,
        }
        for s in strategies:
            vals = [r.get(f"score_{s}", 0) for r in results]
            summary[f"avg_score_{s}"] = sum(vals) / len(vals) if vals else 0

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")
        return

    # Single strategy mode
    print(f"Evaluating {len(instances)} instances with {args.strategy}")
    print(f"Eval model: {args.model}")
    print()

    print(f"{'ID':<40} {'Score':>7} {'Recall':>7} {'MemF':>5}")
    print("-" * 62)

    all_results = []
    scores = []
    recalls = []

    for inst in instances:
        result = evaluate_with_manager(
            inst, eval_client,
            strategy=args.strategy,
            max_size=args.max_size,
            summarization_model=args.summarization_model,
        )
        s = result["score"]
        r = result["note_fact_recall"]
        mf = result["num_memory_required_facts"]
        scores.append(s)
        recalls.append(r)
        all_results.append(result)

        short_id = inst.instance_id[-38:]
        print(f"{short_id:<40} {s:>7.2f} {r:>7.2f} {mf:>5}")

    print("-" * 62)
    avg_s = sum(scores) / len(scores) if scores else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    print(f"{'AVERAGE':<40} {avg_s:>7.3f} {avg_r:>7.3f}")

    print(f"\nStrategy: {args.strategy}")
    print(f"Avg score: {avg_s:.1%}")
    print(f"Avg note recall: {avg_r:.1%}")

    # Save results
    out_path = args.output or args.instances.replace(".jsonl", f"_eval_{args.strategy}.json")
    summary = {
        "eval_model": args.model,
        "strategy": args.strategy,
        "num_instances": len(instances),
        "avg_score": avg_s,
        "avg_note_fact_recall": avg_r,
        "per_instance": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
