"""Test Condition A (all docs at once) vs Condition D (delayed reveal).

Measures whether delayed question reveal actually makes the task harder.
Usage: python -m pipelines.memgym_ir.eval_delayed_reveal <instances.jsonl> [--limit N]
"""

import argparse
import json
import sys

from ..types.schemas import MemGymIRInstance
from ..stages.validate import (
    _ask_and_judge,
    _ask_and_judge_delayed,
    _format_documents,
    _get_all_docs,
)
from ..llm.client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="Test delayed reveal impact")
    parser.add_argument("instances", help="Path to memgym_ir_instances.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Max instances to test")
    parser.add_argument(
        "--model",
        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model for evaluation",
    )
    args = parser.parse_args()

    instances = []
    with open(args.instances) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            instances.append(MemGymIRInstance(**json.loads(line)))

    client = LLMClient(model=args.model)

    print(f"Testing {len(instances)} instances: Condition A vs Condition D")
    print(f"Model: {args.model}")
    print()
    print(f"{'ID':<50} {'A(all@once)':>11} {'D(delayed)':>11} {'Drop':>7}")
    print("-" * 82)

    scores_a = []
    scores_d = []

    for inst in instances:
        # Condition A: all docs at once
        all_docs = _get_all_docs(inst)
        docs_a = _format_documents(all_docs)
        sa = _ask_and_judge(client, inst.question, inst.answer, docs_a)

        # Condition D: delayed reveal (sequential turns, question last)
        turns_docs = [turn.documents for turn in inst.turns]
        sd = _ask_and_judge_delayed(client, inst.question, inst.answer, turns_docs)

        scores_a.append(sa)
        scores_d.append(sd)
        drop = sa - sd
        short_id = inst.instance_id.replace("memgym_ir__musique__", "")
        print(f"{short_id:<50} {sa:>11.2f} {sd:>11.2f} {drop:>7.2f}")

    print("-" * 82)
    avg_a = sum(scores_a) / len(scores_a)
    avg_d = sum(scores_d) / len(scores_d)
    print(f"{'AVERAGE':<50} {avg_a:>11.3f} {avg_d:>11.3f} {avg_a - avg_d:>7.3f}")
    print()
    print(f"Condition A (all docs at once):     {avg_a:.1%}")
    print(f"Condition D (delayed reveal):       {avg_d:.1%}")
    print(f"Drop from delayed reveal:           {(avg_a - avg_d)*100:.1f} pp")

    # Save results
    out_path = args.instances.replace(".jsonl", "_delayed_reveal_test.json")
    results = {
        "model": args.model,
        "num_instances": len(instances),
        "avg_score_a": avg_a,
        "avg_score_d": avg_d,
        "drop": avg_a - avg_d,
        "per_instance": [
            {
                "id": inst.instance_id,
                "score_a": sa,
                "score_d": sd,
                "drop": sa - sd,
            }
            for inst, sa, sd in zip(instances, scores_a, scores_d)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
