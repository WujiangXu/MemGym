#!/usr/bin/env python3
"""Run agent-based solvability evaluation on generated coding-synthetic instances.

Usage:
    python examples/memgym_codeqa/eval_solvability.py results/pilot_100/coding_synthetic_instances.jsonl --sample-size 20
"""
from memgym.pipelines.coding_synthetic.eval.eval_solvability_agent import main

if __name__ == "__main__":
    main()
