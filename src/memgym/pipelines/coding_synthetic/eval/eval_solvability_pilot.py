#!/usr/bin/env python3
"""Evaluate solvability of pilot instances: can an LLM solve them WITH memory?

Tests two conditions:
  A) WITH memory files (task_prompt + memory_files + repo_context) → should score HIGH
  B) WITHOUT memory files (task_prompt + repo_context only) → should score LOW

A good instance has high A and low B (large gap).
"""

import json
import sys
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

PATCH_GEN_SYSTEM = """You are a senior software engineer. Given a bug report and context documents,
generate a unified diff patch that fixes the issue. Output ONLY the diff patch, nothing else."""

JUDGE_SYSTEM = "You are a code evaluation judge. Output only valid JSON."

JUDGE_PROMPT = """Compare the generated patch against the gold (correct) fix.

## Generated Patch
{generated}

## Gold Fix
{gold_fix}

Score how well the generated patch addresses the same issue as the gold fix.
Consider:
- Does it modify the same files/functions?
- Does it address the same root cause?
- Would it fix the same tests?

Output JSON: {{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}

Score guide:
- 1.0: Essentially equivalent fix
- 0.7-0.9: Same root cause, minor differences
- 0.4-0.6: Partially correct, misses some aspects
- 0.1-0.3: Touches related code but wrong approach
- 0.0: Completely wrong or empty"""


def generate_patch(task_prompt, memory_files, repo_context, model, with_memory=True):
    """Generate a fix patch given task + optional memory."""
    messages = [{"role": "system", "content": PATCH_GEN_SYSTEM}]

    if with_memory and memory_files:
        mem_text = ""
        for fname, content in memory_files.items():
            mem_text += f"\n=== {fname} ===\n{content[:4000]}\n"
        messages.append({
            "role": "user",
            "content": f"Here are your memory/context documents:\n{mem_text[:30000]}",
        })
        messages.append({
            "role": "assistant",
            "content": "I've read the context documents. What's the issue?",
        })

    if repo_context:
        messages.append({
            "role": "user",
            "content": f"Repository structure:\n{repo_context[:8000]}",
        })
        messages.append({
            "role": "assistant",
            "content": "I see the repo structure. What needs to be fixed?",
        })

    messages.append({
        "role": "user",
        "content": f"{task_prompt}\n\nGenerate the unified diff patch that fixes this. Output ONLY the diff:",
    })

    try:
        resp = litellm.completion(
            model=model,
            messages=messages,
            max_completion_tokens=8000,
            temperature=0.3,
        )
        patch = resp.choices[0].message.content.strip()
        if patch.startswith("```"):
            lines = patch.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            patch = "\n".join(lines)
        return patch
    except Exception as e:
        print(f"  Patch generation error: {e}")
        return ""


def score_patch(generated, gold_fix, model):
    """Score generated patch vs gold fix using LLM judge."""
    if not generated.strip():
        return 0.0, "empty patch"

    prompt = JUDGE_PROMPT.format(
        generated=generated[:3000],
        gold_fix=gold_fix[:3000],
    )

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2000,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        result = json.loads(text)
        return float(result.get("score", 0.0)), result.get("reasoning", "")
    except Exception as e:
        print(f"  Scoring error: {e}")
        return 0.0, str(e)


def evaluate_instance(inst, model, idx):
    """Evaluate one instance: with-memory vs without-memory."""
    iid = inst["instance_id"]
    print(f"  [{idx}] {iid}...")

    task_prompt = inst["task_prompt"]
    memory_files = inst.get("memory_files", {})
    repo_context = inst.get("repo_context", "")
    gold_fix = inst["gold_fix"]

    # Condition A: WITH memory
    patch_a = generate_patch(task_prompt, memory_files, repo_context, model, with_memory=True)
    score_a, reason_a = score_patch(patch_a, gold_fix, model)

    # Condition B: WITHOUT memory
    patch_b = generate_patch(task_prompt, memory_files, repo_context, model, with_memory=False)
    score_b, reason_b = score_patch(patch_b, gold_fix, model)

    gap = score_a - score_b
    print(f"  [{idx}] {iid}: A={score_a:.2f} B={score_b:.2f} gap={gap:.2f}")

    return {
        "instance_id": iid,
        "score_with_memory": score_a,
        "score_without_memory": score_b,
        "gap": gap,
        "reason_with": reason_a,
        "reason_without": reason_b,
    }


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pilot_instances.jsonl"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    model = sys.argv[3] if len(sys.argv) > 3 else "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    print(f"Loading instances from {input_path}")
    instances = []
    with open(input_path) as f:
        for line in f:
            instances.append(json.loads(line))

    if sample_size < len(instances):
        random.seed(42)
        instances = random.sample(instances, sample_size)

    print(f"Evaluating {len(instances)} instances with model={model}, workers={num_workers}")
    print(f"Condition A: task + memory + repo_context")
    print(f"Condition B: task + repo_context only")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(evaluate_instance, inst, model, i): i
            for i, inst in enumerate(instances)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  Error: {e}")

    # Sort by original index
    results.sort(key=lambda r: r["instance_id"])

    # Summary
    print("\n" + "=" * 60)
    print("SOLVABILITY RESULTS")
    print("=" * 60)

    avg_a = sum(r["score_with_memory"] for r in results) / len(results)
    avg_b = sum(r["score_without_memory"] for r in results) / len(results)
    avg_gap = sum(r["gap"] for r in results) / len(results)

    solvable_with = sum(1 for r in results if r["score_with_memory"] >= 0.5)
    solvable_without = sum(1 for r in results if r["score_without_memory"] >= 0.5)
    good_gap = sum(1 for r in results if r["gap"] >= 0.2)

    print(f"Instances evaluated: {len(results)}")
    print(f"Avg score WITH memory:    {avg_a:.3f}")
    print(f"Avg score WITHOUT memory: {avg_b:.3f}")
    print(f"Avg gap (A - B):          {avg_gap:.3f}")
    print()
    print(f"Solvable with memory (≥0.5):    {solvable_with}/{len(results)} ({solvable_with/len(results)*100:.0f}%)")
    print(f"Solvable without memory (≥0.5): {solvable_without}/{len(results)} ({solvable_without/len(results)*100:.0f}%)")
    print(f"Good gap (≥0.2):                {good_gap}/{len(results)} ({good_gap/len(results)*100:.0f}%)")

    # Distribution
    print("\nScore distribution (with memory):")
    for bucket in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        count = sum(1 for r in results if bucket[0] <= r["score_with_memory"] < bucket[1])
        print(f"  [{bucket[0]:.1f}-{bucket[1]:.1f}): {count}")

    # Save
    out_path = Path(input_path).parent / "solvability_pilot.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "n": len(results),
                "avg_with_memory": avg_a,
                "avg_without_memory": avg_b,
                "avg_gap": avg_gap,
                "solvable_with": solvable_with,
                "solvable_without": solvable_without,
                "good_gap": good_gap,
                "model": model,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
