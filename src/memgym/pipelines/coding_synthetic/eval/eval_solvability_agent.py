#!/usr/bin/env python3
"""Agent-based solvability evaluation for MemGym instances.

Uses a lightweight tool-calling agent that can read/search memory files,
then generates a patch. Scored by a stronger judge model (Sonnet).

Two conditions:
  A) WITH memory: agent can read/search memory files → should score HIGH
  B) WITHOUT memory: agent gets empty files → should score LOW

Usage:
    python eval_solvability_agent.py <input_path> [sample_size] [agent_model] [judge_model] [num_workers] [max_steps]
"""

import json
import re
import sys
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Tool definitions for the agent
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all available memory/context files from previous debugging sessions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full content of a memory/context file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search all memory files for a keyword or pattern. Returns matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or keyword to look for",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

AGENT_SYSTEM_IMPLICIT = """\
You are a senior developer debugging a codebase issue. You have access to memory/context \
files from previous debugging sessions that may contain relevant information about the bug.

Your workflow:
1. List available files to see what context you have
2. Read and search files to find clues about the bug
3. Based on what you learn, generate a unified diff patch that fixes the issue

When you're ready to submit your fix, output the patch in a code block like:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
...
```"""

AGENT_SYSTEM_EXPLICIT = """\
You are a senior developer debugging a codebase issue. You have access to memory/context \
files that contain **critical information** needed to solve this bug. These files contain \
knowledge from past debugging sessions, incident reports, and code reviews that CANNOT be \
found by reading the repository alone. You should thoroughly read and use these memory files \
before attempting a fix.

Your workflow:
1. List available files to see what context you have
2. Read ALL files carefully — they contain essential clues about the root cause
3. Search files for keywords related to the symptoms described
4. Based on what you learn from memory files, generate a unified diff patch

When you're ready to submit your fix, output the patch in a code block like:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
...
```"""

AGENT_SYSTEM_NONE = """\
You are a senior developer debugging a codebase issue. You have access to the repository \
context below. Analyze the problem and generate a unified diff patch that fixes the issue.

When you're ready to submit your fix, output the patch in a code block like:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
...
```"""

MEMORY_MODES = {
    "implicit": AGENT_SYSTEM_IMPLICIT,
    "explicit": AGENT_SYSTEM_EXPLICIT,
    "none": AGENT_SYSTEM_NONE,
}

# Keep backward compat
AGENT_SYSTEM = AGENT_SYSTEM_IMPLICIT

JUDGE_SYSTEM = "You are a code evaluation judge. Output only valid JSON."

JUDGE_PROMPT = """\
Compare the generated patch against the gold (correct) fix.

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


# ---------------------------------------------------------------------------
# Tool execution (local, no Docker)
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, args: dict, memory_files: dict) -> str:
    """Execute a tool call against the local memory files."""
    if tool_name == "list_files":
        if not memory_files:
            return "No memory files available."
        file_list = "\n".join(f"- {fn}" for fn in sorted(memory_files.keys()))
        return f"Available files ({len(memory_files)}):\n{file_list}"

    elif tool_name == "read_file":
        filename = args.get("filename", "")
        if filename in memory_files:
            return memory_files[filename]
        # Try fuzzy match
        for fn in memory_files:
            if fn.lower() == filename.lower() or filename in fn:
                return memory_files[fn]
        return f"File not found: {filename}. Use list_files to see available files."

    elif tool_name == "search_files":
        query = args.get("query", "").lower()
        if not query:
            return "Empty search query."
        matches = []
        for fn, content in memory_files.items():
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if query in line.lower():
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    snippet = "\n".join(lines[start:end])
                    matches.append(f"=== {fn} (line {i+1}) ===\n{snippet}")
                    if len(matches) >= 15:
                        break
            if len(matches) >= 15:
                break
        if not matches:
            return f"No matches found for '{query}'."
        return "\n\n".join(matches)

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    task_prompt: str,
    memory_files: dict,
    repo_context: str,
    agent_model: str,
    max_steps: int = 10,
    memory_mode: str = "implicit",
) -> tuple:
    """Run lightweight tool-calling agent. Returns (patch, trace).

    Args:
        memory_mode: "implicit" (baseline), "explicit" (emphasize importance), "none" (no memory)
    trace is a list of dicts: {step, tool, args, result_preview}
    """
    trace = []
    system_prompt = MEMORY_MODES.get(memory_mode, AGENT_SYSTEM_IMPLICIT)

    user_content = task_prompt
    if repo_context:
        user_content += f"\n\nRepository structure:\n{repo_context[:6000]}"
    if memory_mode != "none":
        user_content += "\n\nUse the available tools to investigate, then generate your fix patch."
    else:
        user_content += "\n\nGenerate your fix as a unified diff patch."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for step in range(max_steps):
        try:
            resp = litellm.completion(
                model=agent_model,
                messages=messages,
                tools=TOOLS,
                max_completion_tokens=4000,
                temperature=0.2,
            )
        except Exception as e:
            print(f"    Agent LLM error at step {step}: {e}")
            break

        choice = resp.choices[0]
        msg = choice.message

        # Append assistant message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # If no tool calls, agent is done — check for patch in content
        if not msg.tool_calls:
            return _extract_patch(msg.content or ""), trace

        # Execute tool calls and record trace
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args, memory_files)
            trace.append({
                "step": step,
                "tool": tc.function.name,
                "args": args,
                "result_preview": result[:2000],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:8000],
            })

    # Max steps reached — ask for final patch
    messages.append({
        "role": "user",
        "content": "You've used all your investigation steps. Now output your best fix as a unified diff patch.",
    })
    try:
        resp = litellm.completion(
            model=agent_model,
            messages=messages,
            max_completion_tokens=4000,
            temperature=0.2,
        )
        return _extract_patch(resp.choices[0].message.content or ""), trace
    except Exception:
        return "", trace


def _extract_patch(text: str) -> str:
    """Extract unified diff patch from agent output."""
    if not text:
        return ""

    # Try to find ```diff block
    diff_match = re.search(r"```(?:diff)?\s*\n(diff --git.*?)```", text, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Try to find raw diff
    diff_match = re.search(r"(diff --git .+)", text, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Try to find --- / +++ style patch
    patch_match = re.search(r"(---\s+\S+.*?\+\+\+\s+\S+.*?)(?:\n\n|\Z)", text, re.DOTALL)
    if patch_match:
        return patch_match.group(1).strip()

    # If text looks like a patch itself
    if text.strip().startswith("diff --git") or text.strip().startswith("---"):
        return text.strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def score_patch(generated: str, gold_fix: str, judge_model: str) -> tuple:
    """Score generated patch vs gold fix using LLM judge. Returns (score, reasoning)."""
    if not generated.strip():
        return 0.0, "empty patch"

    prompt = JUDGE_PROMPT.format(
        generated=generated[:4000],
        gold_fix=gold_fix[:4000],
    )

    try:
        resp = litellm.completion(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2000,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences
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
        print(f"    Judge error: {e}")
        return 0.0, str(e)


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(inst: dict, agent_model: str, judge_model: str,
                      max_steps: int, idx: int,
                      memory_modes: list = None) -> dict:
    """Evaluate one instance across specified memory modes.

    Args:
        memory_modes: list of modes to test. Default: ["implicit", "none"]
            Options: "implicit", "explicit", "none"
    """
    if memory_modes is None:
        memory_modes = ["implicit", "none"]

    iid = inst["instance_id"]
    print(f"  [{idx}] {iid}...")

    task_prompt = inst["task_prompt"]
    memory_files = inst.get("memory_files", {})
    repo_context = inst.get("repo_context", "")
    gold_fix = inst["gold_fix"]

    result = {
        "instance_id": iid,
        "task_prompt_preview": task_prompt[:300],
        "gold_fix_preview": gold_fix[:500],
    }

    for mode in memory_modes:
        mf = memory_files if mode != "none" else {}
        patch, trace = run_agent(
            task_prompt, mf, repo_context, agent_model, max_steps,
            memory_mode=mode,
        )
        score, reason = score_patch(patch, gold_fix, judge_model)
        result[f"score_{mode}"] = score
        result[f"reason_{mode}"] = reason
        result[f"patch_{mode}_preview"] = patch[:500] if patch else ""
        result[f"trace_{mode}"] = trace

    # Compute gaps between modes
    if "implicit" in memory_modes and "none" in memory_modes:
        result["gap_implicit_vs_none"] = result["score_implicit"] - result["score_none"]
    if "explicit" in memory_modes and "none" in memory_modes:
        result["gap_explicit_vs_none"] = result["score_explicit"] - result["score_none"]
    if "explicit" in memory_modes and "implicit" in memory_modes:
        result["gap_explicit_vs_implicit"] = result["score_explicit"] - result["score_implicit"]

    # Backward compat aliases
    if "implicit" in memory_modes:
        result["score_with_memory"] = result["score_implicit"]
    if "none" in memory_modes:
        result["score_without_memory"] = result["score_none"]
    if "gap_implicit_vs_none" in result:
        result["gap"] = result["gap_implicit_vs_none"]

    scores_str = " ".join(f"{m}={result[f'score_{m}']:.2f}" for m in memory_modes)
    print(f"  [{idx}] {iid}: {scores_str}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_progress(output_dir: str):
    """Check pipeline progress by counting checkpoint file lines."""
    from pathlib import Path
    base = Path(output_dir)
    if not base.exists():
        print(f"Directory not found: {output_dir}")
        return

    def count_lines(p):
        if not p.exists():
            return 0
        with open(p) as f:
            return sum(1 for line in f if line.strip())

    filtered = count_lines(base / "filtered.jsonl")
    extracted = count_lines(base / "extracted.jsonl")
    crafted = count_lines(base / "crafted.jsonl")
    difficulty = count_lines(base / "difficulty.jsonl")
    final = count_lines(base / "coding_synthetic_instances.jsonl")

    prescreen_kept = prescreen_rejected = 0
    ps = base / "prescreen.json"
    if ps.exists():
        data = json.loads(ps.read_text())
        prescreen_kept = sum(1 for r in data if not r.get("too_easy"))
        prescreen_rejected = sum(1 for r in data if r.get("too_easy"))

    print(f"Pipeline progress: {output_dir}")
    print(f"  Stage 1 (filtered):     {filtered}")
    print(f"  Stage 1.75 (prescreen): {prescreen_kept} kept, {prescreen_rejected} rejected")
    print(f"  Stage 2 (extracted):    {extracted}")
    print(f"  Stage 3 (crafted):      {crafted}")
    print(f"  Stage 4 (difficulty):   {difficulty}")
    print(f"  Final output:           {final}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent-based solvability evaluation")
    parser.add_argument("input_path", nargs="?",
                        default="results/pilot_100/coding_synthetic_instances.jsonl")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--agent-model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--judge-model", default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--memory-modes", default="implicit,none",
                        help="Comma-separated modes: implicit,explicit,none")
    args, _unknown = parser.parse_known_args()

    input_path = args.input_path

    # Special mode: if input_path is a directory, check pipeline progress
    if not input_path.endswith(".jsonl") and Path(input_path).is_dir():
        _check_progress(input_path)
        return

    sample_size = args.sample_size
    agent_model = args.agent_model
    judge_model = args.judge_model
    num_workers = args.workers
    max_steps = args.max_steps
    memory_modes = [m.strip() for m in args.memory_modes.split(",")]

    print(f"Loading instances from {input_path}")
    instances = []
    with open(input_path) as f:
        for line in f:
            instances.append(json.loads(line))

    if sample_size < len(instances):
        random.seed(42)
        instances = random.sample(instances, sample_size)

    print(f"Agent-based solvability evaluation")
    print(f"  Instances: {len(instances)}")
    print(f"  Agent model: {agent_model}")
    print(f"  Judge model: {judge_model}")
    print(f"  Workers: {num_workers}")
    print(f"  Max steps/instance: {max_steps}")
    print(f"  Memory modes: {memory_modes}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(evaluate_instance, inst, agent_model, judge_model,
                        max_steps, i, memory_modes): i
            for i, inst in enumerate(instances)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  Error: {e}")
                traceback.print_exc()

    results.sort(key=lambda r: r["instance_id"])

    # Summary
    print("\n" + "=" * 60)
    print("AGENT SOLVABILITY RESULTS")
    print("=" * 60)

    n = len(results)
    if n == 0:
        print("No results!")
        return

    print(f"Instances evaluated: {n}")
    print(f"Agent model: {agent_model}")
    print(f"Judge model: {judge_model}")
    print(f"Memory modes: {memory_modes}")
    print()

    summary = {"n": n, "agent_model": agent_model, "judge_model": judge_model,
               "max_steps": max_steps, "memory_modes": memory_modes}

    for mode in memory_modes:
        key = f"score_{mode}"
        scores = [r[key] for r in results if key in r]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        solvable = sum(1 for s in scores if s >= 0.5)
        summary[f"avg_{mode}"] = avg
        summary[f"solvable_{mode}"] = solvable
        print(f"Avg score ({mode:>8s}): {avg:.3f}  |  Solvable (>=0.5): {solvable}/{n} ({solvable/n*100:.0f}%)")

        print(f"  Score distribution:")
        for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            count = sum(1 for s in scores if lo <= s < hi)
            bar = "#" * count
            print(f"    [{lo:.1f}-{hi:.1f}): {count:2d} {bar}")

    # Print gaps
    print()
    gap_keys = [k for k in results[0] if k.startswith("gap_")]
    for gk in gap_keys:
        vals = [r[gk] for r in results if gk in r]
        if vals:
            avg_gap = sum(vals) / len(vals)
            good = sum(1 for v in vals if v >= 0.2)
            summary[f"avg_{gk}"] = avg_gap
            summary[f"good_{gk}"] = good
            print(f"Avg {gk}: {avg_gap:.3f}  |  Good gap (>=0.2): {good}/{n} ({good/n*100:.0f}%)")

    # Save summary (without traces for compact view)
    trace_keys = [k for k in results[0] if k.startswith("trace_")]
    summary_results = [{k: v for k, v in r.items() if k not in trace_keys}
                       for r in results]

    modes_tag = "_".join(memory_modes)
    out_path = Path(input_path).parent / f"solvability_agent_{modes_tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": summary_results}, f, indent=2)
    print(f"\nSaved summary to {out_path}")

    # Save detailed report with traces
    report_path = Path(input_path).parent / f"solvability_agent_{modes_tag}_report.jsonl"
    with open(report_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved detailed report to {report_path}")

    # Print example traces for quick review
    print("\n" + "=" * 60)
    print("EXAMPLE TRACES (top 5 by best gap)")
    print("=" * 60)
    # Use the first gap key for sorting
    sort_key = gap_keys[0] if gap_keys else "score_implicit"
    by_gap = sorted(results, key=lambda r: r.get(sort_key, 0), reverse=True)[:5]
    for r in by_gap:
        scores_str = " ".join(f"{m}={r.get(f'score_{m}', 0):.2f}" for m in memory_modes)
        gaps_str = " ".join(f"{gk}={r.get(gk, 0):.2f}" for gk in gap_keys)
        print(f"\n--- {r['instance_id']} ({scores_str} {gaps_str}) ---")
        print(f"Task: {r.get('task_prompt_preview', '')[:150]}...")
        # Show trace from first memory mode that has one
        for mode in memory_modes:
            trace = r.get(f"trace_{mode}", [])
            if trace:
                print(f"Agent trace ({mode}, {len(trace)} tool calls):")
                for t in trace[:6]:
                    args_str = json.dumps(t["args"])[:80] if t["args"] else ""
                    result_line = t["result_preview"].split("\n")[0][:100]
                    print(f"  step {t['step']}: {t['tool']}({args_str}) -> {result_line}")
                break


if __name__ == "__main__":
    main()
