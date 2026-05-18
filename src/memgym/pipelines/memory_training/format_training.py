#!/usr/bin/env python
"""
Format extracted training steps into LLM fine-tuning JSONL.

Takes the output of extract_steps.py and converts to prompt/completion format
suitable for fine-tuning (OpenAI, Hugging Face, etc).

Usage:
    python -m pipelines.memory_training.format_training \
        --input pipelines/memory_training/data/training_steps.jsonl \
        --output pipelines/memory_training/data/finetune.jsonl \
        --format openai \
        --max-context-tokens 8000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_openai_chat(
    example: Dict,
    max_context_tokens: int = 8000,
) -> Optional[Dict]:
    """Format a training example as OpenAI chat fine-tuning format.

    Returns: {"messages": [{"role": ..., "content": ...}, ...]}
    """
    step_type = example.get("step_type", "unknown")
    strategy = example.get("strategy", "unknown")
    instance_id = example.get("instance_id", "")
    step = example.get("step", 0)
    total_steps = example.get("total_steps", 0)
    episode_outcome = example.get("episode_outcome", {})
    memory_info = example.get("memory_info", {})

    # Build system message
    system_msg = (
        "You are a memory management expert for coding agents. "
        "You analyze agent conversation trajectories and evaluate "
        "whether applying context compression (memory management) "
        "would be beneficial at each step. You consider both task "
        "resolution and token efficiency."
    )

    # Build user message (input)
    context_before = example.get("context_before", {})
    context_text = context_before.get("messages_text", "")

    # Truncate context to fit token budget
    max_chars = max_context_tokens * 4  # rough chars-to-tokens
    if len(context_text) > max_chars:
        # Keep beginning and end
        half = max_chars // 2
        context_text = (
            context_text[:half]
            + "\n\n...[MIDDLE TRUNCATED FOR BREVITY]...\n\n"
            + context_text[-half:]
        )

    user_parts = [
        f"## Instance: {instance_id}",
        f"## Strategy: {strategy}",
        f"## Step: {step}/{total_steps}",
        f"## Context ({context_before.get('num_messages', '?')} messages, ~{context_before.get('num_tokens', '?')} tokens)",
        "",
        context_text,
    ]

    if example.get("action"):
        user_parts.extend(["", "## Agent Action at This Step", example["action"][:1000]])

    if example.get("observation"):
        user_parts.extend(["", "## Observation", example["observation"][:1000]])

    user_msg = "\n".join(user_parts)

    # Build assistant message (output/label)
    assistant_parts = []

    if step_type == "compression":
        # For compression steps: show compression decision + metadata + outcome
        comp_meta = example.get("compression_metadata", {})
        context_after = example.get("context_after", {})

        assistant_parts.append("## Memory Decision: COMPRESS")
        assistant_parts.append(f"Compression triggered at step {step}.")
        assistant_parts.append(
            f"Messages: {comp_meta.get('messages_before', '?')} → {comp_meta.get('messages_after', '?')}"
        )
        assistant_parts.append(
            f"Tokens: {comp_meta.get('tokens_before', '?')} → {comp_meta.get('tokens_after', '?')}"
        )
        assistant_parts.append(
            f"Compression ratio: {comp_meta.get('compression_ratio', 1.0):.2f}x"
        )

        if memory_info.get("summary"):
            assistant_parts.append(f"\n## Summary Generated\n{memory_info['summary'][:2000]}")

    elif step_type == "first":
        # First step: describe initial state and strategy choice
        assistant_parts.append("## Memory Decision: INITIAL STATE")
        assistant_parts.append(f"Strategy: {strategy}")
        assistant_parts.append(f"No compression needed at step 1 (context is small).")

    elif step_type in ("pre_compression", "post_compression"):
        # Neighboring steps: describe context growth / recovery
        was_compacted = memory_info.get("was_compacted", False)
        if was_compacted:
            assistant_parts.append("## Memory Decision: COMPRESS")
        else:
            assistant_parts.append("## Memory Decision: NO COMPRESSION NEEDED")

        assistant_parts.append(
            f"Context: {context_before.get('num_messages', '?')} messages, "
            f"~{context_before.get('num_tokens', '?')} tokens"
        )

        if memory_info.get("compression_ratio", 1.0) > 1.01:
            assistant_parts.append(
                f"Compression ratio: {memory_info['compression_ratio']:.2f}x"
            )

    # Always include episode outcome
    resolve_score = episode_outcome.get("resolve_score", 0)
    token_eff = episode_outcome.get("token_efficiency_score", 0)
    reward = episode_outcome.get("reward", 0)

    assistant_parts.extend([
        "",
        "## Episode Outcome",
        f"Resolved: {'yes' if resolve_score > 0 else 'no'}",
        f"Token efficiency: {token_eff:+.3f}",
        f"Combined reward: {reward:.3f}",
        f"Reward delta vs baseline: {episode_outcome.get('reward_delta', 0):+.1f}",
    ])

    assistant_msg = "\n".join(assistant_parts)

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def format_completion(
    example: Dict,
    max_context_tokens: int = 8000,
) -> Optional[Dict]:
    """Format as simple prompt/completion pair (Hugging Face style)."""
    chat = format_openai_chat(example, max_context_tokens)
    if chat is None:
        return None

    messages = chat["messages"]
    prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    completion = messages[2]["content"]

    return {"prompt": prompt, "completion": completion}


def main():
    parser = argparse.ArgumentParser(description="Format training data for LLM fine-tuning")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL from extract_steps.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL for fine-tuning",
    )
    parser.add_argument(
        "--format",
        choices=["openai", "completion"],
        default="openai",
        help="Output format: openai (chat messages) or completion (prompt/completion)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=8000,
        help="Max tokens for context in each example (default: 8000)",
    )
    parser.add_argument(
        "--compression-only",
        action="store_true",
        help="Only include compression steps (skip first/pre/post)",
    )

    args = parser.parse_args()

    formatter = format_openai_chat if args.format == "openai" else format_completion

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    skipped = 0

    with open(input_path) as in_f, open(output_path, "w") as out_f:
        for line in in_f:
            total_in += 1
            example = json.loads(line)

            if args.compression_only and example.get("step_type") != "compression":
                skipped += 1
                continue

            formatted = formatter(example, args.max_context_tokens)
            if formatted is not None:
                out_f.write(json.dumps(formatted) + "\n")
                total_out += 1

    print(f"Formatted {total_out}/{total_in} examples ({skipped} skipped)")
    print(f"Output: {output_path}")

    # Show a sample
    if total_out > 0:
        with open(output_path) as f:
            sample = json.loads(f.readline())
        if args.format == "openai":
            print(f"\nSample (truncated):")
            for msg in sample["messages"]:
                content = msg["content"][:200]
                print(f"  [{msg['role']}]: {content}...")
        else:
            print(f"\nSample prompt (first 200 chars): {sample['prompt'][:200]}...")
            print(f"Sample completion (first 200 chars): {sample['completion'][:200]}...")


if __name__ == "__main__":
    main()
