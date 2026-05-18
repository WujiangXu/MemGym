#!/usr/bin/env python
"""
Extract compression event neighborhoods from trajectories for LLM training.

For each trajectory with a memory strategy, extracts:
  - First message (problem setup, initial context)
  - 5 steps before each compression event
  - The compression step itself (context before + after)
  - 5 steps after each compression event

Output: JSONL file with one training example per selected step.

Usage:
    python -m pipelines.memory_training.extract_steps \
        --trajectories results/trajectories_all \
        --pairs pipelines/memory_training/data/paired.json \
        --strategy llm_summarizing \
        --output pipelines/memory_training/data/training_steps.jsonl \
        --context-window 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Approx chars per token for estimation
CHARS_PER_TOKEN = 4


def count_tokens_approx(text: str) -> int:
    """Approximate token count from character length."""
    return len(text) // CHARS_PER_TOKEN


def count_message_tokens(messages: List[Dict]) -> int:
    """Count approximate tokens in a message list."""
    total = 0
    for msg in messages:
        total += count_tokens_approx(str(msg.get("content", "")))
        if "tool_calls" in msg:
            total += count_tokens_approx(json.dumps(msg["tool_calls"]))
    return total


def find_compression_steps(replay_data: Dict) -> List[int]:
    """Find step numbers where compression occurred.

    Looks for steps where the message count drops or a [Summary] marker appears.
    """
    messages = replay_data.get("messages", [])
    steps = replay_data.get("steps", [])
    compression_step_numbers = []

    for i, step_info in enumerate(steps):
        assistant_idx = step_info.get("assistant_msg_index")
        if assistant_idx is None:
            continue

        # Check if there's a summary marker in recent messages before this assistant turn
        # Summaries are inserted as user messages with "[Summary]" prefix
        for j in range(max(0, assistant_idx - 3), assistant_idx):
            if j < len(messages):
                content = str(messages[j].get("content", ""))
                if content.startswith("[Summary]"):
                    compression_step_numbers.append(step_info["step"])
                    break

    return compression_step_numbers


def find_compression_steps_from_training(training_data: Dict) -> List[int]:
    """Find compression steps from _training.json data."""
    steps = training_data.get("steps", [])
    return [
        s["step"]
        for s in steps
        if s.get("memory", {}).get("was_compacted", False)
    ]


def select_training_steps(
    total_steps: int,
    compression_steps: List[int],
    context_window: int = 5,
) -> List[Tuple[int, str]]:
    """Select which steps to include as training examples.

    Returns: [(step_number, step_type), ...]
    step_type: "first", "pre_compression", "compression", "post_compression"
    """
    selected = set()
    typed = {}

    # Always include step 1 (first message / problem setup)
    selected.add(1)
    typed[1] = "first"

    for comp_step in compression_steps:
        # Steps before compression
        for offset in range(context_window, 0, -1):
            s = comp_step - offset
            if s >= 1 and s not in selected:
                selected.add(s)
                typed[s] = "pre_compression"

        # The compression step itself
        selected.add(comp_step)
        typed[comp_step] = "compression"

        # Steps after compression
        for offset in range(1, context_window + 1):
            s = comp_step + offset
            if s <= total_steps and s not in selected:
                selected.add(s)
                typed[s] = "post_compression"

    return sorted([(s, typed[s]) for s in selected], key=lambda x: x[0])


def extract_context_at_step(
    replay_data: Dict,
    step_number: int,
) -> Optional[Dict]:
    """Extract the full message context at a given step.

    Returns the messages the agent would have seen before querying at this step.
    """
    messages = replay_data.get("messages", [])
    steps = replay_data.get("steps", [])

    # Find the step entry
    step_info = None
    for s in steps:
        if s["step"] == step_number:
            step_info = s
            break

    if step_info is None:
        return None

    assistant_idx = step_info.get("assistant_msg_index")
    if assistant_idx is None:
        return None

    # Context = all messages before this assistant turn
    context_messages = messages[:assistant_idx]

    return {
        "messages": context_messages,
        "num_messages": len(context_messages),
        "num_tokens": count_message_tokens(context_messages),
        "assistant_msg_index": assistant_idx,
    }


def extract_step_data(
    replay_data: Dict,
    training_data: Optional[Dict],
    step_number: int,
    step_type: str,
    instance_id: str,
    strategy: str,
    episode_outcome: Dict,
) -> Optional[Dict]:
    """Extract a single training example for a step."""
    context = extract_context_at_step(replay_data, step_number)
    if context is None:
        return None

    messages = replay_data.get("messages", [])
    steps = replay_data.get("steps", [])

    # Find step info
    step_info = None
    for s in steps:
        if s["step"] == step_number:
            step_info = s
            break
    if step_info is None:
        return None

    # Get assistant message (the action taken)
    assistant_idx = step_info.get("assistant_msg_index")
    observation_idx = step_info.get("observation_msg_index")

    assistant_content = ""
    if assistant_idx is not None and assistant_idx < len(messages):
        assistant_content = str(messages[assistant_idx].get("content", ""))

    observation_content = ""
    if observation_idx is not None and observation_idx < len(messages):
        observation_content = str(messages[observation_idx].get("content", ""))

    # For compression steps, also extract context AFTER compression
    # (the next step's context will have the compressed view)
    context_after = None
    if step_type == "compression":
        next_step = step_number + 1
        context_after = extract_context_at_step(replay_data, next_step)

    # Get memory info from training.json if available
    memory_info = {}
    if training_data:
        for s in training_data.get("steps", []):
            if s.get("step") == step_number:
                memory_info = s.get("memory", {})
                break

    example = {
        "instance_id": instance_id,
        "strategy": strategy,
        "step": step_number,
        "step_type": step_type,
        "total_steps": len(steps),
        # Context before this step's LLM query
        "context_before": {
            "num_messages": context["num_messages"],
            "num_tokens": context["num_tokens"],
            # Include truncated message content for training
            "messages_text": _format_messages(context["messages"], max_chars=50000),
        },
        # The action taken and its result
        "action": assistant_content[:2000],
        "observation": observation_content[:2000],
        # Memory state at this step
        "memory_info": memory_info,
        # Episode-level outcome
        "episode_outcome": episode_outcome,
    }

    # Add compression-specific data
    if step_type == "compression" and context_after:
        example["context_after"] = {
            "num_messages": context_after["num_messages"],
            "num_tokens": context_after["num_tokens"],
            "messages_text": _format_messages(
                context_after["messages"], max_chars=50000
            ),
        }
        example["compression_metadata"] = {
            "messages_before": context["num_messages"],
            "messages_after": context_after["num_messages"],
            "tokens_before": context["num_tokens"],
            "tokens_after": context_after["num_tokens"],
            "compression_ratio": (
                context["num_tokens"] / context_after["num_tokens"]
                if context_after["num_tokens"] > 0
                else 1.0
            ),
        }

    return example


def _format_messages(messages: List[Dict], max_chars: int = 50000) -> str:
    """Format messages as readable text, truncated to max_chars."""
    parts = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))

        # Truncate individual long messages
        if len(content) > 5000:
            content = content[:2500] + "\n...[TRUNCATED]...\n" + content[-2500:]

        line = f"[{role}]: {content}"
        parts.append(line)
        total_chars += len(line)

        if total_chars > max_chars:
            parts.append("...[CONTEXT TRUNCATED]...")
            break

    return "\n\n".join(parts)


def process_instance(
    instance_id: str,
    strategy: str,
    trajectories_dir: Path,
    pair_data: Dict,
    context_window: int = 5,
) -> List[Dict]:
    """Process a single instance and extract training steps."""
    strategy_traj_dir = trajectories_dir / strategy / "trajectories"

    replay_path = strategy_traj_dir / f"{instance_id}_replay.json"
    training_path = strategy_traj_dir / f"{instance_id}_training.json"

    if not replay_path.exists():
        return []

    with open(replay_path) as f:
        replay_data = json.load(f)

    training_data = None
    if training_path.exists():
        with open(training_path) as f:
            training_data = json.load(f)

    # Find compression steps
    if training_data:
        compression_steps = find_compression_steps_from_training(training_data)
    else:
        compression_steps = find_compression_steps(replay_data)

    # Select which steps to extract
    total_steps = len(replay_data.get("steps", []))
    selected_steps = select_training_steps(total_steps, compression_steps, context_window)

    if not selected_steps:
        # No compression events — just include first step
        selected_steps = [(1, "first")]

    # Build episode outcome from pair data
    episode_outcome = pair_data.get("reward", {})
    episode_outcome["baseline_resolved"] = pair_data.get("baseline", {}).get("resolved", False)
    episode_outcome["strategy_resolved"] = pair_data.get("strategy_run", {}).get("resolved", False)

    # Extract each selected step
    examples = []
    for step_number, step_type in selected_steps:
        example = extract_step_data(
            replay_data=replay_data,
            training_data=training_data,
            step_number=step_number,
            step_type=step_type,
            instance_id=instance_id,
            strategy=strategy,
            episode_outcome=episode_outcome,
        )
        if example is not None:
            examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Extract training steps from trajectories")
    parser.add_argument(
        "--trajectories",
        default="results/trajectories_all",
        help="Consolidated trajectories directory",
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to paired dataset JSON (from build_pairs.py)",
    )
    parser.add_argument(
        "--strategy",
        default="llm_summarizing",
        help="Strategy to extract compression events from",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Steps before/after compression to include (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="pipelines/memory_training/data/training_steps.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Max instances to process (for testing)",
    )

    args = parser.parse_args()

    # Load paired data
    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    # Build instance lookup
    pair_lookup = {p["instance_id"]: p for p in pairs}

    # Process instances
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    instances_processed = 0
    step_type_counts = {}

    with open(output_path, "w") as out_f:
        instance_ids = sorted(pair_lookup.keys())
        if args.max_instances:
            instance_ids = instance_ids[: args.max_instances]

        for instance_id in instance_ids:
            pair_data = pair_lookup[instance_id]
            examples = process_instance(
                instance_id=instance_id,
                strategy=args.strategy,
                trajectories_dir=Path(args.trajectories),
                pair_data=pair_data,
                context_window=args.context_window,
            )

            for ex in examples:
                out_f.write(json.dumps(ex) + "\n")
                step_type = ex.get("step_type", "unknown")
                step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1

            total_examples += len(examples)
            instances_processed += 1

            if instances_processed % 50 == 0:
                print(f"  Processed {instances_processed}/{len(instance_ids)} instances, {total_examples} examples so far")

    print(f"\nExtracted {total_examples} training examples from {instances_processed} instances")
    print(f"Step type distribution: {json.dumps(step_type_counts, indent=2)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
