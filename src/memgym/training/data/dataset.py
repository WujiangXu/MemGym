"""PyTorch Dataset for world model SFT training.

Converts compaction events and augmentation pairs into chat-formatted
training examples for supervised fine-tuning.

Each example:
  System: "You evaluate memory compression events in a coding agent..."
  User: [task + context stats + summary + agent action]
  Assistant: "PREDICTION: SAFE|HARMFUL\nREASONING: ..."

Usage:
    from memgym.training.data.dataset import WorldModelDataset, save_dataset_jsonl

    dataset = WorldModelDataset.from_events_and_pairs(events, pairs)
    save_dataset_jsonl(dataset.examples, "training_output/dataset.jsonl")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from memgym.training.augmentation.replay import CounterfactualPair
from memgym.training.data.compaction import CompactionEvent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a memory management evaluator for coding agents. "
    "You analyze compression events in agent conversation trajectories "
    "and predict whether the compression will help or hurt the agent's "
    "ability to solve the coding task. Consider what information was "
    "preserved vs lost, and whether the agent's subsequent actions "
    "show signs of missing critical context."
)

USER_TEMPLATE = """\
## Coding Task
{problem_statement}

## Compression Event (Step {step}/{total_steps})
- Model: {model}
- Memory strategy: {memory_strategy}
- Context before: {before_tokens} tokens ({before_msgs} messages)
- Context after: {after_tokens} tokens ({after_msgs} messages)

## Raw Trajectory Before Compression
{pre_steps_text}

## Summary Generated (replaces the compressed messages above)
{summary}

## Agent's Action After Compression
{agent_action}

Predict whether this compression event was SAFE (information preserved adequately) \
or HARMFUL (critical information was lost). Explain your reasoning."""

ASSISTANT_TEMPLATE = "PREDICTION: {prediction}\nREASONING: {reasoning}"

# Single-token assistant target — concentrates the entire CE-loss signal on
# the one classification token. Use with WeightedSFTTrainer + per-row
# class_weight to fix the 88/12 imbalance.
ASSISTANT_TEMPLATE_SINGLE_TOKEN = "{prediction}"


@dataclass
class TrainingExample:
    """A single SFT training example in chat format."""

    instance_id: str
    step: int
    label: int  # 1=SAFE, 0=HARMFUL
    source: str  # "episode_label", "counterfactual", "llm_judge"
    messages: List[Dict[str, str]] = field(default_factory=list)
    class_weight: float = 1.0


def _format_pre_steps(pre_steps: List[Dict[str, str]]) -> str:
    """Format the pre-compaction neighborhood for the user prompt.

    We deliberately do NOT truncate — the user-visible signal (what got
    compressed) is the entire raw window before the compression event.
    seq_len=32k is sized to cover 100% of the pilot set.
    """
    if not pre_steps:
        return "(no prior steps available)"

    parts = []
    for s in pre_steps:
        step_no = s.get("step", "?")
        thought = s.get("thought", "")
        action = s.get("action", "")
        observation = s.get("observation", "")
        block = [f"--- Step {step_no} ---"]
        if thought:
            block.append(f"[thought] {thought}")
        if action:
            block.append(f"[action] {action}")
        if observation:
            block.append(f"[observation] {observation}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _event_to_example(
    event: CompactionEvent,
    label: int,
    source: str,
    reasoning: str = "",
) -> TrainingExample:
    """Convert a CompactionEvent into a chat-formatted TrainingExample."""
    prediction = "SAFE" if label == 1 else "HARMFUL"

    if not reasoning:
        if label == 1:
            reasoning = (
                f"The compression at step {event.step} preserved key task context "
                f"with a {event.compression_ratio:.1f}x ratio. The summary captures "
                f"the essential information needed for the coding task."
            )
        else:
            reasoning = (
                f"The compression at step {event.step} lost critical information. "
                f"With {event.forgotten_count} messages forgotten and "
                f"{event.compression_ratio:.1f}x compression, important context "
                f"about the coding task may have been discarded."
            )

    user_content = USER_TEMPLATE.format(
        problem_statement=event.problem_statement[:2000],
        step=event.step,
        total_steps=event.total_steps,
        model=event.model,
        memory_strategy=event.memory_strategy,
        before_tokens=event.context_before_tokens,
        before_msgs=event.context_before_msgs,
        after_tokens=event.context_after_tokens,
        after_msgs=event.context_after_msgs,
        summary=event.summary_generated[:3000],
        agent_action=event.agent_action[:1000],
        pre_steps_text=_format_pre_steps(event.pre_steps),
    )

    assistant_content = ASSISTANT_TEMPLATE.format(
        prediction=prediction,
        reasoning=reasoning,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return TrainingExample(
        instance_id=event.instance_id,
        step=event.step,
        label=label,
        source=source,
        messages=messages,
    )


def _event_to_single_token_example(
    event: CompactionEvent,
    label: int,
    source: str,
    class_weight: float = 1.0,
) -> TrainingExample:
    """Build a TrainingExample whose assistant turn is exactly one classification token.

    Used by `prepare_dataset.py` to feed `WeightedSFTTrainer`. The user
    prompt is identical to `_event_to_example`, but the assistant content
    is just `"SAFE"` or `"HARMFUL"` (no reasoning) so 100% of the loss
    signal lands on the classification token.
    """
    prediction = "SAFE" if label == 1 else "HARMFUL"

    user_content = USER_TEMPLATE.format(
        problem_statement=event.problem_statement[:2000],
        step=event.step,
        total_steps=event.total_steps,
        model=event.model,
        memory_strategy=event.memory_strategy,
        before_tokens=event.context_before_tokens,
        before_msgs=event.context_before_msgs,
        after_tokens=event.context_after_tokens,
        after_msgs=event.context_after_msgs,
        summary=event.summary_generated[:3000],
        agent_action=event.agent_action[:1000],
        pre_steps_text=_format_pre_steps(event.pre_steps),
    )

    assistant_content = ASSISTANT_TEMPLATE_SINGLE_TOKEN.format(
        prediction=prediction,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return TrainingExample(
        instance_id=event.instance_id,
        step=event.step,
        label=label,
        source=source,
        messages=messages,
        class_weight=class_weight,
    )


def _pair_to_example(
    pair: CounterfactualPair,
    event: CompactionEvent,
) -> TrainingExample:
    """Convert a CounterfactualPair into a training example."""
    if pair.diverged:
        reasoning = (
            f"Under {pair.perturbation_name} perturbation, the agent's action "
            f"diverged from the recorded baseline. The recorded action was: "
            f"'{pair.recorded_action[:200]}...' but the model predicted: "
            f"'{pair.predicted_action[:200]}...'. This indicates the compression "
            f"at this step affects the agent's decision-making."
        )
    else:
        reasoning = (
            f"Under {pair.perturbation_name} perturbation, the agent's action "
            f"remained consistent with the baseline. The compression at this step "
            f"did not affect the agent's decision-making."
        )

    return _event_to_example(
        event,
        label=pair.label,
        source=f"counterfactual_{pair.perturbation_name}",
        reasoning=reasoning,
    )


class WorldModelDataset:
    """Collection of training examples for the world model.

    Combines examples from:
    - Episode-level labels (both_pass → SAFE, memory_hurt → HARMFUL)
    - Counterfactual replay pairs (diverged → HARMFUL)
    - LLM-as-a-judge results (optional)
    """

    def __init__(self, examples: Optional[List[TrainingExample]] = None):
        self.examples: List[TrainingExample] = examples or []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def add_from_events(
        self,
        events: List[CompactionEvent],
        default_label: int = 1,
    ) -> int:
        """Add examples from compaction events with episode-level labels.

        Events where episode_resolved is known get their episode label.
        Events without resolve info get default_label.

        Returns count of examples added.
        """
        added = 0
        for event in events:
            if event.episode_resolved is not None:
                label = 1 if event.episode_resolved else 0
            else:
                label = default_label

            example = _event_to_example(event, label, source="episode_label")
            self.examples.append(example)
            added += 1
        return added

    def add_from_counterfactual_pairs(
        self,
        pairs: List[CounterfactualPair],
        events_by_key: Dict[str, CompactionEvent],
    ) -> int:
        """Add examples from counterfactual augmentation pairs.

        Args:
            pairs: CounterfactualPair from replay augmentation.
            events_by_key: {f"{instance_id}_{step}": CompactionEvent} lookup.

        Returns count of examples added.
        """
        added = 0
        for pair in pairs:
            key = f"{pair.instance_id}_{pair.step}"
            event = events_by_key.get(key)
            if event is None:
                continue
            example = _pair_to_example(pair, event)
            self.examples.append(example)
            added += 1
        return added

    def add_from_judge_results(
        self,
        judge_results: List[Any],
        events_by_key: Dict[str, CompactionEvent],
    ) -> int:
        """Add examples from LLM-as-a-judge results.

        Args:
            judge_results: List of JudgeResult from llm_judge.py.
            events_by_key: Event lookup.

        Returns count of examples added.
        """
        added = 0
        for jr in judge_results:
            key = f"{jr.instance_id}_{jr.step}"
            event = events_by_key.get(key)
            if event is None:
                continue
            example = _event_to_example(
                event,
                label=jr.label,
                source="llm_judge",
                reasoning=jr.explanation,
            )
            self.examples.append(example)
            added += 1
        return added

    @classmethod
    def from_events_and_pairs(
        cls,
        events: List[CompactionEvent],
        counterfactual_pairs: Optional[List[CounterfactualPair]] = None,
        judge_results: Optional[List[Any]] = None,
    ) -> "WorldModelDataset":
        """Convenience constructor combining all data sources."""
        dataset = cls()

        # Build event lookup
        events_by_key = {f"{e.instance_id}_{e.step}": e for e in events}

        dataset.add_from_events(events)

        if counterfactual_pairs:
            dataset.add_from_counterfactual_pairs(counterfactual_pairs, events_by_key)

        if judge_results:
            dataset.add_from_judge_results(judge_results, events_by_key)

        return dataset

    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        n = len(self.examples)
        if n == 0:
            return {"total": 0}

        sources = {}
        labels = {0: 0, 1: 0}
        instances = set()

        for ex in self.examples:
            sources[ex.source] = sources.get(ex.source, 0) + 1
            labels[ex.label] = labels.get(ex.label, 0) + 1
            instances.add(ex.instance_id)

        return {
            "total": n,
            "instances": len(instances),
            "label_safe": labels.get(1, 0),
            "label_harmful": labels.get(0, 0),
            "safe_pct": labels.get(1, 0) / n * 100,
            "sources": sources,
        }


def save_dataset_jsonl(
    examples: List[TrainingExample],
    output_path: str | Path,
    extra_fields: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save training examples as JSONL (OpenAI chat fine-tuning format).

    Each line: {"messages": [...], "metadata": {...}, ...extra_fields[i]}

    `extra_fields[i]` is merged at the top level for example `i`. Used by
    `prepare_dataset.py` to attach `split`, `class_weight`, `perturbation`,
    `source_dir` etc. without forking the writer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if extra_fields is not None and len(extra_fields) != len(examples):
        raise ValueError(
            f"extra_fields length {len(extra_fields)} != examples {len(examples)}"
        )

    with open(output_path, "w") as f:
        for i, ex in enumerate(examples):
            record = {
                "messages": ex.messages,
                "metadata": {
                    "instance_id": ex.instance_id,
                    "step": ex.step,
                    "label": ex.label,
                    "source": ex.source,
                },
                "class_weight": ex.class_weight,
            }
            if extra_fields is not None:
                record.update(extra_fields[i])
            f.write(json.dumps(record) + "\n")

    logger.info("Saved %d examples to %s", len(examples), output_path)


def load_dataset_jsonl(path: str | Path) -> List[TrainingExample]:
    """Load training examples from JSONL."""
    path = Path(path)
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            meta = record.get("metadata", {})
            examples.append(TrainingExample(
                instance_id=meta.get("instance_id", ""),
                step=meta.get("step", 0),
                label=meta.get("label", 1),
                source=meta.get("source", "loaded"),
                messages=record["messages"],
                class_weight=float(record.get("class_weight", 1.0)),
            ))
    return examples
