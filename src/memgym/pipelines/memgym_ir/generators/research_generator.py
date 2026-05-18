"""Deep Research Trajectory Generator for MemGym-IR v3.

Generates benchmark instances by running a lightweight research agent that
searches the web to answer complex questions. The agent's trajectory
(queries, results, extracted facts) becomes the instance.

Primary generation method — produces real multi-hop trajectories with genuine
search results as supporting documents and natural distractors.

Usage:
    # Generate from LLM-created questions
    python -m memgym.pipelines.memgym_ir.research_generator \
        --topic "science" --num-questions 50 --max-steps 6

    # Generate from a questions file
    python -m memgym.pipelines.memgym_ir.research_generator \
        --questions-file questions.jsonl --max-steps 6

    # Generate from existing dataset questions
    python -m memgym.pipelines.memgym_ir.research_generator \
        --dataset 2wikimultihop --num-questions 100 --max-steps 4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..llm.client import LLMClient
from ..llm.prompts import (
    RESEARCH_PLAN_PROMPT,
    RESEARCH_PLAN_SCHEMA,
    RESEARCH_ANALYZE_PROMPT,
    RESEARCH_ANALYZE_SCHEMA,
    RESEARCH_FINAL_ANSWER_PROMPT,
    RESEARCH_FINAL_ANSWER_SCHEMA,
    QUESTION_GENERATION_PROMPT,
    QUESTION_GENERATION_SCHEMA,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)
from ..utils.search_tools import get_search_fn
from ..types.schemas import (
    MemGymIRInstance,
    SearchTurn,
    GroundingFactIR,
    EvictionPolicy,
    RubricIR,
)
from ..stages.generate import _compute_memory_requirements

logger = logging.getLogger(__name__)


class ResearchTrajectoryGenerator:
    """Generates research trajectories by running a search-analyze-plan loop."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_fn,
        max_steps: int = 6,
        early_stop_confidence: float = 0.9,
    ):
        self.llm = llm_client
        self.search = search_fn
        self.max_steps = max_steps
        self.early_stop_confidence = early_stop_confidence

    def generate(self, question: str) -> Tuple[List[Dict], str]:
        """Run the research loop and return (trajectory, answer)."""
        trajectory = []

        for step in range(self.max_steps):
            # Plan: decide what to search next
            history_str = self._format_history(trajectory)
            plan = self.llm.get_json_completion(
                prompt=RESEARCH_PLAN_PROMPT.format(
                    question=question, history=history_str
                ),
                response_format=RESEARCH_PLAN_SCHEMA,
                temperature=0.3,
            )

            if plan.get("action") == "answer":
                logger.info(f"  Step {step}: Agent decided to answer (reason: {plan.get('reasoning', '')[:80]})")
                break

            query = plan.get("query", question)
            logger.info(f"  Step {step}: Searching '{query}'")

            # Search: execute the query (with inter-query delay to avoid rate-limiting)
            if step > 0:
                time.sleep(1.5)
            results = self.search(query, num_results=5)
            if not results:
                logger.warning(f"  Step {step}: No results for '{query}', skipping")
                continue

            # Analyze: extract key facts from results
            results_str = self._format_results(results)
            analysis = self.llm.get_json_completion(
                prompt=RESEARCH_ANALYZE_PROMPT.format(
                    question=question,
                    query=query,
                    results=results_str,
                    history=history_str,
                ),
                response_format=RESEARCH_ANALYZE_SCHEMA,
                temperature=0.1,
            )

            trajectory.append({
                "step": step,
                "query": query,
                "results": results,
                "key_facts": analysis.get("facts", []),
                "reasoning": analysis.get("reasoning", ""),
                "contradictions": analysis.get("contradictions", []),
                "confidence": analysis.get("confidence", 0.0),
            })

            # Early stop if confidence is high enough
            if analysis.get("confidence", 0) >= self.early_stop_confidence:
                logger.info(f"  Step {step}: High confidence ({analysis['confidence']:.2f}), stopping")
                break

        answer = self._get_final_answer(question, trajectory)
        return trajectory, answer

    def generate_questions(
        self,
        topic: str = "general knowledge",
        num_questions: int = 10,
        difficulty: str = "medium",
    ) -> List[Dict]:
        """Generate complex multi-hop questions using LLM."""
        questions = []
        for i in range(num_questions):
            result = self.llm.get_json_completion(
                prompt=QUESTION_GENERATION_PROMPT.format(
                    topic=topic, difficulty=difficulty,
                ),
                response_format=QUESTION_GENERATION_SCHEMA,
                temperature=0.8,  # Higher temperature for diversity
            )
            if result.get("question"):
                questions.append(result)
                logger.info(f"  Generated question {i+1}/{num_questions}: {result['question'][:80]}...")
        return questions

    def _get_final_answer(self, question: str, trajectory: List[Dict]) -> str:
        """Get the final answer from the research trajectory."""
        if not trajectory:
            return ""
        summary = self._format_trajectory_summary(trajectory)
        result = self.llm.get_json_completion(
            prompt=RESEARCH_FINAL_ANSWER_PROMPT.format(
                question=question, trajectory_summary=summary,
            ),
            response_format=RESEARCH_FINAL_ANSWER_SCHEMA,
            temperature=0.0,
        )
        return result.get("answer", "")

    def _format_history(self, trajectory: List[Dict]) -> str:
        if not trajectory:
            return "(no research done yet)"
        lines = []
        for step in trajectory:
            facts_str = "; ".join(f["content"] for f in step.get("key_facts", []))
            lines.append(
                f"Step {step['step']}: Searched '{step['query']}' → "
                f"Found: {facts_str or '(no key facts)'}"
            )
        return "\n".join(lines)

    def _format_results(self, results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.get('title', 'Untitled')}\n{r.get('text', '')}")
        return "\n\n".join(parts)

    def _format_trajectory_summary(self, trajectory: List[Dict]) -> str:
        lines = []
        for step in trajectory:
            lines.append(f"--- Step {step['step']}: Query '{step['query']}' ---")
            for f in step.get("key_facts", []):
                lines.append(f"  FACT ({f.get('type', '?')}): {f['content']} [from: {f.get('source', '?')}]")
            if step.get("reasoning"):
                lines.append(f"  Reasoning: {step['reasoning']}")
        return "\n".join(lines)


# =========================================================================
# Trajectory → MemGymIRInstance conversion
# =========================================================================


def _find_downstream_usage(fact: Dict, trajectory: List[Dict], current_step: int) -> int:
    """Find the latest step where a fact's content is referenced downstream."""
    fact_content_lower = fact["content"].lower()
    # Extract key terms from the fact
    key_terms = [
        w.strip(".,!?\"'()") for w in fact_content_lower.split()
        if len(w.strip(".,!?\"'()")) > 4
    ]
    if not key_terms:
        key_terms = fact_content_lower.split()[:3]

    latest_usage = current_step
    for step in trajectory:
        if step["step"] <= current_step:
            continue
        # Check if the fact's key terms appear in the query or reasoning
        search_text = (step.get("query", "") + " " + step.get("reasoning", "")).lower()
        matched = sum(1 for term in key_terms if term in search_text)
        if key_terms and matched / len(key_terms) >= 0.3:
            latest_usage = step["step"]

    return latest_usage


def trajectory_to_instance(
    question: str,
    answer: str,
    trajectory: List[Dict],
    instance_id: str = "",
    eviction_mode: str = "full_eviction",
    window_size: int = 1,
    notes_budget: int = 150,
) -> Optional[MemGymIRInstance]:
    """Convert a research trajectory into a MemGymIRInstance.

    Each search step becomes a SearchTurn with real documents.
    Facts extracted by the agent become GroundingFactIR.
    Memory requirements are computed structurally.
    """
    if not trajectory:
        return None

    # Build SearchTurns from trajectory
    turns = []
    for step in trajectory:
        docs = []
        for result in step.get("results", []):
            # Determine if this result is supporting (contains a key fact)
            is_supporting = any(
                f.get("source", "").lower() == result.get("title", "").lower()
                for f in step.get("key_facts", [])
            )
            docs.append({
                "title": result.get("title", ""),
                "text": result.get("text", ""),
                "is_supporting": is_supporting,
            })
        turns.append(SearchTurn(
            turn_index=step["step"],
            sub_query=step.get("query", ""),
            sub_answer=step.get("reasoning", ""),
            documents=docs,
        ))

    # Build GroundingFactIR from agent's extracted facts
    # For research trajectories, facts from non-final steps are bridge facts
    # that must be retained until the answer phase. Unlike structured QA where
    # hops explicitly link, research steps connect semantically — the agent
    # needs ALL earlier findings to synthesize the final answer.
    last_step = max(s["step"] for s in trajectory) if trajectory else 0
    facts = []
    fact_counter = 0
    for step in trajectory:
        for f in step.get("key_facts", []):
            fact_counter += 1
            is_final_step = (step["step"] == last_step)
            # Try fuzzy match first; fall back to "needed at final step"
            downstream = _find_downstream_usage(f, trajectory, step["step"])
            if downstream == step["step"] and not is_final_step:
                # Fuzzy match found nothing, but this isn't the last step —
                # conservatively assume needed at final step
                downstream = last_step
            fact_type = "bridge_fact" if downstream > step["step"] else "terminal_fact"

            facts.append(GroundingFactIR(
                fact_id=f"F{fact_counter}",
                content=f["content"],
                fact_type=fact_type,
                source_paragraph_title=f.get("source", ""),
                needed_at_hop=downstream,
                first_available_at_hop=step["step"],
                retention_span=downstream - step["step"],
                intermediate_answer=f["content"][:100] if fact_type == "bridge_fact" else "",
            ))

    # Compute memory requirements
    memory_required_ids, per_turn_retention = _compute_memory_requirements(
        turns, facts, eviction_mode, window_size,
    )

    # Build eviction policy
    eviction_policy = EvictionPolicy(
        mode=eviction_mode,
        window_size=window_size,
        allow_notes=True,
        notes_budget_tokens=notes_budget,
    )

    # Build memory files (one per turn)
    memory_files = {}
    for turn in turns:
        docs_text = "\n\n".join(
            f"--- {d['title']} ---\n{d['text']}"
            for d in turn.documents
        )
        memory_files[f"search_results_turn_{turn.turn_index}.txt"] = docs_text

    # Build rubric
    retention_facts = [f.fact_id for f in facts if f.retention_span > 0]
    per_hop_answers = [
        {"hop": step["step"], "answer": step.get("reasoning", "")[:200]}
        for step in trajectory
    ]
    bridge_deps = [
        {
            "fact_id": f.fact_id,
            "from_hop": f.first_available_at_hop,
            "to_hop": f.needed_at_hop,
        }
        for f in facts if f.fact_type == "bridge_fact"
    ]

    rubric = RubricIR(
        expected_answer=answer,
        retention_facts=retention_facts,
        noise_contradictions=[],
        temporal_facts=[],
        per_hop_answers=per_hop_answers,
        bridge_dependencies=bridge_deps,
        per_turn_retention=per_turn_retention,
    )

    # Collect all contradictions found
    contradictions = []
    for step in trajectory:
        for c in step.get("contradictions", []):
            contradictions.append({"step": step["step"], "contradiction": c})

    if not instance_id:
        instance_id = f"memgym_ir__research__{hash(question) % 100000:05d}"

    return MemGymIRInstance(
        instance_id=instance_id,
        base_instance=instance_id,
        source_dataset="deep_research",
        question=question,
        original_question=question,
        answer=answer,
        num_hops=len(turns),
        decomposition=[
            {"sub_question": t.sub_query, "sub_answer": t.sub_answer}
            for t in turns
        ],
        turns=turns,
        grounding_facts=facts,
        distractors=[],
        memory_files=memory_files,
        gold_supporting_paragraphs=[
            {"title": d["title"], "text": d["text"]}
            for turn in turns for d in turn.documents if d.get("is_supporting")
        ],
        contradictions=contradictions,
        transforms_applied=["deep_research"],
        difficulty_preset="research",
        reveal_mode="immediate",
        eviction_policy=eviction_policy,
        memory_required_facts=memory_required_ids,
        total_tokens=sum(
            len(d.get("text", "").split()) for turn in turns for d in turn.documents
        ),
        rubric=rubric,
    )


# =========================================================================
# Question sources
# =========================================================================


def load_questions_from_file(path: str) -> List[Dict]:
    """Load questions from a JSONL file. Each line: {"question": "...", ...}."""
    questions = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    return questions


def load_questions_from_dataset(
    dataset_name: str,
    num_questions: int = 100,
    min_hops: int = 3,
) -> List[Dict]:
    """Load questions from an existing QA dataset."""
    from .stages.adapters import ADAPTERS

    adapter_cls = ADAPTERS.get(dataset_name)
    if not adapter_cls:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(ADAPTERS.keys())}")

    adapter = adapter_cls()
    raw_instances = adapter.load_raw(limit=num_questions * 3)

    questions = []
    for raw in raw_instances:
        parsed = adapter.parse(raw)
        if parsed and parsed.num_hops >= min_hops:
            questions.append({
                "question": parsed.question,
                "expected_answer": parsed.answer,
                "source_id": parsed.instance_id,
                "num_hops": parsed.num_hops,
            })
        if len(questions) >= num_questions:
            break

    return questions


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Trajectory Generator for MemGym-IR v3"
    )

    # Question source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--topic", type=str,
        help="Generate questions on this topic via LLM",
    )
    source.add_argument(
        "--questions-file", type=str,
        help="JSONL file with pre-made questions ({\"question\": \"...\"})",
    )
    source.add_argument(
        "--dataset", type=str,
        help="Use questions from existing dataset (2wikimultihop, musique, etc.)",
    )

    # Generation config
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--search-tool", type=str, default="ddg+wikipedia",
                        choices=["ddg", "wikipedia", "ddg+wikipedia"])
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--min-hops", type=int, default=3)

    # LLM config
    parser.add_argument("--model", type=str,
                        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

    # Instance config
    parser.add_argument("--eviction-mode", type=str, default="full_eviction",
                        choices=["full_eviction", "sliding_window"])
    parser.add_argument("--notes-budget", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=1)

    # Output
    parser.add_argument("--output", type=str, default="output_ir/research_instances.jsonl")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize LLM client and search function
    client = LLMClient(model=args.model)
    search_fn = get_search_fn(args.search_tool)
    generator = ResearchTrajectoryGenerator(
        llm_client=client,
        search_fn=search_fn,
        max_steps=args.max_steps,
    )

    # Load or generate questions
    if args.topic:
        logger.info(f"Generating {args.num_questions} questions on topic '{args.topic}'...")
        questions = generator.generate_questions(
            topic=args.topic,
            num_questions=args.num_questions,
            difficulty=args.difficulty,
        )
    elif args.questions_file:
        logger.info(f"Loading questions from {args.questions_file}...")
        questions = load_questions_from_file(args.questions_file)[:args.num_questions]
    else:
        logger.info(f"Loading questions from dataset '{args.dataset}'...")
        questions = load_questions_from_dataset(
            args.dataset,
            num_questions=args.num_questions,
            min_hops=args.min_hops,
        )

    logger.info(f"Loaded {len(questions)} questions")

    # Generate trajectories and convert to instances
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_generated = 0
    num_failed = 0
    t0 = time.time()

    with open(output_path, "w") as f:
        for i, q_data in enumerate(questions):
            question = q_data["question"]
            logger.info(f"\n[{i+1}/{len(questions)}] {question[:80]}...")

            try:
                trajectory, answer = generator.generate(question)

                if not trajectory:
                    logger.warning(f"  Empty trajectory, skipping")
                    num_failed += 1
                    continue

                instance = trajectory_to_instance(
                    question=question,
                    answer=answer,
                    trajectory=trajectory,
                    instance_id=f"memgym_ir__research__{i:05d}",
                    eviction_mode=args.eviction_mode,
                    window_size=args.window_size,
                    notes_budget=args.notes_budget,
                )

                if instance and instance.memory_required_facts:
                    f.write(json.dumps(instance.to_jsonl_dict()) + "\n")
                    num_generated += 1
                    logger.info(
                        f"  → {len(trajectory)} steps, {len(instance.grounding_facts)} facts, "
                        f"{len(instance.memory_required_facts)} memory-required"
                    )
                else:
                    logger.info(f"  → No memory-required facts, skipping")
                    num_failed += 1

            except Exception as e:
                logger.error(f"  Error: {e}")
                num_failed += 1

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.0f}s: {num_generated} instances generated, {num_failed} failed")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
