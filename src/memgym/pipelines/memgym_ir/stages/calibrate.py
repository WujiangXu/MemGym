"""Stage 3: Harden + Verify via memory fact ablation.

Replaces the old separate harden (Stage 5) and verify (Stage 4) stages
with a single iterative calibration loop. Tests whether the instance
genuinely requires memory by ablating facts one at a time and measuring
the accuracy trend.

Calibration criteria:
  - score_all_memory >= 0.5  → solvable with curated notes
  - memory_gap >= 0.3        → curated memory helps
  - score_no_memory <= 0.5   → not solvable without memory
  - score_long_context <= 0.85 → long context dump is still hard
  - score_all_memory >= score_long_context → notes match or beat raw dump
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types.schemas import MemGymIRInstance, GroundingFactIR
from ..llm.client import LLMClient
from ..llm.prompts import (
    MEMORY_ABLATION_QA_PROMPT,
    MEMORY_ABLATION_QA_SCHEMA,
    ANSWER_JUDGE_PROMPT,
    ANSWER_JUDGE_SCHEMA,
)

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of memory fact ablation testing."""
    scores: List[float] = field(default_factory=list)  # [score_0, score_1, ..., score_N]
    score_no_memory: float = 0.0       # score with 0 facts
    score_all_memory: float = 0.0      # score with all facts
    score_long_context: float = 0.0    # score with all docs (no notes)
    memory_gap: float = 0.0            # all_memory - no_memory
    context_gap: float = 0.0           # all_memory - long_context
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "score_no_memory": self.score_no_memory,
            "score_all_memory": self.score_all_memory,
            "score_long_context": self.score_long_context,
            "memory_gap": self.memory_gap,
            "context_gap": self.context_gap,
            "passed": self.passed,
        }


def memory_fact_ablation(
    instance: MemGymIRInstance,
    verifier_client: LLMClient,
) -> AblationResult:
    """Test accuracy trend as memory facts increase.

    For an instance with N bridge facts, tests with 0, 1, ..., N facts
    given as "notes" alongside the last turn's visible documents.
    Also tests with all documents as a long context dump.
    """
    bridge_facts = [f for f in instance.grounding_facts if f.retention_span > 0]
    if not bridge_facts:
        return AblationResult(passed=False)

    # Get last turn's visible documents
    last_turn = instance.turns[-1] if instance.turns else None
    if not last_turn:
        return AblationResult(passed=False)

    all_last_docs_text = _format_documents(last_turn.documents)
    non_supporting = [d for d in last_turn.documents
                      if not d.get("is_supporting", False)]
    non_supporting_text = _format_documents(non_supporting)

    # Test with 0, 1, ..., N facts. Each k is independent — run all
    # ablation calls plus the long-context probe concurrently. For 6-hop
    # instances this collapses 8 sequential LLM roundtrips (each of which
    # is answer+judge, so 16 calls serialized) into one concurrent batch.
    all_docs_text = _format_documents(
        [doc for turn in instance.turns for doc in turn.documents]
    )

    def _run_k(k: int) -> float:
        facts_given = bridge_facts[:k]
        notes = "\n".join(f"- {f.content}" for f in facts_given) if facts_given else "(no notes available)"
        # k=0 (no memory): exclude supporting docs so the model can't
        # read the answer from the grounding-fact source paragraph.
        # k>0 (has memory): show all docs — supporting docs provide
        # context but the model already has the facts from notes.
        docs_for_test = non_supporting_text if k == 0 else all_last_docs_text
        return _ask_and_judge(
            question=instance.question,
            notes=notes,
            visible_documents=docs_for_test,
            expected_answer=instance.answer,
            client=verifier_client,
        )

    def _run_long_ctx() -> float:
        return _ask_and_judge(
            question=instance.question,
            notes="(no notes available)",
            visible_documents=all_docs_text,
            expected_answer=instance.answer,
            client=verifier_client,
        )

    n_calls = len(bridge_facts) + 2  # k=0..N inclusive + long_ctx
    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        ablation_futures = [pool.submit(_run_k, k) for k in range(len(bridge_facts) + 1)]
        long_ctx_future = pool.submit(_run_long_ctx)
        scores = [f.result() for f in ablation_futures]
        score_long_context = long_ctx_future.result()

    result = AblationResult(
        scores=scores,
        score_no_memory=scores[0],
        score_all_memory=scores[-1],
        score_long_context=score_long_context,
        memory_gap=scores[-1] - scores[0],
        context_gap=scores[-1] - score_long_context,
    )

    # Check calibration criteria.
    # - no_memory: only non-supporting docs visible (supporting docs
    #   excluded since they contain grounding fact text). Threshold 0.5.
    # - long_context threshold is 0.90: with 100K tokens, the model can find
    #   some answers by reading everything. 0.85 was too strict.
    # - gap >= 0.3: memory must add meaningful value over no-memory baseline.
    result.passed = (
        result.score_all_memory >= 0.5
        and result.memory_gap >= 0.3
        and result.score_no_memory <= 0.5
        and result.score_long_context <= 0.90
        and result.score_all_memory >= result.score_long_context
    )

    return result


def harden_and_verify(
    instance: MemGymIRInstance,
    verifier_client: LLMClient,
    worker_client: Optional[LLMClient] = None,
    max_retries: int = 2,
) -> Optional[MemGymIRInstance]:
    """Iterative calibration: harden then verify via fact ablation.

    Returns the calibrated instance, or None if calibration fails.
    For now, applies no hardening transforms (the grow-by-search pipeline
    already produces structured instances). Future: add query fuzz,
    distractor scaling, temporal shift from v2 harden.py.
    """
    for attempt in range(max_retries):
        logger.info(f"  Calibration attempt {attempt + 1}/{max_retries}")

        ablation = memory_fact_ablation(instance, verifier_client)

        logger.info(
            f"    no_memory={ablation.score_no_memory:.2f}, "
            f"all_memory={ablation.score_all_memory:.2f}, "
            f"long_context={ablation.score_long_context:.2f}, "
            f"gap={ablation.memory_gap:.2f}, "
            f"passed={ablation.passed}"
        )

        if ablation.passed:
            instance.verification = ablation.to_dict()
            return instance

        # Diagnose and log reason for failure
        if ablation.score_all_memory < 0.5:
            logger.info("    → Too hard: even with all facts, score < 0.5")
        elif ablation.score_long_context > 0.85:
            logger.info("    → Distractors too weak: long context dump scores > 0.85")
        elif ablation.memory_gap < 0.3:
            logger.info("    → Memory doesn't help enough: gap < 0.3")
        elif ablation.score_no_memory > 0.5:
            logger.info("    → Too easy: solvable without any memory")

        # No automated adjustment yet — just retry with different verifier temperature
        # Future: adjust difficulty config (add distractors, fuzz query, etc.)

    logger.info("  Calibration failed after all retries")
    return None


def _ask_and_judge(
    question: str,
    notes: str,
    visible_documents: str,
    expected_answer: str,
    client: LLMClient,
) -> float:
    """Ask the question with given notes + docs, then judge the answer."""
    # Get answer
    response = client.get_json_completion(
        prompt=MEMORY_ABLATION_QA_PROMPT.format(
            question=question,
            notes=notes,
            visible_documents=visible_documents,
        ),
        response_format=MEMORY_ABLATION_QA_SCHEMA,
        temperature=0.0,
    )
    predicted = response.get("answer", "")

    if not predicted or predicted.lower().startswith("insufficient"):
        return 0.0

    # Judge
    judge = client.get_json_completion(
        prompt=ANSWER_JUDGE_PROMPT.format(
            gold_answer=expected_answer,
            predicted_answer=predicted,
            question=question,
        ),
        response_format=ANSWER_JUDGE_SCHEMA,
        temperature=0.0,
    )
    return judge.get("score", 0.0)


def _format_documents(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs):
        title = doc.get("title", "Untitled") if isinstance(doc, dict) else "Untitled"
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        parts.append(f"[{i+1}] {title}\n{text}")
    return "\n\n".join(parts) if parts else "(no documents available)"
