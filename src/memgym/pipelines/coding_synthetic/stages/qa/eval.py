"""Evaluation protocols for Coding QA benchmark.

Three protocols:
- Protocol A (full_context): All memory files visible → answers questions
- Protocol B (evicted): Memory files shown one at a time with eviction + note-taking
- Protocol C (fact_retrieval): Agent lists all relevant facts from documents

Supports strategy comparison for Protocol B.
"""

import itertools
import json
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from tqdm import tqdm

from ...types.schemas import (
    CodingQAInstance,
    QAEvalResult,
    FactScore,
    CODING_QA_ANSWER_SCHEMA,
    NOTE_TAKING_QA_SCHEMA,
    FACT_RETRIEVAL_SCHEMA,
)
from ...llm.client import LLMClient
from ...llm.prompts import (
    CODING_QA_ANSWER_PROMPT,
    NOTE_TAKING_QA_PROMPT,
    EVICTED_ANSWER_QA_PROMPT,
    EVICTED_ANSWER_QA_PROMPT_MEMORY_ONLY,
    FACT_RETRIEVAL_PROMPT,
    FACT_JUDGE_PROMPT,
)
from .score import (
    score_fact_recall_keyword,
    score_fact_recall_llm,
    score_noise_rejection,
    score_noise_rejection_keyword,
    aggregate_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_memory_files(memory_files: Dict[str, str]) -> str:
    """Format all memory files into a single text block for prompts."""
    parts = []
    for fname, content in memory_files.items():
        parts.append(f"=== {fname} ===\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# the training phase ingestion grouping
# ---------------------------------------------------------------------------
#
# IR-track memory eval concatenates a turn's documents into one observation
# and makes a single ingest call per turn (~3 calls / instance). Coding eval
# used to call ``memory_method.ingest`` once per ``memory_files`` entry —
# typically 6–10 calls / instance. For LightMem that is 6–10 × 2 = 12–20
# extraction LLM calls before the first QA answer. This helper groups
# ``memory_files`` into a small fixed number of concatenated observations
# so we mirror IR's pattern.

# Defaults tuned to match the in-adapter 15 000-char cap (amem_method.py:58,
# lightmem_method.py:175). Override per call from CLI / mock harness.
DEFAULT_INGEST_CHARS_PER_CALL = 15000
DEFAULT_INGEST_MAX_CALLS = 3


def _group_memory_files_for_ingest(
    memory_files: Dict[str, str],
    chars_per_call: int = DEFAULT_INGEST_CHARS_PER_CALL,
    max_calls: int = DEFAULT_INGEST_MAX_CALLS,
) -> List[tuple]:
    """Pack ``memory_files`` into ≤ ``max_calls`` concatenated observations.

    Each group's text length is capped at ``chars_per_call``. Files are kept
    on whole-file boundaries when possible. Once ``max_calls`` groups are
    open we keep packing into the last one rather than dropping files; the
    per-call truncation absorbs overflow. Returns ``[(group_name,
    group_text), ...]`` with length ≤ ``max_calls``.
    """
    if not memory_files:
        return []

    chars_per_call = max(int(chars_per_call), 1)
    max_calls = max(int(max_calls), 1)

    items: List[tuple] = []
    for fname, content in memory_files.items():
        items.append((fname, f"=== {fname} ===\n{content}"))

    groups: List[List[tuple]] = []
    current: List[tuple] = []
    current_len = 0
    for fname, body in items:
        body_len = len(body)
        will_overflow = current_len + body_len > chars_per_call
        room_for_new_group = len(groups) < max_calls - 1
        if not current or (not will_overflow) or not room_for_new_group:
            current.append((fname, body))
            current_len += body_len
        else:
            groups.append(current)
            current = [(fname, body)]
            current_len = body_len
    if current:
        groups.append(current)

    out: List[tuple] = []
    for i, group in enumerate(groups):
        text = "\n\n".join(body for _, body in group)
        if len(text) > chars_per_call:
            text = text[:chars_per_call]
        names = [fname for fname, _ in group]
        group_name = f"memgroup_{i}__{len(names)}files"
        out.append((group_name, text))
    return out


def _format_repo_files(
    repo_files: Dict[str, str],
    per_file_char_cap: int = 8000,
) -> str:
    """Format repo source files as code blocks for prompts.

    A per-file cap (default 8 KB) is applied before joining so the budget is
    spread across files instead of biasing toward whichever ones happen to
    appear first in dict order. The combined result is still truncated by
    the caller's ``--answerer-prompt-char-cap``; this is the inner cap that
    keeps a single mega-file from squeezing the others out.
    """
    if not repo_files:
        return "(no source files available)"
    parts = []
    for fpath, content in repo_files.items():
        if per_file_char_cap and len(content) > per_file_char_cap:
            dropped = len(content) - per_file_char_cap
            content = content[:per_file_char_cap] + f"\n... [truncated, {dropped} more chars]"
        parts.append(f"### {fpath}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _ask_qa(
    client: LLMClient,
    question: str,
    task_prompt: str,
    repo_context: str,
    memory_files_text: str,
    repo_files_text: str = "(no source files available)",
) -> Dict:
    """Ask a single QA question and parse the response."""
    prompt = CODING_QA_ANSWER_PROMPT.format(
        task_prompt=task_prompt,
        repo_context=repo_context[:8000],
        repo_files_text=repo_files_text[:500000],
        memory_files_text=memory_files_text[:500000],
        question=question,
    )
    result = client.get_json_completion(
        prompt=prompt,
        response_format=CODING_QA_ANSWER_SCHEMA,
        temperature=0.0,
    )
    return {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "sources_used": result.get("sources_used", []),
    }


# ---------------------------------------------------------------------------
# Protocol A: Full-Context QA
# ---------------------------------------------------------------------------

def evaluate_full_context(
    instance: CodingQAInstance,
    client: LLMClient,
    use_llm_scoring: bool = True,
) -> QAEvalResult:
    """Protocol A: Agent sees all memory files + repo context at once."""
    memory_text = _format_memory_files(instance.memory_files)
    repo_files_text = _format_repo_files(getattr(instance, "repo_files", {}))

    qa_scores = []
    all_answers = []

    for qp in instance.qa_pairs:
        response = _ask_qa(
            client, qp.question, instance.task_prompt,
            instance.repo_context, memory_text, repo_files_text,
        )
        answer = response["answer"]
        all_answers.append(answer)

        # Score this QA pair against its linked facts
        linked_facts = [
            f for f in instance.grounding_facts if f.id in qp.linked_fact_ids
        ]
        if use_llm_scoring and linked_facts:
            fact_scores = score_fact_recall_llm(answer, linked_facts, qp.question, client)
        else:
            fact_scores = score_fact_recall_keyword(answer, linked_facts)

        score = sum(1 for fs in fact_scores if fs.recalled) / max(1, len(fact_scores))
        qa_scores.append({
            "qa_id": qp.qa_id,
            "score": score,
            "predicted_answer": answer,
            "gold_answer": qp.gold_answer,
        })

    # Score all facts (not just linked ones) against the combined answers
    combined_answer = " ".join(all_answers)
    if use_llm_scoring:
        all_fact_scores = score_fact_recall_llm(
            combined_answer, instance.grounding_facts, instance.task_prompt, client,
        )
        noise_score = score_noise_rejection(
            combined_answer, instance.distractors, instance.task_prompt, client,
        )
    else:
        all_fact_scores = score_fact_recall_keyword(combined_answer, instance.grounding_facts)
        noise_score = score_noise_rejection_keyword(
            combined_answer, instance.distractors, instance.grounding_facts)

    result = aggregate_scores(all_fact_scores, noise_score, qa_scores, instance)
    result.strategy = "full_context"
    return result


# ---------------------------------------------------------------------------
# Protocol B: Evicted-Context QA
# ---------------------------------------------------------------------------

# Strategy-specific note-taking instructions.
# These are injected into the note-taking prompt to guide HOW the agent
# takes notes. The key design axis: basic/structured/selective differ in
# FORMAT; hypothesis/entity_graph/question_driven differ in SELECTION.
STRATEGY_INSTRUCTIONS = {
    # --- Format-varying strategies (same selection, different output) ---
    "basic": "Write concise notes capturing key facts relevant to the bug.",
    "structured": (
        "Record notes using ONLY this format — one fact per line:\n"
        "  ENTITY: <name> | RELATION: <relationship> | VALUE: <value>\n"
        "  BEHAVIOR: <component> | EXPECTED: <expected> | ACTUAL: <actual>\n"
        "Do NOT write prose or general observations."
    ),
    "selective": (
        "You have VERY LIMITED memory. Write ONLY the most critical facts — "
        "those about specific behavior, error conditions, or design constraints. "
        "Discard everything else. One fact per line, as brief as possible."
    ),
    # --- Selection-varying strategies (different what, not just how) ---
    "hypothesis": (
        "Maintain a SINGLE hypothesis about the root cause of the bug. "
        "After reading each document, state your current hypothesis in one "
        "sentence, then list ONLY facts that SUPPORT or CONTRADICT it. "
        "If the document changes your hypothesis, update it. Format:\n"
        "  HYPOTHESIS: <your current theory>\n"
        "  SUPPORTS: <fact>\n"
        "  CONTRADICTS: <fact>\n"
        "Discard facts unrelated to your hypothesis."
    ),
    "entity_graph": (
        "Track how code entities (functions, classes, variables, modules) "
        "relate to each other and to the bug. Record ONLY relationships "
        "between entities — not descriptions of individual entities. Format:\n"
        "  <entity_A> --[calls/uses/overrides/returns/raises]--> <entity_B>\n"
        "  <entity> --[bug: unexpected behavior]--> <description>\n"
        "Build a connection map, not a fact list."
    ),
    "question_driven": (
        "Before taking notes, predict 3-5 questions someone might ask about "
        "this bug (e.g., 'What function is responsible?', 'What is the "
        "expected behavior?', 'What triggers the failure?'). Then ONLY "
        "record information that directly answers one of your predicted "
        "questions. Format:\n"
        "  Q: <predicted question>\n"
        "  A: <answer from this document>\n"
        "Ignore information that doesn't answer any of your questions."
    ),
}


def evaluate_evicted(
    instance: CodingQAInstance,
    client: LLMClient,
    strategy: str = "basic",
    notes_budget: int = 500,
    use_llm_scoring: bool = True,
    no_repo_files: bool = False,
    memory_method: Optional[object] = None,
    ingest_chars_per_call: int = DEFAULT_INGEST_CHARS_PER_CALL,
    ingest_max_calls: int = DEFAULT_INGEST_MAX_CALLS,
    answerer_prompt_char_cap: int = 60000,
) -> QAEvalResult:
    """Protocol B: Memory files shown one at a time with eviction.

    Args:
        strategy: "none" (no notes), "basic", "structured", "selective"
        notes_budget: Max tokens for notes per document
        no_repo_files: If True, answering phase gets only notes + repo_context
            (no full source code). Tests pure memory retention.
        memory_method: Optional pluggable memory method implementing
            ``ingest(doc_name, doc_content, task_prompt)`` and
            ``retrieve(question, task_prompt)``.  When provided, replaces
            the prompt-based note-taking loop entirely.
        ingest_chars_per_call: Per-ingest text budget (default matches the
            15 000-char cap inside amem_method / lightmem_method).
        ingest_max_calls: Cap on the training phase ingest calls per instance.
            Default 3 mirrors the IR-track 3-turn pattern; this is the main
            knob that prevents the N-files × M-LLM-calls hang.
        answerer_prompt_char_cap: Per-QA-pair, per-section soft cap applied
            to ``notes_text`` and ``repo_files_text`` before prompt
            assembly. Independent of which memory method is in use.
    """
    accumulated_notes = ""
    turn_notes = []
    ingest_calls = 0

    use_memory_method = memory_method is not None

    groups = _group_memory_files_for_ingest(
        instance.memory_files,
        chars_per_call=ingest_chars_per_call,
        max_calls=ingest_max_calls,
    )

    if use_memory_method:
        # the training phase (memory method): one ingest per group instead of per file.
        # Net effect on a typical 8-file instance: amem 8→3 LLM calls,
        # lightmem 16→6 LLM calls.
        memory_method.reset()
        for group_name, group_text in groups:
            memory_method.ingest(group_name, group_text, instance.task_prompt)
            ingest_calls += 1
    else:
        # the training phase (prompt-based): note-taking once per group, accumulating.
        allow_notes = strategy != "none"
        strategy_instruction = STRATEGY_INSTRUCTIONS.get(
            strategy, STRATEGY_INSTRUCTIONS["basic"]
        )

        if allow_notes:
            for grp_idx, (group_name, group_text) in enumerate(groups):
                if accumulated_notes:
                    prev_section = (
                        f"Your notes from previous documents:\n{accumulated_notes}"
                    )
                else:
                    prev_section = "This is the first document — no previous notes."

                prompt = NOTE_TAKING_QA_PROMPT.format(
                    task_prompt=instance.task_prompt,
                    filename=group_name,
                    doc_index=grp_idx + 1,
                    total_docs=len(groups),
                    document_content=group_text,
                    previous_notes_section=prev_section,
                    strategy_instruction=strategy_instruction,
                    budget=notes_budget,
                )
                result = client.get_json_completion(
                    prompt=prompt,
                    response_format=NOTE_TAKING_QA_SCHEMA,
                    temperature=0.0,
                )
                new_notes = result.get("notes", "")
                accumulated_notes = new_notes
                ingest_calls += 1
                turn_notes.append({
                    "doc_index": grp_idx,
                    "filename": group_name,
                    "notes": new_notes,
                })

    # the training phase: Answer questions using notes/memories + context
    qa_scores = []
    all_answers = []
    if not use_memory_method:
        notes_text = accumulated_notes if accumulated_notes else "(no notes taken)"
    if not no_repo_files:
        repo_files_text = _format_repo_files(getattr(instance, "repo_files", {}))

    answerer_cap = max(int(answerer_prompt_char_cap), 1)
    repo_files_dropped_chars = 0

    for qp in instance.qa_pairs:
        # Memory methods do per-question retrieval; prompt path uses full notes.
        if use_memory_method:
            notes_text = memory_method.retrieve(qp.question, instance.task_prompt)

        # B': cap the per-QA-pair payload. Without this, even --no-repo-files
        # can ship 60 KB+ of notes_text per call × N qa_pairs × workers.
        notes_text_capped = notes_text[:answerer_cap] if isinstance(notes_text, str) else str(notes_text)[:answerer_cap]

        if no_repo_files:
            prompt = EVICTED_ANSWER_QA_PROMPT_MEMORY_ONLY.format(
                task_prompt=instance.task_prompt,
                notes=notes_text_capped,
                repo_context=instance.repo_context[:8000],
                question=qp.question,
            )
        else:
            repo_capped = repo_files_text[:answerer_cap]
            if len(repo_files_text) > answerer_cap:
                repo_files_dropped_chars = max(
                    repo_files_dropped_chars, len(repo_files_text) - answerer_cap
                )
            prompt = EVICTED_ANSWER_QA_PROMPT.format(
                task_prompt=instance.task_prompt,
                notes=notes_text_capped,
                repo_files_text=repo_capped,
                repo_context=instance.repo_context[:8000],
                question=qp.question,
            )
        result = client.get_json_completion(
            prompt=prompt,
            response_format=CODING_QA_ANSWER_SCHEMA,
            temperature=0.0,
        )
        answer = result.get("answer", "")
        all_answers.append(answer)

        linked_facts = [
            f for f in instance.grounding_facts if f.id in qp.linked_fact_ids
        ]
        if use_llm_scoring and linked_facts:
            fact_scores = score_fact_recall_llm(answer, linked_facts, qp.question, client)
        else:
            fact_scores = score_fact_recall_keyword(answer, linked_facts)

        score = sum(1 for fs in fact_scores if fs.recalled) / max(1, len(fact_scores))
        qa_scores.append({
            "qa_id": qp.qa_id,
            "score": score,
            "predicted_answer": answer,
            "gold_answer": qp.gold_answer,
        })

    # Score all facts against combined answers
    combined_answer = " ".join(all_answers)
    if use_llm_scoring:
        all_fact_scores = score_fact_recall_llm(
            combined_answer, instance.grounding_facts, instance.task_prompt, client,
        )
        noise_score = score_noise_rejection(
            combined_answer, instance.distractors, instance.task_prompt, client,
        )
    else:
        all_fact_scores = score_fact_recall_keyword(combined_answer, instance.grounding_facts)
        noise_score = score_noise_rejection_keyword(
            combined_answer, instance.distractors, instance.grounding_facts)

    eval_result = aggregate_scores(all_fact_scores, noise_score, qa_scores, instance)
    if use_memory_method:
        method_name = getattr(memory_method, '__class__', type(memory_method)).__name__
        eval_result.strategy = f"method_{method_name}"
    else:
        eval_result.strategy = f"evicted_{strategy}"
    # V1 verification + observability — cheap, fixed-shape diagnostics. Read by
    # mock_eval_smoke.py to assert that Change B kept ingest call counts low.
    eval_result.metadata.update({
        "ingest_calls": ingest_calls,
        "num_memory_files": len(instance.memory_files),
        "num_ingest_groups": len(groups),
        "answerer_prompt_char_cap": answerer_cap,
        "repo_files_dropped_chars": repo_files_dropped_chars,
    })
    if os.environ.get("MEMGYM_DEBUG_INGEST"):
        print(
            f"[ingest] inst={instance.instance_id} method="
            f"{eval_result.strategy} files={len(instance.memory_files)} "
            f"groups={len(groups)} calls={ingest_calls} "
            f"repo_dropped={repo_files_dropped_chars}",
            flush=True,
        )
    return eval_result


# ---------------------------------------------------------------------------
# Protocol C: Fact Retrieval
# ---------------------------------------------------------------------------

def evaluate_fact_retrieval(
    instance: CodingQAInstance,
    client: LLMClient,
    use_llm_scoring: bool = True,
) -> QAEvalResult:
    """Protocol C: Agent lists all relevant facts from documents."""
    memory_text = _format_memory_files(instance.memory_files)

    prompt = FACT_RETRIEVAL_PROMPT.format(
        task_prompt=instance.task_prompt,
        repo_context=instance.repo_context[:4000],
        memory_files_text=memory_text[:60000],
    )
    result = client.get_json_completion(
        prompt=prompt,
        response_format=FACT_RETRIEVAL_SCHEMA,
        temperature=0.0,
    )
    retrieved_facts = result.get("facts", [])

    # Combine all retrieved fact content for scoring
    combined_answer = " ".join(f.get("content", "") for f in retrieved_facts)

    if use_llm_scoring:
        all_fact_scores = score_fact_recall_llm(
            combined_answer, instance.grounding_facts, instance.task_prompt, client,
        )
        noise_score = score_noise_rejection(
            combined_answer, instance.distractors, instance.task_prompt, client,
        )
    else:
        all_fact_scores = score_fact_recall_keyword(combined_answer, instance.grounding_facts)
        noise_score = score_noise_rejection_keyword(
            combined_answer, instance.distractors, instance.grounding_facts)

    # No per-QA scores for retrieval protocol
    eval_result = aggregate_scores(all_fact_scores, noise_score, [], instance)
    eval_result.strategy = "fact_retrieval"
    return eval_result


# ---------------------------------------------------------------------------
# Main evaluation dispatcher
# ---------------------------------------------------------------------------

def evaluate_instance(
    instance: CodingQAInstance,
    client: LLMClient,
    protocol: Literal["full_context", "evicted", "fact_retrieval"] = "evicted",
    strategy: str = "basic",
    notes_budget: int = 500,
    use_llm_scoring: bool = True,
    no_repo_files: bool = False,
    memory_method: Optional[object] = None,
    ingest_chars_per_call: int = DEFAULT_INGEST_CHARS_PER_CALL,
    ingest_max_calls: int = DEFAULT_INGEST_MAX_CALLS,
    answerer_prompt_char_cap: int = 60000,
) -> QAEvalResult:
    """Evaluate one CodingQA instance with the specified protocol."""
    if protocol == "full_context":
        return evaluate_full_context(instance, client, use_llm_scoring)
    elif protocol == "evicted":
        return evaluate_evicted(
            instance, client, strategy, notes_budget, use_llm_scoring,
            no_repo_files=no_repo_files,
            memory_method=memory_method,
            ingest_chars_per_call=ingest_chars_per_call,
            ingest_max_calls=ingest_max_calls,
            answerer_prompt_char_cap=answerer_prompt_char_cap,
        )
    elif protocol == "fact_retrieval":
        return evaluate_fact_retrieval(instance, client, use_llm_scoring)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def _load_checkpoint(checkpoint_path: Optional[str]) -> set:
    """Load instance IDs already completed from a checkpoint JSONL file."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return set()
    done = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done.add(row.get("instance_id", ""))
    return done


def _load_checkpoint_full(checkpoint_path: Optional[str]) -> List[QAEvalResult]:
    """Load full QAEvalResult records from a checkpoint JSONL file.

    Paired with `_load_checkpoint` (which only returns instance IDs for resume
    skip). This helper fixes R-11: after a resume, `evaluate_all` must reload
    the full set of results from disk because its in-memory `results` list
    only contains items processed in the current run.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return []
    loaded: List[QAEvalResult] = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            loaded.append(QAEvalResult(**json.loads(line)))
    return loaded


def _append_checkpoint(checkpoint_path: str, result: Dict, lock: threading.Lock):
    """Thread-safe append of one result to checkpoint JSONL."""
    with lock:
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(result) + "\n")


def evaluate_all(
    instances: Iterable[CodingQAInstance],
    client: LLMClient,
    protocol: str = "evicted",
    strategy: str = "basic",
    notes_budget: int = 500,
    use_llm_scoring: bool = True,
    show_progress: bool = True,
    no_repo_files: bool = False,
    num_workers: int = 1,
    checkpoint_path: Optional[str] = None,
    memory_method: Optional[object] = None,
    total_hint: Optional[int] = None,
    ingest_chars_per_call: int = DEFAULT_INGEST_CHARS_PER_CALL,
    ingest_max_calls: int = DEFAULT_INGEST_MAX_CALLS,
    answerer_prompt_char_cap: int = 60000,
) -> List[QAEvalResult]:
    """Evaluate instances with optional parallelism and checkpointing.

    Accepts any iterable so the caller can stream a multi-GB JSONL file
    without materializing every row up front. Filtering for resume happens
    inline; in parallel mode submission is bounded so the in-flight set
    stays bounded by ``num_workers`` regardless of input size.

    Args:
        instances: iterable of CodingQAInstance — one row at a time is fine.
        num_workers: Number of parallel threads (1 = sequential)
        checkpoint_path: JSONL file for incremental saves + resume
        memory_method: Optional pluggable memory method (see memory_methods/).
            When set, num_workers is forced to 1 because memory methods carry
            mutable state that is not thread-safe across instances.
        total_hint: optional total instance count for tqdm; when ``None`` the
            progress bar advances without a known total.
    """
    if memory_method is not None and num_workers > 1:
        print("[warn] memory_method is stateful — forcing num_workers=1")
        num_workers = 1

    done_ids = _load_checkpoint(checkpoint_path)

    # Resume sanity check: peek a bounded prefix of the input to detect a
    # stale checkpoint (e.g. rebuilt for a different bucket / survivor cut),
    # then chain the peeked rows back so the streaming generator is intact.
    # The previous code printed `len(done_ids)` as the "resume" count without
    # ever verifying the IDs match the input — a stale checkpoint silently
    # produced a from-scratch run with a misleading "Resuming: N" banner.
    adjusted_total = total_hint
    if done_ids:
        peek_n = 200
        peeked: List[CodingQAInstance] = list(
            itertools.islice(instances, peek_n)
        )
        matches_in_peek = sum(
            1 for i in peeked if i.instance_id in done_ids
        )
        if peeked and matches_in_peek == 0:
            print(
                f"[warn] checkpoint at {checkpoint_path} has "
                f"{len(done_ids)} entries but NONE match the first "
                f"{len(peeked)} input instances — checkpoint is likely "
                f"stale (input file changed?). Run will proceed FROM "
                f"SCRATCH; delete the checkpoint to silence this warning."
            )
            # No skips expected — leave total alone.
        else:
            print(
                f"Resuming: {len(done_ids)} checkpoint entries, "
                f"{matches_in_peek}/{len(peeked)} of peeked input matched"
            )
            if total_hint is not None:
                # Optimistic upper-bound: actual skips can't exceed
                # min(total_hint, len(done_ids)). Clamp so the bar never
                # goes past 100%.
                adjusted_total = max(
                    0, total_hint - min(len(done_ids), total_hint)
                )
        instances = itertools.chain(peeked, instances)

    method_label = type(memory_method).__name__ if memory_method else strategy
    desc = f"Evaluating ({protocol}/{method_label})"
    if no_repo_files:
        desc += " [no-repo-files]"

    skip_count_box = [0]

    def _filtered() -> Iterable[CodingQAInstance]:
        for inst in instances:
            if inst.instance_id in done_ids:
                skip_count_box[0] += 1
                continue
            yield inst

    results: List[QAEvalResult] = []
    lock = threading.Lock()

    def _work(inst):
        return evaluate_instance(
            inst, client, protocol, strategy, notes_budget, use_llm_scoring,
            no_repo_files=no_repo_files,
            memory_method=memory_method,
            ingest_chars_per_call=ingest_chars_per_call,
            ingest_max_calls=ingest_max_calls,
            answerer_prompt_char_cap=answerer_prompt_char_cap,
        )

    if num_workers <= 1:
        iterator = _filtered()
        if show_progress:
            iterator = tqdm(iterator, desc=desc, total=adjusted_total)
        for inst in iterator:
            result = _work(inst)
            results.append(result)
            if checkpoint_path:
                _append_checkpoint(checkpoint_path, result.to_jsonl_dict(), lock)
    else:
        pbar = tqdm(total=adjusted_total, desc=desc) if show_progress else None
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            it = iter(_filtered())
            in_flight = set()
            # Prime the pool with up to 2x num_workers so a slow worker doesn't
            # idle the others, but still bound RAM (each in-flight Future pins
            # one CodingQAInstance — the 1m bucket is ~5MB per row).
            cap = max(num_workers * 2, num_workers)
            for _ in range(cap):
                try:
                    in_flight.add(pool.submit(_work, next(it)))
                except StopIteration:
                    break
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    if checkpoint_path:
                        _append_checkpoint(checkpoint_path, result.to_jsonl_dict(), lock)
                    if pbar:
                        pbar.update(1)
                    try:
                        in_flight.add(pool.submit(_work, next(it)))
                    except StopIteration:
                        pass
        if pbar:
            pbar.close()

    if done_ids:
        print(
            f"Resume summary: skipped {skip_count_box[0]} matched "
            f"checkpoint entries, processed {len(results)} new instances "
            f"(checkpoint had {len(done_ids)} total entries)"
        )
        if skip_count_box[0] == 0 and len(results) > 0:
            print(
                f"[warn] checkpoint at {checkpoint_path} matched 0 input "
                f"instances — the run effectively started from scratch. "
                f"Verify that the checkpoint was written for THIS input file."
            )

    # R-11 fix: if resuming, `results` only contains items processed in this
    # run. Reload the full set from the checkpoint so the caller's printed
    # report and --output JSON reflect all 670 rows, not just the new fragment.
    if checkpoint_path and Path(checkpoint_path).exists():
        full = _load_checkpoint_full(checkpoint_path)
        if len(full) > len(results):
            print(
                f"  Loaded {len(full)} instances from checkpoint "
                f"({len(results)} new this run, {len(full) - len(results)} from resume)"
            )
            return full

    return results


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

STRATEGIES = ["none", "basic", "structured", "selective", "hypothesis", "entity_graph", "question_driven"]


def _compare_one_instance(
    inst: CodingQAInstance,
    client: LLMClient,
    protocol: str,
    strategies: List[str],
    notes_budget: int,
    use_llm_scoring: bool,
    no_repo_files: bool,
) -> Dict:
    """Evaluate all strategies for a single instance. Thread-safe."""
    inst_result = {
        "instance_id": inst.instance_id,
        "num_qa_pairs": len(inst.qa_pairs),
        "num_critical_facts": inst.num_critical_facts,
        "num_external_facts": inst.num_external_facts,
    }
    for strat in strategies:
        result = evaluate_instance(
            inst, client, protocol, strat, notes_budget, use_llm_scoring,
            no_repo_files=no_repo_files,
        )
        inst_result[f"fact_recall_{strat}"] = result.fact_recall
        inst_result[f"external_recall_{strat}"] = result.external_fact_recall
        inst_result[f"noise_rejection_{strat}"] = result.noise_rejection
        inst_result[f"qa_accuracy_{strat}"] = result.qa_accuracy
    return inst_result


def compare_strategies(
    instances: List[CodingQAInstance],
    client: LLMClient,
    protocol: str = "evicted",
    strategies: Optional[List[str]] = None,
    notes_budget: int = 500,
    use_llm_scoring: bool = True,
    show_progress: bool = True,
    no_repo_files: bool = False,
    num_workers: int = 1,
    checkpoint_path: Optional[str] = None,
) -> List[Dict]:
    """Run all strategies on each instance and return comparison results.

    Args:
        num_workers: Number of parallel threads (1 = sequential)
        checkpoint_path: JSONL file for incremental saves + resume
    """
    if strategies is None:
        strategies = STRATEGIES

    # Resume from checkpoint
    done_ids = _load_checkpoint(checkpoint_path)
    all_results = []
    if done_ids:
        # Reload completed results
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        print(f"Resuming: {len(done_ids)} instances already done, skipping")

    todo = [inst for inst in instances if inst.instance_id not in done_ids]

    desc = "Comparing strategies"
    if no_repo_files:
        desc += " [no-repo-files]"
    lock = threading.Lock()

    def _work(inst):
        return _compare_one_instance(
            inst, client, protocol, strategies, notes_budget,
            use_llm_scoring, no_repo_files,
        )

    if num_workers <= 1:
        for inst in (tqdm(todo, desc=desc) if show_progress else todo):
            inst_result = _work(inst)
            all_results.append(inst_result)
            if checkpoint_path:
                _append_checkpoint(checkpoint_path, inst_result, lock)
    else:
        pbar = tqdm(total=len(todo), desc=desc) if show_progress else None
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_work, inst): inst for inst in todo}
            for future in as_completed(futures):
                inst_result = future.result()
                all_results.append(inst_result)
                if checkpoint_path:
                    _append_checkpoint(checkpoint_path, inst_result, lock)
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()

    # Print summary
    _print_strategy_summary(all_results, strategies)

    return all_results


def _print_strategy_summary(all_results: List[Dict], strategies: List[str]):
    """Print formatted strategy comparison table."""
    print("\nStrategy Comparison Summary")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Fact Recall':>12} {'External':>12} {'Noise Rej':>12} {'QA Acc':>12}")
    print("-" * 70)
    for strat in strategies:
        fr = sum(r.get(f"fact_recall_{strat}", 0) for r in all_results) / max(1, len(all_results))
        er = sum(r.get(f"external_recall_{strat}", 0) for r in all_results) / max(1, len(all_results))
        nr = sum(r.get(f"noise_rejection_{strat}", 0) for r in all_results) / max(1, len(all_results))
        qa = sum(r.get(f"qa_accuracy_{strat}", 0) for r in all_results) / max(1, len(all_results))
        print(f"{strat:<15} {fr:>11.1%} {er:>11.1%} {nr:>11.1%} {qa:>11.1%}")
