"""Convert coding_synthetic instances to Coding QA format (v2).

v2 improvements over v1:
- Fact clustering (union-find) for diverse 1-N fact linking per question
- Multi-tier QA: hard (1 fact) / medium (2-3) / easy (4+)
- Adversarial QA-specific distractors per question
- Full repo file extraction (Docker-based, falls back to skeleton)
- Token budget scaling via --target-tokens

Usage:
    from .convert import convert_all
    convert_all("results/full_11k/coding_synthetic_instances.jsonl",
                "results/coding_qa/coding_qa_instances.jsonl",
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .distractor_pool import DistractorPool

from ...types.schemas import (
    MemGymInstance,
    GroundingFact,
    DistractorFact,
    QAPair,
    CodingQAInstance,
    FACT_QA_GENERATION_SCHEMA,
    GOLD_ANSWER_SCHEMA,
    MULTI_FACT_QA_SCHEMA,
    ADVERSARIAL_QA_DISTRACTOR_SCHEMA,
)
from ...llm.client import LLMClient
from ...llm.prompts import (
    PER_FACT_QA_PROMPT,
    GOLD_ANSWER_PROMPT,
    MULTI_FACT_QA_PROMPT,
    ADVERSARIAL_QA_DISTRACTOR_PROMPT,
)
from ...utils.parallel import parallel_map
from ...utils.token_counter import count_tokens


# ---------------------------------------------------------------------------
# Token budget presets
# ---------------------------------------------------------------------------

TOKEN_PRESETS = {
    "small": 10_000,
    "medium": 50_000,
    "large": 100_000,
    "xl": 500_000,
    "extreme": 1_000_000,
}


def resolve_target_tokens(value: Optional[str]) -> Optional[int]:
    """Resolve a target-tokens CLI value to an int.

    Accepts preset names ("small", "large") or raw integers ("100000").
    Returns None if value is None.
    """
    if value is None:
        return None
    if value in TOKEN_PRESETS:
        return TOKEN_PRESETS[value]
    return int(value)


# ---------------------------------------------------------------------------
# Fact clustering (union-find)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Simple union-find for grouping facts by shared hunks."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cluster_facts(facts: List[GroundingFact]) -> List[List[GroundingFact]]:
    """Group facts into independent clusters by hunk coverage.

    Algorithm:
    1. Build hunk-to-facts mapping: {hunk_idx: [fact_idx, ...]}
    2. Union-find: facts sharing ANY hunk go in the same cluster
    3. Within each cluster, sub-split non-discoverable critical facts into
       their own singleton clusters (for hard single-fact probes)
    4. Sort clusters: single-fact clusters first (hard), then larger (easier)
    """
    if not facts:
        return []

    n = len(facts)
    uf = _UnionFind(n)

    # Build hunk → fact indices mapping
    hunk_to_facts: Dict[int, List[int]] = {}
    for i, f in enumerate(facts):
        for h in f.hunks_enabled:
            hunk_to_facts.setdefault(h, []).append(i)

    # Union facts that share any hunk
    for hunk_facts in hunk_to_facts.values():
        for j in range(1, len(hunk_facts)):
            uf.union(hunk_facts[0], hunk_facts[j])

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    clusters: List[List[GroundingFact]] = []

    for indices in groups.values():
        group_facts = [facts[i] for i in indices]

        # Sub-split: pull out non-discoverable critical facts as singletons
        singletons = []
        remaining = []
        for f in group_facts:
            if not f.discoverable and f.relevance == "critical":
                singletons.append(f)
            else:
                remaining.append(f)

        # Each non-discoverable critical fact gets its own hard cluster
        for f in singletons:
            clusters.append([f])

        # Remaining facts stay as one cluster (if non-empty)
        if remaining:
            clusters.append(remaining)

    # Sort: single-fact clusters first (hard probes), then larger
    clusters.sort(key=lambda c: len(c))
    return clusters


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_LEAK_CHECK_ANSWER_PROMPT = (
    "Answer the following question based on your general software-engineering "
    "knowledge. Do NOT speculate about a specific codebase.\n\n"
    "## Question\n{question}\n\n"
    'Return JSON: {{"answer": "<1-3 sentences>"}}'
)
_LEAK_CHECK_JUDGE_PROMPT = (
    "You are deciding whether an answer contains the specific value / detail "
    "in a grounding fact.\n\n"
    "## Grounding fact (the detail we don't want leaked)\n{fact_content}\n\n"
    "## Question\n{question}\n\n"
    "## Answer (produced WITHOUT seeing any code)\n{answer}\n\n"
    "Does the answer demonstrate knowledge of the specific detail in the "
    "fact (the specific value / identifier / behavior)? Ignore generic "
    "restatements of the question.\n\n"
    'Return JSON: {{"contains_fact": bool, "explanation": string}}'
)
_LEAK_CHECK_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "leak_answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
    },
}
_LEAK_CHECK_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "leak_judge", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contains_fact": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["contains_fact", "explanation"],
            "additionalProperties": False,
        },
    },
}


def _question_leaks(
    question: str, fact_content: str, client: LLMClient,
) -> bool:
    """Inline P4 check: does Q leak the fact's detail value?

    Mirrors scripts/post_check_coding_qa.py's check_question_leakage but
    collapses the cost to 2 calls per Q. Failures are treated as "did not
    leak" so a flaky judge call doesn't spuriously reject good Qs.
    """
    try:
        a = client.get_json_completion(
            prompt=_LEAK_CHECK_ANSWER_PROMPT.format(question=question),
            response_format=_LEAK_CHECK_ANSWER_SCHEMA, temperature=0.0,
        ).get("answer", "")
        if not a:
            return False
        j = client.get_json_completion(
            prompt=_LEAK_CHECK_JUDGE_PROMPT.format(
                fact_content=fact_content, question=question, answer=a[:1500],
            ),
            response_format=_LEAK_CHECK_JUDGE_SCHEMA, temperature=0.0,
        )
        return bool(j.get("contains_fact", False))
    except Exception:
        return False


def _generate_cluster_question(
    cluster: List[GroundingFact],
    task_prompt: str,
    existing_questions: List[str],
    client: LLMClient,
    leak_check: bool = True,
    max_retries: int = 2,
) -> Optional[Dict]:
    """Generate one question for a fact cluster.

    For single-fact clusters, uses PER_FACT_QA_PROMPT and (when leak_check
    is True) rejects questions that leak the fact detail, retrying up to
    ``max_retries`` times at rising temperature. Multi-fact clusters skip
    the inline check because the post-check already handles them well.
    """
    single = len(cluster) == 1
    if single:
        fact = cluster[0]
        # Prefer topic (non-leaking seed). Fall back to content for legacy
        # instances whose facts pre-date the topic/detail split (Option A).
        fact_topic = fact.topic or fact.content
        prompt = PER_FACT_QA_PROMPT.format(
            category=fact.category,
            fact_topic=fact_topic,
            task_prompt=task_prompt[:2000],
        )
        schema = FACT_QA_GENERATION_SCHEMA
    else:
        facts_list = "\n".join(
            f"- [{f.category}] {f.content}" for f in cluster
        )
        existing_str = "\n".join(
            f"- {q}" for q in existing_questions[-10:]
        ) if existing_questions else "(none yet)"

        prompt = MULTI_FACT_QA_PROMPT.format(
            facts_list=facts_list,
            task_prompt=task_prompt[:2000],
            existing_questions=existing_str,
        )
        schema = MULTI_FACT_QA_SCHEMA

    last_question = ""
    for attempt in range(max_retries + 1):
        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=schema,
                temperature=0.3 + 0.2 * attempt,
            )
            question = result.get("question", "")
        except Exception as e:
            print(f"  Question generation failed for cluster of {len(cluster)}: {e}")
            continue

        if not question or len(question) <= 15:
            continue
        last_question = question

        if single and leak_check:
            if _question_leaks(question, cluster[0].content, client):
                continue  # retry

        return {
            "question": question,
            "category": result.get("category", cluster[0].category),
            "linked_fact_ids": [f.id for f in cluster],
        }

    # Exhausted retries — return the last candidate (even if it leaked) so we
    # still emit a QA pair; the post-check will drop it downstream.
    if last_question:
        return {
            "question": last_question,
            "category": cluster[0].category,
            "linked_fact_ids": [f.id for f in cluster],
        }
    return None


def _generate_gold_answer(
    question: str,
    linked_facts: List[GroundingFact],
    repo_context: str,
    client: LLMClient,
) -> str:
    """Generate a gold-standard answer from the linked facts."""
    facts_str = "\n".join(
        f"- [{f.category}] {f.content}" for f in linked_facts
    )
    prompt = GOLD_ANSWER_PROMPT.format(
        question=question,
        facts=facts_str,
        repo_context=repo_context[:3000],
    )
    try:
        result = client.get_json_completion(
            prompt=prompt,
            response_format=GOLD_ANSWER_SCHEMA,
            temperature=0.1,
        )
        return result.get("answer", "")
    except Exception:
        return ""


def _classify_difficulty(linked_facts: List[GroundingFact]) -> str:
    """Classify QA difficulty based on memory requirement and synthesis complexity.

    New mapping (v3):
    - easy: All facts are discoverable (answerable from code alone)
    - medium: 1-2 non-discoverable facts (memory recall needed)
    - hard: 3+ non-discoverable facts (multi-fact memory synthesis)
    """
    requires_memory = any(not f.discoverable for f in linked_facts)
    if not requires_memory:
        return "easy"
    n = len(linked_facts)
    if n >= 3:
        return "hard"
    # 1 or 2 memory-required facts
    return "medium"


def _recompute_difficulty_for_context(
    qa_pairs: List[QAPair],
    total_context_tokens: int,
) -> None:
    """Upgrade difficulty labels based on context length (needle-in-haystack).

    Single-fact recall in a 100K+ context is harder than in 10K context.
    Mutates qa_pairs in place.
    """
    for qp in qa_pairs:
        if qp.difficulty == "medium" and total_context_tokens > 100_000:
            qp.difficulty = "hard"


# ---------------------------------------------------------------------------
# Adversarial QA distractor generation
# ---------------------------------------------------------------------------

def _generate_qa_distractors(
    qa_pairs: List[QAPair],
    facts: List[GroundingFact],
    client: LLMClient,
    num_per_question: int = 2,
) -> List[DistractorFact]:
    """Generate targeted adversarial distractors for each QA pair."""
    fact_map = {f.id: f for f in facts}
    all_distractors: List[DistractorFact] = []
    distractor_idx = 0

    for qp in qa_pairs:
        linked = [fact_map[fid] for fid in qp.linked_fact_ids if fid in fact_map]
        if not linked:
            continue

        correct_facts_str = "\n".join(f"- {f.content}" for f in linked)

        prompt = ADVERSARIAL_QA_DISTRACTOR_PROMPT.format(
            num_distractors=num_per_question,
            question=qp.question,
            gold_answer_summary=qp.gold_answer[:300],
            correct_facts=correct_facts_str,
        )

        try:
            result = client.get_json_completion(
                prompt=prompt,
                response_format=ADVERSARIAL_QA_DISTRACTOR_SCHEMA,
                temperature=0.5,
            )
            for d in result.get("distractors", [])[:num_per_question]:
                distractor_idx += 1
                all_distractors.append(DistractorFact(
                    id=f"DQ{distractor_idx}",
                    content=d.get("content", ""),
                    source="adversarial_qa",
                    target_qa_id=qp.qa_id,
                ))
        except Exception as e:
            print(f"  Distractor generation failed for {qp.qa_id}: {e}")

    return all_distractors


# ---------------------------------------------------------------------------
# Repo file extraction
# ---------------------------------------------------------------------------

def _extract_repo_files(
    instance: MemGymInstance,
    output_dir: Path,
) -> Dict[str, str]:
    """Extract full file contents from Docker image.

    Falls back to empty dict if Docker is not available.
    """
    if not instance.image_name or not instance.referenced_files:
        return {}

    try:
        from ..extract import extract_repo_context
        from ...types.schemas import FilteredInstance

        # Build minimal FilteredInstance for the extraction function
        fi = FilteredInstance(
            instance_id=instance.base_instance or instance.instance_id,
            problem_statement=instance.task_prompt,
            patch=instance.patch,
            patch_metrics={"lines_added": 0, "lines_removed": 0,
                           "lines_changed": 0, "num_hunks": 0,
                           "files_changed": []},
            FAIL_TO_PASS=[],
            PASS_TO_PASS=[],
            image_name=instance.image_name,
            repo="",
            base_commit="",
        )

        return extract_repo_context(
            instance=fi,
            referenced_files=instance.referenced_files,
            output_dir=output_dir / "repo_context_cache",
        )
    except Exception as e:
        print(f"  Repo file extraction failed (Docker unavailable?): {e}")
        return {}


# ---------------------------------------------------------------------------
# Batch repo file extraction (full Python codebase, Docker-based)
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = ("__pycache__/", ".git/", "node_modules/", ".tox/", ".eggs/", "dist/", "build/", ".egg-info/")


def _extract_full_repo_files(
    image_name: str,
    max_files: int = 300,
    max_file_size: int = 50_000,
) -> Dict[str, str]:
    """Extract all Python files from a Docker image's /testbed directory.

    Starts a container, discovers .py files, reads each one, then removes
    the container. Skips files in __pycache__, .git, node_modules, etc.
    and files larger than max_file_size bytes.
    """
    container_name = f"repo_extract_{random.randint(100000, 999999)}"
    repo_files: Dict[str, str] = {}

    try:
        # Start container
        subprocess.run(
            ["docker", "create", "--name", container_name, image_name, "sleep", "3600"],
            capture_output=True, text=True, check=True, timeout=300,
        )
        subprocess.run(
            ["docker", "start", container_name],
            capture_output=True, text=True, check=True, timeout=120,
        )

        # Discover Python files
        find_result = subprocess.run(
            ["docker", "exec", container_name, "find", "/testbed",
             "-name", "*.py", "-type", "f", "-size", f"-{max_file_size}c"],
            capture_output=True, text=True, timeout=120,
        )
        if find_result.returncode != 0:
            print(f"  find failed: {find_result.stderr[:200]}")
            return {}

        all_paths = [p.strip() for p in find_result.stdout.strip().split("\n") if p.strip()]

        # Filter out unwanted directories
        filtered = []
        for p in all_paths:
            if any(skip in p for skip in _SKIP_PATTERNS):
                continue
            filtered.append(p)

        # Sort by path depth (shallow first) and truncate
        filtered.sort(key=lambda p: p.count("/"))
        filtered = filtered[:max_files]

        # Read each file
        for fpath in filtered:
            try:
                cat_result = subprocess.run(
                    ["docker", "exec", container_name, "cat", fpath],
                    capture_output=True, text=True, timeout=10,
                )
                if cat_result.returncode == 0 and cat_result.stdout:
                    # Store relative to /testbed
                    rel_path = fpath.replace("/testbed/", "", 1)
                    repo_files[rel_path] = cat_result.stdout
            except (subprocess.TimeoutExpired, Exception):
                continue

    except Exception as e:
        print(f"  Docker extraction failed for {image_name}: {e}")
    finally:
        # Cleanup
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, text=True,
        )

    return repo_files


def batch_extract_repo_files(
    input_path: str,
    output_path: str,
    max_files: int = 300,
    max_file_size: int = 50_000,
    limit: Optional[int] = None,
) -> List[CodingQAInstance]:
    """Extract full repo files from Docker for all CodingQA instances.

    Reads instances from input_path, extracts Python files from Docker images,
    populates repo_files, and writes to output_path. Caches by image_name
    so identical repos are only extracted once.

    Args:
        input_path: Path to coding_qa_instances.jsonl (repo_files may be empty)
        output_path: Path to write updated instances
        max_files: Max Python files to extract per repo
        max_file_size: Max file size in bytes to include
        limit: Max instances to process

    Returns:
        List of updated CodingQAInstance objects
    """
    instances = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(CodingQAInstance(**json.loads(line)))
            if limit and len(instances) >= limit:
                break

    print(f"Loaded {len(instances)} instances for repo extraction")

    # Cache: image_name -> repo_files dict
    cache: Dict[str, Dict[str, str]] = {}
    updated = []

    for i, inst in enumerate(instances):
        if inst.repo_files:
            print(f"  [{i+1}/{len(instances)}] {inst.instance_id}: already has {len(inst.repo_files)} repo files, skipping")
            updated.append(inst)
            continue

        image = inst.image_name
        if not image:
            print(f"  [{i+1}/{len(instances)}] {inst.instance_id}: no image_name, skipping")
            updated.append(inst)
            continue

        if image in cache:
            repo_files = cache[image]
            print(f"  [{i+1}/{len(instances)}] {inst.instance_id}: cached ({len(repo_files)} files)")
        else:
            print(f"  [{i+1}/{len(instances)}] {inst.instance_id}: extracting from {image}...")
            repo_files = _extract_full_repo_files(image, max_files, max_file_size)
            cache[image] = repo_files
            print(f"    Extracted {len(repo_files)} files")

        updated.append(inst.model_copy(update={"repo_files": repo_files}))

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for inst in updated:
            f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    n_with_repo = sum(1 for inst in updated if inst.repo_files)
    avg_files = sum(len(inst.repo_files) for inst in updated) / max(1, n_with_repo)
    print(f"\nDone: {n_with_repo}/{len(updated)} instances have repo files ({avg_files:.0f} avg files)")
    print(f"Output: {out}")

    return updated


# ---------------------------------------------------------------------------
# Token budget expansion for CodingQA instances
# ---------------------------------------------------------------------------

def _current_qa_tokens(instance: CodingQAInstance) -> int:
    """Count total tokens in a CodingQA instance."""
    total = count_tokens(instance.task_prompt)
    for content in instance.memory_files.values():
        total += count_tokens(content)
    total += count_tokens(instance.repo_context or "")
    for fp, content in instance.repo_files.items():
        total += count_tokens(content)
    for f in instance.grounding_facts:
        total += count_tokens(f.content)
    for d in instance.distractors:
        total += count_tokens(d.content)
    return total


def _select_repo_files_for_budget(
    repo_files: Dict[str, str],
    referenced_files: List[str],
    token_budget: int,
) -> Dict[str, str]:
    """Select repo files to include within a token budget.

    Priority order:
    1. Files referenced in the patch (most relevant)
    2. Files in same directories as patch files
    3. Remaining files, sorted by path depth (shallow first)
    """
    if not repo_files or token_budget <= 0:
        return {}

    selected: Dict[str, str] = {}
    used_tokens = 0

    # Normalize referenced files for matching
    ref_set = set()
    ref_dirs = set()
    for rf in (referenced_files or []):
        norm = rf.lstrip("/").replace("/testbed/", "")
        ref_set.add(norm)
        parts = norm.rsplit("/", 1)
        if len(parts) > 1:
            ref_dirs.add(parts[0])

    # Priority 1: Referenced (patch) files
    for fpath in sorted(repo_files.keys()):
        if fpath in ref_set and used_tokens < token_budget:
            tok = count_tokens(repo_files[fpath])
            if used_tokens + tok <= token_budget:
                selected[fpath] = repo_files[fpath]
                used_tokens += tok

    # Priority 2: Same-directory files
    for fpath in sorted(repo_files.keys()):
        if fpath in selected:
            continue
        fdir = fpath.rsplit("/", 1)[0] if "/" in fpath else ""
        if fdir in ref_dirs and used_tokens < token_budget:
            tok = count_tokens(repo_files[fpath])
            if used_tokens + tok <= token_budget:
                selected[fpath] = repo_files[fpath]
                used_tokens += tok

    # Priority 3: Remaining files (shallow first)
    remaining = [f for f in repo_files if f not in selected]
    remaining.sort(key=lambda p: p.count("/"))
    for fpath in remaining:
        if used_tokens >= token_budget:
            break
        tok = count_tokens(repo_files[fpath])
        if used_tokens + tok <= token_budget:
            selected[fpath] = repo_files[fpath]
            used_tokens += tok

    return selected


def expand_qa_instance(
    instance: CodingQAInstance,
    target_tokens: int,
    worker_client: LLMClient,
    filler_mode: str = "local",
    distractor_pool: Optional["DistractorPool"] = None,
    seed: Optional[int] = None,
) -> CodingQAInstance:
    """Scale a CodingQA instance to target token count.

    Strategy (in priority order):
    1. Base content (memory_files + grounding_facts + task_prompt + repo_skeleton)
    2. Repo files — include progressively more files until budget approaches
       (high-quality real code, not filler)
    3. Adversarial QA distractors (targeted noise)
    4. Generic adversarial distractors (reuse expand.py pattern)
    5. Filler — either LLM prose (legacy) or repo-file excerpts (default).

    Args:
        filler_mode: "local" samples real source from repo_files (no LLM, fast),
            "llm" calls the model for prose (legacy behaviour), "hybrid" tries
            local first and falls back to LLM if the pool is exhausted.
        distractor_pool: Optional pre-built cross-instance pool. When provided,
            the training phase/3 sample from it before falling back to per-instance LLM
            generation, cutting LLM calls roughly 10-30x.
        seed: Optional RNG seed for deterministic filler sampling.

    The grounding facts and QA pairs stay the same — only the noise floor rises.
    """
    from ..expand import (
        _generate_more_distractors,
        _expand_file_prose,
        _expand_file_prose_local,
        FILLER_MODES,
    )
    from ...utils.expansion import inject_distractors_into_files as _inject_distractors_into_files

    if filler_mode not in FILLER_MODES:
        raise ValueError(f"filler_mode must be one of {FILLER_MODES}, got {filler_mode!r}")

    current = _current_qa_tokens(instance)
    if current >= target_tokens:
        return instance

    rng = random.Random(seed if seed is not None else hash(instance.instance_id) & 0xFFFFFFFF)

    memory_files = dict(instance.memory_files)
    all_distractors = list(instance.distractors)
    repo_files = dict(instance.repo_files)

    # the training phase: Include more repo files (high-quality context padding)
    if instance.repo_files:
        # Budget for repo files: up to 70% of the deficit
        deficit = target_tokens - current
        repo_budget = int(deficit * 0.7)
        repo_files = _select_repo_files_for_budget(
            instance.repo_files, instance.referenced_files, repo_budget,
        )
        # Recount with selected repo files
        current = count_tokens(instance.task_prompt)
        for content in memory_files.values():
            current += count_tokens(content)
        current += count_tokens(instance.repo_context or "")
        for content in repo_files.values():
            current += count_tokens(content)

    # the training phase: Generate more adversarial QA distractors
    deficit = target_tokens - current
    extra_per_q = max(1, min(5, deficit // (50 * max(1, len(instance.qa_pairs)))))
    if deficit > 1000 and instance.qa_pairs:
        new_qa_distractors: List[DistractorFact] = []
        if distractor_pool is not None:
            need = extra_per_q * max(1, len(instance.qa_pairs))
            new_qa_distractors = distractor_pool.sample(
                kind="qa",
                count=need,
                exclude_ids=instance.instance_id,
                rng=rng,
            )
        if not new_qa_distractors:
            new_qa_distractors = _generate_qa_distractors(
                instance.qa_pairs,
                instance.grounding_facts,
                worker_client,
                num_per_question=extra_per_q,
            )
        if new_qa_distractors:
            memory_files = _inject_distractors_into_files(memory_files, new_qa_distractors)
            all_distractors.extend(new_qa_distractors)

    # Recount
    current = count_tokens(instance.task_prompt)
    for content in memory_files.values():
        current += count_tokens(content)
    current += count_tokens(instance.repo_context or "")
    for content in repo_files.values():
        current += count_tokens(content)

    # the training phase: Generic adversarial distractors via expand.py
    deficit = target_tokens - current
    if deficit > 2000:
        max_new = min(deficit // 50, 50)
        if max_new >= 3:
            new_generic: List[DistractorFact] = []
            if distractor_pool is not None:
                new_generic = distractor_pool.sample(
                    kind="generic",
                    count=max_new,
                    exclude_ids=instance.instance_id,
                    rng=rng,
                )
            if not new_generic:
                temp_instance = MemGymInstance(
                    instance_id=instance.instance_id,
                    base_instance=instance.base_instance,
                    task_prompt=instance.task_prompt,
                    memory_files=memory_files,
                    grounding_facts=instance.grounding_facts,
                    distractors=all_distractors,
                    patch=instance.patch,
                    image_name=instance.image_name,
                )
                new_generic = _generate_more_distractors(temp_instance, worker_client, max_new)
            if new_generic:
                memory_files = _inject_distractors_into_files(memory_files, new_generic)
                all_distractors.extend(new_generic)

    # Recount
    current = count_tokens(instance.task_prompt)
    for content in memory_files.values():
        current += count_tokens(content)
    current += count_tokens(instance.repo_context or "")
    for content in repo_files.values():
        current += count_tokens(content)

    # the training phase: Pad memory files to hit the budget. Default uses real source
    # excerpts (no LLM); legacy LLM mode is opt-in via filler_mode="llm".
    deficit = target_tokens - current
    if deficit > 500 and memory_files:
        n_files = len(memory_files)
        extra_per_file = deficit // n_files
        expanded_files: Dict[str, str] = {}

        if filler_mode in ("local", "hybrid"):
            # Build a pool of (path, source) excerpts from this instance's
            # repo_files, excluding files we already injected as repo context
            # (the training phase) and the actual referenced files (signal-bearing).
            excluded = set(repo_files.keys()) | set(instance.referenced_files or [])
            filler_pool = [
                (p, s) for p, s in instance.repo_files.items() if p not in excluded
            ]
            for fn, content in memory_files.items():
                expanded_files[fn] = _expand_file_prose_local(
                    content, extra_per_file, filler_pool, rng,
                )

            # Hybrid: if we still fell short (e.g. tiny repo, large target), top
            # up with one LLM round per file using the remaining deficit.
            if filler_mode == "hybrid":
                memory_files = expanded_files
                running = sum(count_tokens(c) for c in memory_files.values())
                running += count_tokens(instance.task_prompt)
                running += count_tokens(instance.repo_context or "")
                running += sum(count_tokens(c) for c in repo_files.values())
                short = target_tokens - running
                if short > 1000:
                    extra_words = short // n_files
                    expanded_files = {
                        fn: _expand_file_prose(c, worker_client, extra_words)
                        for fn, c in memory_files.items()
                    }
        else:  # filler_mode == "llm"
            for fn, content in memory_files.items():
                expanded_files[fn] = _expand_file_prose(
                    content, worker_client, extra_per_file
                )
        memory_files = expanded_files

    return instance.model_copy(
        update={
            "memory_files": memory_files,
            "repo_files": repo_files,
            "distractors": all_distractors,
            "target_tokens": target_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_instance(
    instance: MemGymInstance,
    client: LLMClient,
    output_dir: Optional[Path] = None,
    extract_repo: bool = True,
) -> Optional[CodingQAInstance]:
    """Convert one MemGymInstance to CodingQAInstance (v2).

    v2 flow:
    1. Cluster facts by hunk independence (union-find)
    2. Generate one question per cluster (variable 1-N fact linking)
    3. Generate gold answers
    4. Generate adversarial QA-specific distractors
    5. Extract repo files (if Docker available)
    6. Quality gate
    """
    facts = [GroundingFact(**f) if isinstance(f, dict) else f
             for f in instance.grounding_facts]
    fact_map = {f.id: f for f in facts}

    # Step 1: Cluster facts
    clusters = _cluster_facts(facts)
    if not clusters:
        return None

    # Step 2: Generate one question per cluster
    existing_questions: List[str] = []
    qa_dicts: List[Dict] = []

    for cluster in clusters:
        q_dict = _generate_cluster_question(
            cluster, instance.task_prompt, existing_questions, client,
        )
        if q_dict:
            qa_dicts.append(q_dict)
            existing_questions.append(q_dict["question"])

    # Step 3: Build QAPair objects with gold answers
    qa_pairs: List[QAPair] = []
    for i, q_dict in enumerate(qa_dicts):
        linked_ids = q_dict.get("linked_fact_ids", [])
        linked_facts = [fact_map[fid] for fid in linked_ids if fid in fact_map]

        gold_answer = _generate_gold_answer(
            question=q_dict["question"],
            linked_facts=linked_facts,
            repo_context=instance.repo_context,
            client=client,
        )
        if not gold_answer:
            continue

        requires_memory = any(not f.discoverable for f in linked_facts)
        difficulty = _classify_difficulty(linked_facts)
        signal_tokens = sum(count_tokens(f.content) for f in linked_facts)

        qa_pairs.append(QAPair(
            qa_id=f"Q{i+1}",
            question=q_dict["question"],
            gold_answer=gold_answer,
            linked_fact_ids=linked_ids,
            category=q_dict.get("category", ""),
            requires_memory=requires_memory,
            difficulty=difficulty,
            num_linked_facts=len(linked_facts),
            signal_tokens=signal_tokens,
            requires_synthesis=len(linked_facts) >= 2,
        ))

    # Step 4: Generate adversarial distractors
    qa_distractors = _generate_qa_distractors(qa_pairs, facts, client)

    # Combine with original distractors
    original_distractors = [
        DistractorFact(**d) if isinstance(d, dict) else d
        for d in instance.distractors
    ]
    all_distractors = original_distractors + qa_distractors

    # Inject QA distractors into memory files
    memory_files = dict(instance.memory_files)
    if qa_distractors:
        from ...utils.expansion import inject_distractors_into_files
        memory_files = inject_distractors_into_files(memory_files, qa_distractors)

    # Step 5: Extract repo files
    repo_files: Dict[str, str] = {}
    if extract_repo and output_dir:
        repo_files = _extract_repo_files(instance, output_dir)

    # Step 6: Quality gate
    if len(qa_pairs) < 3:
        return None
    if not any(q.requires_memory for q in qa_pairs):
        return None

    return CodingQAInstance(
        instance_id=f"coding_qa__{instance.base_instance}",
        base_instance=instance.base_instance if instance.base_instance else instance.instance_id,
        task_prompt=instance.task_prompt,
        memory_files=memory_files,
        repo_context=instance.repo_context,
        repo_files=repo_files,
        image_name=instance.image_name,
        referenced_files=instance.referenced_files,
        qa_pairs=qa_pairs,
        grounding_facts=facts,
        distractors=all_distractors,
        patch=instance.patch,
        num_critical_facts=sum(1 for f in facts if f.relevance == "critical"),
        num_external_facts=sum(1 for f in facts if not f.discoverable),
        difficulty_preset=instance.difficulty_preset,
    )


def convert_all(
    input_path: str,
    output_path: str,
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    limit: Optional[int] = None,
    num_workers: int = 4,
    target_tokens: Optional[int] = None,
    extract_repo: bool = True,
    show_progress: bool = True,
    checkpoint_path: Optional[str] = None,
) -> List[CodingQAInstance]:
    """Convert coding_synthetic instances to Coding QA format (v2).

    Args:
        input_path: Path to coding_synthetic_instances.jsonl
        output_path: Path to write coding_qa_instances.jsonl
        model: LLM for question/answer/distractor generation
        limit: Max instances to convert
        num_workers: Parallel workers
        target_tokens: Target token budget per instance (None=no expansion)
        extract_repo: Whether to extract repo files from Docker
        show_progress: Show progress bar
        checkpoint_path: Path to a JSONL checkpoint file. Each successfully
            converted instance is appended immediately, so a killed run can be
            resumed by re-invoking with the same checkpoint. Defaults to
            ``output_path + ".checkpoint.jsonl"``. Pass empty string to
            disable checkpointing entirely (legacy in-memory mode).
    """
    instances = []
    with open(input_path) as f:
        for line in f:
            instances.append(MemGymInstance(**json.loads(line)))
            if limit and len(instances) >= limit:
                break

    print(f"Loaded {len(instances)} coding_synthetic instances")

    client = LLMClient(model=model)
    out_dir = Path(output_path).parent

    def _process(inst: MemGymInstance) -> Optional[CodingQAInstance]:
        return convert_instance(inst, client, output_dir=out_dir, extract_repo=extract_repo)

    # Default checkpoint path lives next to the final output. Setting
    # checkpoint_path="" explicitly opts out (legacy behavior — no resume).
    if checkpoint_path is None:
        checkpoint_path = str(output_path) + ".checkpoint.jsonl"

    ckpt: Optional[Path] = Path(checkpoint_path) if checkpoint_path else None
    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if ckpt.exists():
            existing = sum(1 for _ in open(ckpt))
            print(f"Found existing checkpoint at {ckpt} ({existing} entries)")
        else:
            print(f"Checkpoint will be written to {ckpt}")

    # item_id_fn must produce the SAME key that gets written to the checkpoint
    # so parallel_map's resume logic can match input items against done IDs.
    # Output instance_id format is "coding_qa__<base_instance>" (line ~825).
    def _item_id(inst: MemGymInstance) -> str:
        base = inst.base_instance or inst.instance_id
        return f"coding_qa__{base}"

    def _serialize(inst: CodingQAInstance) -> str:
        return json.dumps(inst.to_jsonl_dict())

    new_results = parallel_map(
        _process, instances,
        num_workers=num_workers,
        desc="Converting to QA (v2)",
        show_progress=show_progress,
        filter_none=True,
        item_id_fn=_item_id,
        output_path=ckpt,
        serialize_fn=_serialize,
    )

    # parallel_map only returns NEW results in resume mode. Reload the full set
    # from the checkpoint so downstream stages (expansion, summary, final write)
    # see every instance — both freshly converted AND previously checkpointed.
    if ckpt is not None and ckpt.exists():
        results: List[CodingQAInstance] = []
        with open(ckpt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(CodingQAInstance(**json.loads(line)))
        print(f"Loaded {len(results)} instances from checkpoint ({len(new_results)} new this run)")
    else:
        results = new_results

    # Optional: expand to target token budget
    if target_tokens and target_tokens > 0:
        print(f"\nExpanding instances to {target_tokens} tokens...")
        expanded = []
        for inst in results:
            current = _current_qa_tokens(inst)
            if current < target_tokens:
                expanded.append(expand_qa_instance(inst, target_tokens, client))
            else:
                expanded.append(inst)
        results = expanded

    # Recompute difficulty labels based on final context length
    for inst in results:
        total_tokens = _current_qa_tokens(inst)
        _recompute_difficulty_for_context(inst.qa_pairs, total_tokens)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for inst in results:
            f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    # Summary
    total_qs = sum(len(r.qa_pairs) for r in results)
    memory_qs = sum(
        sum(1 for q in r.qa_pairs if q.requires_memory) for r in results
    )
    hard_qs = sum(
        sum(1 for q in r.qa_pairs if q.difficulty == "hard") for r in results
    )
    med_qs = sum(
        sum(1 for q in r.qa_pairs if q.difficulty == "medium") for r in results
    )
    easy_qs = sum(
        sum(1 for q in r.qa_pairs if q.difficulty == "easy") for r in results
    )
    qa_distractors = sum(
        sum(1 for d in r.distractors if d.source == "adversarial_qa") for r in results
    )

    print(f"\nConverted {len(results)}/{len(instances)} instances")
    print(f"Total QA pairs: {total_qs} ({total_qs / max(1, len(results)):.1f} avg)")
    print(f"  hard: {hard_qs}, medium: {med_qs}, easy: {easy_qs}")
    print(f"Memory-required: {memory_qs} ({memory_qs / max(1, total_qs) * 100:.0f}%)")
    print(f"QA distractors: {qa_distractors}")
    if target_tokens:
        avg_tokens = sum(_current_qa_tokens(r) for r in results) / max(1, len(results))
        print(f"Avg tokens/instance: {avg_tokens:.0f} (target: {target_tokens})")
    print(f"Output: {output_file}")

    return results
