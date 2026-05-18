"""
Pipeline orchestrator for Coding-Synthetic generation v2.

Coordinates the full pipeline:
  Mine → Split → Craft+Verify (integrated per-instance loop)
with checkpointing, progress tracking, and optional adaptive feedback.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from .types.schemas import MemGymInstance, FilteredInstance
from .stages.filter import filter_instances, load_filtered_instances
from .stages.extract import (
    split_all_instances, load_extractions,
    extract_all_repo_contexts, load_repo_contexts,
)
from .stages.generate import craft_all_instances, craft_and_verify_all
from .stages.expand import expand_all_instances
from .stages.harden import apply_difficulty_all
from .stages.prescreen import prescreen_all
from .stages.validate import verify_all_instances
from .llm.client import LLMClient, VerifierClient


@dataclass
class PipelineConfig:
    """Configuration for the v2 pipeline."""
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Data limits
    limit: Optional[int] = None

    # Model settings (Worker/Verifier split)
    worker_model: str = "gpt-5-mini"
    verifier_model: str = "claude-opus-4-20250514"
    api_key: Optional[str] = None

    # Filter criteria
    min_lines_changed: int = 5
    max_lines_changed: int = 500
    min_fail_to_pass: int = 1
    min_hunks: int = 1
    min_problem_length: int = 100

    # Difficulty settings
    difficulty_preset: str = "medium"

    # Verification settings
    skip_verification: bool = False
    docker_available: bool = False

    # Adaptive feedback
    adaptive_rounds: int = 1

    # Checkpointing
    resume: bool = True

    # Distractor count
    num_distractors: int = 5

    # Target context tokens per instance
    target_context_tokens: int = 12000

    # Token expansion (post-craft)
    expand_target_tokens: Optional[int] = None  # None = no expansion

    # Exclude already-generated instance IDs (for multi-batch runs)
    exclude_file: Optional[str] = None  # Path to JSONL with instance_ids to skip

    # Parallelism
    num_workers: int = 4


def run_pipeline(config: Optional[PipelineConfig] = None) -> List[MemGymInstance]:
    """
    Run the full Coding-Synthetic v2 generation pipeline.

    Flow: Mine → Split → Craft → Difficulty → Verify (with optional adaptive feedback)

    Args:
        config: PipelineConfig object (uses defaults if None)

    Returns:
        List of generated MemGymInstance objects
    """
    if config is None:
        config = PipelineConfig()

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define checkpoint paths
    filtered_path = output_dir / "filtered.jsonl"
    extracted_path = output_dir / "extracted.jsonl"
    crafted_path = output_dir / "crafted.jsonl"
    difficulty_path = output_dir / "difficulty.jsonl"
    final_path = output_dir / "coding_synthetic_instances.jsonl"

    # Stage count: Mine, Repo-context, Split, then either Craft+Verify (1) or Craft+Difficulty (2)
    total_stages = 4  # Mine, Repo-context, Split, Craft+Verify
    if config.skip_verification:
        total_stages = 5  # Mine, Repo-context, Split, Craft, Difficulty

    # Initialize LLM clients
    worker_client = LLMClient(
        model=config.worker_model,
        api_key=config.api_key,
    )
    verifier_client = VerifierClient(
        model=config.verifier_model,
        api_key=config.api_key,
    )

    print("=" * 60)
    print("Coding-Synthetic Pipeline v2")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Worker model: {config.worker_model}")
    print(f"Verifier model: {config.verifier_model}")
    print(f"Difficulty preset: {config.difficulty_preset}")
    if config.limit:
        print(f"Limit: {config.limit} instances")
    if config.docker_available:
        print("Docker: ENABLED")
    print(f"Workers: {config.num_workers}")
    print()

    # =========================================================================
    # Stage 1: Mine (filter.py)
    # =========================================================================
    stage = 1
    print(f"[{stage}/{total_stages}] Stage 1 — Mine: Filtering SWE-smith instances...")
    if config.resume and filtered_path.exists():
        print(f"  Loading from checkpoint: {filtered_path}")
        filtered = load_filtered_instances(filtered_path)
        print(f"  Loaded {len(filtered)} filtered instances")
    else:
        if filtered_path.exists():
            filtered_path.unlink()

        filtered = filter_instances(
            limit=config.limit,
            min_lines_changed=config.min_lines_changed,
            max_lines_changed=config.max_lines_changed,
            min_fail_to_pass=config.min_fail_to_pass,
            min_hunks=config.min_hunks,
            min_problem_length=config.min_problem_length,
            output_path=filtered_path,
            show_progress=True,
        )
    print()

    # Exclude already-generated instance IDs (for multi-batch runs)
    if config.exclude_file:
        exclude_ids = set()
        exclude_path = Path(config.exclude_file)
        if exclude_path.exists():
            with open(exclude_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        iid = data.get("instance_id", "")
                        # Normalize: store both with and without prefix
                        bare = iid.replace("memgym__", "") if iid.startswith("memgym__") else iid
                        exclude_ids.add(iid)
                        exclude_ids.add(bare)
                    except Exception:
                        continue
            before = len(filtered)
            filtered = [f for f in filtered if f.instance_id not in exclude_ids]
            print(f"  Excluded {before - len(filtered)} already-generated instances "
                  f"({len(filtered)} remaining)")
        else:
            print(f"  Warning: exclude file not found: {exclude_path}")

    if not filtered:
        print("No instances passed filtering. Exiting.")
        return []

    # =========================================================================
    # Stage 1.5: Extract repo context from Docker
    # =========================================================================
    stage += 1
    print(f"[{stage}/{total_stages}] Extracting repo context from Docker...")
    repo_contexts: Dict[str, Dict[str, str]] = {}
    cached = load_repo_contexts(output_dir)
    if config.resume and cached:
        print(f"  Loaded cached repo context for {len(cached)} instances")
        repo_contexts = cached
    else:
        # For split stage, we need a preliminary extraction using patch files
        # The split_all_instances function handles the extractions list internally
        # First pass: use patch files as referenced_files for Docker extraction
        preliminary_extractions = [
            {
                "instance_id": inst.instance_id,
                "referenced_files": [],
            }
            for inst in filtered
        ]
        repo_contexts = extract_all_repo_contexts(
            instances=filtered,
            extractions=preliminary_extractions,
            output_dir=output_dir,
            show_progress=True,
        )
    print()

    # =========================================================================
    # Stage 1.75: Pre-screen (reject instances solvable without memory)
    # =========================================================================
    prescreen_path = output_dir / "prescreen.json"
    if config.resume and prescreen_path.exists():
        print(f"  Loading pre-screen results from checkpoint: {prescreen_path}")
        with open(prescreen_path) as f:
            prescreen_results = json.load(f)
        # Filter to kept instances
        rejected_ids = {
            r["instance_id"] for r in prescreen_results if r["too_easy"]
        }
        filtered = [i for i in filtered if i.instance_id not in rejected_ids]
        print(f"  After pre-screen: {len(filtered)} instances kept")
    else:
        filtered, prescreen_results = prescreen_all(
            instances=filtered,
            repo_contexts=repo_contexts,
            worker_model=config.worker_model,
            show_progress=True,
            num_workers=config.num_workers,
        )
        with open(prescreen_path, "w") as f:
            json.dump(prescreen_results, f, indent=2)
    print()

    if not filtered:
        print("All instances rejected by pre-screen. Exiting.")
        return []

    # =========================================================================
    # Stage 2: Split (extract.py)
    # =========================================================================
    stage += 1
    print(f"[{stage}/{total_stages}] Stage 2 — Split: Extracting grounding facts...")
    if config.resume and extracted_path.exists():
        existing_extractions = load_extractions(extracted_path)
        if len(existing_extractions) >= len(filtered):
            print(f"  Loading from checkpoint: {extracted_path} ({len(existing_extractions)} extractions)")
            extractions = existing_extractions
        else:
            print(f"  Partial checkpoint: {len(existing_extractions)}/{len(filtered)}, resuming...")
            extractions = split_all_instances(
                instances=filtered,
                repo_contexts=repo_contexts,
                worker_client=worker_client,
                verifier_client=verifier_client,
                output_path=extracted_path,
                show_progress=True,
                num_workers=config.num_workers,
            )
            # Reload to include both old and new results
            extractions = load_extractions(extracted_path)
    else:
        if extracted_path.exists():
            extracted_path.unlink()

        extractions = split_all_instances(
            instances=filtered,
            repo_contexts=repo_contexts,
            worker_client=worker_client,
            verifier_client=verifier_client,
            output_path=extracted_path,
            show_progress=True,
            num_workers=config.num_workers,
        )
    print()

    # =========================================================================
    # Stage 3: Craft + Difficulty + Verify (integrated per-instance loop)
    # =========================================================================
    stage += 1
    verification_results = []

    if config.skip_verification:
        # --- Batch path: Craft → Difficulty (no verification) ---
        print(f"[{stage}/{total_stages}] Stage 3 — Craft: Generating task prompts & memory files...")
        if config.resume and crafted_path.exists():
            existing_crafted = []
            with open(crafted_path) as f:
                for line in f:
                    data = json.loads(line)
                    existing_crafted.append(MemGymInstance(**data))
            # Check if complete (crafted count close to extraction count)
            expected = sum(1 for e in extractions if e.get("quality_gate_passed", False))
            if len(existing_crafted) >= expected * 0.95:
                print(f"  Loading from checkpoint: {crafted_path} ({len(existing_crafted)} crafted)")
                crafted = existing_crafted
            else:
                print(f"  Partial checkpoint: {len(existing_crafted)}/{expected}, resuming...")
                crafted = craft_all_instances(
                    instances=filtered,
                    extractions=extractions,
                    repo_contexts=repo_contexts,
                    worker_client=worker_client,
                    output_path=crafted_path,
                    show_progress=True,
                    num_distractors=config.num_distractors,
                    target_context_tokens=config.target_context_tokens,
                    num_workers=config.num_workers,
                )
                # Reload all results from file
                crafted = []
                with open(crafted_path) as f:
                    for line in f:
                        data = json.loads(line)
                        crafted.append(MemGymInstance(**data))
        else:
            if crafted_path.exists():
                crafted_path.unlink()

            crafted = craft_all_instances(
                instances=filtered,
                extractions=extractions,
                repo_contexts=repo_contexts,
                worker_client=worker_client,
                output_path=crafted_path,
                show_progress=True,
                num_distractors=config.num_distractors,
                target_context_tokens=config.target_context_tokens,
                num_workers=config.num_workers,
            )
        print()

        if not crafted:
            print("No instances crafted. Exiting.")
            return []

        stage += 1
        print(f"[{stage}/{total_stages}] Stage 4 — Difficulty: Applying transforms (preset={config.difficulty_preset})...")
        if config.resume and difficulty_path.exists():
            print(f"  Loading from checkpoint: {difficulty_path}")
            difficult = []
            with open(difficulty_path) as f:
                for line in f:
                    data = json.loads(line)
                    difficult.append(MemGymInstance(**data))
            print(f"  Loaded {len(difficult)} instances with difficulty applied")
        else:
            if difficulty_path.exists():
                difficulty_path.unlink()

            difficult = apply_difficulty_all(
                instances=crafted,
                preset=config.difficulty_preset,
                worker_client=worker_client,
                repo_contexts=repo_contexts,
                show_progress=True,
                num_workers=config.num_workers,
            )

            with open(difficulty_path, "w") as f:
                for instance in difficult:
                    f.write(json.dumps(instance.to_jsonl_dict()) + "\n")
        print()

    else:
        # --- Integrated path: Craft + Difficulty + Verify per instance ---
        print(
            f"[{stage}/{total_stages}] Stage 3 — Craft+Verify: "
            f"Generating, applying difficulty ({config.difficulty_preset}), "
            f"and verifying per instance..."
        )

        if config.resume and final_path.exists():
            print(f"  Loading from checkpoint: {final_path}")
            difficult = []
            with open(final_path) as f:
                for line in f:
                    data = json.loads(line)
                    difficult.append(MemGymInstance(**data))
            print(f"  Loaded {len(difficult)} final instances")
        else:
            if crafted_path.exists():
                crafted_path.unlink()

            difficult, verification_results = craft_and_verify_all(
                instances=filtered,
                extractions=extractions,
                repo_contexts=repo_contexts,
                worker_client=worker_client,
                verifier_model=config.verifier_model,
                docker_available=config.docker_available,
                difficulty_preset=config.difficulty_preset,
                adaptive_rounds=config.adaptive_rounds,
                output_path=crafted_path,
                show_progress=True,
                num_distractors=config.num_distractors,
                target_context_tokens=config.target_context_tokens,
                num_workers=config.num_workers,
            )

            # Save verification metadata
            if verification_results:
                meta_path = output_dir / "verification_metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(verification_results, f, indent=2, default=str)
                print(f"  Verification metadata saved to {meta_path}")
        print()

        if not difficult:
            print("No instances crafted/verified. Exiting.")
            return []

    # =========================================================================
    # Optional: Token expansion
    # =========================================================================
    if config.expand_target_tokens and config.expand_target_tokens > 0:
        expanded_path = output_dir / "expanded.jsonl"
        if config.resume and expanded_path.exists():
            print(f"  Loading expanded instances from checkpoint: {expanded_path}")
            difficult = []
            with open(expanded_path) as f:
                for line in f:
                    data = json.loads(line)
                    difficult.append(MemGymInstance(**data))
            print(f"  Loaded {len(difficult)} expanded instances")
        else:
            print(f"Expanding instances to {config.expand_target_tokens} tokens...")
            difficult = expand_all_instances(
                instances=difficult,
                target_tokens=config.expand_target_tokens,
                worker_client=worker_client,
                output_path=expanded_path,
                num_workers=config.num_workers,
                show_progress=True,
            )
        print()

    # =========================================================================
    # Save final output
    # =========================================================================
    print(f"Saving {len(difficult)} instances to {final_path}")
    with open(final_path, "w") as f:
        for instance in difficult:
            f.write(json.dumps(instance.to_jsonl_dict()) + "\n")

    # Summary
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total instances: {len(difficult)}")
    print(f"Output file: {final_path}")

    # Difficulty distribution
    presets = {}
    for g in difficult:
        presets[g.difficulty_preset] = presets.get(g.difficulty_preset, 0) + 1
    print(f"Difficulty presets: {presets}")

    # Transform distribution
    transforms = {}
    for g in difficult:
        for t in g.transforms_applied:
            transforms[t] = transforms.get(t, 0) + 1
    if transforms:
        print(f"Transforms applied: {transforms}")

    # Quality stats
    has_gold_fix = sum(1 for g in difficult if g.gold_fix)
    avg_facts = (
        sum(len(g.grounding_facts) for g in difficult) / max(len(difficult), 1)
    )
    avg_distractors = (
        sum(len(g.distractors) for g in difficult) / max(len(difficult), 1)
    )
    avg_files = (
        sum(len(g.memory_files) for g in difficult) / max(len(difficult), 1)
    )
    print(f"Instances with gold_fix: {has_gold_fix}/{len(difficult)}")
    print(f"Average grounding facts: {avg_facts:.1f}")
    print(f"Average distractors: {avg_distractors:.1f}")
    print(f"Average memory files: {avg_files:.1f}")

    # Token stats
    instances_with_tokens = [g for g in difficult if g.context_tokens]
    if instances_with_tokens:
        n = len(instances_with_tokens)
        avg_task_tok = sum(g.context_tokens.task_prompt for g in instances_with_tokens) / n
        avg_mem_tok = sum(g.context_tokens.memory_files for g in instances_with_tokens) / n
        avg_repo_tok = sum(g.context_tokens.repo_context for g in instances_with_tokens) / n
        avg_facts_tok = sum(g.context_tokens.grounding_facts for g in instances_with_tokens) / n
        avg_total_tok = sum(g.context_tokens.total for g in instances_with_tokens) / n
        print(f"\nToken counts (avg over {n} instances):")
        print(f"  Task prompt:     {avg_task_tok:,.0f}")
        print(f"  Memory files:    {avg_mem_tok:,.0f}")
        print(f"  Repo context:    {avg_repo_tok:,.0f}")
        print(f"  Grounding facts: {avg_facts_tok:,.0f}")
        print(f"  Total:           {avg_total_tok:,.0f}")

    # Verification stats
    if verification_results:
        accepted = sum(1 for v in verification_results if v.get("accepted"))
        by_docker = sum(1 for v in verification_results if v.get("method") == "docker")
        by_llm = sum(1 for v in verification_results if v.get("method") == "llm_judge")
        avg_a = sum(v.get("cond_a_llm_score", 0) for v in verification_results) / max(len(verification_results), 1)
        avg_b = sum(v.get("cond_b_llm_score", 0) for v in verification_results) / max(len(verification_results), 1)
        print(f"\nAblation verification:")
        print(f"  Accepted: {accepted}/{len(verification_results)}")
        print(f"  By Docker: {by_docker}, By LLM judge: {by_llm}")
        print(f"  Avg LLM scores: with_memory={avg_a:.2f}, without={avg_b:.2f}, gap={avg_a - avg_b:.2f}")

    return difficult


def run_pipeline_from_args(
    output: str = "./output",
    limit: Optional[int] = None,
    worker_model: str = "gpt-5-mini",
    verifier_model: str = "claude-opus-4-20250514",
    max_lines_changed: int = 500,
    difficulty_preset: str = "medium",
    skip_verification: bool = False,
    docker_available: bool = False,
    adaptive_rounds: int = 1,
    no_resume: bool = False,
    num_distractors: int = 5,
    target_context_tokens: int = 12000,
    num_workers: int = 4,
    expand_target_tokens: Optional[int] = None,
    exclude_file: Optional[str] = None,
) -> List[MemGymInstance]:
    """
    Run pipeline from CLI arguments.

    Args:
        output: Output directory path
        limit: Limit on instances to process
        worker_model: Worker LLM model name (cheap, generative)
        verifier_model: Verifier LLM model name (expensive, judgment)
        max_lines_changed: Max patch size in lines
        difficulty_preset: Difficulty preset (easy/medium/hard)
        skip_verification: Skip ablation verification
        docker_available: Whether Docker is available for test execution
        adaptive_rounds: Number of adaptive difficulty rounds (0 to disable)
        no_resume: Don't resume from checkpoints
        num_distractors: Target number of distractors per instance
        target_context_tokens: Target total context tokens per instance
        num_workers: Number of parallel workers (1 = sequential)
        exclude_file: Path to JSONL with instance_ids to skip (for multi-batch)

    Returns:
        List of generated instances
    """
    config = PipelineConfig(
        output_dir=Path(output),
        limit=limit,
        worker_model=worker_model,
        verifier_model=verifier_model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_lines_changed=max_lines_changed,
        difficulty_preset=difficulty_preset,
        skip_verification=skip_verification,
        docker_available=docker_available,
        adaptive_rounds=adaptive_rounds,
        resume=not no_resume,
        num_distractors=num_distractors,
        target_context_tokens=target_context_tokens,
        num_workers=num_workers,
        expand_target_tokens=expand_target_tokens,
        exclude_file=exclude_file,
    )

    return run_pipeline(config)
