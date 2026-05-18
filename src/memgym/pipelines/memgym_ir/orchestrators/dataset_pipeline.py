"""Pipeline orchestrator for MemGym-IR v2 generation (dataset-based).

Flow: Mine -> Pre-screen -> Split -> Craft+Verify -> Harden
With checkpointing at each stage.
"""

import json
from pathlib import Path
from typing import List, Optional

from ..config import DifficultyConfig, PipelineConfig
from ..types.schemas import FilteredIRInstance, MemGymIRInstance
from ..stages.filter import filter_instances, load_filtered_instances
from ..stages.upgrade import upgrade_all, load_upgraded_instances
from ..stages.prescreen import prescreen_all
from ..stages.extract import extract_all, load_extractions
from ..stages.generate import craft_all
from ..stages.validate import verify_all
from ..stages.harden import harden_all
from ..stages.report import generate_quality_report
from ..llm.client import LLMClient, VerifierClient


def run_pipeline(config: Optional[PipelineConfig] = None) -> List[MemGymIRInstance]:
    """Run the full MemGym-IR v2 pipeline."""
    if config is None:
        config = PipelineConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulty = config.get_difficulty_config()

    # Checkpoint paths
    filtered_path = output_dir / "filtered.jsonl"
    prescreen_path = output_dir / "prescreen.json"
    extracted_path = output_dir / "extracted.jsonl"
    crafted_path = output_dir / "crafted.jsonl"
    final_path = output_dir / "memgym_ir_instances.jsonl"

    # Init LLM clients
    worker_client = LLMClient(model=config.worker_model)
    verifier_client = VerifierClient(model=config.verifier_model)

    print("=" * 60)
    print("MemGym-IR Pipeline v2")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Dataset: {config.dataset}")
    print(f"Worker: {config.worker_model}")
    print(f"Verifier: {config.verifier_model}")
    print(f"Difficulty: {config.difficulty_preset}")
    print(f"Target tokens: {difficulty.target_tokens}")
    print(f"Adversarial distractors: {difficulty.adversarial_count}")
    print(f"Contradiction distractors: {difficulty.contradiction_count}")
    print(f"Red herring turns: {difficulty.red_herring_turns}")
    print(f"Reveal mode: {difficulty.reveal_mode}")
    if difficulty.expand_paragraphs:
        print(f"Paragraph expansion: ON ({difficulty.expansion_target_tokens} tokens/para)")
    if difficulty.overlap_count > 0:
        print(f"Overlapping docs: {difficulty.overlap_count} per supporting para")
    if difficulty.conflicting_sources_count > 0:
        print(f"Conflicting sources: {difficulty.conflicting_sources_count} pairs")
    if difficulty.eviction_mode != "none":
        print(f"Eviction mode: {difficulty.eviction_mode} (window={difficulty.eviction_window_size})")
        print(f"Agent notes: {'ON' if difficulty.allow_agent_notes else 'OFF'} ({difficulty.agent_notes_budget} tokens)")
    if config.upgrade_hops is not None:
        print(f"Hop upgrade: {config.upgrade_hops} target hops (fictionalize + extend)")
    if config.min_memory_facts > 1:
        print(f"Min memory facts: {config.min_memory_facts}")
    print(f"Max concurrent: {config.max_concurrent}")
    if config.limit:
        print(f"Limit: {config.limit}")
    if config.dry_run:
        print("Mode: DRY RUN (no LLM calls)")
    print()

    # Track stats for quality report
    pipeline_stats: dict = {}
    verification_results = None

    # =========================================================================
    # Stage 0: Mine
    # =========================================================================
    print(f"[1/5] Stage 0 -- Mine: Filtering {config.dataset} instances...")
    if config.resume and filtered_path.exists():
        print(f"  Loading from checkpoint: {filtered_path}")
        filtered = load_filtered_instances(filtered_path)
        print(f"  Loaded {len(filtered)} filtered instances")
    else:
        filtered = filter_instances(
            dataset=config.dataset,
            limit=config.limit,
            min_hops=config.min_hops,
            min_supporting=config.min_supporting,
            min_answer_length=config.min_answer_length,
            min_question_length=config.min_question_length,
            output_path=filtered_path,
            worker_client=worker_client,
            dry_run=config.dry_run,
        )
    pipeline_stats["raw_loaded"] = config.limit or "all"
    pipeline_stats["after_filter"] = len(filtered)
    print()

    if not filtered:
        print("No instances passed filtering. Exiting.")
        return []

    # =========================================================================
    # Stage 0.5: Upgrade (optional -- fictionalize + extend hops)
    # =========================================================================
    upgraded_path = output_dir / "upgraded.jsonl"
    if config.upgrade_hops is not None:
        print(f"[1.5/5] Stage 0.5 -- Upgrade: Fictionalizing + extending to {config.upgrade_hops} hops...")
        if config.resume and upgraded_path.exists():
            filtered = load_upgraded_instances(upgraded_path)
        else:
            if upgraded_path.exists():
                upgraded_path.unlink()
            filtered = upgrade_all(
                instances=filtered,
                target_hops=config.upgrade_hops,
                client=worker_client,
                output_path=upgraded_path,
                max_concurrent=config.max_concurrent,
                dry_run=config.dry_run,
            )
        pipeline_stats["after_upgrade"] = len(filtered)
        print()

        if not filtered:
            print("No instances survived upgrade. Exiting.")
            return []

    # =========================================================================
    # Stage 1: Pre-screen
    # =========================================================================
    print("[2/5] Stage 1 -- Pre-screen: Rejecting parametric questions...")
    if config.resume and prescreen_path.exists():
        print(f"  Loading from checkpoint: {prescreen_path}")
        with open(prescreen_path) as f:
            prescreen_results = json.load(f)
        rejected_ids = {
            r["instance_id"] for r in prescreen_results if r.get("rejected")
        }
        filtered = [i for i in filtered if i.instance_id not in rejected_ids]
        print(f"  After pre-screen: {len(filtered)} kept")
    else:
        filtered, prescreen_results = prescreen_all(
            instances=filtered,
            client=worker_client,
            dry_run=config.dry_run,
            max_concurrent=config.max_concurrent,
        )
        with open(prescreen_path, "w") as f:
            json.dump(prescreen_results, f, indent=2)
    pipeline_stats["after_prescreen"] = len(filtered)
    print()

    if not filtered:
        print("All instances rejected. Exiting.")
        return []

    # =========================================================================
    # Stage 2: Split
    # =========================================================================
    print("[3/5] Stage 2 -- Split: Extracting grounding facts...")
    if config.resume and extracted_path.exists():
        extractions = load_extractions(extracted_path)
        print(f"  Loaded {len(extractions)} extractions from checkpoint")
    else:
        if extracted_path.exists():
            extracted_path.unlink()
        extractions = extract_all(
            instances=filtered,
            worker_client=worker_client,
            verifier_client=verifier_client,
            output_path=extracted_path,
            dry_run=config.dry_run,
            max_concurrent=config.max_concurrent,
        )
    pipeline_stats["after_extract"] = len(extractions)
    print()

    # =========================================================================
    # Stage 3: Craft
    # =========================================================================
    print("[4/5] Stage 3 -- Craft: Building search turns + distractors...")
    if config.resume and crafted_path.exists():
        crafted = []
        with open(crafted_path) as f:
            for line in f:
                crafted.append(MemGymIRInstance(**json.loads(line)))
        print(f"  Loaded {len(crafted)} crafted instances from checkpoint")
    else:
        if crafted_path.exists():
            crafted_path.unlink()
        crafted = craft_all(
            instances=filtered,
            extractions=extractions,
            worker_client=worker_client,
            target_tokens=difficulty.target_tokens,
            adversarial_count=difficulty.adversarial_count,
            contradiction_count=difficulty.contradiction_count,
            output_path=crafted_path,
            dry_run=config.dry_run,
            max_concurrent=config.max_concurrent,
            red_herring_count=difficulty.red_herring_turns,
            reveal_mode=difficulty.reveal_mode,
            expand_paragraphs=difficulty.expand_paragraphs,
            expansion_target_tokens=difficulty.expansion_target_tokens,
            overlap_count=difficulty.overlap_count,
            conflicting_sources_count=difficulty.conflicting_sources_count,
            eviction_mode=difficulty.eviction_mode,
            eviction_window_size=difficulty.eviction_window_size,
            allow_agent_notes=difficulty.allow_agent_notes,
            agent_notes_budget=difficulty.agent_notes_budget,
            min_memory_facts=config.min_memory_facts,
        )
    print()

    pipeline_stats["after_craft"] = len(crafted)

    if not crafted:
        print("No instances crafted. Exiting.")
        return []

    # =========================================================================
    # Stage 4: Verify (optional)
    # =========================================================================
    if not config.skip_verification:
        print("[4.5/5] Stage 4 -- Verify: Running ablation tests...")
        verification_results = verify_all(
            instances=crafted,
            client=verifier_client,
            worker_client=worker_client,
            min_a_score=difficulty.min_a_score,
            min_gap_ab=difficulty.min_gap_ab,
            max_c_score=difficulty.max_c_score,
            pre_verify_threshold=difficulty.pre_verify_threshold,
            dry_run=config.dry_run,
            max_concurrent=config.max_concurrent,
            max_docs_a=difficulty.max_docs_a,
        )
        # Filter to accepted only
        accepted_ids = {
            r["instance_id"] for r in verification_results if r.get("accepted")
        }
        crafted = [c for c in crafted if c.instance_id in accepted_ids]
        print(f"  After verification: {len(crafted)} instances")

        pipeline_stats["after_verify"] = len(crafted)

        meta_path = output_dir / "verification_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(verification_results, f, indent=2)
        print()

    # =========================================================================
    # Stage 5: Harden
    # =========================================================================
    print("[5/5] Stage 5 -- Harden: Applying difficulty transforms...")
    final = harden_all(
        instances=crafted,
        preset=config.difficulty_preset,
        difficulty=difficulty,
        client=worker_client,
        dry_run=config.dry_run,
        max_concurrent=config.max_concurrent,
    )
    print()

    pipeline_stats["after_harden"] = len(final)

    # =========================================================================
    # Save output
    # =========================================================================
    print(f"Saving {len(final)} instances to {final_path}")
    with open(final_path, "w") as f:
        for inst in final:
            f.write(json.dumps(inst.to_jsonl_dict()) + "\n")

    # =========================================================================
    # Quality report
    # =========================================================================
    print()
    print("Generating quality report...")
    report = generate_quality_report(
        output_dir=output_dir,
        instances=final,
        pipeline_stats=pipeline_stats,
        verification_results=verification_results,
    )

    # Summary
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total instances: {len(final)}")
    print(f"Output: {final_path}")

    if final:
        ds = report.get("dataset_statistics", {})
        print(f"Avg hops: {ds.get('avg_hops', 'N/A')}")
        print(f"Avg grounding facts: {ds.get('avg_grounding_facts', 'N/A')}")
        print(f"Avg distractors: {ds.get('avg_distractors', 'N/A')}")
        print(f"Avg turns: {ds.get('avg_turns', 'N/A')}")
        print(f"Avg tokens: {ds.get('avg_total_tokens', 'N/A')}")

    return final
