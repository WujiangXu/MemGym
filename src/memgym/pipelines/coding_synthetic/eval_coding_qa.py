"""CLI entry point for Coding QA benchmark.

Two subcommands:
  convert — Transform coding_synthetic instances to QA format
  eval    — Evaluate QA instances with different protocols/strategies

Usage:
    # Convert
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa convert \
        --input results/full_11k/coding_synthetic_instances.jsonl \
        --output results/coding_qa/coding_qa_instances.jsonl \
        --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --limit 100

    # Evaluate
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa eval \
        results/coding_qa/coding_qa_instances.jsonl \
        --protocol evicted --strategy structured --limit 20

    # Compare strategies
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa eval \
        results/coding_qa/coding_qa_instances.jsonl \
        --protocol evicted --compare-strategies --limit 20

    # Evaluate with A-MEM memory system
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa eval \
        results/coding_qa/coding_qa_instances.jsonl \
        --protocol evicted --method amem --method-config '{"retrieve_k": 5}'

    # Evaluate with truncated-input baseline
    python -m memgym.pipelines.coding_synthetic.eval_coding_qa eval \
        results/coding_qa/coding_qa_instances.jsonl \
        --protocol evicted --method truncated --method-config '{"max_chars": 4000}'
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .types.schemas import CodingQAInstance
from .llm.client import LLMClient


def cmd_convert(args):
    """Run conversion: coding_synthetic → coding_qa."""
    from .stages.qa.convert import convert_all, resolve_target_tokens

    target_tokens = resolve_target_tokens(args.target_tokens)

    convert_all(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        limit=args.limit,
        num_workers=args.num_workers,
        target_tokens=target_tokens,
        extract_repo=not args.no_repo,
        show_progress=True,
        checkpoint_path=args.checkpoint,
    )


def cmd_extract_repo(args):
    """Extract full repo files from Docker images into CodingQA instances."""
    from .stages.qa.convert import batch_extract_repo_files

    batch_extract_repo_files(
        input_path=args.input,
        output_path=args.output,
        max_files=args.max_files,
        max_file_size=args.max_file_size,
        limit=args.limit,
    )


def cmd_expand_instances(args):
    """Scale verified instances to one or more target token budgets.

    For each --target-tokens value, writes a separate jsonl. Re-running with the
    same paths resumes from the per-bucket checkpoint via parallel_map's
    item_id_fn handshake.
    """
    from .stages.qa.convert import (
        expand_qa_instance,
        resolve_target_tokens,
        _current_qa_tokens,
    )
    from .stages.qa.distractor_pool import build_pool, load_pool_from_jsonl
    from .stages.expand import FILLER_MODES
    from .utils.parallel import parallel_map
    from .types.schemas import CodingQAInstance

    if args.filler_mode not in FILLER_MODES:
        raise SystemExit(f"--filler-mode must be one of {FILLER_MODES}")

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    print(f"Loading instances from {src}")
    instances: List[CodingQAInstance] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(CodingQAInstance(**json.loads(line)))
    if args.limit:
        instances = instances[:args.limit]
    print(f"Loaded {len(instances)} instances")

    # Distractor pool: if --pool-from is set, build once and reuse for all
    # buckets. The pool typically draws from a *larger* reference set than the
    # subset being expanded (e.g. all 5K verified instances → expand 1K).
    pool = None
    if args.pool_from:
        pool_path = Path(args.pool_from)
        pool = load_pool_from_jsonl(pool_path) if pool_path != src else build_pool(instances)
    elif args.use_pool:
        pool = build_pool(instances)

    # Lazy-build LLM client only if some path actually needs it.
    needs_llm = (
        args.filler_mode in ("llm", "hybrid")
        or pool is None  # the training phase/3 may still need to LLM-generate distractors
    )
    client = None
    if needs_llm:
        client = LLMClient(model=args.model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [resolve_target_tokens(t) for t in args.target_tokens]
    targets = [t for t in targets if t]

    manifest: dict = {"input": str(src), "buckets": []}
    for target in targets:
        tag = _format_token_tag(target)
        out_path = out_dir / f"{src.stem}_{tag}.jsonl"
        print(f"\n→ Bucket {tag} → {out_path}")

        def _expand_one(inst: CodingQAInstance, _t=target) -> CodingQAInstance:
            if _current_qa_tokens(inst) >= _t:
                return inst
            return expand_qa_instance(
                inst, _t, client,
                filler_mode=args.filler_mode,
                distractor_pool=pool,
                seed=args.seed,
            )

        parallel_map(
            _expand_one,
            instances,
            num_workers=args.num_workers,
            desc=f"Expanding to {tag}",
            output_path=out_path,
            serialize_fn=lambda r: json.dumps(r.to_jsonl_dict()),
            show_progress=True,
            item_id_fn=lambda inst: inst.instance_id,
        )
        manifest["buckets"].append({"target_tokens": target, "path": str(out_path)})

    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"\nWrote manifest → {args.manifest}")


def _format_token_tag(t: int) -> str:
    if t >= 1_000_000 and t % 1_000_000 == 0:
        return f"{t // 1_000_000}m"
    if t >= 1_000 and t % 1_000 == 0:
        return f"{t // 1_000}k"
    return str(t)


def cmd_verify_qa(args):
    """Run QA verification: solvability + distractor confusion + question leakage + dedup."""
    from .stages.qa.verify import verify_all

    verify_all(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        positive_threshold=args.positive_threshold,
        distractor_threshold=args.distractor_threshold,
        leakage_threshold=args.leakage_threshold,
        num_workers=args.num_workers,
        limit=args.limit,
        metadata_path=args.metadata_output,
        use_llm_scoring=args.llm_scoring,
        max_qa_per_instance=args.max_qa_per_instance,
        dedup_overlap=args.dedup_overlap,
        checkpoint_path=getattr(args, "checkpoint", None),
    )


def cmd_dedup(args):
    """Post-hoc dedup on already-verified instances (no LLM calls)."""
    from .stages.qa.verify import dedup_qa_pairs

    instances = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    print(f"Loaded {len(instances)} verified instances")
    total_before = sum(len(d.get("qa_pairs", [])) for d in instances)

    kept = []
    total_after = 0
    for inst_dict in instances:
        inst = CodingQAInstance(**inst_dict)
        deduped = dedup_qa_pairs(
            inst.qa_pairs,
            overlap_threshold=args.dedup_overlap,
            max_per_instance=args.max_qa_per_instance,
        )
        has_memory = any(qp.requires_memory for qp in deduped)
        if len(deduped) >= 2 and has_memory:
            filtered = inst.model_copy(update={"qa_pairs": deduped})
            kept.append(filtered.to_jsonl_dict())
            total_after += len(deduped)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for inst in kept:
            f.write(json.dumps(inst) + "\n")

    print(f"QA pairs: {total_before} -> {total_after} ({total_before - total_after} removed)")
    print(f"Instances: {len(instances)} -> {len(kept)} ({len(instances) - len(kept)} dropped)")
    print(f"Output: {args.output}")


def cmd_eval(args):
    """Run evaluation on coding_qa instances."""
    from .stages.qa.eval import evaluate_all, compare_strategies
    from .stages.qa.score import format_report
    from .memory_methods import create_memory_method

    # Stream the JSONL — at the 100k/500k/1m buckets each row can exceed 5 MB,
    # so loading the whole file up front would OOM or burn minutes before the
    # first LLM call. Count lines once for an honest tqdm total; the count is
    # read-only and finishes in seconds even for 20 GB files.
    instances_path = Path(args.instances)
    total_hint: Optional[int] = None
    try:
        with open(instances_path, "rb") as f:
            total_hint = sum(1 for _ in f)
        if args.limit and args.limit < total_hint:
            total_hint = args.limit
    except OSError:
        total_hint = None

    def _stream_instances():
        with open(instances_path) as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                yield CodingQAInstance(**json.loads(line))

    instances = _stream_instances()
    print(
        f"Streaming Coding QA instances from {instances_path}"
        + (f" (total={total_hint})" if total_hint is not None else "")
    )

    client = LLMClient(model=args.model)

    no_repo_files = getattr(args, "no_repo_files", False)
    num_workers = getattr(args, "num_workers", 1)
    checkpoint = getattr(args, "checkpoint", None)

    # Build memory method (None when --method prompt — uses legacy strategy path)
    method_name = getattr(args, "method", "prompt")
    method_config = json.loads(args.method_config) if args.method_config else {}
    # CLI flag wins unless the user already set max_ingest_chars in
    # --method-config; that lets advanced users override per-method.
    method_config.setdefault("max_ingest_chars", args.ingest_adapter_cap)
    memory_method = None

    if method_name != "prompt":
        memory_method = create_memory_method(method_name, config=method_config)
        print(f"Memory method: {method_name}  config={method_config}")
    else:
        print(f"Memory method: prompt  strategy={args.strategy}")

    if args.compare_strategies:
        if memory_method is not None:
            print("[warn] --compare-strategies ignores --method; using prompt strategies")
        strategies = args.strategies.split(",") if args.strategies else None
        results = compare_strategies(
            instances, client,
            protocol=args.protocol,
            strategies=strategies,
            notes_budget=args.notes_budget,
            use_llm_scoring=args.llm_scoring,
            no_repo_files=no_repo_files,
            num_workers=num_workers,
            checkpoint_path=checkpoint,
        )
        # Save comparison results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
    else:
        results = evaluate_all(
            instances, client,
            protocol=args.protocol,
            strategy=args.strategy,
            notes_budget=args.notes_budget,
            use_llm_scoring=args.llm_scoring,
            no_repo_files=no_repo_files,
            num_workers=num_workers,
            checkpoint_path=checkpoint,
            memory_method=memory_method,
            total_hint=total_hint,
            ingest_chars_per_call=args.ingest_chars_per_call,
            ingest_max_calls=args.ingest_max_calls,
            answerer_prompt_char_cap=args.answerer_prompt_char_cap,
        )

        # Print report
        report = format_report(results)
        print(f"\n{report}")

        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump([r.to_jsonl_dict() for r in results], f, indent=2)
            print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Coding QA Benchmark — fact-level memory evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── convert ──
    p_convert = subparsers.add_parser(
        "convert", help="Convert coding_synthetic instances to QA format"
    )
    p_convert.add_argument(
        "--input", type=str, required=True,
        help="Path to coding_synthetic_instances.jsonl",
    )
    p_convert.add_argument(
        "--output", type=str, required=True,
        help="Path to write coding_qa_instances.jsonl",
    )
    p_convert.add_argument(
        "--model", type=str,
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="LLM for question/answer generation",
    )
    p_convert.add_argument("--limit", type=int, default=None)
    p_convert.add_argument("--num-workers", type=int, default=4)
    p_convert.add_argument(
        "--target-tokens", type=str, default=None,
        help="Target token budget per instance (e.g., 100000 or preset: small/medium/large/xl/extreme)",
    )
    p_convert.add_argument(
        "--no-repo", action="store_true",
        help="Skip Docker repo file extraction",
    )
    p_convert.add_argument(
        "--checkpoint", type=str, default=None,
        help="JSONL checkpoint file for resume support. Each converted "
             "instance is appended immediately so a killed run can resume. "
             "Defaults to <output>.checkpoint.jsonl. Pass empty string to "
             "disable checkpointing (legacy in-memory mode).",
    )

    # ── extract-repo ──
    p_extract = subparsers.add_parser(
        "extract-repo", help="Extract full repo files from Docker into CodingQA instances"
    )
    p_extract.add_argument(
        "--input", type=str, required=True,
        help="Path to coding_qa_instances.jsonl (repo_files may be empty)",
    )
    p_extract.add_argument(
        "--output", type=str, required=True,
        help="Path to write updated instances with repo_files populated",
    )
    p_extract.add_argument("--max-files", type=int, default=300,
                           help="Max Python files to extract per repo (default: 300)")
    p_extract.add_argument("--max-file-size", type=int, default=50000,
                           help="Max file size in bytes to include (default: 50000)")
    p_extract.add_argument("--limit", type=int, default=None)

    # ── dedup ──
    p_dedup = subparsers.add_parser(
        "dedup", help="Post-hoc dedup on verified instances (no LLM calls)"
    )
    p_dedup.add_argument("--input", type=str, required=True,
                         help="Path to verified coding_qa instances")
    p_dedup.add_argument("--output", type=str, required=True,
                         help="Path to write deduped instances")
    p_dedup.add_argument("--max-qa-per-instance", type=int, default=5,
                         help="Max QA pairs per instance (default: 5)")
    p_dedup.add_argument("--dedup-overlap", type=float, default=0.7,
                         help="Question word overlap threshold (default: 0.7)")

    # ── eval ──
    p_eval = subparsers.add_parser(
        "eval", help="Evaluate Coding QA instances"
    )
    p_eval.add_argument(
        "instances", type=str,
        help="Path to coding_qa_instances.jsonl",
    )
    p_eval.add_argument(
        "--protocol", type=str, default="evicted",
        choices=["full_context", "evicted", "fact_retrieval"],
        help="Evaluation protocol",
    )
    p_eval.add_argument(
        "--strategy", type=str, default="basic",
        choices=["none", "basic", "structured", "selective",
                 "hypothesis", "entity_graph", "question_driven"],
        help="Note-taking strategy (for evicted protocol)",
    )
    p_eval.add_argument(
        "--compare-strategies", action="store_true",
        help="Compare all strategies side-by-side",
    )
    p_eval.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated strategies for comparison (default: all)",
    )
    p_eval.add_argument(
        "--model", type=str,
        default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="LLM for evaluation",
    )
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--notes-budget", type=int, default=500)
    p_eval.add_argument(
        "--llm-scoring", default=True, action=argparse.BooleanOptionalAction,
        help="Use LLM judge for scoring (default: on). Pass --no-llm-scoring "
             "for a fast keyword-overlap smoke test.",
    )
    p_eval.add_argument("--no-repo-files", action="store_true",
                        help="Evicted protocol: exclude repo source files from answering phase. "
                             "Tests pure memory retention (notes + repo skeleton only).")
    p_eval.add_argument("--num-workers", type=int, default=1,
                        help="Parallel threads for evaluation (default: 1 = sequential)")
    p_eval.add_argument("--checkpoint", type=str, default=None,
                        help="JSONL checkpoint file for resume support. "
                             "Results saved incrementally; existing results skipped on restart.")
    p_eval.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    p_eval.add_argument(
        "--method", type=str, default="prompt",
        choices=["prompt", "amem", "truncated", "lightmem", "hipporag", "simplemem",
                 "mem0", "memorybank"],
        help="Memory method: prompt (use --strategy), amem, truncated, lightmem, "
             "hipporag, simplemem, mem0, memorybank (default: prompt)",
    )
    p_eval.add_argument(
        "--method-config", type=str, default=None,
        help='JSON config for memory method, e.g. \'{"retrieve_k": 5}\' '
             'or \'{"max_chars": 4000}\'',
    )
    p_eval.add_argument(
        "--ingest-chars-per-call", type=int, default=15000,
        help="Per-ingest text budget in chars (default 15000, matching the "
             "in-adapter cap). Lower = faster ingest, less context per call.",
    )
    p_eval.add_argument(
        "--ingest-max-calls", type=int, default=3,
        help="Cap the training phase ingest calls per instance (default 3). The main "
             "knob: at 8-10 memory_files/instance the old per-file loop fired "
             "10+ ingest calls, often hanging on lightmem (2 LLM calls each).",
    )
    p_eval.add_argument(
        "--answerer-prompt-char-cap", type=int, default=60000,
        help="Per-QA-pair char cap on notes_text and repo_files_text before "
             "prompt assembly (default 60000 ~= 15k tokens). Independent of "
             "memory method; primary knob to keep prompt mode from hanging.",
    )
    p_eval.add_argument(
        "--ingest-adapter-cap", type=int, default=15000,
        help="Per-call hard cap (chars) inside amem/lightmem ingest. Default "
             "15000 matches the original in-adapter slice. Raise this for "
             "500k/1m buckets so more content fits per ingest LLM call. "
             "Total content ingested per instance is bounded by "
             "--ingest-max-calls × this value.",
    )

    # ── expand-instances ──
    p_expand = subparsers.add_parser(
        "expand-instances",
        help="Scale verified QA instances to one or more target token budgets "
             "(parallel, checkpointed; no-LLM filler by default).",
    )
    p_expand.add_argument("--input", type=str, required=True,
                          help="Path to a verified coding_qa_instances.jsonl")
    p_expand.add_argument("--output-dir", type=str, required=True,
                          help="Directory for per-bucket jsonl outputs")
    p_expand.add_argument("--target-tokens", nargs="+", required=True,
                          help="One or more targets (presets or ints), e.g. "
                               "10000 50000 100000 500000 1000000, or "
                               "small medium large xl extreme")
    p_expand.add_argument("--filler-mode", type=str, default="local",
                          choices=["local", "llm", "hybrid"],
                          help="the training phase filler source: local=repo excerpts (no LLM, "
                               "default), llm=legacy prose, hybrid=local then LLM "
                               "top-up if budget unmet")
    p_expand.add_argument("--use-pool", action="store_true",
                          help="Build cross-instance distractor pool from --input "
                               "and recycle for the training phase/3 (skips most LLM calls)")
    p_expand.add_argument("--pool-from", type=str, default=None,
                          help="Build the distractor pool from a different jsonl "
                               "(e.g. the full verified set when expanding a subset). "
                               "Implies --use-pool.")
    p_expand.add_argument("--num-workers", type=int, default=4,
                          help="Parallel threads (default: 4)")
    p_expand.add_argument("--limit", type=int, default=None)
    p_expand.add_argument("--seed", type=int, default=None,
                          help="RNG seed for filler sampling (default: derived "
                               "from instance_id for reproducibility)")
    p_expand.add_argument("--model", type=str,
                          default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                          help="LLM client model (only used if filler-mode != local "
                               "or pool is empty)")
    p_expand.add_argument("--manifest", type=str, default=None,
                          help="Optional path to write a JSON manifest of bucket → file")

    # ── verify-qa ──
    p_verify = subparsers.add_parser(
        "verify-qa", help="Verify QA pair solvability and anti-hack properties"
    )
    p_verify.add_argument(
        "--input", type=str, required=True,
        help="Path to coding_qa_instances.jsonl (with repo_files)",
    )
    p_verify.add_argument(
        "--output", type=str, required=True,
        help="Path to write verified/filtered instances",
    )
    p_verify.add_argument(
        "--model", type=str,
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="LLM for verification checks (default: Haiku)",
    )
    p_verify.add_argument("--positive-threshold", type=float, default=0.4,
                          help="Min score for solvability check (default: 0.4)")
    p_verify.add_argument("--distractor-threshold", type=float, default=0.5,
                          help="Max score for distractor confusion check (default: 0.5)")
    p_verify.add_argument("--leakage-threshold", type=float, default=0.3,
                          help="Max score for question leakage check (default: 0.3)")
    p_verify.add_argument("--num-workers", type=int, default=8)
    p_verify.add_argument("--limit", type=int, default=None)
    p_verify.add_argument(
        "--llm-scoring", default=True, action=argparse.BooleanOptionalAction,
        help="Use LLM judge for scoring (default: on). Pass --no-llm-scoring "
             "for a fast keyword-overlap smoke test.",
    )
    p_verify.add_argument("--metadata-output", type=str, default=None,
                          help="Path for verification metadata JSONL (default: auto)")
    p_verify.add_argument("--max-qa-per-instance", type=int, default=5,
                          help="Max QA pairs to keep per instance after dedup (default: 5)")
    p_verify.add_argument("--dedup-overlap", type=float, default=0.7,
                          help="Question word overlap threshold for dedup (default: 0.7)")
    p_verify.add_argument("--checkpoint", type=str, default=None,
                          help="JSONL checkpoint file for resume support. "
                               "Each verified instance is appended immediately so a killed "
                               "run can resume. Defaults to <output>.checkpoint.jsonl. "
                               "Pass empty string to disable.")

    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "extract-repo":
        cmd_extract_repo(args)
    elif args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "verify-qa":
        cmd_verify_qa(args)
    elif args.command == "expand-instances":
        cmd_expand_instances(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
