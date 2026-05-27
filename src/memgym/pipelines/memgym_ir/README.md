# MemGym-IR: Memory Benchmark for Multi-Turn Information Retrieval

Generates benchmark instances that test whether an agentic memory system can selectively retain critical facts across multi-turn searches to answer complex multi-hop questions.

## Pipeline (6 stages)

```
MuSiQue dataset (25K multi-hop QA)
       |
[Stage 0: Mine]      Filter for >= 3-hop, >= 3 supporting paragraphs
       |
[Stage 1: Pre-screen] Reject questions answerable from parametric knowledge
       |
[Stage 2: Split]     Extract grounding facts + dependency graph (LLM)
       |
[Stage 3: Craft]     Build search turns with 4-source distractors (LLM)
       |
[Stage 4: Verify]    Ablation test: with-memory vs without-memory gap
       |
[Stage 5: Harden]    Apply difficulty transforms (query fuzz, scaling, etc.)
       |
   final dataset + quality report
```

## Data Source

| Source | Description | Size |
|--------|-------------|------|
| **MuSiQue** | Multi-hop QA with per-hop supporting paragraphs | 25K train |

## Models

Uses Claude via AWS Bedrock by default:
- **Worker** (cheap): Claude Haiku 4.5 -- generation, fact extraction, distractor creation
- **Verifier** (strong): Claude Sonnet 4.5 -- ablation verification, fact cross-checking

## Quick Start

```bash
# Dry-run (no LLM calls, no cost)
cd ${REPO_ROOT}
python -m pipelines.memgym_ir --dry-run --limit 50

# Full run with Claude models
python -m pipelines.memgym_ir --limit 50 --output results/ir_test

# Via the job-runner API (optional remote server)
curl -X POST http://<YOUR_HOST>:30000/run-ir-pipeline \
  -H "Content-Type: application/json" \
  -d '{"limit": 50, "output": "results/ir_test"}'
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config FILE` | YAML config file | None |
| `--output DIR` | Output directory | `./output_ir` |
| `--limit N` | Max raw instances to load | None (all) |
| `--worker-model MODEL` | Cheap LLM for generation | Claude Haiku 4.5 |
| `--verifier-model MODEL` | Strong LLM for verification | Claude Sonnet 4.5 |
| `--difficulty PRESET` | `easy` / `medium` / `hard` | `easy` |
| `--target-tokens N` | Token budget per instance | `10000` |
| `--skip-verification` | Skip ablation (Stage 4) | false |
| `--no-resume` | Don't resume from checkpoints | false |
| `--dry-run` | Only run filter stage, no LLM | false |

## Output

```
output_dir/
  filtered.jsonl              # Stage 0
  prescreen.json              # Stage 1
  extracted.jsonl             # Stage 2
  crafted.jsonl               # Stage 3
  verification_metadata.json  # Stage 4
  memgym_ir_instances.jsonl   # Final dataset
  quality_report.json         # Machine-readable report
  quality_report.md           # Human-readable report
```

## File Structure

```
pipelines/memgym_ir/
  __init__.py, __main__.py, pipeline.py
  configs/          # YAML configs (default, smoke_test)
  types/schemas.py  # Pydantic models
  stages/           # Pipeline stages (filter, prescreen, extract, generate, validate, harden, report)
  llm/              # LLM client + prompt templates
```
