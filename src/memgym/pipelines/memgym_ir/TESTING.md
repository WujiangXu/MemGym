# MemGym-IR Testing Guide

## Setup

```bash
source ${HOME}/env/memgym-ir-pipeline/bin/activate
cd ${REPO_ROOT}

python -c "from pipelines.memgym_ir.pipeline import run_pipeline; print('OK')"
```

---

## Test 1: Dry Run (no LLM, no cost)

```bash
python -m pipelines.memgym_ir --dry-run --limit 50
```

Expected: Loads MuSiQue, filters ~50 raw to ~4 instances (>=3 hops), generates quality report.

---

## Test 2: Full Pipeline via EC2 (Claude Bedrock)

Submit to EC2 server which has Bedrock credentials:

```bash
EC2=http://<YOUR_EC2_HOST>:30000

# Skip verification (cheaper)
curl -X POST $EC2/run-ir-pipeline -H "Content-Type: application/json" \
  -d '{"limit": 50, "skip_verification": true, "output": "results/ir_test"}'

# Full with verification
curl -X POST $EC2/run-ir-pipeline -H "Content-Type: application/json" \
  -d '{"limit": 50, "output": "results/ir_full_test", "difficulty": "medium"}'

# Check job status
curl $EC2/jobs

# Download quality report
curl $EC2/download/ir_test/quality_report.md
```

---

## Test 3: Batch Jobs (IR + Coding + SWE-gym in parallel)

```bash
curl -X POST $EC2/batch-jobs -H "Content-Type: application/json" -d '{
  "jobs": [
    {"run_ir_pipeline": {"limit": 50, "output": "results/ir_test"}},
    {"run_pipeline": {"limit": 20, "output": "results/coding_test"}},
    {"evaluate": {"dataset": "lite", "output": "results/swe_test"}}
  ]
}'
```

---

## Test 4: Local with Custom Models

```bash
# OpenAI models
OPENAI_API_KEY="sk-..." python -m pipelines.memgym_ir \
    --limit 50 --skip-verification \
    --worker-model gpt-4o-mini --verifier-model gpt-4o

# Local LLM (SGLang/vLLM)
OPENAI_API_BASE="http://localhost:30001/v1" python -m pipelines.memgym_ir \
    --limit 50 --worker-model openai/Qwen3-32B --verifier-model openai/Qwen3-32B
```

---

## Output Files

```
output_dir/
  filtered.jsonl              # Stage 0: filtered MuSiQue instances
  prescreen.json              # Stage 1: parametric test results
  extracted.jsonl             # Stage 2: grounding facts
  crafted.jsonl               # Stage 3: full instances
  verification_metadata.json  # Stage 4: ablation scores
  memgym_ir_instances.jsonl   # Final dataset
  quality_report.json         # Machine-readable report
  quality_report.md           # Human-readable report
```

---

## What to Check in Quality Report

1. **Pipeline funnel**: How many instances survive each stage
2. **Ablation gap**: score_A - score_B >= 0.30 (memory genuinely helps)
3. **Distractor composition**: Adversarial and contradiction tokens should be non-zero
4. **Per-instance**: Each instance should have >= 3 grounding facts

---

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config FILE` | YAML config file | None |
| `--output DIR` | Output directory | `./output_ir` |
| `--limit N` | Max raw instances to load | None |
| `--worker-model MODEL` | Cheap LLM | Claude Haiku 4.5 |
| `--verifier-model MODEL` | Strong LLM | Claude Sonnet 4.5 |
| `--difficulty PRESET` | `easy`/`medium`/`hard` | `easy` |
| `--target-tokens N` | Token budget per instance | `10000` |
| `--skip-verification` | Skip Stage 4 | false |
| `--no-resume` | Ignore checkpoints | false |
| `--dry-run` | No LLM calls | false |
