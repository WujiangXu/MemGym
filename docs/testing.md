# Manual Testing Guide

A tiered set of **copy-pasteable** commands to exercise every MemGym ability by
hand. Every command here was checked against the actual argparse / imports — the
flags are the ones the code really accepts.

Tiers escalate by cost and required resources:

| Tier | Needs | Cost | What it proves |
|------|-------|------|----------------|
| **0** | Python env only | free | Package installs and imports |
| **1** | nothing extra | free | Every CLI parses; all strategies registered |
| **2** | one LLM API key (GPU for MemRM) | ~$0.01–$1 | One real episode / one reward-model read |
| **3** | Docker / GPU / WebArena server | hours | Full per-track eval reproduction |

### Track → paper map

| Track | Entry point | Paper |
|-------|-------------|-------|
| SWE-Gym (coding) | `memgym.gym.swe_bench` | Tab. `tab:wrapped-gyms` |
| τ²-bench (dialogue) | `memgym.gym.tau2_bench` | Tab. `tab:wrapped-gyms` |
| WebArena-Infinity (computer use) | `memgym.gym.webarena` | Tab. `tab:wrapped-gyms` |
| MemGym-DR (deep research) | `memgym.pipelines.memgym_ir` + `examples/memgym_dr/run_ir_benchmark.py` | Fig. 2b |
| MemGym-CodeQA (coding QA) | `memgym.pipelines.coding_synthetic` | Fig. 2a |
| MemRM (reward model) | `memgym-eval-rm` | Tab. `tab:memrm` |

---

## Tier 0 — install & import (free, no key)

```bash
# From a clean venv (Python 3.10+; 3.12 recommended)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install -e ".[dev,eval]"
```

Verify imports and the release-critical unit suite:

```bash
pytest tests/unit/test_imports.py            # expect: 4 passed

pytest tests/unit/test_imports.py tests/unit/test_rm_eval.py \
       tests/unit/test_world_model.py tests/unit/test_memory_strategies_roundtrip.py \
       tests/unit/test_known_bugs.py tests/unit/test_memory_eval.py -q
# expect: 67 passed
```

> There are no `xfail`s: the config-fallback contract bug (`gym/swe_bench/env.py`
> used `print()` instead of `warnings.warn`) is now fixed, so its `xfail` marker
> was removed.

With all track extras installed, the full `tests/unit/` run reports
**301 passed, 1 skipped, 0 xfailed** (302 collected). The lone skip is the
API-key-gated integration test in `test_naive_summarization.py` (set
`OPENAI_API_KEY` and `RUN_INTEGRATION_TESTS=1` to include it). On `[dev,eval]`
alone, the `[swe]`-only tests (`minisweagent`/`unidiff`, etc.) self-skip via
`pytest.importorskip`; the tau2 trajectory-loader tests run against the
committed fixture at `tests/fixtures/trajectories/tau2_bench_run/` and self-skip
only if it is removed.

---

## Tier 1 — zero-cost wiring smoke (free, no key/Docker/GPU)

These prove the entry points parse and the strategies are wired. **No model is
ever called.**

```bash
# Every entry point parses its args and exits 0
python -m memgym.gym.tau2_bench --help
python -m memgym.gym.swe_bench --help
python -m memgym.gym.webarena --help
python examples/run_episode.py --help
memgym-eval-rm --help
memgym-eval-memory --help
python -m memgym.pipelines.memgym_ir --help
```

```bash
# τ²-bench: list domains + every registered memory strategy
python -m memgym.gym.tau2_bench --list_domains          # mock, telecom, airline, retail
python -m memgym.gym.tau2_bench --list_memory_models
```

```bash
# Prove the README "Memory Strategies" table is real — print the registry
python -c "from memgym.memory.base import list_memory_models; print(sorted(list_memory_models()))"
```

The registry should include the paper's baselines:
`passthrough` (None), `llm_summarizing`/`naive` (Summary), `structured_summary`
(Structured), `ir_bm25` (BM25), `ir_naive_rag` (Naive RAG), `ir_memorybank`
(MemoryBank), `ir_simplemem` (SimpleMem), `ir_lightmem` (LightMem),
`ir_passthrough` (None for DR). **A-Mem** (`amem`, `ir_amem`) only registers when
the optional dependency is installed — see Caveats.

**Caveats for Tier 1:**
- `webarena --list_apps` / `--list_memory_models` are **not** zero-cost: `main()`
  resolves the WebArena-Infinity directory before those flags short-circuit, so
  they need WebArena installed. Only `webarena --help` is a true no-dep smoke.
- `coding_synthetic --help` and `examples/memgym_dr/run_ir_benchmark.py --help` import
  `unidiff` / `tenacity` at module load, so they need the `[swe]` extra even just
  to print help. See Tier 3.

---

## Tier 2 — cheap LLM-key runs (~$0.01–$1)

Set a backend. MemGym uses LiteLLM, so any provider works:

```bash
export OPENAI_API_KEY='sk-...'        # then use --agent_llm gpt-4o-mini, etc.
# For the paper's Bedrock models (bedrock/...claude-haiku-4-5...), instead:
#   uv pip install boto3 && configure AWS creds (litellm needs botocore for Bedrock)
```

### τ²-bench — one mock task, memory OFF then ON

```bash
# Baseline (no memory) — one task, ~$0.01
python -m memgym.gym.tau2_bench --domains mock --task_ids 0 --limit 1 \
    --agent_llm gpt-4o-mini --memory_model none

# Same task, with dialogue summarization memory (symmetric: agent + user sim)
python -m memgym.gym.tau2_bench --domains mock --task_ids 0 --limit 1 \
    --agent_llm gpt-4o-mini --memory_model tau2_summarizing \
    --memory_side both --keep_first 1 --keep_last 4 --max_size 30
```

Success: each prints a task result / success rate and writes to `--result_dir`.
The `--memory_side {both,agent,user}` flag is the paper's "whose memory matters"
ablation.

### Generic single episode (tau2 or swe)

```bash
python examples/run_episode.py --env tau2 --domain mock --task-id 0 \
    --agent-llm gpt-4o-mini --context-management --max-tokens 2000 -v
```

### MemRM — reward-model read (needs 1 GPU + ~3.4 GB HF download)

```bash
# Small slice first (fast); drop --limit for the full paper number
memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b --limit 8
```

Success: prints AUROC / accuracy / ECE. The full IID split (n=3,007, no `--limit`)
reproduces the paper's **AUROC 0.985** at threshold `t*=0.88` (the CLI default).
4-bit QLoRA inference needs CUDA + `bitsandbytes`; pass `--no-4bit` for fp16 or
`--device cpu` (slow).

### Score your own memory against MemRM (relative metric)

`memgym-eval-memory` applies a memory strategy to a raw-trajectory JSONL and
scores it. The signal is the **AUROC delta** between two strategies on the same
file (the schema is documented in `src/memgym/training/eval/memory_eval.py`;
`tests/fixtures/trajectories/*.json` are example trajectories to adapt).

```bash
memgym-eval-memory --memory-model structured_summary \
    --trajectories your_trajs.jsonl --checkpoint MemGym/memgym-rm-1p7b \
    --output results/structured.json

memgym-eval-memory --memory-model passthrough \
    --trajectories your_trajs.jsonl --checkpoint MemGym/memgym-rm-1p7b \
    --output results/passthrough.json
# Compare AUROC between the two runs — that delta is the memory-quality read.
```

---

## Tier 3 — Docker / GPU / server runs (hours, real resources)

### SWE-Gym — agent + official swebench harness (Docker + key)

```bash
# Smoke: 5 instances of SWE-bench Lite, no memory
python examples/swe_bench/evaluate_swe_bench.py --model openai/gpt-4o-mini \
    --dataset lite --slice 0:5 --workers 4 -v -o results/lite5

# Same slice, with OpenHands-style rolling summary memory
python examples/swe_bench/evaluate_swe_bench.py --model openai/gpt-4o-mini \
    --dataset lite --slice 0:5 --memory llm_summarizing --max-size 100 --keep-first 1 \
    --workers 4 -v -o results/lite5_summ
```

`-o/--output` is required. Valid `--memory`: `none`, `naive`,
`observation_masking`, `llm_summarizing`, `structured_summary`,
`pipeline_masking_summarizing`, `pipeline_masking_structured`, `sliding_window`,
`adaptive_token_budget`. Results land in `results/<name>/` with
`preds.json` + `summary.json`. To skip the Docker eval and only collect
trajectories, add `--skip-eval`.

### WebArena-Infinity — needs install + running server

```bash
python -m memgym.gym.webarena.install         # one-time: package + Playwright/Chromium
export WEBARENA_BASE_URL='http://localhost:7770'

python -m memgym.gym.webarena \
    --policy_model gpt-4o-mini --app_name gitlab \
    --task_ids 0,1,2 --observation_mode text \
    --memory_model webarena_structured --max_tokens 16000
```

`--policy_model` and `--app_name` are required. After install,
`python -m memgym.gym.webarena --app_name gitlab --policy_model x --list_apps`
lists valid apps.

### MemGym-DR (deep research) — needs `[swe]` extra + key

```bash
uv pip install -e ".[swe]"     # brings tenacity/unidiff/datasets

# Generate a tiny dataset -> data/dr_smoke/memgym_ir_instances.jsonl
python -m memgym.pipelines.memgym_ir dataset --limit 2 --output data/dr_smoke

# Benchmark memory strategies. One JSONL per hop stratum; >=1 required.
python examples/memgym_dr/run_ir_benchmark.py \
    --data-4hop data/dr_smoke/memgym_ir_instances.jsonl \
    --strategies ir_bm25,ir_naive_rag,ir_summarizing \
    --limit 2 --output data/dr_results.json
```

Maps to Fig. 2b. IR strategy names: `ir_bm25`, `ir_naive_rag`, `ir_memorybank`,
`ir_simplemem`, `ir_lightmem`, `ir_amem` (needs `requirements-amem.txt`),
`ir_passthrough` (None).

### MemGym-CodeQA — needs `[swe]` extra + key

```bash
uv pip install -e ".[swe]"

python -m memgym.pipelines.coding_synthetic \
    --limit 2 --worker-model gpt-4o-mini --verifier-model gpt-4o-mini \
    --output output/codeqa_smoke
```

Generates verified QA items from SWE-smith instances (Fig. 2a). The three-check
verifier (solvability / distractor-confusion / question-leakage) runs unless you
pass `--skip-verification`.

### Full MemRM table (GPU)

```bash
memgym-eval-rm --dataset all --checkpoint MemGym/memgym-rm-1p7b \
    --output results/memrm_table.json
```

Registered datasets: `iid-heldout`, `train-sanity`, `scenario-ood-tau2`,
`scenario-ood-wa-long`, `scenario-ood-webarena`, `strategy-ood`. Reproduces
Tab. `tab:memrm`. (`strategy-ood` is a 22-row covered subset — the CLI prints a
warning that it does not reproduce the paper's n=166 headline.)

---

## Caveats / known issues

- **A-Mem requires an optional dep.** `amem` / `ir_amem` only register after
  `uv pip install -r requirements-amem.txt` (pulls `sentence-transformers`).
  Without it you'll see `ir_amem unavailable ...` on stderr and the name is
  absent from `--list_memory_models`. This is expected.
- **Bedrock needs boto3.** The paper's reasoners are `bedrock/...` models. Plain
  `[dev,eval]`/`[swe]` extras don't install `botocore`, so you'll see LiteLLM
  Bedrock warnings until you `uv pip install boto3`. OpenAI models work out of
  the box with `[swe]`.
- **`webarena --list_apps` is gated** behind WebArena-Infinity being installed
  (see Tier 1 caveats).
- **`coding_synthetic` / `run_ir_benchmark` need `[swe]`** to even print `--help`.
- See `reproducibility.md` for the original structural-smoke transcript and what
  each tier of eval actually requires.
