# MemGym Quick Start

## 1. Install (One Command)

```bash
./install.sh --all  # Installs UV + all dependencies + tau2-bench + SWE-bench
```

**Options:**
- `./install.sh --tau2` - Only tau2-bench
- `./install.sh --swe` - Only SWE-bench
- `./install.sh` - Core only

## 2. Setup LLM Backend

**OpenAI:**
```bash
export OPENAI_API_KEY='sk-...'
```

**Local Model (SGLang - Free):**
```bash
# Terminal 1: Start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --port 8000

# Terminal 2: Set env
export OPENAI_API_BASE='http://localhost:8000/v1'
export OPENAI_API_KEY='dummy'
```

**Note:** Tau2-bench already uses LiteLLM - supports all backends (OpenAI, Anthropic, local models, Ollama).

## 3. Run Evaluation

**Tau2-bench (gym/tau2_bench, uses original tau2 agents):**
```bash
# Install once: clones sierra-research/tau2-bench at pinned SHA
python -m memgym.gym.tau2_bench.install --venv /path/to/venv

# List domains / registered memory models (no API key needed)
python -m memgym.gym.tau2_bench --list_domains
python -m memgym.gym.tau2_bench --list_memory_models

# All four domains in one sweep (mock + telecom + airline + retail)
python -m memgym.gym.tau2_bench --domains all --agent_llm gpt-4o-mini

# Mock only (quick smoke test)
python -m memgym.gym.tau2_bench --domains mock --task_ids 0,1,2 --agent_llm gpt-4o-mini

# Single domain
python -m memgym.gym.tau2_bench --domains airline --agent_llm gpt-4o-mini

# Limit tasks per domain (per-domain cap; 0 = no limit)
python -m memgym.gym.tau2_bench --domains all --agent_llm gpt-4o-mini --limit 3

# With symmetric memory (agent + user simulator both wrapped)
python -m memgym.gym.tau2_bench --domains all --agent_llm gpt-4o-mini \
    --memory_model tau2_summarizing --keep_first 1 --keep_last 4 --max_size 30
```

**Note:** solo_mode is auto-detected by domain:
- **mock, telecom**: solo_mode=True (single agent, no user simulator)
- **airline, retail**: solo_mode=False (agent + user simulator, both wrappable with memory)

**SWE-bench (uses original SWE evaluation):**

`-o/--output` is **required**. Valid `--dataset` values: `lite`, `verified`,
`full`, `swe-gym`, `swe-smith` (default: `lite`).

```bash
# Lite (300 instances), first 100
python scripts/evaluate_swe_bench.py --model openai/gpt-4o-mini \
    --dataset lite --slice 0:100 -o results/lite_smoke

# Full SWE-bench (2,294 instances)
python scripts/evaluate_swe_bench.py --model openai/gpt-4o \
    --dataset full --slice 0:500 -o results/full

# Equivalent module form (no scripts/ path dependency)
python -m memgym.gym.swe_bench --model openai/gpt-4o-mini \
    --dataset lite --slice 0:100 -o results/lite_smoke
```

**WebArena (gym/webarena, uses WebArena-Infinity):**

`--policy_model` and `--app_name` are **required**. WebArena needs the
WebArena-Infinity package installed and a running WebArena server.

```bash
# One-time: install WebArena-Infinity + Playwright/Chromium
python -m memgym.gym.webarena.install

# WebArena requires a running WebArena server. Point the env at its base URL:
export WEBARENA_BASE_URL='http://localhost:7770'

# Smoke test: one app, first few tasks
python -m memgym.gym.webarena \
    --policy_model gpt-4o-mini --app_name gitlab \
    --task_ids 0,1,2 --observation_mode text

# With structured memory
python -m memgym.gym.webarena \
    --policy_model gpt-4o-mini --app_name gitlab \
    --memory_model webarena_structured --max_tokens 16000
```

> After install, `python -m memgym.gym.webarena --app_name <x> --policy_model <x> --list_apps`
> enumerates the valid `--app_name` values.

**MemRM (Memory Reward Model evaluation):**
```bash
uv pip install -e ".[eval]"   # bootstrap uv once with `pip install uv`

# Reproduce paper IID-heldout (n=3,007) — downloads checkpoint from HF
memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b

# Sweep all six RM datasets and write JSON
memgym-eval-rm --dataset all --checkpoint MemGym/memgym-rm-1p7b \
    --output results/memrm_table.json

# Evaluate on a custom JSONL (rows need at least `prompt` and `label`)
memgym-eval-rm --dataset /path/to/pairs.jsonl --checkpoint MemGym/memgym-rm-1p7b
```

**Evaluating your own memory against the RM (lightweight):**
```bash
# 1) Write a memory subclassing BaseMemoryManager and call
#    register_memory_model("my_memory", MyMemory) at module import time.
# 2) Score it against a raw-trajectory JSONL (schema in
#    src/memgym/training/eval/memory_eval.py):
memgym-eval-memory \
    --memory-model my_memory \
    --memory-module my_memory \
    --trajectories sample_trajectories.jsonl \
    --checkpoint MemGym/memgym-rm-1p7b

# Baseline: passthrough on the same JSONL — AUROC delta is the signal
memgym-eval-memory --memory-model passthrough \
    --trajectories sample_trajectories.jsonl \
    --checkpoint MemGym/memgym-rm-1p7b
```
The score is a *relative* memory-quality metric (see CLI banner).

## 4. Quick Test

```bash
# Test mock domain (fast smoke test, 3 tasks)
python -m memgym.gym.tau2_bench --domains mock --task_ids 0,1,2 --agent_llm gpt-4o-mini

# Verify package imports
pytest tests/unit/test_imports.py
```

## Common Commands

| Task | Command |
|------|---------|
| Install tau2-bench | `python -m memgym.gym.tau2_bench.install --venv /path/to/venv` |
| Quick test (mock) | `python -m memgym.gym.tau2_bench --domains mock --task_ids 0,1,2 --agent_llm gpt-4o-mini` |
| Eval all domains | `python -m memgym.gym.tau2_bench --domains all --agent_llm gpt-4o-mini` |
| Eval limited tasks | `python -m memgym.gym.tau2_bench --domains all --agent_llm gpt-4o-mini --limit 10` |
| Eval SWE-bench | `python scripts/evaluate_swe_bench.py --model openai/gpt-4o-mini --dataset lite --slice 0:100 -o results/lite_smoke` |
| Eval WebArena (one app) | `python -m memgym.gym.webarena --policy_model gpt-4o-mini --app_name gitlab --task_ids 0,1,2` |
| Eval MemRM (paper IID) | `memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b` |
| Eval MemRM (all RM datasets) | `memgym-eval-rm --dataset all --checkpoint MemGym/memgym-rm-1p7b` |
| Eval your memory vs RM | `memgym-eval-memory --memory-model my_memory --memory-module my_memory --trajectories sample.jsonl --checkpoint MemGym/memgym-rm-1p7b` |

## Key Features

✅ **Original agent implementations** (tau2's LLMAgent, SWE-bench's harness)
✅ **LiteLLM support** (OpenAI, Anthropic, SGLang, vLLM, Ollama)
✅ **Memory operations** (with/without context management)
✅ **Fast installation** (UV is 10-100x faster than pip)

## Troubleshooting

**UV not found:**
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

**tau2-bench not found:**
```bash
python -m memgym.gym.tau2_bench.install --venv /path/to/venv
# (clones sierra-research/tau2-bench at pinned SHA into third_party/tau2-bench
#  then uv pip installs -e . into the chosen venv)
```

**Local model connection error:**
```bash
# Check server is running
curl http://localhost:8000/v1/models

# Enable debug
export LITELLM_LOG=DEBUG
```

## Documentation

- **Main README**: `README.md` - Full documentation
- **Architecture**: `ARCHITECTURE.md` - System design
- **This guide**: `QUICKSTART.md` - You are here

---

**Next:** Run `python -m memgym.gym.tau2_bench.install --venv /path/to/venv` then `python -m memgym.gym.tau2_bench --domains mock --task_ids 0,1,2 --agent_llm gpt-4o-mini` for a quick test.

**Note on Memory Operations:**
- All environments track memory operations (context updates, summarization)
- For solo_mode=False domains (airline, retail), both the agent AND user simulator have their own memory operations tracked
- Use `--context-management --max-tokens 2000` to enable automatic summarization when context exceeds threshold
- Use `--debug` to see full error tracebacks when debugging issues
