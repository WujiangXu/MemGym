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

# All four domains in one sweep (mock + telecom + airline + retail)
python -m memgym.gym.tau2_bench --all_domains --agent_llm gpt-4o-mini

# Mock only (quick smoke test)
python -m memgym.gym.tau2_bench --domain mock --task_ids 0,1,2 --agent_llm gpt-4o-mini

# Single domain
python -m memgym.gym.tau2_bench --domain airline --agent_llm gpt-4o-mini

# Limit tasks per domain
python -m memgym.gym.tau2_bench --all_domains --agent_llm gpt-4o-mini --max_tasks_per_domain 3

# With symmetric memory (agent + user simulator both wrapped)
python -m memgym.gym.tau2_bench --all_domains --agent_llm gpt-4o-mini \
    --memory_model tau2_summarizing --keep_first 1 --keep_last 4 --max_size 30
```

**Note:** solo_mode is auto-detected by domain:
- **mock, telecom**: solo_mode=True (single agent, no user simulator)
- **airline, retail**: solo_mode=False (agent + user simulator, both wrappable with memory)

**SWE-bench (uses original SWE evaluation):**
```bash
# Lite (300 instances)
python scripts/evaluate_swe_bench.py --model openai/gpt-4o-mini --slice 0:100

# Full (2,294 instances)
python scripts/evaluate_swe_bench.py --model openai/gpt-4o --dataset swe-bench --slice 0:500
```

## 4. Quick Test

```bash
# Test mock domain (fast smoke test, 3 tasks)
python -m memgym.gym.tau2_bench --domain mock --task_ids 0,1,2 --agent_llm gpt-4o-mini

# Verify package imports
pytest tests/unit/test_imports.py
```

## Common Commands

| Task | Command |
|------|---------|
| Install tau2-bench | `python -m memgym.gym.tau2_bench.install --venv /path/to/venv` |
| Quick test (mock) | `python -m memgym.gym.tau2_bench --domain mock --task_ids 0,1,2 --agent_llm gpt-4o-mini` |
| Eval all domains | `python -m memgym.gym.tau2_bench --all_domains --agent_llm gpt-4o-mini` |
| Eval limited tasks | `python -m memgym.gym.tau2_bench --all_domains --agent_llm gpt-4o-mini --max_tasks_per_domain 10` |
| Eval SWE-bench | `python scripts/evaluate_swe_bench.py --model openai/gpt-4o-mini --slice 0:100` |

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
#  then pip installs -e . into the chosen venv)
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

**Next:** Run `python -m memgym.gym.tau2_bench.install --venv /path/to/venv` then `python -m memgym.gym.tau2_bench --domain mock --task_ids 0,1,2 --agent_llm gpt-4o-mini` for a quick test.

**Note on Memory Operations:**
- All environments track memory operations (context updates, summarization)
- For solo_mode=False domains (airline, retail), both the agent AND user simulator have their own memory operations tracked
- Use `--context-management --max-tokens 2000` to enable automatic summarization when context exceeds threshold
- Use `--debug` to see full error tracebacks when debugging issues
