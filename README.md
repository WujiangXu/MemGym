# MemGym

A framework for testing memory systems in long-context LLM agents.

## Why MemGym?

As conversations grow, context windows fill up and agent performance degrades. MemGym provides:

- **Standardized benchmarks** for evaluating memory across five agent tracks
- **Memory-reasoning separation** to test different memory strategies
- **Training trajectories** (step-based CoT format) for training memory-aware models
- **Memory Reward Model (MemRM)** — a Qwen3-1.7B classifier that scores memory-conditioned actions and a `memgym-eval-rm` CLI for paper-style validation
- **Official SWE-bench evaluation** via the `swebench` harness (same as sb-cli / leaderboard)

## Tracks

| Track | Module | Source data | Paper § |
|-------|--------|-------------|---------|
| SWE-Gym (code repair) | `memgym.gym.swe_bench` | SWE-Gym / SWE-bench Lite | Tab. `tab:main` |
| tau2-bench (dialogue) | `memgym.gym.tau2_bench` | airline / retail / telecom / mock | Tab. `tab:main` |
| WebArena (web nav) | `memgym.gym.webarena` | WebArena-Infinity | Tab. `tab:main` |
| MemGymCodeQA | `memgym.pipelines.coding_synthetic` | 4,289 verified items | Tab. `tab:codeqa` |
| MemGymDR (multi-hop QA) | `memgym.pipelines.memgym_ir` | 2WikiMultihopQA + MuSiQue | Tab. `tab:dr` |

> Per-track run commands: [`docs/tracks.md`](docs/tracks.md). Released datasets,
> schemas, and download snippets: [`docs/data.md`](docs/data.md).

## Installation

```bash
./install.sh --all        # Full install (core + SWE-bench + tau2 + OpenHands)
./install.sh --swe        # SWE-bench only (mini-swe-agent scaffold)
./install.sh --tau2       # Dialogue tasks only
./install.sh --openhands  # OpenHands CodeAct agent scaffold
```

**Requirements:** Python 3.12+, Docker (for SWE-bench eval), `swebench>=4.1.0`

> All install steps in this repo use `uv pip install` (≈10–100× faster than pip and the standard installer on the project's EC2 fleet). Bootstrap uv once with `pipx install uv` or `pip install uv` — every subsequent `pip` line in the docs is then `uv pip`.

### OpenHands Setup (CodeAct agent)

The OpenHands CodeAct agent runs as an alternative scaffold alongside mini-swe-agent. It is **not** bundled — clone it into `third_party/`:

```bash
git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands
./install.sh --openhands   # or: cd third_party/OpenHands && uv pip install -e .
```

## Memory Strategies

| Strategy | Description | Key Args |
|----------|-------------|----------|
| `none` / `passthrough` | Baseline — no filtering | — |
| `llm_summarizing` | Rolling LLM summary (OpenHands default) | `--max-size`, `--keep-first` |
| `observation_masking` | Truncate old observations | `--attention-window` |
| `naive` | Summarize when over token limit | `--max-tokens` |
| `adaptive_token_budget` | Pin critical context, keep recent tail, summarize middle | `--max-tokens`, `--keep-recent`, `--keep-first` |
| `structured_summary` | LLM function-calling summary | `--max-size` |
| `pipeline_masking_summarizing` | Masking + LLM summary | combined args |

All LLM-based strategies support any provider via [litellm](https://github.com/BerriAI/litellm).

## Memory Reward Model (MemRM)

MemRM is a Qwen3-1.7B classifier (QLoRA NF4) that scores whether a proposed agent action is *consistent with* the trajectory's recorded memory. It exposes:

- A Python API — `memgym.training.models.world_model.MemoryWorldModel.from_checkpoint(...)`, plus `WorldModelGate` (inference-time gate) and `WorldModelEvaluator` (batch eval).
- A console script — `memgym-eval-rm` — that downloads the checkpoint from Hugging Face and reproduces the paper's per-dataset AUROC / ECE / coverage table.

```bash
uv pip install -e ".[eval]"   # bootstrap uv once with `pip install uv` or `pipx install uv`

# Reproduce paper Tab. tab:memrm IID-heldout (n=3,007)
memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b

# Sweep all six RM-compatible datasets and print a markdown table
memgym-eval-rm --dataset all --checkpoint MemGym/memgym-rm-1p7b \
    --output results/memrm_table.json

# Evaluate on the training set as a sanity check (paper §app:memrm)
memgym-eval-rm --dataset train-sanity --checkpoint MemGym/memgym-rm-1p7b
```

Registered short names: `iid-heldout`, `train-sanity`, `scenario-ood-tau2`, `scenario-ood-wa-long`, `scenario-ood-webarena`, `strategy-ood`. Pass a local `*.jsonl` path to evaluate on custom data.

> **Note on `strategy-ood`.** The released artifact (`MemGym/memgym-rm-strategy-ood`) is a 22-row covered subset (2 slices that passed the data-integrity check). The CLI reports per-slice AUROC honestly and prints a warning that this *does not* reproduce the paper's headline n=166 number — see the dataset card's "Known Limitations" section.

Model card: [`MemGym/memgym-rm-1p7b`](https://huggingface.co/MemGym/memgym-rm-1p7b).
Full config, metrics, and reproduction details: [`docs/memrm.md`](docs/memrm.md).

### Evaluating your own memory against MemRM

Third-party memories register through the same `BaseMemoryManager` interface the built-in strategies use (`src/memgym/memory/base.py`). After registering, score the memory with `memgym-eval-memory`:

```python
# my_memory.py
from memgym.memory.base import BaseMemoryManager, FilteredContext, register_memory_model

class MyMemory(BaseMemoryManager):
    def manage_context(self, original_context, current_observation, metadata=None):
        ...
        return FilteredContext(content=..., metadata={"tokens": ..., "strategy": "mine"})
    def reset(self) -> None:
        ...

register_memory_model("my_memory", MyMemory)
```

```bash
# Score the new memory on a raw-trajectory JSONL (schema documented in
# `src/memgym/training/eval/memory_eval.py`)
memgym-eval-memory \
    --memory-model my_memory \
    --memory-module my_memory \
    --trajectories sample_trajectories.jsonl \
    --checkpoint MemGym/memgym-rm-1p7b \
    --output results/my_memory.json

# Compare against the passthrough baseline on the same JSONL
memgym-eval-memory \
    --memory-model passthrough \
    --trajectories sample_trajectories.jsonl \
    --checkpoint MemGym/memgym-rm-1p7b \
    --output results/passthrough.json
```

The reported AUROC is a *relative* memory-quality metric — the RM was trained on action-pair prompts without a memory-output block, so absolute numbers drift from the paper's MemRM table. Use the delta between two memories on the same JSONL to read memory quality. The CLI prints this caveat on every run.

For an end-to-end "memory ON vs memory OFF" comparison on real coding instances (Docker + LLM API key + ~hours), see `src/memgym/training/scripts/eval_memory_ab.py`.

## In-house Synthetic Tracks (MemGym-DR & MemGym-CodeQA)

These two tracks are generated by length-controllable pipelines (paper §3.4).
Both pipelines import `unidiff`/`tenacity`/`datasets`, so install the `[swe]`
extra first: `uv pip install -e ".[swe]"`.

**MemGym-DR (deep-research, multi-hop QA).** Generate a small dataset, then
benchmark memory strategies across hop strata:

```bash
# 1) Generate a tiny DR dataset (needs an LLM key; --dry-run skips model calls).
#    Writes data/dr_smoke/memgym_ir_instances.jsonl (one file, mixed hop counts).
python -m memgym.pipelines.memgym_ir dataset --limit 2 --output data/dr_smoke

# 2) Benchmark IR memory strategies. run_ir_benchmark.py takes one JSONL per hop
#    stratum (--data-3hop / --data-4hop / --data-56hop); at least one is required.
#    Split memgym_ir_instances.jsonl by its hop field, or pass the file to the
#    stratum that matches your generated instances.
python scripts/run_ir_benchmark.py \
    --data-4hop data/dr_smoke/memgym_ir_instances.jsonl \
    --strategies ir_bm25,ir_naive_rag,ir_summarizing \
    --limit 2 --output data/dr_results.json
```

IR strategy names map to the paper's MemGym-DR baselines: `ir_bm25` (BM25),
`ir_naive_rag` (Naive RAG), `ir_memorybank` (MemoryBank), `ir_simplemem`
(SimpleMem), `ir_lightmem` (LightMem), `ir_amem` (A-Mem; needs
`requirements-amem.txt`), `ir_passthrough` (None). `--list_memory_models` on any
gym entrypoint enumerates all registered names.

**MemGym-CodeQA (evicted-protocol coding QA).** Generate verified QA items from
SWE-smith instances:

```bash
python -m memgym.pipelines.coding_synthetic \
    --limit 2 --worker-model gpt-4o-mini --verifier-model gpt-4o-mini \
    --output output/codeqa_smoke
```

## Data & Artifacts

MemGym publishes its corpora and the MemRM checkpoint on the Hugging Face Hub
under [`MemGym/`](https://huggingface.co/MemGym). Full schemas, row counts, and
load snippets are in **[`docs/data.md`](docs/data.md)**; the MemRM model details
are in **[`docs/memrm.md`](docs/memrm.md)**.

| HF repo (`MemGym/…`) | What | Rows | License | Status |
|---|---|---|---|---|
| `memgym-rm-1p7b` | MemRM QLoRA checkpoint | 24.5 MB | Apache-2.0 | Public |
| `memgym-rm-train` + `memgym-rm-iid-heldout` | Paired-trajectory corpus (train + eval) | 15,630 + 3,007 | MIT | Public |
| `memgym-rm-scenario-ood-*`, `memgym-rm-strategy-ood` | MemRM OOD eval splits | 6,768 | MIT | Public |
| `memgym-dr-instances` | MemGym-DR deep-research instances | 1,194 | MIT | Public |
| `memgym-codeqa-instances` | MemGym-CodeQA coding-QA instances | 4,289 | MIT | Pending¹ |

¹ Card published; data withheld pending a collaborating institution's review (the generation pipeline is usable today). All released data is plain MIT — see [`docs/licenses.md`](docs/licenses.md).

## SWE-bench Evaluation

### Running the agent

```bash
export OPENAI_API_KEY='sk-...'

# Baseline (no memory)
python scripts/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory none \
    --slice 0:50 --workers 4 -v -o results/baseline

# LLM summarizing (OpenHands default)
python scripts/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory llm_summarizing \
    --max-size 100 --keep-first 1 \
    --slice 0:50 --workers 4 -v -o results/llm_summarizing

# Adaptive token-budget memory
python scripts/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory adaptive_token_budget \
    --max-tokens 4000 --keep-first 2 --keep-recent 8 \
    --slice 0:50 --workers 4 -v -o results/adaptive_token_budget
```

### CodeAct agent (OpenHands)

Use `--agent codeact` to run with the OpenHands CodeAct scaffold. Memory is handled by OpenHands' native condensers (mapped from the same `--memory` CLI args):

```bash
python -m memgym.gym.swe_bench \
    --model openai/gpt-4o-mini --agent codeact \
    --memory llm_summarizing --max-size 100 \
    --dataset swe-gym --instances getmoto__moto-7365 \
    --step-limit 50 --skip-eval -v -o results/codeact
```

### Evaluating patches

`scripts/evaluate_swe_bench.py` invokes the **official SWE-bench harness** (`swebench.harness.run_evaluation`) end-to-end — the same pipeline used by `sb-cli` and the SWE-bench leaderboard. Tests run inside Docker containers with project-specific test runners and log parsers.

To re-evaluate an existing `preds.json` without re-running the agent, call the harness directly:

```bash
python -m swebench.harness.run_evaluation \
    --predictions_path results/baseline/preds.json \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --max_workers 4 --run_id my-eval
```

Results are written to `./logs/run_evaluation/<run_id>/` and a summary JSON report.

### Results (GPT-5, SWE-bench Lite first 50)

| Strategy | Resolved | Rate | Notes |
|----------|----------|------|-------|
| Baseline (no memory) | 36/50 | 72% | Matches sb-cli exactly |
| OpenHands `llm_summarizing` | 35/50 | 70% | -2% vs baseline |

The memory wrapper has minimal impact here because conversations are short (~16-32 messages), well below the `--max-size 100` summarization threshold.

### Strategy-Specific CLI Args

| Arg | Default | Strategies | Description |
|-----|---------|-----------|-------------|
| `--max-size` | 100 | llm_summarizing, structured_summary | Max events before condensation |
| `--keep-first` | 1 | llm_summarizing, structured_summary | Events to always keep from start |
| `--attention-window` | 100 | observation_masking | Recent observations to keep |
| `--summarization-model` | gpt-4o-mini | all LLM-based | Model for summarization calls |
| `--max-tokens` | 4000 | naive, adaptive_token_budget | Token threshold for summarization |
| `--keep-recent` | 8 | adaptive_token_budget | Recent raw messages to preserve verbatim |
| `--preserve-first-user` | true | adaptive_token_budget | Keep the first user issue statement pinned |

## Output Structure

```
results/<run_name>/
  preds.json                              # SWE-bench submission format
  summary.json                            # Aggregate stats
  trajectories/
    <instance_id>.traj.json               # Full agent trajectory
    <instance_id>.traj_concise.json       # Messages only
    <instance_id>_training.json           # Step-based CoT training data
```

### Training Trajectory Format

`_training.json` — compact step-based format for CoT training:

```json
{
  "instance_id": "astropy__astropy-12907",
  "repo": "astropy/astropy",
  "problem": "CompoundModel separability matrix is wrong...",
  "outcome": "resolved",
  "num_steps": 8,
  "steps": [
    {
      "step": 1,
      "thought": "I need to find the separability code...",
      "action": "find /testbed -name '*.py' | grep separab",
      "observation": "astropy/modeling/separable.py",
      "memory": null
    }
  ],
  "patch": "diff --git a/..."
}
```

## Documentation

Start with [`docs/`](docs/README.md) — the documentation index. Highlights:

- [docs/data.md](docs/data.md) — **Data catalog**: every released dataset, schema, license, and load snippet
- [docs/memrm.md](docs/memrm.md) — MemRM config, metrics, and reproduction
- [docs/tracks.md](docs/tracks.md) — the five tracks and how to run each
- [docs/licenses.md](docs/licenses.md) — full license matrix
- [QUICKSTART.md](QUICKSTART.md) — install + first runs
- [TESTING.md](TESTING.md) — tier-labeled, copy-pasteable manual test commands
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — smoke-run transcript; what each eval tier needs
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design

## License

Licenses vary by artifact — full matrix in [docs/licenses.md](docs/licenses.md):

- **Code** (this repository): Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **Data & synthetic instances** (the paired-trajectory corpus, MemGym-DR /
  MemGym-CodeQA): MIT. Every released MemGym-DR row is `deep_research`-derived
  synthetic text, so no Wikipedia content (and no CC-BY-SA-4.0 obligation) ships.
- **MemRM weights** (`MemGym/memgym-rm-1p7b`): Apache-2.0, inherited from the
  Qwen3-1.7B-Base model.
