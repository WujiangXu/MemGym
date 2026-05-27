# SWE-bench / SWE-Gym Evaluation

How to run the MemGym SWE-bench agent, evaluate patches with the official harness, and interpret the output structure.

## Running the agent

```bash
export OPENAI_API_KEY='sk-...'

# Baseline (no memory)
python examples/swe_bench/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory none \
    --slice 0:50 --workers 4 -v -o results/baseline

# LLM summarizing (OpenHands default)
python examples/swe_bench/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory llm_summarizing \
    --max-size 100 --keep-first 1 \
    --slice 0:50 --workers 4 -v -o results/llm_summarizing

# Adaptive token-budget memory
python examples/swe_bench/evaluate_swe_bench.py \
    --model openai/gpt-5 --memory adaptive_token_budget \
    --max-tokens 4000 --keep-first 2 --keep-recent 8 \
    --slice 0:50 --workers 4 -v -o results/adaptive_token_budget
```

## CodeAct agent (OpenHands)

Use `--agent codeact` to run with the OpenHands CodeAct scaffold. Memory is handled by OpenHands' native condensers (mapped from the same `--memory` CLI args):

```bash
python -m memgym.gym.swe_bench \
    --model openai/gpt-4o-mini --agent codeact \
    --memory llm_summarizing --max-size 100 \
    --dataset swe-gym --instances getmoto__moto-7365 \
    --step-limit 50 --skip-eval -v -o results/codeact
```

## Evaluating patches

`examples/swe_bench/evaluate_swe_bench.py` invokes the **official SWE-bench harness** (`swebench.harness.run_evaluation`) end-to-end — the same pipeline used by `sb-cli` and the SWE-bench leaderboard. Tests run inside Docker containers with project-specific test runners and log parsers.

To re-evaluate an existing `preds.json` without re-running the agent, call the harness directly:

```bash
python -m swebench.harness.run_evaluation \
    --predictions_path results/baseline/preds.json \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --max_workers 4 --run_id my-eval
```

Results are written to `./logs/run_evaluation/<run_id>/` and a summary JSON report.

## Results (GPT-5, SWE-bench Lite first 50)

| Strategy | Resolved | Rate | Notes |
|----------|----------|------|-------|
| Baseline (no memory) | 36/50 | 72% | Matches sb-cli exactly |
| OpenHands `llm_summarizing` | 35/50 | 70% | -2% vs baseline |

The memory wrapper has minimal impact here because conversations are short (~16-32 messages), well below the `--max-size 100` summarization threshold.

## Strategy-Specific CLI Args

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

## Training Trajectory Format

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
