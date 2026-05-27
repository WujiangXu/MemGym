# Examples

Runnable, copy-pasteable how-tos for each MemGym evaluation track. These assume
the package is installed in editable mode:

```bash
pip install -e ".[dev]"   # add track extras as needed, e.g. .[swe], .[tau2], .[eval]
```

Examples import `memgym` as an installed package (no `sys.path` hacks), so run
them from anywhere once installed.

| Path | Track | What it does |
|------|-------|--------------|
| `run_episode.py` | any (`--env {tau2,swe}`) | Run one episode end-to-end and dump the trajectory. |
| `swe_bench/evaluate_swe_bench.py` | SWE-Gym | One strategy over a dataset slice (== `memgym-evaluate`). |
| `swe_bench/run_all_strategies.sh` | SWE-Gym | Sweep every strategy on the same slice, then print a comparison table. |
| `swe_bench/run_official_eval.sh` | SWE-Gym | Score predictions with the official `swebench` Docker harness. |
| `swe_bench/compare.py` | SWE-Gym | Tabulate results across a comparison run directory. |
| `swe_bench/replay.py` | SWE-Gym | Replay / re-score a recorded trajectory. |
| `memgym_dr/run_ir_benchmark.py` | MemGymDR (IR) | Benchmark IR memory methods across hop-depth strata. |
| `memgym_codeqa/run_pipeline.py` | MemGymCodeQA | Run the coding-synthetic generation pipeline. |
| `memgym_codeqa/eval_solvability.py` | MemGymCodeQA | Solvability check for generated instances. |

For dev/ops tooling (GPU cleanup, config dumps, HF dataset staging) see
[`../scripts/`](../scripts/). For per-track flag references see [`../docs/`](../docs/).
