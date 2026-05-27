# MemGym Tracks

MemGym evaluates memory across five agent tracks. Three **wrap** existing
benchmarks; two are **in-house synthetic** pipelines with length-controllable
difficulty. This page is the map from track → module → data → run command. For
copy-pasteable, tier-labeled commands see [`../TESTING.md`](../TESTING.md); for
the data each track produces or consumes see [data.md](data.md).

| Track | Module | Data source | Paper |
|---|---|---|---|
| SWE-Gym | `memgym.gym.swe_bench` | SWE-Gym / SWE-bench Lite | Tab. `tab:main` |
| τ²-bench | `memgym.gym.tau2_bench` | mock / telecom / airline / retail | Tab. `tab:main` |
| WebArena-Infinity | `memgym.gym.webarena` | WebArena-Infinity | Tab. `tab:main` |
| MemGym-DR | `memgym.pipelines.memgym_ir` | [`memgym-dr-instances`](data.md#3-memgym-dr--deep-research-instances) | Fig. 2b |
| MemGym-CodeQA | `memgym.pipelines.coding_synthetic` | [`memgym-codeqa-instances`](data.md#4-memgym-codeqa--coding-qa-instances--pending) | Fig. 2a |

## Wrapped tracks

### SWE-Gym (code repair)
Runs an agent on SWE-bench/SWE-Gym instances and evaluates patches with the
**official `swebench` harness** (same pipeline as sb-cli / the leaderboard).
Entry: `python scripts/evaluate_swe_bench.py … -o results/<name>` (`-o` required;
`--dataset {lite,verified,full,swe-gym,swe-smith}`). See README "SWE-bench
Evaluation" and `TESTING.md` Tier 3.

### τ²-bench (dialogue)
Wraps `sierra-research/tau2-bench` agents. Solo-mode domains (mock, telecom) run
a single agent; airline/retail add a user simulator, and `--memory_side` selects
whose memory is wrapped. Entry: `python -m memgym.gym.tau2_bench --domains …
--agent_llm …`. See `QUICKSTART.md` and `TESTING.md` Tier 2.

### WebArena-Infinity (web navigation)
Wraps WebArena-Infinity; needs the package installed
(`python -m memgym.gym.webarena.install`) and a running WebArena server
(`WEBARENA_BASE_URL`). Entry: `python -m memgym.gym.webarena --policy_model …
--app_name …` (both required). See `TESTING.md` Tier 3.

## In-house synthetic tracks

Both pipelines import `unidiff`/`tenacity`/`datasets`, so install the `[swe]`
extra first: `uv pip install -e ".[swe]"`.

### MemGym-DR
Length-controllable multi-hop deep-research QA (paper §3.4, Fig. 2b). Generate,
then benchmark IR memory strategies across hop strata:

```bash
# Generate -> data/dr_smoke/memgym_ir_instances.jsonl (mixed hop counts)
python -m memgym.pipelines.memgym_ir dataset --limit 2 --output data/dr_smoke

# Benchmark (one JSONL per hop stratum: --data-3hop/--data-4hop/--data-56hop)
python scripts/run_ir_benchmark.py \
    --data-4hop data/dr_smoke/memgym_ir_instances.jsonl \
    --strategies ir_bm25,ir_naive_rag,ir_summarizing \
    --limit 2 --output data/dr_results.json
```

IR strategy names: `ir_bm25`, `ir_naive_rag`, `ir_memorybank`, `ir_simplemem`,
`ir_lightmem`, `ir_amem` (needs `requirements-amem.txt`), `ir_passthrough`.
Released instances: [`memgym-dr-instances`](data.md#3-memgym-dr--deep-research-instances).

### MemGym-CodeQA
Evicted-protocol coding QA generated from SWE-smith bugs (paper §3.4, Fig. 2a):

```bash
python -m memgym.pipelines.coding_synthetic \
    --limit 2 --worker-model gpt-4o-mini --verifier-model gpt-4o-mini \
    --output output/codeqa_smoke
```

A three-check verifier (solvability / distractor-confusion / question-leakage)
runs unless you pass `--skip-verification`. The released instance set
([`memgym-codeqa-instances`](data.md#4-memgym-codeqa--coding-qa-instances--pending))
is pending review, but the pipeline runs today.
