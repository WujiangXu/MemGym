# Reproducibility Smoke Run

Transcript of the structural-smoke audit run against a clean install on
2026-05-20. The goal: prove every command in `README.md` and
`quickstart.md` exists, parses its arguments, and exits cleanly *without*
needing an LLM key, Docker, a WebArena server, or a GPU. This is the
"cheap tier" — full eval reproduction needs the resources documented in
each track's section of the README.

The audit policy is **fix on red, do not fudge**: any `--help` failure
means the docs reference something that no longer exists; the doc is
patched, the test is not.

## Environment

```bash
python3 -m venv ~/.venvs/memgym-smoke
source ~/.venvs/memgym-smoke/bin/activate
python -m pip install --upgrade pip uv
uv pip install -e ".[dev,eval]"
```

- Python 3.12.3
- uv 0.11.15
- memgym 0.2.0 with the `dev` and `eval` extras (matches `.github/workflows/ci.yml`)

The `[swe]`, `[train]`, and `[rl-way-a]` extras are intentionally
excluded — they pull in heavy deps (`swebench`, `vllm`, `flash-attn`,
`unidiff`) that are not part of the reviewer-readiness surface and would
mask install-path bugs the CI install is meant to catch.

## Tier 1 — structural smoke (~2 min, no external systems)

| # | Command | Result |
|---|---|---|
| 1 | `pytest tests/unit/test_imports.py` | 4 passed |
| 2 | `python -m memgym.gym.tau2_bench --help` | exits 0; argparse prints `--domains`, `--task_ids`, `--task_split_name` |
| 3 | `python -m memgym.gym.tau2_bench.install --help` | **eagerly runs `pip install`** instead of parsing `--help` — see known issues below |
| 4 | `python -m memgym.gym.swe_bench --help` | exits 0; argparse prints `--dataset {lite,verified,full,swe-gym,swe-smith}`, `--slice` |
| 5 | `python examples/swe_bench/evaluate_swe_bench.py --help` | exits 0; matches the module form |
| 6 | `python -m memgym.gym.webarena --help` | exits 0; argparse prints `--webarena_dir`, `--app_name`, `--task_ids` |
| 7 | `memgym-eval-rm --help` | exits 0; required args `--dataset` and `--checkpoint`, threshold defaults to paper t*=0.88 |
| 7b | `memgym-eval-memory --help` | exits 0; required args `--memory-model`, `--trajectories`, `--checkpoint` |
| 8 | `pytest tests/integration/ --collect-only` | 28 tests collected (smoke-only, all gated behind `@pytest.mark.integration`) |

### Targeted release-critical unit suite

```bash
pytest tests/unit/test_imports.py tests/unit/test_rm_eval.py \
       tests/unit/test_world_model.py tests/unit/test_memory_strategies_roundtrip.py \
       tests/unit/test_known_bugs.py tests/unit/test_memory_eval.py -q
```

→ **66 passed, 1 xfailed**.

The xfail is the documented open contract bug in
`gym/swe_bench/env.py` (config fallback uses `print()` instead of
`warnings.warn`). All known-bug regression tests pass; the xfail is one extra.

## Known issues surfaced by the smoke run

1. **`python -m memgym.gym.tau2_bench.install --help` is not a true
   `--help`.** The module runs the install pipeline immediately on
   import; argparse never gets to intercept the `--help` flag. This is
   a usability bug in the installer, not a release blocker — the
   documented form `python -m memgym.gym.tau2_bench.install --venv
   <path>` works. Tracking: noted for a future PR.
2. **Whole-suite `pytest tests/unit -q` on `[dev,eval]`**: 286 passed,
   15 skipped, 1 xfailed. The skips are tau2 trajectory-loader tests
   that need a captured-episode fixture (not committed; regenerate
   locally per the docstring at `tests/unit/test_tau2_trajectory_loader.py`).
   The rest of the suite is green without extras beyond `[dev,eval]`.
3. **stderr noise from optional deps.** `litellm` warns about missing
   `botocore` and `memgym.pipelines.memgym_ir.ir_amem` warns about
   missing `sentence_transformers`. Both are optional, both are
   documented as "install the relevant extras if you need them," and
   both go to stderr only — no command exits non-zero.

## What is *not* covered by this run

These tiers were deferred — they need resources the smoke venv cannot
provide:

- **Tier 2** — one-task tau2 mock dry run. Needs `OPENAI_API_KEY`. ~$0.01.
- **Tier 3** — one-instance SWE-bench eval + 8-row MemRM eval against
  the HF checkpoint. Needs Docker, an LLM API key, and one GPU + ~3.4
  GB HF download.
- **Tier 4** — full reproduction matrix (`memgym-eval-rm --dataset all`,
  full WebArena 1100-task sweep, full SWE-bench Lite). This is the
  *paper's* reproduction surface and is documented per-track in
  `README.md`.

A reviewer can run Tiers 2 and 3 with the commands already in
`quickstart.md`; this document's purpose is to prove the install path
those commands sit on top of is itself clean.
