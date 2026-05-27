# Changelog

All notable changes to MemGym are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — repository reorganization

Structural cleanup distilling the layout conventions of
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (library core) and
[verl](https://github.com/volcengine/verl) (training/examples split). No public CLI
or `--memory` strategy name changes; internal import paths were rationalized.

### Added

- `pyproject.toml` (PEP 621) with a `[tool.ruff]` block (line-length 120); `setup.py`
  reduced to a thin editable-install shim.
- `CONTRIBUTING.md` — dev setup, the `examples/` vs `scripts/` split, and the
  one-call recipe for registering a new memory strategy.
- `assets/` — slot for an optional README hero banner (`assets/banner.png`).

### Removed

- Legacy and one-off diagnostic code deleted to keep the public tree minimal. The
  paper's reproduction home is the **private** `MemGym` repo, so nothing here was
  load-bearing for results. **Everything removed is recoverable from git history at
  commit `3f867062a40993e6e1021960d2246727465718e3`** (the last commit before this
  reorganization):
  - `src/memgym/pipelines/memgym_ir/` flat backward-compat shims (real code lives in
    the `eval/`, `generators/`, `orchestrators/`, `utils/` subpackages).
  - `src/memgym/training/scripts/probe_*.py` (25 environment/schema diagnostics),
    `audit_*.py`, and `inventory_*.py` one-offs.
  - `src/memgym/pipelines/coding_synthetic/eval/expanded_output{,2,3,4}/` pilot dumps.
  - `src/memgym/pipelines/memory_training/` (superseded by `src/memgym/training/data/`).
  - Deprecated `src/memgym/runner.py` (use `memgym.runners`); the unused
    `adapter/llm_summarization_manager.py` and `eval/docker_prune.py`.
  - The suppressed reinforcement-learning / mid-training stack the paper's v1 does
    not claim: `src/memgym/training/rl/` (GRPO loop, verl agent-loop, env reward,
    FSDP wrap), `training/models/world_model_gate.py`, the `midtrain_sft_pairs*.py`
    and summarizer-pair/dataset builders, and the internal handoff/diagnostic
    launchers under `training/scripts/`. The released training surface is the
    **MemRM** reward-model stack only (`MemoryWorldModel`, `WorldModelEvaluator`,
    SFT augmentation, and the corpus builders behind the `memgym-eval-rm` /
    `memgym-eval-memory` CLIs).
  - The unused `WorldModelGate` class and its tests — the MemRM API ships
    `MemoryWorldModel` + `WorldModelEvaluator` only.

### Changed

- `memory/` reorganized into `strategies/`, `backends/`, `adapters/`, `external/`, `ir/`.
- `utility/` renamed to `utils/`; absorbed `trajectory_recorder` and `swegym_specs`.
- Runnable per-track scripts moved to top-level `examples/`; dev/ops utilities and
  cluster scripts consolidated under `scripts/`.
- Root-level guides (`ARCHITECTURE`, `QUICKSTART`, `TESTING`, `REPRODUCIBILITY`) moved
  into `docs/` (lowercased); the heavy SWE-bench walkthrough extracted to
  `docs/swe_bench.md`; `README.md` trimmed to a ~150-line orientation page with two
  Mermaid diagrams.
- `src/memgym/docs/datasets.md` moved to `docs/datasets-upstream.md` so the installed
  package ships no documentation.
- Orphan `coding_synthetic/tests/*` moved into `tests/unit/` (now collected by CI);
  test fixtures de-duplicated under `tests/fixtures/`.
- Packaging migrated to `pyproject.toml` (same three console scripts).

### Fixed

- Repaired a botched global find-replace that had turned internal phase labels into
  the ungrammatical string "the baseline-train phase" across the WebArena runner,
  the WebArena evaluator, and the MemGym-DR pipeline.

### Security

- Scrubbed internal-infrastructure references from the public source: deployment
  host names and a job-runner URL (now generic `<runner-host>` / `<YOUR_HOST>`
  placeholders), internal experimental-plan codenames (`Phase A/B/C`) replaced with
  behavior-describing names, collaborator/monorepo-path mentions, and orphaned prose
  referencing the now-removed GRPO training loop.

## [0.2.0] — 2026-05-19 — first public release

Companion to the NeurIPS 2026 submission. Establishes the reviewer-readiness
baseline: every paper claim has either a runnable script, a unit test, or an
acknowledged limitation in the docs.

### Added

- **Five evaluation tracks.** SWE-Gym (`memgym.gym.swe_bench`), tau2-bench
  (`memgym.gym.tau2_bench`), WebArena (`memgym.gym.webarena`), MemGymCodeQA
  (`memgym.pipelines.coding_synthetic`), and MemGymDR
  (`memgym.pipelines.memgym_ir`).
- **Seven memory strategies.** `passthrough`, `naive_summarization`,
  `llm_summarizing`, `structured_summary`, `observation_masking`,
  `adaptive_token_budget`, `sliding_window_summary`, plus the
  `PipelineMemory` composer.
- **Memory Reward Model (MemRM).** `memgym.training.models.world_model`
  (`MemoryWorldModel`, `WorldModelGate`, `WorldModelEvaluator`), checkpoint
  published as [`MemGym/memgym-rm-1p7b`](https://huggingface.co/MemGym/memgym-rm-1p7b).
- **`memgym-eval-rm` console script.** Single entry-point that reproduces
  the paper's MemRM table over the six RM-compatible Hugging Face datasets
  (`iid-heldout`, `train-sanity`, `scenario-ood-tau2`,
  `scenario-ood-wa-long`, `scenario-ood-webarena`, `strategy-ood`).
  Reports AUROC with bootstrap CI, ECE, accuracy/coverage at the paper's
  threshold (t*=0.88).
- **`memgym-eval-memory` console script.** Lightweight memory-quality
  probe: applies a registered memory strategy to raw trajectory steps,
  prepends the compressed view to the standard pair-prompt template,
  scores with the MemRM. Designed for third parties testing their own
  `BaseMemoryManager` subclass — `--memory-module` imports user code so
  `register_memory_model(...)` is in effect before lookup. The CLI
  prints a "RELATIVE METRIC" banner: AUROC here is OOD vs the MemRM's
  training prompt and should be read as a delta between memories on
  the same JSONL.
- **Unit-test suite.** `tests/unit/` covers the MemRM CLI
  (`test_rm_eval.py`), the world-model + gate plumbing
  (`test_world_model.py`), round-trip behaviour of every registered memory
  strategy (`test_memory_strategies_roundtrip.py`), and the five known
  contract bugs from internal tracking (`test_known_bugs.py`, four passing
  + one `xfail`).
- **Integration smoke tests.** `tests/integration/test_track_smoke.py`
  exercises one passthrough run per track, gated behind
  `@pytest.mark.integration` so CI does not need Docker / WebArena /
  GPU.
- **GitHub Actions CI.** `.github/workflows/ci.yml` runs the unit suite on
  Python 3.10 and 3.12 and smoke-tests the `memgym-eval-rm --help` entry
  point on every PR.
- **Docs.** Tracks summary table and MemRM section in `README.md`; MemRM
  + WebArena quickstart blocks in `QUICKSTART.md`.

### Changed

- `setup.py` license classifier corrected from MIT to Apache-2.0 (matches
  `LICENSE`).
- `setup.py` ships a new `[eval]` extras group (`huggingface-hub`,
  `datasets`, `scikit-learn`) so reviewers can install only what the
  MemRM CLI needs.

### Removed

- Twenty-three internal `(see paper)` annotations left behind by the
  release scrubber. Replaced with concrete phrasing or removed entirely so
  the public source compiles cleanly on its own.

### Known limitations

- `strategy-ood` artifact is a 22-row covered subset and **does not**
  reproduce the paper's n=166 headline (per-slice numbers only). See the
  dataset card's "Known Limitations" section and the CLI's runtime
  warning.
- One open contract bug remains documented as `xfail`: the
  `mini-swe-agent` config-fallback path in `gym/swe_bench/env.py` still
  uses `print()` rather than `warnings.warn`.
