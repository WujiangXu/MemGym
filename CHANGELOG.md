# Changelog

All notable changes to MemGym are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
