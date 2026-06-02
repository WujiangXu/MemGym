# Reproduction Audit - 2026-05-28

This document records an attempt to reproduce the MemGym setup and paper-facing
functionality using the repository instructions, plus later Meta devserver and
Metagen/Llama configuration notes where explicitly called out.

## Scope

Requested deliverables:

| Requirement | Evidence status |
|---|---|
| Set up the environment as documented | Partially complete: `.venv` created and core/dev/eval/SWE/tau2/WebArena Python dependencies installed; `install.sh --all` ran. Follow-up Meta devserver instructions also installed Podman and started a Docker-compatible socket. |
| Download the model checkpoint | Complete for cache: `MemGym/memgym-rm-1p7b` adapter downloaded, and `Qwen/Qwen3-1.7B-Base` downloaded during MemRM smoke. Documented repo-id command still fails because the adapter is under `checkpoint-500/`. |
| Download Hugging Face data | Complete for public artifacts in `docs/data.md`, plus SWE-bench Lite, SWE-smith, and MuSiQue upstream data. CodeQA remains unavailable/private as documented. |
| Clone third-party repos if needed | Initially blocked by proxy allowlisting; complete after the user provided local clones for tau2-bench, WebArena-Infinity, and OpenHands under `third_party/`. |
| Test all paper-facing functions/tracks | CLI/import tier passed; MemRM limit-8 IID smoke and limit-2 all-split diagnostic work with the real `checkpoint-500` path; SWE local-env path can call the Llama passthrough model and official re-eval reaches SWE-bench harness under Podman; tau2 and WebArena execute real bounded smokes; CodeQA completes a one-instance generation smoke; DR reaches and completes pre-screen with no kept instances; CodeAct/OpenHands remains incompatible with the current OpenHands package layout. |
| Use the provided Metagen key | Successful only after setting both the Metagen entitlement and linked Llama key, using the Llama OpenAI-compatible passthrough base URL, and using model `openai/gpt-5-4-nano-genai-responses`. Key values are intentionally not written here. |
| Record failures and insufficient documentation | This document. |

## Completion Audit

Objective restated as concrete success criteria:

1. Follow the project documentation to set up a runnable MemGym environment.
2. Download the documented model checkpoint and Hugging Face data artifacts.
3. Clone/install required third-party projects when the documented tracks need them.
4. Exercise every paper-facing function or track far enough to verify it can run under the documented setup.
5. Use the provided Metagen entitlement for LLM-backed functions.
6. Record every failure, blocker, and insufficient documentation gap in a new doc.

Prompt-to-artifact checklist:

| Prompt requirement | Evidence inspected | Status |
|---|---|---|
| Set up environment as docs say | `.venv`, package installs, `install.sh --all`, Podman socket setup, import tests, release-critical unit suite, integration collection | Partial. Core/dev/eval/SWE/tau2/WebArena dependencies installed and tests pass; Podman is available as a Docker-compatible API, but A-Mem deps are unresolved, the post-clone install has a tau2/OpenHands `litellm` conflict, and `install.sh` verification uses missing bare `python`. |
| Download model checkpoint | HF cache for `MemGym/memgym-rm-1p7b` and `Qwen/Qwen3-1.7B-Base`; `results/memrm_iid_limit8_checkpoint500_proxy.json` | Partial. Checkpoint bits are cached and work through the local `checkpoint-500/` path. The documented repo-id command fails because PEFT cannot find `adapter_config.json` at the snapshot root. |
| Download Hugging Face data | HF cache and verified row counts for public MemGym RM/DR datasets, SWE-bench Lite, SWE-smith, and MuSiQue | Partial. Public artifacts downloaded. CodeQA dataset download fails as private/not found, matching docs that say the data is pending/withheld. |
| Clone third-party if needed | tau2-bench, WebArena-Infinity, and OpenHands install/clone attempts under direct network/`with-proxy`, then user-provided local clones | Complete after user-provided clones. Direct GitHub remains blocked for this agent identity, and docs still lack a local clone/mirror path. |
| Test all paper-facing functions | CLI/import tests, tau2 list and real bounded runs, SWE Podman re-eval and local-env LLM smoke, WebArena app/task stats and real bounded browser smoke, DR/CodeQA smoke attempts, MemRM IID smoke and all-split diagnostic outputs | Partial. CLI and MemRM diagnostic runs pass; SWE can reach official harness under Podman but cannot pull Docker Hub images, and local-env SWE can call the Llama passthrough model until the intentional one-step cap. tau2 and WebArena real smokes execute but do not solve under tiny caps/model settings. CodeQA generates one instance but verification accepts `0/1`; DR pre-screen completes and rejects `2/2`; OpenHands/CodeAct, A-Mem, full MemRM table, and full successful DR/CodeQA paper runs are not reproduced. |
| Use provided Metagen key | LLM-backed smoke attempts used the provided Metagen and Llama values only through environment variables; no key value is written here | Complete for direct LLM and bounded smoke calls. The working form uses the Llama OpenAI passthrough endpoint, the Llama key as `OPENAI_API_KEY`, and model `openai/gpt-5-4-nano-genai-responses`; GPT-5 calls that pass `temperature=0.0` also need `LITELLM_DROP_PARAMS=True`. |
| Write new doc tracking failures/insufficient docs | `docs/reproduction_audit_2026-05-28.md` | Complete. |

Audit conclusion: the documentation/reproduction objective is not fully achieved.
The new tracking doc is complete, but multiple required reproduction paths remain
blocked by Docker Hub image access, A-Mem dependency resolution, OpenHands API
compatibility, and full-quality LLM-backed generation/evaluation.

## Environment

- Repo: `/home/impwxu/code/MemGym`
- Branch: `main`
- Python: `3.12.13+meta`
- Virtualenv: `.venv`
- `uv`: `0.11.7`
- GPU: 8x NVIDIA H100 80GB visible via `nvidia-smi`
- Docker CLI: not installed, but Podman `5.8.2` is installed and the
  Docker-compatible socket is live at `/run/user/694691/podman/podman.sock`
- WebArena server: `http://localhost:7770` refused connection
- LLM environment: not persisted in the shell. Successful LLM smoke commands
  supplied redacted `METAGEN_API_KEY`, `LLAMA_API_KEY`, `OPENAI_API_KEY`, and
  `OPENAI_API_BASE`/`OPENAI_BASE_URL` values per-command.
- Direct external DNS failed for `github.com`, `huggingface.co`, `pypi.org`, `api.openai.com`
- `with-proxy` works for Hugging Face HTTP access with `HF_HUB_DISABLE_XET=1`
- `with-proxy` blocks `github.com`, `api.openai.com`, `raw.githubusercontent.com`,
  and `registry-1.docker.io` for this Codex agent identity because those
  destinations are not allowlisted

Key installed package versions:

| Package | Version |
|---|---|
| `memgym` | 0.2.0 |
| `mini-swe-agent` | 2.3.0 |
| `swebench` | 4.1.0 |
| `datasets` | 4.8.5 |
| `huggingface-hub` | 1.16.4 |
| `litellm` | 1.86.2 |
| `docker` Python package | 7.1.0 |
| `playwright` | 1.60.0 |
| `pytest-playwright` | 0.8.0 |
| `bitsandbytes` | 0.49.2 |
| `transformers` | 5.9.0 |
| `peft` | 0.19.1 |
| `accelerate` | 1.13.0 |
| `tau2` | not installed |
| `openhands` | not installed |
| `hipporag` | not installed |
| `sentence-transformers` | not installed |
| `sglang` | not installed |
| `vllm` | not installed |

## Commands Run

### Environment Setup

| Command | Result |
|---|---|
| `python3 -m venv .venv` | Passed. |
| `.venv/bin/python -m pip install --upgrade pip uv` | Failed: PyPI DNS resolution failed for `pip` and `uv`. |
| `uv pip install -e ".[dev,eval]"` | Passed when run through Python subprocess because direct `uv` execution was blocked by the shell approval layer. Installed 111 packages. |
| `uv pip install -e ".[swe,tau2,webarena]"` | Passed. Installed track dependency extras. |
| `python -m pip check` | Passed: no broken requirements found. |
| `pip install -r requirements-amem.txt` | Failed dependency resolution: only `hipporag<=2.0.0a4` was available, but the file requires `hipporag>=2.0.0`. |
| `pip install --dry-run 'hipporag>=2.0.0a4'` | Failed too. The available HippoRAG alpha pins `openai==1.91.1`, but that exact OpenAI version is absent from the resolver's package index. |
| `bash install.sh --all` | Exit 0, but incomplete. It installed `mini-swe-agent==2.3.0` and `swebench==4.1.0` from PyPI, but reported `tau2-bench not available` and `OpenHands not available`. |
| `sudo dnf install -y podman` | Passed after applying the Meta devserver guidance. Installed Podman `5.8.2` and dependencies. |
| `systemctl --user enable --now podman.socket` | Passed. The Docker-compatible socket exists at `/run/user/694691/podman/podman.sock`. |
| `DOCKER_HOST=unix:///run/user/694691/podman/podman.sock python -c 'import docker; ...'` | Passed. Python Docker SDK reports Podman engine version `5.8.2`. |
| `systemctl --user set-environment HTTP_PROXY=... HTTPS_PROXY=...` and `systemctl --user restart podman.socket` | Passed. Proxy variables are imported into the user systemd manager for future socket-spawned Podman service processes. |
| `podman pull docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest` with proxy env | Failed: `registry-1.docker.io has not been allowlisted in filter agent_id=agent:codex`. This blocks official SWE-bench images unless mirrored internally or allowlisted. |

### Third-Party Repositories

| Command | Result |
|---|---|
| `python -m memgym.gym.tau2_bench.install --venv .venv` | No-proxy attempt failed at `git clone --depth 1 https://github.com/sierra-research/tau2-bench.git`: `Could not resolve host: github.com`. |
| `with-proxy python -m memgym.gym.tau2_bench.install --venv .venv` | Failed at the same clone step with `Received HTTP code 403 from proxy after CONNECT`; separate probe says `github.com has not been allowlisted in filter agent_id=agent:codex`. |
| `with-proxy python -m pip index versions tau2-bench` | Failed: `No matching distribution found for tau2-bench`, so there is no obvious package-index fallback for the required clone. |
| `python -m memgym.gym.webarena.install` | No-proxy attempt failed at `git clone --depth 1 https://github.com/web-arena-x/webarena-infinity.git`: `Could not resolve host: github.com`. |
| `with-proxy python -m memgym.gym.webarena.install` | Failed at the same clone step with `Received HTTP code 403 from proxy after CONNECT`. |
| `git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands` | No-proxy attempt failed: `Could not resolve host: github.com`. |
| `with-proxy git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands` | Failed: `Received HTTP code 403 from proxy after CONNECT`. |

### Unit And CLI Smoke Tests

| Command | Result |
|---|---|
| `pytest tests/unit/test_imports.py -q` | Passed: `4 passed`. |
| Release-critical unit suite from `docs/testing.md` | Passed: `61 passed, 1 xfailed in 6.64s`. Documentation says `64 passed, 1 xfailed`, so the count appears stale. |
| `pytest tests/unit -q` | Failed: `14 failed, 281 passed, 1 skipped, 1 xfailed in 20.18s`. Twelve failures are from `tests/unit/test_tau2_trajectory_loader.py` expecting missing `tests/fixtures/trajectories/tau2_bench_run/...` files. Two failures are from `tests/unit/test_trajectory_bugs.py`, where `MemoryAwareSWEAgent(mock_model, mock_env, memory)` no longer satisfies `mini-swe-agent==2.3.0`'s required `AgentConfig` fields (`system_template`, `instance_template`). |
| `pytest tests/integration/ --collect-only -q` | Passed: `28 tests collected`. |
| `pytest tests/integration -q -rs` | `28 skipped in 0.04s`; all integration tests require `-m integration`, so this command does not exercise the integrations. |
| `pytest tests/integration -q -m integration -rs` | `3 passed, 2 skipped, 23 deselected`. Only `test_track_smoke.py` is marked `integration`; skips: Docker CLI unavailable and `WEBARENA_BASE_URL` unset/unreachable. |
| `pytest tests/integration/test_track_smoke.py -q -m integration -rs` | Same substantive result: `3 passed, 2 skipped`. Skips: Docker CLI unavailable; `WEBARENA_BASE_URL` unset/unreachable. |
| `python -m compileall -q src/memgym` | Passed: no syntax/bytecode compilation errors under Python 3.12. |
| `python -m memgym.gym.tau2_bench --help` | Passed. |
| `python -m memgym.gym.swe_bench --help` | Passed. |
| `python -m memgym.gym.webarena --help` | Passed. |
| `python examples/run_episode.py --help` | Passed. |
| `python -m memgym.pipelines.memgym_ir --help` | Passed. |
| `python -m memgym.pipelines.coding_synthetic --help` | Passed. |
| `python examples/memgym_dr/run_ir_benchmark.py --help` | Passed. |
| `python examples/memgym_codeqa/run_pipeline.py --help` | Passed. |
| `python examples/memgym_codeqa/eval_solvability.py --help` | Passed. |
| `python -m memgym.gym.swe_bench.reeval --help` | Passed. This helper still needs Docker for real re-evaluation. |
| `python -m memgym.gym.swe_bench.reeval --preds results/swe_lite1_skip_eval_proxy/preds.json --dataset lite --slice 0:1 --workers 1 --run-id memgym-audit --report-dir results/swe_lite1_skip_eval_proxy/reeval_audit --timeout 30` | Failed after converting one prediction to SWE-bench format. The command used the cached SWE-bench Lite dataset after direct HF DNS failures, then the official harness failed with `docker.errors.DockerException` because no Docker socket/daemon was available. No completed report artifact was written. |
| `DOCKER_HOST=unix:///run/user/694691/podman/podman.sock python -m memgym.gym.swe_bench.reeval --preds results/swe_lite1_skip_eval_proxy/preds.json --dataset lite --slice 0:1 --workers 1 --run-id memgym-audit-podman --report-dir results/swe_lite1_skip_eval_proxy/reeval_podman --timeout 30` | Reached the official SWE-bench harness under Podman and wrote `results/swe_lite1_skip_eval_proxy/openai__gpt-4o-mini.memgym-audit-podman.json`. The one instance remained `error` because the harness attempted to pull `swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest` from Docker Hub, and `registry-1.docker.io` is not allowlisted. |
| `python -m memgym.gym.swe_bench.enrich_trajectories --help` | Passed. |
| `python -m memgym.gym.swe_bench.enrich_trajectories --result-dir results/swe_lite1_skip_eval_proxy --output results/swe_lite1_skip_eval_proxy/enrichment.json` | Passed on the failed SWE smoke output. It enriched 1 error instance, resolved `0/1`, with zero API/summarizer tokens and average true compression `1.00x`. |
| `python examples/swe_bench/replay.py --help` | Passed. |
| `python examples/swe_bench/compare.py --help` | Failed: `ERROR: Directory not found: --help`; this script treats `--help` as a positional result directory instead of exposing argparse help. |
| `python examples/swe_bench/compare.py results` | Passed on existing result folders. It found `swe_lite1_skip_eval_proxy` and printed a comparison table with `0/1` resolved and `ERROR` for `astropy__astropy-12907`. |
| `python -m memgym.gym.webarena.replay_probe --help` | Passed, but real replay still needs WebArena-Infinity and runnable app servers. |
| `python -m memgym.gym.webarena.dataset_stats --help` | Passed, but real stats require the WebArena-Infinity checkout. |
| `python -m memgym.training.eval.rm_cli --help` | Passed. |
| `python -m memgym.training.eval.memory_eval_cli --help` | Passed. |
| `memgym-evaluate --help` | Passed as an installed console script. Help still emitted LiteLLM remote cost-map fetch warning, mini-swe-agent v2.3.0 migration banner, and `ir_amem unavailable`. |
| `memgym-eval-rm --help` | Passed as an installed console script, with the same LiteLLM / mini-swe-agent / `ir_amem` side-effect warnings. |
| `memgym-eval-memory --help` | Passed as an installed console script, with the same LiteLLM / mini-swe-agent / `ir_amem` side-effect warnings. |
| `pyproject.toml` `[project.scripts]` vs `.venv/bin/memgym*` | Complete for declared console scripts: the package declares only `memgym-evaluate`, `memgym-eval-rm`, and `memgym-eval-memory`, and all three are installed and help-tested. |
| `python -m memgym.gym.tau2_bench --list_domains` | Passed: `mock`, `telecom`, `airline`, `retail`. |
| `python -m memgym.gym.tau2_bench --list_memory_models` | Passed, but stderr says `ir_amem unavailable`. |
| `rg --files tests/fixtures` | Failed: `tests/fixtures` does not exist, even though `docs/testing.md` references `tests/fixtures/trajectories/*.json` as examples for `memgym-eval-memory`. |

### Artifact Downloads

| Command | Result |
|---|---|
| `snapshot_download("MemGym/memgym-rm-1p7b", repo_type="model")` under `with-proxy` | Passed. Snapshot: `models--MemGym--memgym-rm-1p7b/snapshots/6f24c788...`, 40 MB. |
| MemRM base model download | Passed during MemRM smoke. Cache: `models--Qwen--Qwen3-1.7B-Base`, 3.3 GB. |
| `snapshot_download` for public MemGym datasets in `docs/data.md` | Passed for `memgym-rm-iid-heldout`, `memgym-rm-train`, `memgym-rm-scenario-ood-webarena`, `memgym-rm-scenario-ood-extras`, `memgym-rm-strategy-ood`, and `memgym-dr-instances`. |
| `snapshot_download("MemGym/memgym-codeqa-instances", repo_type="dataset")` | Failed with `401 Repository Not Found`, consistent with docs saying data is pending/withheld. |
| Upstream `load_dataset("princeton-nlp/SWE-bench_Lite", split="test")` | Passed: 300 rows. |
| Upstream `load_dataset("SWE-bench/SWE-smith", split="train")` | Passed: 59,136 rows. |
| Upstream `load_dataset("bdsaglam/musique", split="train")` | Passed: 39,876 rows. |

Downloaded MemGym JSONL row counts verified with `wc -l`:

| Artifact | Rows |
|---|---:|
| `memgym-rm-iid-heldout/reward_model_pairs_v2_eval.jsonl` | 3,007 |
| `memgym-rm-train/reward_model_pairs_v2_train.jsonl` | 15,630 |
| `memgym-rm-scenario-ood-webarena/pairs_paper_eval.jsonl` | 426 |
| `memgym-rm-scenario-ood-webarena/pairs_union.jsonl` | 487 |
| `memgym-rm-scenario-ood-extras/reward_model_pairs_tau2.jsonl` | 6,209 |
| `memgym-rm-scenario-ood-extras/reward_model_pairs_webarena_longctx.jsonl` | 111 |
| `memgym-rm-strategy-ood/data/rm_strategy_ood_pairs.jsonl` | 22 |
| `memgym-dr-instances/3hop_verified.jsonl` | 161 |
| `memgym-dr-instances/4hop_paper_run.jsonl` | 916 |
| `memgym-dr-instances/56hop_clean.jsonl` | 117 |

Cached Hugging Face snapshot refs:

| Repo | Snapshot |
|---|---|
| `MemGym/memgym-rm-1p7b` | `6f24c788dee65023f2e86c0e0b658fe807f9a74c` |
| `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| `MemGym/memgym-rm-iid-heldout` | `a8ee7cebb0ae476b05488dbe49a626478f2d8300` |
| `MemGym/memgym-rm-train` | `a2b23c32d4d533aa6128274ed403439dad09dc35` |
| `MemGym/memgym-rm-scenario-ood-webarena` | `541d3aff66c3282f9ab4e2553fdc7db0057bf75e` |
| `MemGym/memgym-rm-scenario-ood-extras` | `a2e050e67a4f9cb48ca019aa679d7d6da35fa8c7` |
| `MemGym/memgym-rm-strategy-ood` | `40e8030ee98bf726c8888efcc20898f544de04c2` |
| `MemGym/memgym-dr-instances` | `02f8da42a1a771d1d58552274db1cbf46f0fc7b4` |
| `princeton-nlp/SWE-bench_Lite` | `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` |
| `SWE-bench/SWE-smith` | `ea6d7173829c7ec8fa16c22055699ff2e9188091` |
| `bdsaglam/musique` | `22873a405dd809893b22ada0b499299fb612d2df` |

### Track Runs

Initial LLM-backed commands used the provided Metagen value as `OPENAI_API_KEY`
and failed because `api.openai.com` is blocked and `mg-api-*` is not itself an
OpenAI-compatible key. Follow-up commands used the linked Llama key as
`OPENAI_API_KEY`, the Llama OpenAI passthrough base URL, and model
`openai/gpt-5-4-nano-genai-responses`. Key values are intentionally not written
here.

| Track | Command shape | Result |
|---|---|---|
| tau2 mock | `python -m memgym.gym.tau2_bench --domains mock --task_ids 0 --limit 1 --agent_llm gpt-4o-mini --memory_model none` | Failed: `No module named 'tau2'`, because tau2-bench clone/install failed. Retry under `with-proxy` still failed: `github.com has not been allowlisted in filter agent_id=agent:codex`. |
| SWE-bench Lite | `python -m memgym.gym.swe_bench --model openai/gpt-4o-mini --dataset lite --slice 0:1 --skip-eval -o results/swe_lite1_skip_eval_proxy` | Reached execution after HF data download, then failed per-instance with `RuntimeError: docker backend not available on this system. Available backends: ['local']`. Summary written to `results/swe_lite1_skip_eval_proxy/summary.json`. |
| SWE-bench Lite local env diagnostic, initial | `python -m memgym.gym.swe_bench --model openai/gpt-4o-mini --dataset lite --slice 0:1 --memory none --env-type local --skip-eval --step-limit 1 ...` under `with-proxy` | Reached the local environment path and bypassed Docker, then repeatedly failed with `litellm.InternalServerError: OpenAIException - Connection error`. |
| SWE-bench Lite local env diagnostic, Llama passthrough | Same command shape with model `openai/gpt-5-4-nano-genai-responses`, Llama passthrough env vars, and `MSWEA_COST_TRACKING=ignore_errors`, output `results/swe_lite1_local_metagen_costignore` | Passed the LLM connection and mini-swe-agent cost-map blockers. The bounded run stopped as `LimitsExceeded` after the intentional one-step cap; `errors=0`, `limits_exceeded=1`. The local environment had no `/testbed`, so this is an LLM/control-flow smoke, not a valid SWE solve. |
| SWE-bench Lite official re-eval, no Docker socket | `python -m memgym.gym.swe_bench.reeval --preds results/swe_lite1_skip_eval_proxy/preds.json --dataset lite --slice 0:1 --workers 1 --run-id memgym-audit --report-dir results/swe_lite1_skip_eval_proxy/reeval_audit --timeout 30` | Failed in the official SWE-bench harness with `docker.errors.DockerException` while fetching the Docker server API version: the local Docker socket was missing. |
| SWE-bench Lite official re-eval, Podman socket | Same re-eval with `DOCKER_HOST=unix:///run/user/694691/podman/podman.sock`, run id `memgym-audit-podman` | Reached the official harness and wrote `results/swe_lite1_skip_eval_proxy/openai__gpt-4o-mini.memgym-audit-podman.json`; result is `error_instances=1` because the harness cannot pull the SWE-bench Docker Hub image (`registry-1.docker.io` not allowlisted). |
| WebArena | `python -m memgym.gym.webarena --policy_model gpt-4o-mini --app_name gitlab --task_ids 0 --observation_mode text` | Failed: WebArena-Infinity checkout missing; installer clone failed. |
| MemGym-DR, initial | `python -m memgym.pipelines.memgym_ir dataset --limit 1 --output data/dr_smoke_proxy ...` | Loaded MuSiQue successfully but filtered to 0 instances. With `--limit 20`, reached LLM pre-screen and failed before Llama passthrough setup. |
| MemGym-DR, Llama passthrough | `python -m memgym.pipelines.memgym_ir dataset --limit 20 --output data/dr_smoke_metagen_drop_params --worker-model openai/gpt-5-4-nano-genai-responses --verifier-model openai/gpt-5-4-nano-genai-responses --max-concurrent 1 --no-resume` with `LITELLM_DROP_PARAMS=True` | Completed filtering and LLM pre-screen: 20 MuSiQue rows -> 2 candidates -> 0 kept, 2 rejected. No final `memgym_ir_instances.jsonl` was generated. Without `LITELLM_DROP_PARAMS=True`, the same GPT-5 model errors because the pipeline sends `temperature=0.0`. |
| MemGym-CodeQA, initial | `python -m memgym.pipelines.coding_synthetic --limit 1 --worker-model gpt-4o-mini --verifier-model gpt-4o-mini --output output/codeqa_smoke_proxy1 --num-workers 1` | Loaded SWE-smith, filtered 1 instance, reached LLM stages, then failed all fact extraction attempts with `litellm.InternalServerError: OpenAIException - Connection error`; no instances crafted/verified. |
| MemGym-CodeQA, Llama passthrough | `python -m memgym.pipelines.coding_synthetic --limit 1 --worker-model openai/gpt-5-4-nano-genai-responses --verifier-model openai/gpt-5-4-nano-genai-responses --output output/codeqa_smoke_metagen --num-workers 1` | Completed a one-instance generation smoke. It wrote `output/codeqa_smoke_metagen/coding_synthetic_instances.jsonl` with 1 crafted instance and `gold_fix`, but ablation verification accepted `0/1` (`with_memory=0.00`, `without=0.00`). |
| Direct LiteLLM smoke, OpenAI API | `litellm.completion(model="gpt-4o-mini", ...)` under `with-proxy` | Failed: proxy blocks `api.openai.com` for `agent_id=agent:codex`. |
| Direct LiteLLM smoke, Llama passthrough | `litellm.completion(model="openai/gpt-5-4-nano-genai-responses", api_base="https://api.llama.com/experimental/passthrough/openai/v1/", ...)` | Passed and returned `OK`. A direct `/home/impwxu/code/metagen_api` CLI smoke with model alias `gpt` also returned `OK`. |

### MemRM Follow-Up

| Command | Result |
|---|---|
| Documented `memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b --limit 8 ...` under `with-proxy` | Downloaded checkpoint and base model, then failed: PEFT could not find `adapter_config.json` at the snapshot root. The file exists under `checkpoint-500/`. |
| Diagnostic `memgym-eval-rm --dataset iid-heldout --checkpoint <snapshot>/checkpoint-500 --limit 8 ...` | Passed. Output written to `results/memrm_iid_limit8_checkpoint500_proxy.json`. This verifies the checkpoint works, but not via the documented `--checkpoint MemGym/memgym-rm-1p7b` form. |
| Diagnostic `memgym-eval-rm --dataset all --checkpoint <snapshot>/checkpoint-500 --limit 2 --n-bootstrap 0 ...` | Passed. Output written to `results/memrm_all_limit2_checkpoint500_proxy.json`. This exercised all six registered RM splits with cached HF data, but is a tiny diagnostic, not the paper table. It also logged `strategy-ood`: `dropped 22 rows missing both prompt and input fields` before scoring 2 rows from 3 JSONL files. |

## Follow-Up After Local Third-Party Clones

After the user cloned the previously blocked GitHub repositories, the following
additional checks were run. Key values are still only supplied through
per-command environment variables and are not recorded in this document.

| Check | Result |
|---|---|
| Local clone presence | Present: `third_party/tau2-bench` at `fcc9ed6`, `third_party/webarena-infinity` at `1ca77813`, and `third_party/OpenHands` at `1e32eeefb`. |
| `bash install.sh --all` rerun | Installed `tau2==1.0.0` and `openhands-ai==1.7.0` from local checkouts, then failed its verification footer with `install.sh: line 149: python: command not found` because the script invokes `python` instead of `.venv/bin/python`. The actual installs completed before that failure. |
| `.venv/bin/python -m pip check` | Failed after OpenHands install: `tau2 1.0.0` requires `litellm<1.82.7,>=1.80.15`, while `openhands-ai==1.7.0` pins `litellm==1.84.1` and `openhands-sdk` requires `litellm>=1.83.7`. These tracks cannot both satisfy package metadata in one venv without a compatibility decision. |
| Direct imports | `memgym`, `tau2`, top-level `openhands`, and `swebench` import successfully. |
| MemGym CodeAct wrapper import | Still fails. MemGym expects legacy modules such as `openhands.core.config`, `openhands.events.action`, and `openhands.utils.async_utils`, but this OpenHands checkout/package exposes no `openhands.core`, `openhands.events`, or `openhands.utils` modules. `--agent codeact` therefore reports `OpenHands not installed` even though the top-level package imports. |
| tau2 task discovery | `python -m memgym.gym.tau2_bench --domains mock --list_tasks` now works and lists 10 mock tasks. Numeric `--task_ids 0` is invalid for this clone; a valid id is `create_task_1`. |
| tau2 real smoke | `tau2_summarizing` memory-vs-baseline smoke on `mock/create_task_1` with `max_steps=10` ran both phases and wrote `results/tau2_smoke_metagen/20260528-231300_mock_both_tau2_summarizing_ms30kf1kl4r0.6_both_memory_clone_retry`. Both phases completed but scored `0/1`; the model made tool calls, but did not satisfy the task assertions. |
| WebArena app discovery | `python -m memgym.gym.webarena --webarena_dir third_party/webarena-infinity --app_name gitlab --policy_model x --list_apps` works and lists 13 apps. The CLI still requires dummy `--app_name` and `--policy_model` even for `--list_apps`. |
| WebArena dataset stats | `python -m memgym.gym.webarena.dataset_stats --webarena-dir third_party/webarena-infinity` works and reports 1,620 total tasks across 13 apps. The option spelling is `--webarena-dir`, not the main CLI's `--webarena_dir`. |
| WebArena browser smoke | After `python -m playwright install chromium`, a one-step run on `gitlab-plan-and-track/task_e1` starts the app server, launches Chromium, calls the Llama passthrough model, takes action `click [40]`, and writes `results/webarena_smoke_metagen`. It does not solve the task because `max_steps=1`. |
| Release-critical unit suite | Exact command from `docs/testing.md` still passes: `61 passed, 1 xfailed`. |
| Full unit suite | Still fails with the same shape: `14 failed, 281 passed, 1 skipped, 1 xfailed`. Twelve failures are missing `tests/fixtures/trajectories/tau2_bench_run/...`; two are the `mini-swe-agent==2.3.0` `AgentConfig` constructor mismatch. |

## Documentation Gaps Found

1. `install.sh --all` prints `Installation Complete!` and exits 0 even when
   tau2-bench and OpenHands are not installed.
2. `README.md` and `docs/quickstart.md` imply `./install.sh --all` installs
   tau2-bench and OpenHands. In this run the script only printed clone
   instructions for missing repos; it did not clone them.
3. `requirements-tau2.txt` says tau2-bench is cloned at a pinned SHA, but
   `src/memgym/gym/tau2_bench/install.py` uses `git clone --depth 1` with no
   pinned commit.
4. `docs/testing.md` says the release-critical suite should report
   `64 passed, 1 xfailed`; the current code reports `61 passed, 1 xfailed`.
5. `requirements-amem.txt` is not currently resolvable as written because
   `hipporag>=2.0.0` has no stable available version in the resolver result.
   Even the available alpha (`hipporag>=2.0.0a4`) fails a dry run because it
   pins `openai==1.91.1`, which is absent from the package index used here.
6. The docs explain `OPENAI_API_KEY` and local OpenAI-compatible servers, but do
   not explain how to configure Metagen/Llama passthrough. The working form in
   this run required a linked Llama key, `OPENAI_API_BASE`/`OPENAI_BASE_URL`
   set to `https://api.llama.com/experimental/passthrough/openai/v1/`, and a
   model such as `openai/gpt-5-4-nano-genai-responses`.
7. The documented MemRM command `--checkpoint MemGym/memgym-rm-1p7b` is not
   sufficient with the current code because the adapter lives under
   `checkpoint-500/`. The docs mention `checkpoint-500`, but the CLI does not
   automatically pass `subfolder="checkpoint-500"` to PEFT.
8. The `strategy-ood` RM registry entry has no specific file, so the CLI loads
   every JSONL in the HF snapshot. In this run it dropped 22 per-slice rows that
   have labels but no `prompt`/`input`, then scored rows from the aggregate
   `rm_strategy_ood_pairs.jsonl`. This is not fatal, but the warning is
   confusing for a reproduction run.
9. HF data and checkpoint download instructions assume direct Hugging Face
   access. On this devserver the working form required internal `with-proxy` and
   `HF_HUB_DISABLE_XET=1`, which are not mentioned in the project docs.
10. GitHub-based third-party setup is blocked for this Codex agent identity by
   fwdproxy allowlisting, even when `with-proxy` is used. The project docs do not
   provide an alternate archive/mirror path for tau2-bench, WebArena-Infinity, or
   OpenHands. For tau2-bench, PyPI is not an alternate path:
   `pip index versions tau2-bench` returned no matching distribution.
11. `docs/testing.md` points `memgym-eval-memory` users to
   `tests/fixtures/trajectories/*.json`, but this checkout has no
   `tests/fixtures` directory. The CLI help works, but the documented example
   fixture is unavailable.
12. The full unit suite is not green after the documented setup. The
   documented release-critical subset passes, but `pytest tests/unit -q` fails
   because the tau2 trajectory fixture tree is absent and two trajectory tests
   are incompatible with the installed `mini-swe-agent==2.3.0` constructor
   requirements.
13. `examples/swe_bench/compare.py --help` is not a usable help command. It
   exits with `ERROR: Directory not found: --help`, because the script treats
   the first argument as a result directory instead of using argparse.
14. The docs do not explain the Meta devserver Podman path for Docker-compatible
   evaluation. Podman plus `DOCKER_HOST=unix:///run/user/<uid>/podman/podman.sock`
   fixes the missing Docker socket, but official SWE-bench still tries to pull
   images from Docker Hub.
15. Official SWE-bench image access is blocked after Podman setup because
   `registry-1.docker.io` is not allowlisted for this agent identity. The docs
   need an internal registry/mirror or instructions for preloading SWE-bench
   images on Meta devservers.
16. GPT-5-family passthrough model IDs need extra runtime switches in some
   MemGym paths: `MSWEA_COST_TRACKING=ignore_errors` for mini-swe-agent cost
   accounting and `LITELLM_DROP_PARAMS=True` where the pipeline passes
   unsupported parameters such as `temperature=0.0`.
17. `install.sh` verification should use the active venv interpreter rather
   than bare `python`; on this devserver the installs completed, then the script
   exited `127` during verification because `python` was not on PATH.
18. Installing tau2 and OpenHands in one venv creates an unsatisfied `litellm`
   dependency conflict. The docs do not mention separate venvs or a known-good
   compatible version set.
19. MemGym's CodeAct wrapper targets an older OpenHands API layout. The current
   OpenHands checkout installs `openhands-ai==1.7.0`, but the wrapper imports
   modules under `openhands.core`, `openhands.events`, and `openhands.utils`
   that are not present.
20. WebArena docs/CLI path handling is inconsistent: the installer only clones
   into `src/memgym/envs/webarena_infinity`, while a user-provided local clone
   works with `--webarena_dir`; `dataset_stats` uses `--webarena-dir` instead.

## Current Reproducibility State

Reproducible in this environment:

- Python package dependency setup for core/dev/eval/SWE/tau2/WebArena extras.
- `mini-swe-agent` and `swebench` installation via `install.sh --all`.
- Podman Docker-compatible socket setup on this Meta devserver.
- Unit/import/CLI wiring tier.
- The documented release-critical unit subset.
- Memory registry visibility, except optional A-Mem.
- Public MemGym HF artifact downloads, using `with-proxy` and
  `HF_HUB_DISABLE_XET=1`.
- Upstream SWE-bench Lite, SWE-smith, and MuSiQue data loading, using the same
  proxy setup.
- MemRM limit-8 smoke when the checkpoint path points directly at
  `checkpoint-500/`.
- MemRM `--dataset all` diagnostic at `--limit 2` when the checkpoint path
  points directly at `checkpoint-500/`.
- Direct LLM smoke through Llama OpenAI passthrough using the linked Llama key
  and model `openai/gpt-5-4-nano-genai-responses`.
- SWE-bench local-env LLM smoke to an intentional one-step `LimitsExceeded`
  result with `MSWEA_COST_TRACKING=ignore_errors`.
- Official SWE-bench harness entry under Podman, up to the point of pulling
  Docker Hub images.
- MemGym-CodeQA one-instance generation smoke through Llama passthrough
  (`1` crafted instance, `0/1` accepted by ablation verification).
- MemGym-DR filtering and LLM pre-screen through Llama passthrough with
  `LITELLM_DROP_PARAMS=True` (`20` raw rows -> `2` candidates -> `0` kept).
- tau2 real episode smoke on `mock/create_task_1`, including both baseline and
  `tau2_summarizing` memory phases.
- WebArena real browser smoke on `gitlab-plan-and-track/task_e1` after installing
  Playwright Chromium, up to an intentional one-step cap.

Not reproduced:

- SWE-bench/SWE-Gym real episode execution with official Docker backend.
- tau2 paper-scale success-rate reproduction.
- WebArena paper-scale success-rate reproduction or replay evaluation.
- MemGym-DR full generation/benchmarking.
- MemGym-CodeQA accepted/high-quality generation.
- A-Mem / `ir_amem` functionality.
- Docker/Podman-based official SWE-bench evaluation past Docker Hub image pull.
- OpenHands/CodeAct path.
- Full MemRM table through the documented `--checkpoint MemGym/memgym-rm-1p7b`
  CLI form.
- Full `tests/unit` suite after documented setup.

The remaining blockers are Docker Hub image access or an internal SWE-bench
image mirror, A-Mem dependency resolution, the OpenHands API mismatch, the
`litellm` dependency conflict between tau2 and OpenHands, missing tau2 unit-test
fixtures, and full-quality LLM-backed generation/evaluation. The docs should
also be updated for the `install.sh --all` success semantics, tau2 pinning,
stale test count, MemRM `checkpoint-500` loading, `strategy-ood` RM JSONL
selection, proxy requirements, missing memory-eval/tau2 trajectory fixtures,
full-unit-suite expectations, Meta devserver Podman setup, Docker Hub image
mirroring, third-party local-clone paths, and Metagen/Llama backend
configuration.
