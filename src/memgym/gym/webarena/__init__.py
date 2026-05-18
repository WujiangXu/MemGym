"""
WebArena (Web GUI) gym for MemGym.

This replaces the OSWorld-based CUA gym which could not run without `/dev/kvm`.
WebArena-Infinity environments are self-contained Python web servers controlled
via Playwright/Chromium — no hypervisor, no VM, no KVM. The result is a Web GUI
trajectory track that runs on the shared EC2 box (or locally, though EC2 is the
default execution surface) using Docker-only tooling.

Architecture:

- `install.py`: Clones web-arena-x/webarena-infinity into the embedded envs
  directory and installs Playwright/Chromium into the active venv.
- `server_pool.py`: Boots `apps/<name>/server.py` subprocesses on a port range
  and hands free ports to per-task runners; handles teardown.
- `env.py`: `WebArenaMemoryEnv` — subclass of `BaseMemoryEnvironment` wrapping
  a Playwright browser session + the webarena task's programmatic verifier.
  Default observation modality is **text** (pruned accessibility tree);
  `observation_mode="vision"` returns base64 screenshots for ablation runs.
- `agent.py`: `WebArenaAgent` — text-mode (accessibility tree → LiteLLM
  tool-call action) or vision-mode (screenshot → coordinate action).
- `evaluate.py`: Main CLI. `python -m memgym.gym.webarena ...`
- `runners/webarena_runner.py`: Episode loop matching the BaseRunner contract.

Importing this module's `env.py` registers the environment under the names
`webarena` and `web` in the MemGym environment registry. Playwright is NOT
imported at module-load time (it is imported lazily inside the env's
`_launch_browser()`), so this package is safe to import on dev machines
without Playwright installed.
"""

__all__: list[str] = []
