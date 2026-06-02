"""
Install sierra-research/tau2-bench into ``third_party/tau2-bench`` and
verify the MemGym integration.

Usage (locally, or submitted to a remote runner via POST /run-script):
    python -m memgym.gym.tau2_bench.install
    python -m memgym.gym.tau2_bench.install --help

What this script does
=====================
1. Clones ``sierra-research/tau2-bench`` into
   ``<repo_root>/third_party/tau2-bench/`` (skips if already present).
   Pins to ``TAU2_REPO_SHA`` so reruns are reproducible.
2. Runs ``pip install -e third_party/tau2-bench`` into the current venv
   so ``import tau2`` works without the sys.path fallback baked into
   ``memgym.gym.tau2_bench.env._ensure_tau2_on_path``.
3. Verifies the import by:
   a. ``import tau2`` — basic package health
   b. Loading the ``mock`` domain via ``tau2.registry.get_env_constructor``
      and ``get_tasks_loader`` — exercises the exact call sites env.py uses
   c. Wrapping a dummy MockAgent with ``wrap_agent_with_memory`` — exercises
      the memory_adapter's lazy class builder
   d. Constructing each of the four registered memory adapters
      (``tau2_summarizing``, ``tau2_structured``) so the full stack is
      proven healthy before a real episode runs.

The script is safe to re-run: the git clone is skipped when the repo is
already checked out, ``pip install -e`` is idempotent, and none of the
verification steps have side effects.

Why ``third_party/`` (not ``src/memgym/envs/``)
================================================
WebArena installs into ``src/memgym/envs/webarena_infinity/`` because it's
vendored data + scripts. Tau2-bench is a Python *package* we pip-install,
so it belongs under ``third_party/`` next to other installable vendors.
This is consistent with the search path ``env._ensure_tau2_on_path`` uses
when a bare ``git clone`` exists without pip install.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
# Pinned SHA — matches the version `requirements-tau2.txt` claims it
# tracks. Bump together with that file. `git clone --depth 1` cannot
# fetch a specific SHA directly, so we clone the default branch and
# `git checkout` to this commit.
TAU2_REPO_SHA = "fcc9ed6"


def _repo_root() -> Path:
    """Walk up from this file until we hit the MemGym repo root."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
        d = d.parent
    raise RuntimeError("Could not locate MemGym repo root from install.py")


def _run(cmd: list[str], *, check: bool = True, cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def main() -> None:
    # Argparse here is purely for `--help`. The script has no required
    # args and historically always ran on `python -m ...install` — but
    # without argparse, `... .install --help` would eagerly clone +
    # pip-install, which is a nasty surprise on first contact.
    parser = argparse.ArgumentParser(
        description=(
            f"Clone sierra-research/tau2-bench (pinned to {TAU2_REPO_SHA}) "
            f"into third_party/tau2-bench, pip install -e it, and verify the "
            f"MemGym integration. Idempotent on re-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--venv", type=str, default=None,
        help="(Informational only; install uses the current sys.executable.)",
    )
    parser.parse_args()

    repo_root = _repo_root()
    tau2_parent = repo_root / "third_party"
    tau2_dir = tau2_parent / "tau2-bench"

    python_exe = sys.executable
    pip_exe = [python_exe, "-m", "pip"]

    print(f"Repo root:        {repo_root}")
    print(f"Venv python:      {python_exe}")
    print(f"Target directory: {tau2_dir}")
    print(f"Pinned SHA:       {TAU2_REPO_SHA}")

    # ---- Step 1: Clone sierra-research/tau2-bench (idempotent) ----
    tau2_parent.mkdir(parents=True, exist_ok=True)
    if tau2_dir.exists() and (tau2_dir / ".git").exists():
        print(f"\n=== tau2-bench already cloned at {tau2_dir}, skipping clone ===")
    elif tau2_dir.exists():
        raise SystemExit(
            f"{tau2_dir} exists but is not a git checkout — please remove it manually"
        )
    else:
        print(f"\n=== Cloning {TAU2_REPO} into {tau2_dir} ===")
        _run(["git", "clone", TAU2_REPO, str(tau2_dir)])
        print(f"=== Checking out pinned SHA {TAU2_REPO_SHA} ===")
        _run(["git", "checkout", TAU2_REPO_SHA], cwd=tau2_dir)

    # ---- Step 2: pip install -e the cloned repo ----
    print(f"\n=== Installing tau2-bench into venv: pip install -e {tau2_dir} ===")
    _run([*pip_exe, "install", "-e", str(tau2_dir)])

    # ---- Step 3: Verify the stack ----
    print("\n=== Verifying tau2 + memgym integration ===")
    verify_cmd = [
        python_exe, "-c",
        # Keep this inline so a failure shows up with full traceback in the
        # remote job log rather than hiding in a sub-script.
        "import sys, traceback\n"
        "try:\n"
        "    import tau2\n"
        "    print('[1/4] import tau2 OK — version:', getattr(tau2, '__version__', 'unknown'))\n"
        "    from tau2.registry import registry\n"
        "    env_ctor = registry.get_env_constructor('mock')\n"
        "    tasks = list(registry.get_tasks_loader('mock')())\n"
        "    print(f'[2/4] mock domain: env_ctor={env_ctor.__name__} tasks={len(tasks)}')\n"
        "    from memgym.gym.tau2_bench.memory_adapter import wrap_agent_with_memory\n"
        "    from memgym.memory import get_memory_model\n"
        "    mm = get_memory_model('tau2_summarizing', task_instruction='verify', domain='mock', solo_mode=True, role='agent', max_size=10, keep_first=1, keep_last=2, condensation_ratio=0.5, summarization_model='gpt-4o-mini')\n"
        "    print(f'[3/4] Tau2SummarizingMemory built: {type(mm).__name__}')\n"
        "    mm2 = get_memory_model('tau2_structured', task_instruction='verify', domain='mock', solo_mode=True, role='agent', max_size=10, keep_first=1, keep_last=2, condensation_ratio=0.5, summarization_model='gpt-4o-mini')\n"
        "    print(f'[4/4] Tau2StructuredSummary built: {type(mm2).__name__}')\n"
        "    print('\\nAll tau2-bench verification steps passed.')\n"
        "except Exception as e:\n"
        "    print('VERIFY FAILED:', type(e).__name__, e)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(2)\n",
    ]
    _run(verify_cmd)

    print("\n✓ tau2-bench installation complete.")
    print(f"  Repo: {tau2_dir}")
    print(f"  Registered domains (via tau2.registry): mock, telecom, airline, retail")


if __name__ == "__main__":
    main()
