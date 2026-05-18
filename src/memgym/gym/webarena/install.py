"""
Install WebArena-Infinity into the embedded envs directory and set up Playwright.

Usage (on EC2 via POST /run-script):
    python -m memgym.gym.webarena.install

By default this installs into the venv determined by `sys.executable`, which
matches the EC2 launcher convention: the venv is activated before running, and
`sys.executable` points at the venv python. On the user's standard EC2 host
that is `${VENV_DIR}/bin/python`.

What this script does:
1. Clones `web-arena-x/webarena-infinity` into `src/memgym/envs/webarena_infinity/`
   (skips if the directory already exists and is a valid git checkout).
2. Installs webarena-infinity's own requirements into the current venv.
3. Installs Playwright + Chromium browser binaries.
4. Verifies imports.

WebArena-Infinity disk footprint on a fresh install:
- repo clone: ~50 MB
- Python deps (playwright, pytest-playwright, fastapi, uvicorn, etc.): ~200 MB
- Chromium browser binary: ~500 MB
- Total: well under 1 GB — negligible on the EC2 1.6 TB volume.

This script is safe to re-run: git clone is skipped if the repo is present,
pip installs are idempotent, and Playwright's `install chromium` is a no-op
when the browser is already cached.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WEBARENA_REPO = "https://github.com/web-arena-x/webarena-infinity.git"


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
    repo_root = _repo_root()
    webarena_parent = repo_root / "src" / "memgym" / "envs"
    webarena_dir = webarena_parent / "webarena_infinity"

    python_exe = sys.executable
    pip_exe = [python_exe, "-m", "pip"]

    print(f"Repo root:        {repo_root}")
    print(f"Venv python:      {python_exe}")
    print(f"Target directory: {webarena_dir}")

    # ---- Step 1: Clone webarena-infinity (idempotent) ----
    webarena_parent.mkdir(parents=True, exist_ok=True)
    if webarena_dir.exists() and (webarena_dir / ".git").exists():
        print(f"\n=== webarena-infinity already cloned at {webarena_dir}, skipping clone ===")
    elif webarena_dir.exists():
        raise SystemExit(
            f"{webarena_dir} exists but is not a git checkout — please remove it manually"
        )
    else:
        print(f"\n=== Cloning {WEBARENA_REPO} into {webarena_dir} ===")
        _run(["git", "clone", "--depth", "1", WEBARENA_REPO, str(webarena_dir)])

    # ---- Step 2: Install webarena-infinity's requirements ----
    reqs = webarena_dir / "requirements.txt"
    if reqs.exists():
        print(f"\n=== Installing webarena-infinity requirements from {reqs} ===")
        _run([*pip_exe, "install", "-r", str(reqs)])
    else:
        print(f"\n=== No requirements.txt at {reqs} — installing known-needed packages ===")
        _run([*pip_exe, "install", "playwright>=1.40", "pytest-playwright", "fastapi", "uvicorn"])

    # Make sure playwright is installed even if requirements.txt omitted it.
    print("\n=== Ensuring Playwright is installed ===")
    _run([*pip_exe, "install", "--upgrade", "playwright>=1.40"])

    # ---- Step 3: Install Chromium browser binaries ----
    print("\n=== Installing Chromium via `playwright install chromium` ===")
    # --with-deps installs OS-level dependencies (apt-get); safe on EC2 Ubuntu.
    rc = _run([python_exe, "-m", "playwright", "install", "--with-deps", "chromium"], check=False)
    if rc != 0:
        print("WARNING: `playwright install --with-deps chromium` failed. "
              "Retrying without --with-deps (no sudo-requiring system deps)...")
        _run([python_exe, "-m", "playwright", "install", "chromium"])

    # ---- Step 4: Verify imports ----
    print("\n=== Verifying imports ===")
    verify_cmd = [
        python_exe, "-c",
        "import playwright; "
        "from playwright.sync_api import sync_playwright; "
        "import memgym; "
        "from memgym.memory import get_memory_model; "
        "wa_sum = get_memory_model('webarena_summarizing', max_size=20, keep_first=1, condensation_ratio=0.5); "
        "wa_str = get_memory_model('webarena_structured', max_size=20, keep_first=1, condensation_ratio=0.5); "
        "print('All imports OK:', type(wa_sum).__name__, type(wa_str).__name__)"
    ]
    _run(verify_cmd)

    print("\n✓ webarena-infinity installation complete.")
    print(f"  Repo: {webarena_dir}")
    print(f"  Apps: {webarena_dir / 'apps'}")


if __name__ == "__main__":
    main()
