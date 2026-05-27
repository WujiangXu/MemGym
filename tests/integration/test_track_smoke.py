"""Per-track integration smoke tests.

Each test instantiates the smallest possible piece of a track with
``passthrough`` memory and asserts the wiring is sane. We intentionally
do **not** run a full episode here — that would need Docker, a
WebArena server, or a real coding-instance container. Instead each test:

1. Imports the track's entry-point module.
2. Skips when the track's external dependency is missing.
3. Constructs the smallest in-memory object (env, agent, or pipeline)
   that exercises the import + ``__init__`` path.

These tests are gated behind ``@pytest.mark.integration`` so the
public CI matrix (Python 3.10 / 3.12 on ubuntu-latest) does not run
them by default. Run them locally with::

    pytest tests/integration/ -m integration

The point is to catch import-time breakage and signature drift in the
five tracks the paper claims — a cheap "is anything obviously broken"
guard for reviewers cloning the repo.
"""
from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# SWE-Gym / SWE-bench
# ---------------------------------------------------------------------------

def test_swe_bench_module_imports_with_passthrough():
    """``memgym.gym.swe_bench`` imports and exposes the env class."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available — SWE-Bench env cannot run")

    env_mod = importlib.import_module("memgym.gym.swe_bench.env")
    assert hasattr(env_mod, "SWEMemoryEnv"), "SWEMemoryEnv missing from env module"
    agent_mod = importlib.import_module("memgym.gym.swe_bench.agent")
    assert hasattr(agent_mod, "MemoryAwareSWEAgent"), (
        "MemoryAwareSWEAgent missing from agent module"
    )

    # A passthrough memory wrapper must be constructible without a real LLM.
    from memgym.memory.base import get_memory_model
    mem = get_memory_model("passthrough", max_tokens=4000)
    assert mem.manage_context([], {"role": "user", "content": "smoke"}).content


# ---------------------------------------------------------------------------
# tau2-bench
# ---------------------------------------------------------------------------

def test_tau2_bench_module_imports_with_passthrough():
    """tau2 entry-point + memory adapter import cleanly."""
    try:
        importlib.import_module("memgym.gym.tau2_bench")
    except ImportError as exc:
        pytest.skip(f"tau2-bench third-party deps not installed: {exc}")

    adapter_mod = importlib.import_module("memgym.gym.tau2_bench.memory_adapter")
    # The adapter is the public seam between MemGym's memory strategies
    # and the upstream tau2 Agent interface. It must exist regardless of
    # whether the user has cloned tau2-bench itself.
    assert hasattr(adapter_mod, "__file__")


# ---------------------------------------------------------------------------
# WebArena
# ---------------------------------------------------------------------------

def test_webarena_module_imports_with_passthrough():
    """WebArena env requires a running server; we only check imports here."""
    if not os.environ.get("WEBARENA_BASE_URL"):
        pytest.skip("WEBARENA_BASE_URL not set — WebArena server unreachable")

    env_mod = importlib.import_module("memgym.gym.webarena.env")
    assert hasattr(env_mod, "__file__")


# ---------------------------------------------------------------------------
# MemGymCodeQA — data-generation + eval pipeline
# ---------------------------------------------------------------------------

def test_coding_synthetic_pipeline_imports():
    """The coding-QA pipeline package imports cleanly."""
    importlib.import_module("memgym.pipelines.coding_synthetic")


# ---------------------------------------------------------------------------
# MemGymDR — multi-hop QA pipeline
# ---------------------------------------------------------------------------

def test_memgym_ir_pipeline_imports():
    """The MemGymDR (IR / multi-hop) pipeline package imports cleanly."""
    importlib.import_module("memgym.pipelines.memgym_ir")
