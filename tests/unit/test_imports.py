"""Verify clean imports after memgym package namespace reorganization."""

import os
from pathlib import Path

import pytest


MEMGYM_SRC = Path(__file__).parent.parent.parent / "src" / "memgym"


def test_no_sys_path_hacks():
    """Ensure no sys.path.insert hacks remain in the memgym package.

    Exceptions: amem (vendored), webarena_infinity (third-party, installed by
    gym/webarena/install.py), tau2_bench (third-party, installed by
    gym/tau2_bench/install.py), lightmem (third-party adapter under
    third_party/LightMem; loaded lazily by the lightmem memory method).
    """
    EXCEPTIONS = {"amem", "webarena_infinity", "tau2_bench", "lightmem"}

    violations = []
    for py_file in MEMGYM_SRC.rglob("*.py"):
        # Skip known third-party exceptions
        if any(exc in str(py_file) for exc in EXCEPTIONS):
            continue

        content = py_file.read_text()
        for i, line in enumerate(content.split("\n"), 1):
            if "sys.path.insert" in line and not line.lstrip().startswith("#"):
                violations.append(f"{py_file.relative_to(MEMGYM_SRC)}:{i}: {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} sys.path.insert hack(s) in memgym package:\n"
        + "\n".join(violations)
    )


def test_package_imports():
    """Test that core package imports work."""
    import memgym
    assert hasattr(memgym, "__version__")

    from memgym.memory import get_memory_model
    assert callable(get_memory_model)

    from memgym.envs import BaseMemoryEnvironment
    assert BaseMemoryEnvironment is not None

    from memgym.agents import BaseAgent, get_agent
    assert callable(get_agent)


def test_gym_swe_bench_imports():
    """Test that gym.swe_bench imports work."""
    from memgym.gym.swe_bench.env import SWEMemoryEnv
    assert SWEMemoryEnv is not None

    from memgym.gym.swe_bench.container import BACKENDS
    assert "docker" in BACKENDS
    assert "bubblewrap" in BACKENDS


def test_no_reasoning_module():
    """The deprecated reasoning module should be deleted."""
    reasoning_dir = MEMGYM_SRC / "reasoning"
    assert not reasoning_dir.exists(), "reasoning/ module should be deleted (use agents/ instead)"
