"""
Environments for MemGym.

This module provides memory-testing environments that wrap
various benchmarks (SWE-bench, WebArena, tau2-bench, etc.).

Usage:
    >>> from envs import get_env, list_envs
    >>> env = get_env("swe", dataset="princeton-nlp/SWE-bench_Lite")
    >>> env = get_env("webarena", task_id="...")
    >>> env = get_env("tau2-bench", domain="mock", task_id="0")  # via gym/tau2_bench
"""

from .base import (
    BaseMemoryEnvironment,
    BaseMemoryState,
    SimpleMemoryState,
    MemorySnapshot,
    get_env,
    register_env,
    list_envs,
)

# Import environments to register them (optional - may fail if dependencies missing)

# Tau2-bench env now lives at memgym.gym.tau2_bench.env (lazy import — depends on
# third_party/tau2-bench installed via gym/tau2_bench/install.py).
try:
    from memgym.gym.tau2_bench.env import Tau2BenchMemoryEnv
    _TAU2_AVAILABLE = True
except ImportError:
    Tau2BenchMemoryEnv = None
    _TAU2_AVAILABLE = False

# SWE-bench environment
try:
    from memgym.gym.swe_bench.env import SWEMemoryEnv, SWETask
    _SWE_AVAILABLE = True
except ImportError as e:
    SWEMemoryEnv = None
    SWETask = None
    _SWE_AVAILABLE = False
    # Silently skip if swe dependencies not installed

# WebArena (Web GUI) environment — import is lazy because playwright
# is an optional dependency installed by gym/webarena/install.py
try:
    from memgym.gym.webarena.env import WebArenaMemoryEnv, WebArenaTask
    _WEBARENA_AVAILABLE = True
except ImportError:
    WebArenaMemoryEnv = None
    WebArenaTask = None
    _WEBARENA_AVAILABLE = False

__all__ = [
    # Base
    "BaseMemoryEnvironment",
    "BaseMemoryState",
    "SimpleMemoryState",
    "MemorySnapshot",
    # Registry
    "get_env",
    "register_env",
    "list_envs",
]

# Add available environments to __all__
if _TAU2_AVAILABLE:
    __all__.append("Tau2BenchMemoryEnv")

if _SWE_AVAILABLE:
    __all__.extend(["SWEMemoryEnv", "SWETask"])

if _WEBARENA_AVAILABLE:
    __all__.extend(["WebArenaMemoryEnv", "WebArenaTask"])
