"""
gym/tau2_bench — tau2-bench dialogue gym for MemGym.

This is the third pillar of the benchmark suite (coding / GUI / dialogue),
mirroring the file layout of `gym/swe_bench/` and `gym/webarena/`:

    env.py             — Tau2BenchMemoryEnv (delegates to tau2.Orchestrator)
    agent.py           — Tau2BenchAgent (BaseAgent tracking wrapper)
    memory_adapter.py  — MemoryManagerAdapter + UserMemoryManagerAdapter
                         (intercept tau2's generate_next_message for symmetric
                          agent + user memory wrapping)
    evaluate.py        — `python -m memgym.gym.tau2_bench` worker-pool CLI
    install.py         — clone sierra-research/tau2-bench at pinned SHA

The env, agent, and memory_adapter modules each lazy-import the upstream
`tau2.*` modules so this package can be imported even before
`third_party/tau2-bench/` has been installed by `install.py`.
"""
from __future__ import annotations

# Module-level re-exports are intentionally lazy: importing
# `memgym.gym.tau2_bench` should not eagerly trigger `import tau2`, because
# the upstream package is installed by gym/tau2_bench/install.py and may not
# yet be present at first import time.

__all__ = ["Tau2BenchMemoryEnv", "Tau2BenchAgent", "load_domain_tasks"]


def __getattr__(name: str):
    if name == "Tau2BenchMemoryEnv":
        from .env import Tau2BenchMemoryEnv
        return Tau2BenchMemoryEnv
    if name == "Tau2BenchAgent":
        from .agent import Tau2BenchAgent
        return Tau2BenchAgent
    if name == "load_domain_tasks":
        from .env import load_domain_tasks
        return load_domain_tasks
    raise AttributeError(f"module 'memgym.gym.tau2_bench' has no attribute {name!r}")
