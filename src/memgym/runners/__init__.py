"""
Runners for MemGym experiments.

Runners compose environment + agent + memory manager and handle episode orchestration.
This keeps environments pure (just reset/step/close) while runners handle the full loop.

Usage:
    >>> from memgym.runners import SWERunner
    >>> # tau2-bench runner is shipped under gym/tau2_bench (lazy-loaded below).
"""

from .base import BaseRunner, SimpleRunner
from .swe_runner import SWERunner, SWEPureRunner
from .webarena_replay_runner import ObservationReplayRunner, ReplayResult

# Tau2-bench runner is part of the gym/tau2_bench package; lazy import so memgym
# loads cleanly even when third_party/tau2-bench isn't installed.
try:
    from .tau2_bench_runner import Tau2BenchRunner
    _TAU2_AVAILABLE = True
except ImportError:
    Tau2BenchRunner = None
    _TAU2_AVAILABLE = False

__all__ = [
    # Base
    "BaseRunner",
    "SimpleRunner",
    # Task-specific
    "SWERunner",
    "SWEPureRunner",
    # Replay
    "ObservationReplayRunner",
    "ReplayResult",
]

if _TAU2_AVAILABLE:
    __all__.append("Tau2BenchRunner")
