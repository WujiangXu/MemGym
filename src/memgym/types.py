"""
Core types and enums for MemGym.

This module defines the fundamental types used throughout the MemGym framework.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class TopicType(str, Enum):
    """Types of agent topics that can be tested."""

    SEARCH = "search"
    CODING = "coding"
    COMPUTER_USE = "computer_use"  # Web agents, GUI automation, etc.
    GENERAL = "general"


class MemoryTestType(str, Enum):
    """Types of memory tests."""

    SHORT_TERM = "short_term"  # Within single episode
    LONG_TERM = "long_term"    # Across episodes
    SEQUENTIAL = "sequential"   # Sequential multi-task
    PARALLEL = "parallel"       # Parallel multi-task


class MetricType(str, Enum):
    """Types of memory metrics."""

    RECALL_ACCURACY = "recall_accuracy"
    RETENTION_RATE = "retention_rate"
    CONTEXT_UTILIZATION = "context_utilization"
    FORGETTING_CURVE = "forgetting_curve"
    RETRIEVAL_LATENCY = "retrieval_latency"


class MemoryMetrics(TypedDict, total=False):
    """Memory-specific metrics."""

    recall_accuracy: float
    retention_rate: float
    context_utilization: float
    forgetting_curve: List[float]
    retrieval_latency: float
    custom_metrics: Dict[str, Any]


class ObservationType(TypedDict, total=False):
    """Structure for observations in memory environments."""

    observation: Any
    memory_state: Dict[str, Any]
    metadata: Dict[str, Any]
