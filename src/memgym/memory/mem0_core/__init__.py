"""Shared Mem0 core (`pip install mem0ai`).

Wraps the upstream Mem0 SDK behind the same `add_doc` / `retrieve` /
`reset` interface as the other memory backbones. Used by the
coding-synthetic pipeline's `mem0` method and the IR pipeline's
`ir_mem0` strategy.
"""
from .system import Mem0System

__all__ = ["Mem0System"]
