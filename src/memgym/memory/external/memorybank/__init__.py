"""Vendored MemoryBank core (Ebbinghaus decay + retrieval reinforcement).

Implements the algorithmic contribution of MemoryBank (Zhong et al., AAAI 2024)
without the upstream chatbot scaffolding. Used by both the coding-synthetic
pipeline's `memorybank` method and the IR pipeline's `ir_memorybank` strategy.
"""
from .system import MemoryBankSystem

__all__ = ["MemoryBankSystem"]
