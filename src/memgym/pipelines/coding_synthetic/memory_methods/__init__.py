"""Pluggable memory methods for the coding_synthetic evaluation pipeline.

Registry
--------
- ``prompt``      – existing prompt-only strategies (basic, structured, etc.)
- ``amem``        – A-MEM agentic Zettelkasten memory
- ``lightmem``    – LightMEM three-stage memory (extraction + embedding retrieval)
- ``truncated``   – raw sliding-window truncation baseline
- ``hipporag``    – HippoRAG knowledge-graph + Personalized PageRank
- ``simplemem``   – SimpleMem semantic structured compression
- ``mem0``        – Mem0 LLM extract+update fact memory
- ``memorybank``  – MemoryBank Ebbinghaus decay + retrieval reinforcement
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .protocol import CodingMemoryMethod, PromptStrategyMethod
from .truncated_method import TruncatedMethod

__all__ = [
    "CodingMemoryMethod",
    "PromptStrategyMethod",
    "TruncatedMethod",
    "create_memory_method",
]

# Method names accepted by --method CLI flag.
MEMORY_METHODS = {
    "prompt",
    "amem",
    "truncated",
    "lightmem",
    "hipporag",
    "simplemem",
    "mem0",
    "memorybank",
}


def create_memory_method(
    method: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    # Only used when method="prompt":
    strategy: str = "basic",
    client: Any = None,
    budget: int = 500,
) -> CodingMemoryMethod:
    """Factory: build a ``CodingMemoryMethod`` from a name + config dict.

    Parameters
    ----------
    method : str
        One of ``MEMORY_METHODS``.
    config : dict, optional
        Method-specific configuration (e.g. ``{"retrieve_k": 5}``).
    strategy : str
        Only for ``method="prompt"`` — the prompt strategy name.
    client : LLMClient
        Only for ``method="prompt"`` — the LLM client.
    budget : int
        Only for ``method="prompt"`` — the note-taking token budget.
    """
    config = config or {}

    if method == "prompt":
        if client is None:
            raise ValueError("method='prompt' requires a client argument")
        m = PromptStrategyMethod(strategy=strategy, client=client, budget=budget)
        return m  # type: ignore[return-value]

    if method == "truncated":
        max_chars = config.get("max_chars", 2000)
        return TruncatedMethod(max_chars=max_chars)  # type: ignore[return-value]

    if method == "amem":
        from .amem_method import AMemMethod

        return AMemMethod(  # type: ignore[return-value]
            llm_model=config.get(
                "llm_model",
                "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            retrieve_k=config.get("retrieve_k", 5),
            enable_evolution=config.get("enable_evolution", True),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            max_ingest_chars=config.get("max_ingest_chars", 15000),
        )

    if method == "lightmem":
        from .lightmem_method import LightMemMethod

        return LightMemMethod(  # type: ignore[return-value]
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_dims=config.get("embedding_dims", 384),
            retrieve_k=config.get("retrieve_k", 10),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            enable_metadata=config.get("enable_metadata", True),
            enable_summary=config.get("enable_summary", True),
            qdrant_dir=config.get("qdrant_dir"),
            max_ingest_chars=config.get("max_ingest_chars", 15000),
        )

    if method == "hipporag":
        from .hipporag_method import HippoRAGMethod

        return HippoRAGMethod(  # type: ignore[return-value]
            llm_model=config.get("llm_model", "openai/gpt-4o-mini"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            retrieve_k=config.get("retrieve_k", 5),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            max_ingest_chars=config.get("max_ingest_chars", 15000),
        )

    if method == "simplemem":
        from .simplemem_method import SimpleMemMethod

        return SimpleMemMethod(  # type: ignore[return-value]
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            retrieve_k=config.get("retrieve_k", 5),
            max_ingest_chars=config.get("max_ingest_chars", 15000),
            enable_planning=config.get("enable_planning", True),
            enable_reflection=config.get("enable_reflection", False),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_dimension=config.get("embedding_dimension", 384),
        )

    if method == "mem0":
        from .mem0_method import Mem0Method

        return Mem0Method(  # type: ignore[return-value]
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            retrieve_k=config.get("retrieve_k", 5),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            max_ingest_chars=config.get("max_ingest_chars", 10000),
        )

    if method == "memorybank":
        from .memorybank_method import MemoryBankMethod

        return MemoryBankMethod(  # type: ignore[return-value]
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            retrieve_k=config.get("retrieve_k", 10),
            chunk_chars=config.get("chunk_chars", 1500),
            decay_hours_scale=config.get("decay_hours_scale", 24.0),
            reinforcement_factor=config.get("reinforcement_factor", 1.5),
            strength_cap=config.get("strength_cap", 100.0),
            max_ingest_chars=config.get("max_ingest_chars", 15000),
        )

    raise ValueError(
        f"Unknown method {method!r}. Choose from: {sorted(MEMORY_METHODS)}"
    )
