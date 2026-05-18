"""
Pipeline memory manager for MemGym.

Exact port of OpenHands CondenserPipeline.
Chains multiple memory managers (condensers) into a single pipeline,
running them sequentially and passing the output of one as input to the next.

Source: openhands/memory/condenser/impl/pipeline.py
"""

from typing import Any, Dict, List, Optional

from .base import BaseMemoryManager, FilteredContext, register_memory_model


class PipelineMemory(BaseMemoryManager):
    """
    Combines multiple memory managers into a single pipeline.

    Exact port of OpenHands CondenserPipeline.

    Each memory manager is run in sequence, passing the output messages of one
    to the next, until we reach the end. If any stage compacts, the overall
    result is marked as compacted.

    OpenHands original:
        def condense(self, view):
            result = view
            for condenser in self.condensers:
                result = condenser.condense(result)
                if isinstance(result, Condensation):
                    break
            return result

    In MemGym, we don't have the Condensation/short-circuit concept since all
    condensers return FilteredContext. We simply chain the stages.

    Args:
        stages: List of BaseMemoryManager instances to chain.
    """

    def __init__(
        self,
        stages: List[BaseMemoryManager],
        max_tokens: int = 4000,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(max_tokens, tokenizer_name, config)
        self.stages = stages

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """
        Chain multiple memory managers sequentially.

        Each stage receives the output of the previous stage as input.
        """
        messages = list(original_context) + [current_observation]
        original_tokens = self.count_tokens(messages)
        any_compacted = False
        stage_stats = []

        for i, stage in enumerate(self.stages):
            # Split messages into context + observation for the stage API
            if len(messages) > 1:
                ctx = messages[:-1]
                obs = messages[-1]
            else:
                ctx = []
                obs = messages[0] if messages else None

            result = stage.manage_context(
                original_context=ctx,
                current_observation=obs,
                metadata=metadata
            )

            # Use the result's content as input for the next stage
            if isinstance(result.content, list):
                messages = result.content
            # else: keep previous messages (shouldn't happen with well-behaved stages)

            if result.metadata.get("was_compacted"):
                any_compacted = True

            stage_stats.append({
                "stage": i,
                "strategy": result.metadata.get("strategy", "unknown"),
                "was_compacted": result.metadata.get("was_compacted", False),
                "messages_after": len(messages),
            })

        filtered_tokens = self.count_tokens(messages)
        self._update_token_tracking(filtered_tokens)
        if any_compacted:
            self._compaction_count += 1

        return FilteredContext(
            content=messages,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": any_compacted,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "pipeline",
                "stage_stats": stage_stats,
                "step": self._step_count
            }
        )

    def reset(self) -> None:
        """Reset all stages."""
        for stage in self.stages:
            stage.reset()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        if hasattr(self, '_legacy_observations'):
            self._legacy_observations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including per-stage stats."""
        base_stats = super().get_stats()
        base_stats["stages"] = [
            {"index": i, "type": stage.__class__.__name__, "stats": stage.get_stats()}
            for i, stage in enumerate(self.stages)
        ]
        return base_stats


# Register in memory model registry
register_memory_model("pipeline", PipelineMemory)
