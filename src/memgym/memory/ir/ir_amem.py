"""A-MEM (Agentic Memory) adapter for IR evaluation.

Bridges AgenticMemorySystem (Zettelkasten + LLM evolution) into the
BaseMemoryManager interface for IR turn-by-turn evaluation.

Overrides the generic A-MEM prompts with IR-specific versions that
focus on entity tracking, bridge-fact extraction, and cross-hop linking.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..external.amem.system import AgenticMemorySystem
from ..external.amem import note as _amem_note
from ..external.amem import system as _amem_system


# ---------------------------------------------------------------------------
# IR-tuned prompts (override the generic A-MEM defaults)
# ---------------------------------------------------------------------------

# 1. Metadata extraction: focus on entities, relationships, and bridge facts
#    rather than generic "salient keywords"
IR_METADATA_PROMPT = """\
You are a research memory system processing search results for a multi-hop \
research question. Extract structured metadata that will help chain facts \
across multiple search steps.

For the following research document, extract:

1. **keywords**: Named entities, methods, datasets, and specific claims \
(numbers, comparisons). Prioritize proper nouns and technical terms that \
might appear in other documents. Order by specificity (most specific first).

2. **context**: One sentence capturing the core factual claim — what entity \
does what, with what result. Include specific numbers or comparisons if present.

3. **tags**: Categories useful for cross-referencing with other documents: \
the research subfield, the type of contribution (method/result/comparison/dataset), \
and the entities involved.

Format as JSON:
{
    "keywords": ["entity1", "method_name", "specific_claim", ...],
    "context": "One sentence: <entity> achieves <result> via <method>",
    "tags": ["subfield", "contribution_type", "entity_name", ...]
}

Content for analysis:
"""

# 2. Evolution: prioritize cross-hop entity connections over generic similarity
IR_EVOLUTION_PROMPT = """\
You are managing a research memory system for a multi-hop investigation. \
A new note has been added. Analyze whether it shares entities or methods \
with existing notes — these cross-references are critical for answering \
multi-hop questions.

The new memory:
context: {context}
content: {content}
keywords: {keywords}

Existing neighbor memories:
{nearest_neighbors_memories}

Determine:
1. Does this note share any named entities, methods, or datasets with \
neighbors? If yes, STRENGTHEN the connection — these cross-hop links are \
how the agent chains facts to answer the research question.
2. Should neighbor tags/context be updated to reflect the new relationship?

The number of neighbors is {neighbor_number}.
Return JSON:
{{
    "should_evolve": true/false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": [neighbor_indices_with_shared_entities],
    "tags_to_update": ["shared_entity", "bridge_to_hop_N", ...],
    "new_context_neighborhood": ["updated context for each neighbor"],
    "new_tags_neighborhood": [["updated", "tags", "for", "each"], ...]
}}"""


def _install_ir_prompts():
    """Monkey-patch A-MEM module prompts with IR-tuned versions."""
    _amem_note.METADATA_PROMPT = IR_METADATA_PROMPT
    _amem_system.EVOLUTION_PROMPT = IR_EVOLUTION_PROMPT


class IRAMemMemory(BaseMemoryManager):
    """A-MEM memory: Zettelkasten notes with LLM-driven evolution.

    Each incoming observation is ingested as a memory note. The system
    auto-generates metadata (keywords, tags, context) and links notes
    via Zettelkasten-style connections. On retrieval, uses embedding
    similarity + graph neighbors for comprehensive recall.

    Uses IR-tuned prompts that focus on entity tracking and bridge-fact
    extraction rather than generic keyword/theme extraction.
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        llm_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        model_name: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 10,
        enable_evolution: bool = True,
        evo_threshold: int = 50,
        question: str = "",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self._llm_model = llm_model
        self._model_name = model_name
        self._retrieve_k = retrieve_k
        self._enable_evolution = enable_evolution
        self._evo_threshold = evo_threshold
        self._api_base = api_base
        self._api_key = api_key

        # Install IR-tuned prompts before creating the system
        _install_ir_prompts()

        self._system = self._create_system()
        self._all_observations: List[str] = []

    def _create_system(self) -> AgenticMemorySystem:
        system = AgenticMemorySystem(
            model_name=self._model_name,
            llm_model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
            retrieve_k=self._retrieve_k,
            enable_evolution=self._enable_evolution,
            evo_threshold=self._evo_threshold,
            verbose=False,
            # Deferred + batched evolution: ingest-time calls skip the
            # per-note evolution LLM call; one batched call before retrieval
            # inspects the full graph, yielding richer linking with ~1/N
            # the LLM calls.
            defer_evolution=True,
        )
        if self._question:
            system.set_task_description(self._question)
        return system

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Ingest the new observation as an A-MEM note."""
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            # Truncate very long observations to avoid overwhelming the LLM
            self._system.add_note(
                content=text[:15000],
                auto_generate_metadata=True,
            )

        tokens = self.count_tokens(self._all_observations)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_amem",
                "num_memories": len(self._system.memories)
                if hasattr(self._system, "memories")
                else 0,
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Retrieve relevant memories using A-MEM's graph-aware search.

        Runs the deferred batched-evolution pass first so cross-note links
        are populated before retrieval. The flush is a no-op when nothing
        has been added since the last flush.
        """
        self._system.flush_evolution()
        k = top_k or self._retrieve_k
        result = self._system.find_related_memories_raw(query=question, k=k)
        return result if result else "(no relevant memories found)"

    def get_notes_text(self) -> str:
        """Return all A-MEM notes as text (for compatibility)."""
        return self._system.get_full_context()

    def reset(self) -> None:
        self._system = self._create_system()
        self._all_observations.clear()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        amem_stats = self._system.get_stats()
        stats.update({
            "num_memories": amem_stats.get("num_memories", 0),
            "evolution_count": amem_stats.get("evolution_count", 0),
            "retrieve_k": self._retrieve_k,
            "enable_evolution": self._enable_evolution,
        })
        return stats


register_memory_model("ir_amem", IRAMemMemory)
