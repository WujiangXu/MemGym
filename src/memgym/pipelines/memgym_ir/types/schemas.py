"""Pydantic models for MemGym-IR v2 pipeline data structures."""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class GroundingFactIR(BaseModel):
    """A grounding fact extracted from a supporting paragraph."""
    fact_id: str = Field(description="Unique identifier (e.g., F1, F2)")
    content: str = Field(description="The factual statement")
    fact_type: Literal[
        "bridge_fact", "terminal_fact", "temporal_fact", "numerical_fact"
    ] = Field(description="Type of grounding fact")
    source_paragraph_title: str = Field(
        description="Title of the paragraph containing this fact"
    )
    needed_at_hop: int = Field(description="Which hop needs this fact")
    first_available_at_hop: int = Field(
        description="When the system first encounters this fact"
    )
    retention_span: int = Field(
        description="How many turns it must be retained (needed_at_hop - first_available_at_hop)"
    )
    is_discoverable: bool = Field(
        default=False,
        description="Can be inferred without the paragraph",
    )
    intermediate_answer: str = Field(
        default="",
        description="The intermediate answer this fact provides (for bridge facts)",
    )


class DistractorFactIR(BaseModel):
    """A distractor fact injected for noise."""
    id: str = Field(description="Unique identifier (e.g., D1, D2)")
    content: str = Field(description="The distractor paragraph content")
    source: Literal[
        "cross_instance", "same_topic", "adversarial", "same_entity",
        "overlap", "conflicting_authority", "dead_end", "filler",
        "natural", "enhanced", "sibling_near_miss",
    ] = Field(description="Origin of the distractor")
    target_turn: int = Field(description="Which turn it was injected into")
    interferes_with: Optional[str] = Field(
        default=None,
        description="Fact ID of the grounding fact it could confuse",
    )


class SearchTurn(BaseModel):
    """A single search turn in the multi-hop evaluation."""
    turn_index: int = Field(description="0-based turn index")
    sub_query: str = Field(description="The sub-question for this turn")
    sub_answer: str = Field(
        default="", description="The intermediate answer for this hop"
    )
    documents: List[Dict] = Field(
        default_factory=list,
        description="List of documents: [{title, text, is_supporting}]",
    )
    is_red_herring: bool = Field(
        default=False,
        description="True if this turn contains only distractors (no supporting docs)",
    )


class ContextTokensIR(BaseModel):
    """Token counts for each context component."""
    supporting_paragraphs: int = Field(
        default=0, description="Tokens in gold supporting docs"
    )
    cross_instance_noise: int = Field(default=0)
    same_topic_noise: int = Field(default=0)
    adversarial_noise: int = Field(default=0)
    same_entity_contradictions: int = Field(default=0)
    expanded_context: int = Field(
        default=0, description="Extra tokens from paragraph expansion"
    )
    redundant_overlap: int = Field(
        default=0, description="Tokens from overlapping paraphrased docs"
    )
    dead_end_noise: int = Field(
        default=0, description="Tokens from dead-end search result docs"
    )
    conflicting_authority: int = Field(
        default=0, description="Tokens from conflicting authority docs"
    )
    filler_noise: int = Field(default=0)
    total: int = Field(default=0, description="Sum of all components")


class RubricIR(BaseModel):
    """Evaluation rubric for an IR instance."""
    expected_answer: str = Field(description="Gold answer")
    retention_facts: List[str] = Field(
        default_factory=list,
        description="Facts that MUST be in memory at answer time",
    )
    noise_contradictions: List[str] = Field(
        default_factory=list,
        description="Facts that MUST NOT be in memory",
    )
    temporal_facts: List[Dict] = Field(
        default_factory=list,
        description="Facts with retention_span metadata",
    )
    per_hop_answers: List[Dict] = Field(
        default_factory=list,
        description="Intermediate answers for per-hop scoring",
    )
    bridge_dependencies: List[Dict] = Field(
        default_factory=list,
        description="Which facts bridge which hops",
    )
    per_turn_retention: List[Dict] = Field(
        default_factory=list,
        description="Per-turn memory map: [{turn, available_facts, evicted_facts, critical}]",
    )


class EvictionPolicy(BaseModel):
    """Specifies how earlier turn documents are removed from context during evaluation."""
    mode: Literal["full_eviction", "sliding_window"] = Field(
        default="full_eviction",
        description="full_eviction: only current turn visible; "
                    "sliding_window: last K turns visible",
    )
    window_size: int = Field(
        default=1,
        description="For sliding_window: number of recent turns to keep visible",
    )
    allow_notes: bool = Field(
        default=True,
        description="Whether the agent may write notes between turns",
    )
    notes_budget_tokens: int = Field(
        default=500,
        description="Max tokens for agent notes per turn",
    )


class FilteredIRInstance(BaseModel):
    """Intermediate representation of a filtered MuSiQue instance."""
    instance_id: str = Field(description="e.g., musique__2hop__12345")
    source_dataset: str = Field(description="musique or 2wiki")
    question: str = Field(description="The multi-hop question")
    answer: str = Field(description="Gold answer")
    num_hops: int = Field(description="Number of reasoning hops")
    decomposition: List[Dict] = Field(
        default_factory=list,
        description="[{sub_question, sub_answer, paragraph_title, paragraph_text}]",
    )
    supporting_paragraphs: List[Dict] = Field(
        default_factory=list,
        description="[{title, text, is_supporting=True}]",
    )
    distractor_paragraphs: List[Dict] = Field(
        default_factory=list,
        description="[{title, text, is_supporting=False}] from dataset",
    )
    question_type: str = Field(
        default="bridge", description="bridge | comparison | intersection"
    )

    class Config:
        extra = "allow"


class MemGymIRInstance(BaseModel):
    """Complete MemGym-IR v2 benchmark instance."""
    # Identity
    instance_id: str = Field(description="memgym_ir__musique__12345")
    base_instance: str = Field(description="Original dataset instance ID")
    source_dataset: str = Field(description="musique or 2wiki")

    # Task
    question: str = Field(description="The multi-hop question (possibly fuzzed)")
    original_question: str = Field(description="Pre-fuzz question for reference")
    answer: str = Field(description="Gold answer")

    # Structure
    num_hops: int = Field(default=0)
    decomposition: List[Dict] = Field(default_factory=list)

    # Search turns
    turns: List[SearchTurn] = Field(default_factory=list)

    # Grounding
    grounding_facts: List[GroundingFactIR] = Field(default_factory=list)
    distractors: List[DistractorFactIR] = Field(default_factory=list)

    # Memory files
    memory_files: Dict[str, str] = Field(
        default_factory=dict,
        description="filename -> content (assembled search results)",
    )

    # Evaluation metadata
    gold_supporting_paragraphs: List[Dict] = Field(default_factory=list)
    contradictions: List[Dict] = Field(default_factory=list)

    # Difficulty
    transforms_applied: List[str] = Field(default_factory=list)
    difficulty_preset: str = Field(default="easy")
    reveal_mode: Literal["immediate", "delayed"] = Field(
        default="immediate",
        description="immediate: question shown before docs; delayed: question shown after all turns",
    )

    # Memory eviction (for memory-required evaluation)
    eviction_policy: Optional[EvictionPolicy] = Field(
        default=None,
        description="How to present turns with context eviction at eval time",
    )
    memory_required_facts: List[str] = Field(
        default_factory=list,
        description="Fact IDs only available in evicted turns (must be remembered)",
    )

    # Token counts
    context_tokens: Optional[ContextTokensIR] = Field(default=None)
    total_tokens: int = Field(default=0)

    # Verification
    verification: Optional[Dict] = Field(default=None)

    # Rubric
    rubric: Optional[RubricIR] = Field(default=None)

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()


# =========================================================================
# Grow-by-search types
# =========================================================================


class HopRecord(BaseModel):
    """Record of a single hop in the grow-by-search process."""
    hop_index: int
    query: str
    fact: GroundingFactIR
    supporting_doc_indices: List[int] = Field(default_factory=list)
    results: List[Dict] = Field(default_factory=list)  # SearchResult dicts

    def siblings(self) -> List[Dict]:
        """Non-selected search results for this hop.

        By construction, every hop-k search query is derived from the answer of
        hop-(k-1), so all K results in :attr:`results` share the full chain of
        context up to the point of divergence at hop k. The non-selected ones
        are therefore the best possible seed material for near-miss distractors:
        they are globally indistinguishable from the supporting doc until the
        hop-k answer fact, which is exactly where we want the reader to have to
        consult memory.
        """
        supporting = set(self.supporting_doc_indices)
        return [r for i, r in enumerate(self.results) if i not in supporting]


class GrowthResult(BaseModel):
    """Output of the grow-by-search question builder."""
    question: str = Field(description="Final synthesized multi-hop question")
    answer: str = Field(description="Gold answer derived from fact chain")
    topic: str = Field(default="", description="Original seed topic")
    hops: List[HopRecord] = Field(default_factory=list)
    facts: List[GroundingFactIR] = Field(default_factory=list)
    reasoning_trace: str = Field(default="", description="F1 + F2 + ... = answer")
    seed_question: str = Field(default="", description="Original seed question")
    num_hops: int = Field(default=0)

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()
