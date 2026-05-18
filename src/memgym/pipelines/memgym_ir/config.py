"""Configuration dataclasses for MemGym-IR pipelines.

Extracted from pipeline.py so both v2 (dataset) and v3 (research) pipelines
can share DifficultyConfig presets without circular imports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DifficultyConfig:
    """Fine-grained difficulty hyperparameters.

    Use from_preset() for named presets, then override individual fields.
    """
    # Token budget
    target_tokens: int = 10000
    min_tokens: int = 5000

    # Distractor counts (-1 = auto/fill budget)
    cross_instance_count: int = -1
    same_topic_count: int = -1
    adversarial_count: int = 2
    contradiction_count: int = 2

    # Hardening transforms
    query_fuzz_intensity: str = "none"    # none / light / heavy
    distractor_scale_factor: float = 1.0
    enable_temporal_shift: bool = False
    enable_contra_placement: bool = True

    # Red herring turns (extra turns with only distractors)
    red_herring_turns: int = 0

    # Reveal mode: immediate (question before docs) or delayed (question after all turns)
    reveal_mode: str = "immediate"

    # Paragraph expansion (deep research realism)
    expand_paragraphs: bool = False
    expansion_target_tokens: int = 600  # Target tokens per expanded paragraph

    # Overlapping documents (paraphrased duplicates across turns)
    overlap_count: int = 0

    # Conflicting authority sources (wrong-answer docs before correct)
    conflicting_sources_count: int = 0

    # Max docs for Condition A verification (increase for deep research)
    max_docs_a: int = 15

    # Memory eviction settings
    eviction_mode: str = "none"       # none / full_eviction / sliding_window
    eviction_window_size: int = 1     # For sliding_window: turns to keep visible
    allow_agent_notes: bool = True    # Let agent write notes between turns
    agent_notes_budget: int = 500     # Max tokens for notes per turn

    # Verification thresholds
    min_a_score: float = 0.5
    min_gap_ab: float = 0.30
    max_c_score: float = 0.4
    pre_verify_threshold: float = 0.3

    @classmethod
    def from_preset(cls, preset: str) -> "DifficultyConfig":
        presets = {
            "easy": cls(
                adversarial_count=4,
                contradiction_count=4,
            ),
            "medium": cls(
                target_tokens=12000,
                adversarial_count=5,
                contradiction_count=5,
                query_fuzz_intensity="light",
                distractor_scale_factor=1.5,
                enable_temporal_shift=True,
                red_herring_turns=1,
                reveal_mode="delayed",
            ),
            "hard": cls(
                target_tokens=15000,
                adversarial_count=6,
                contradiction_count=6,
                query_fuzz_intensity="heavy",
                distractor_scale_factor=2.5,
                enable_temporal_shift=True,
                red_herring_turns=2,
                reveal_mode="delayed",
            ),
            "deep_research_medium": cls(
                target_tokens=50000,
                adversarial_count=8,
                contradiction_count=6,
                query_fuzz_intensity="light",
                distractor_scale_factor=1.0,
                enable_temporal_shift=True,
                red_herring_turns=3,
                reveal_mode="delayed",
                expand_paragraphs=True,
                expansion_target_tokens=600,
                overlap_count=2,
                conflicting_sources_count=2,
                max_docs_a=30,
                min_a_score=0.4,
            ),
            "deep_research_hard": cls(
                target_tokens=100000,
                adversarial_count=12,
                contradiction_count=10,
                query_fuzz_intensity="heavy",
                distractor_scale_factor=1.0,
                enable_temporal_shift=True,
                red_herring_turns=5,
                reveal_mode="delayed",
                expand_paragraphs=True,
                expansion_target_tokens=800,
                overlap_count=3,
                conflicting_sources_count=4,
                max_docs_a=30,
                min_a_score=0.4,
            ),
            # Memory-focused presets (eviction forces actual memory usage)
            "memory_easy": cls(
                target_tokens=15000,
                adversarial_count=4,
                contradiction_count=4,
                red_herring_turns=1,
                reveal_mode="delayed",
                eviction_mode="sliding_window",
                eviction_window_size=2,
                allow_agent_notes=True,
                agent_notes_budget=400,
            ),
            "memory_medium": cls(
                target_tokens=25000,
                adversarial_count=6,
                contradiction_count=4,
                red_herring_turns=2,
                reveal_mode="delayed",
                expand_paragraphs=True,
                expansion_target_tokens=600,
                eviction_mode="sliding_window",
                eviction_window_size=1,
                allow_agent_notes=True,
                agent_notes_budget=200,
            ),
            "memory_hard": cls(
                target_tokens=50000,
                adversarial_count=8,
                contradiction_count=6,
                query_fuzz_intensity="light",
                red_herring_turns=3,
                reveal_mode="delayed",
                expand_paragraphs=True,
                expansion_target_tokens=800,
                overlap_count=2,
                conflicting_sources_count=2,
                eviction_mode="full_eviction",
                eviction_window_size=1,
                allow_agent_notes=True,
                agent_notes_budget=150,
            ),
            "memory_extreme": cls(
                target_tokens=50000,
                adversarial_count=8,
                contradiction_count=6,
                query_fuzz_intensity="light",
                red_herring_turns=4,
                reveal_mode="delayed",
                expand_paragraphs=True,
                expansion_target_tokens=800,
                overlap_count=2,
                conflicting_sources_count=3,
                eviction_mode="full_eviction",
                eviction_window_size=1,
                allow_agent_notes=True,
                agent_notes_budget=100,
                max_docs_a=30,
                min_a_score=0.4,
                min_gap_ab=0.25,
            ),
            # Research-first pipeline presets (grow-by-search, 3 tiers)
            # 10K: 3 hops, light noise — entry level
            "research_medium": cls(
                target_tokens=10000,
                adversarial_count=3,
                contradiction_count=2,
                expand_paragraphs=True,
                expansion_target_tokens=400,
                eviction_mode="full_eviction",
                allow_agent_notes=True,
                agent_notes_budget=300,
            ),
            # 100K: 5 hops, heavy near-misses — production
            "research_hard": cls(
                target_tokens=100000,
                adversarial_count=6,
                contradiction_count=4,
                expand_paragraphs=True,
                expansion_target_tokens=800,
                eviction_mode="full_eviction",
                allow_agent_notes=True,
                agent_notes_budget=200,
            ),
            # 1M: 8+ hops, massive noise — stress test
            "research_extreme": cls(
                target_tokens=1000000,
                adversarial_count=10,
                contradiction_count=6,
                expand_paragraphs=True,
                expansion_target_tokens=1500,
                eviction_mode="full_eviction",
                allow_agent_notes=True,
                agent_notes_budget=150,
            ),
            # Backward compat alias (= research_hard)
            "research_memory": cls(
                target_tokens=100000,
                adversarial_count=6,
                contradiction_count=4,
                expand_paragraphs=True,
                expansion_target_tokens=800,
                eviction_mode="full_eviction",
                allow_agent_notes=True,
                agent_notes_budget=200,
            ),
        }
        return presets.get(preset, cls())


@dataclass
class PipelineConfig:
    """Configuration for the MemGym-IR v2 pipeline."""
    output_dir: Path = field(default_factory=lambda: Path("./output_ir"))
    limit: Optional[int] = None

    # Dataset
    dataset: str = "musique"

    # Model settings
    worker_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    verifier_model: str = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Filter criteria
    min_hops: int = 3
    min_supporting: int = 2
    min_answer_length: int = 2
    min_question_length: int = 30

    # Difficulty (preset name, used if difficulty_config not provided)
    difficulty_preset: str = "medium"

    # Fine-grained difficulty overrides (applied on top of preset)
    difficulty_config: Optional[DifficultyConfig] = None

    # Token budget (legacy, overridden by difficulty_config.target_tokens)
    target_tokens: int = 10000

    # Verification
    skip_verification: bool = False

    # Checkpointing
    resume: bool = True

    # Dry run (skip LLM calls)
    dry_run: bool = False

    # Hop upgrade (fictionalize + extend chain)
    upgrade_hops: Optional[int] = None  # Target hops (e.g., 6). None = no upgrade.

    # Memory quality gate
    min_memory_facts: int = 1

    # Parallel processing
    max_concurrent: int = 10

    def get_difficulty_config(self) -> DifficultyConfig:
        """Build final DifficultyConfig from preset + overrides."""
        if self.difficulty_config is not None:
            return self.difficulty_config
        config = DifficultyConfig.from_preset(self.difficulty_preset)
        # Legacy target_tokens override
        if self.target_tokens != 10000:
            config.target_tokens = self.target_tokens
        return config
