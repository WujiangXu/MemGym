"""Pydantic models for Coding-Synthetic pipeline v2 data structures.

V2 uses grounding facts (packaged as natural documents) instead of
multi-turn conversations.  Memory items become GroundingFact objects
stored in files alongside the repo, not distributed across dialogue turns.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Kept unchanged from v1
# ---------------------------------------------------------------------------

class PatchMetrics(BaseModel):
    """Metrics extracted from a unified diff patch."""
    lines_added: int = Field(description="Number of lines added")
    lines_removed: int = Field(description="Number of lines removed")
    lines_changed: int = Field(description="Total lines changed (added + removed)")
    num_hunks: int = Field(description="Number of distinct hunks in the patch")
    files_changed: List[str] = Field(description="List of files modified")


class FilteredInstance(BaseModel):
    """Intermediate representation of a filtered SWE-smith instance."""
    instance_id: str = Field(description="Original SWE-smith instance ID")
    problem_statement: str = Field(description="The problem description")
    patch: str = Field(description="The gold patch (unified diff)")
    patch_metrics: PatchMetrics = Field(description="Metrics from patch analysis")
    FAIL_TO_PASS: List[str] = Field(description="Tests that should pass after fix")
    PASS_TO_PASS: List[str] = Field(description="Tests that should continue passing")
    image_name: str = Field(description="Docker image name for testing")
    repo: str = Field(description="Repository name")
    base_commit: str = Field(description="Base commit hash")

    class Config:
        extra = "allow"  # Allow extra fields from SWE-smith dataset


# ---------------------------------------------------------------------------
# V2 core models
# ---------------------------------------------------------------------------

class GroundingFact(BaseModel):
    """A grounding fact extracted from the patch."""
    id: str = Field(description="Unique identifier (e.g., F1, F2)")
    content: str = Field(description="The factual statement (the memory payload — contains the specific value/detail)")
    topic: Optional[str] = Field(
        default=None,
        description=(
            "What the fact is ABOUT, without stating the specific answer. "
            "Seeds QA synthesis so the question never contains the answer. "
            "Example: content='The SYM parser uses 0x7FF as the frame-type "
            "threshold'; topic='the frame-type threshold the SYM parser uses'."
        ),
    )
    category: Literal[
        "user_requirement",
        "bug_description",
        "repro_condition",
        "error_detail",
        "debug_finding",
        "api_behavior",
        "design_decision",
        "constraint",
    ] = Field(description="Category of the grounding fact")
    discoverable: bool = Field(
        description="True if this fact can be found by reading the repo"
    )
    relevance: Literal["critical", "helpful", "background"] = Field(
        description="How important this fact is for generating the fix"
    )
    hunks_enabled: List[int] = Field(
        default_factory=list,
        description="Which patch hunk indices (0-based) this fact enables",
    )
    evidence: str = Field(
        default="",
        description="Why the fact is classified as discoverable or not",
    )


class DistractorFact(BaseModel):
    """A distractor fact for noise."""
    id: str = Field(description="Unique identifier (e.g., D1, D2)")
    content: str = Field(description="The distractor content")
    source: Literal[
        "cross_instance", "same_repo", "adversarial", "same_function",
        "adversarial_qa", "adversarial_expansion",
    ] = Field(description="Origin of the distractor")
    target_qa_id: Optional[str] = Field(
        default=None,
        description="QA pair this distractor targets (for adversarial_qa source)",
    )


class ContextTokens(BaseModel):
    """Token counts for each context component."""
    task_prompt: int = Field(default=0, description="Tokens in task prompt")
    memory_files: int = Field(default=0, description="Total tokens across all memory files")
    repo_context: int = Field(default=0, description="Total tokens in repo context")
    grounding_facts: int = Field(default=0, description="Tokens in raw grounding facts")
    distractors: int = Field(default=0, description="Tokens in distractor content")
    total: int = Field(default=0, description="Sum of all components")


class MemGymInstance(BaseModel):
    """V2 instance -- fact-based, no conversations."""
    instance_id: str = Field(description="MemGym instance ID (memgym__<original_id>)")
    base_instance: str = Field(description="Original SWE-smith instance ID")
    task_prompt: str = Field(description="Vague problem description for the agent")
    memory_files: Dict[str, str] = Field(
        default_factory=dict,
        description="filename -> content (natural documents with grounding facts)",
    )
    grounding_facts: List[GroundingFact] = Field(default_factory=list)
    distractors: List[DistractorFact] = Field(default_factory=list)
    gold_fix: str = Field(default="", description="Reversed patch (buggy->correct)")
    transforms_applied: List[str] = Field(
        default_factory=list,
        description="Names of difficulty transforms applied to this instance",
    )
    difficulty_preset: str = Field(
        default="medium",
        description="Difficulty preset used (easy/medium/hard)",
    )
    # Standard SWE-smith fields
    patch: str = Field(description="The gold patch (bug-introducing diff)")
    FAIL_TO_PASS: List[str] = Field(default_factory=list)
    PASS_TO_PASS: List[str] = Field(default_factory=list)
    image_name: str = Field(default="", description="Docker image for testing")
    referenced_files: List[str] = Field(default_factory=list)
    # Synthetic repo context (built from patch + tests, no Docker needed)
    repo_context: str = Field(
        default="",
        description="Synthetic repo skeleton derived from patch and test info",
    )
    # Token counts
    context_tokens: Optional[ContextTokens] = Field(
        default=None,
        description="Token counts for each context component",
    )
    # Verification results (populated after verify)
    verification: Optional[Dict] = Field(
        default=None,
        description="Ablation verification results",
    )

    def to_jsonl_dict(self) -> dict:
        """Convert to dictionary for JSONL serialization."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# JSON schemas for LLM structured output
# ---------------------------------------------------------------------------

GROUNDING_FACT_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounding_fact_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "topic": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "user_requirement",
                                    "bug_description",
                                    "repro_condition",
                                    "error_detail",
                                    "debug_finding",
                                    "api_behavior",
                                    "design_decision",
                                    "constraint",
                                ],
                            },
                            "discoverable": {"type": "boolean"},
                            "relevance": {
                                "type": "string",
                                "enum": ["critical", "helpful", "background"],
                            },
                            "hunks_enabled": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "evidence": {"type": "string"},
                        },
                        "required": [
                            "id",
                            "content",
                            "topic",
                            "category",
                            "discoverable",
                            "relevance",
                            "hunks_enabled",
                            "evidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "referenced_files": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["facts", "referenced_files"],
            "additionalProperties": False,
        },
    },
}


FACT_VERIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reclassifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "new_discoverable": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["fact_id", "new_discoverable", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["reclassifications"],
            "additionalProperties": False,
        },
    },
}


TASK_PROMPT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_prompt",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
            },
            "required": ["task_prompt"],
            "additionalProperties": False,
        },
    },
}


ADVERSARIAL_DISTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "adversarial_distractors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["id", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
    },
}


MEMORY_FILE_ASSEMBLY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_file_assembly",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["filename", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["files"],
            "additionalProperties": False,
        },
    },
}


PROMPT_FUZZ_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "prompt_fuzz",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fuzzed_prompt": {"type": "string"},
                "changes_made": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["fuzzed_prompt", "changes_made"],
            "additionalProperties": False,
        },
    },
}


INDIRECTION_TRANSFORM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "indirection_transform",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_content": {"type": "string"},
                "indirections_added": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "indirect": {"type": "string"},
                        },
                        "required": ["original", "indirect"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rewritten_content", "indirections_added"],
            "additionalProperties": False,
        },
    },
}


LLM_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "llm_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["score", "explanation"],
            "additionalProperties": False,
        },
    },
}


FACT_SEED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_seeds",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "seeds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "user_requirement",
                                    "bug_description",
                                    "repro_condition",
                                    "error_detail",
                                    "debug_finding",
                                    "api_behavior",
                                    "design_decision",
                                    "constraint",
                                ],
                            },
                            "hunk_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["question", "category", "hunk_indices"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["seeds"],
            "additionalProperties": False,
        },
    },
}


SAME_FUNCTION_DISTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "same_function_distractors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["id", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
    },
}


FACT_FRAGMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_fragmentation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fragments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_fact_id": {"type": "string"},
                            "target_file": {"type": "string"},
                            "partial_content": {"type": "string"},
                        },
                        "required": [
                            "source_fact_id",
                            "target_file",
                            "partial_content",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["fragments"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Coding QA types — fact-level memory evaluation
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    """A behavioral question linked to grounding facts."""
    qa_id: str = Field(description="Unique identifier (e.g., Q1, Q2)")
    question: str = Field(description="Behavioral question about code")
    gold_answer: str = Field(description="Expected answer derived from linked facts")
    linked_fact_ids: List[str] = Field(
        default_factory=list,
        description="GroundingFact IDs needed to answer this question",
    )
    category: str = Field(
        default="",
        description="Primary category (from GroundingFact categories)",
    )
    requires_memory: bool = Field(
        default=False,
        description="True if linked facts include non-discoverable ones",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="easy=all discoverable, medium=1-2 memory facts, hard=3+ memory facts or needle-in-haystack",
    )
    num_linked_facts: int = Field(
        default=1,
        description="Number of grounding facts linked to this question",
    )
    signal_tokens: int = Field(
        default=0,
        description="Total tokens in linked facts (needle size)",
    )
    requires_synthesis: bool = Field(
        default=False,
        description="True if answering requires combining 2+ facts",
    )


class CodingQAInstance(BaseModel):
    """Coding QA benchmark instance for fact-level memory evaluation."""
    instance_id: str = Field(description="coding_qa__<original_id>")
    base_instance: str = Field(description="Original coding_synthetic instance ID")

    # Context (what the agent sees)
    task_prompt: str = Field(description="Vague bug description")
    memory_files: Dict[str, str] = Field(
        default_factory=dict,
        description="filename -> prose content with embedded facts",
    )
    repo_context: str = Field(default="", description="Synthetic repo skeleton")
    repo_files: Dict[str, str] = Field(
        default_factory=dict,
        description="file_path -> full file content extracted from Docker image",
    )
    image_name: str = Field(default="", description="Docker image for full-repo mode")
    referenced_files: List[str] = Field(default_factory=list)

    # Questions
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="5-15 behavioral questions per instance",
    )

    # Ground truth (hidden from agent during eval)
    grounding_facts: List[GroundingFact] = Field(default_factory=list)
    distractors: List[DistractorFact] = Field(default_factory=list)
    patch: str = Field(default="", description="Gold patch for reference")

    # Metadata
    num_critical_facts: int = Field(default=0)
    num_external_facts: int = Field(default=0, description="discoverable=False count")
    difficulty_preset: str = Field(default="medium")
    target_tokens: Optional[int] = Field(
        default=None,
        description="Target token budget used for expansion (None=no expansion)",
    )

    # Runtime fact verification (the training phase — advisory; populated by stages/fact_tests)
    fact_test_results: Dict[str, Dict] = Field(
        default_factory=dict,
        description="fact_id -> FactTestResult.model_dump(); absent=not tested",
    )

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()


class FactScore(BaseModel):
    """Per-fact scoring result."""
    fact_id: str = Field(description="GroundingFact ID")
    recalled: bool = Field(description="Whether the fact was found in the answer")
    confidence: float = Field(default=0.0, description="Judge confidence 0.0-1.0")


class QAEvalResult(BaseModel):
    """Evaluation result for one instance across all questions."""
    instance_id: str
    strategy: str = Field(default="full_context")

    # Per-QA scores
    qa_scores: List[Dict] = Field(
        default_factory=list,
        description="[{qa_id, score, predicted_answer, gold_answer}]",
    )

    # Per-fact scores
    fact_scores: List[FactScore] = Field(default_factory=list)

    # Aggregate metrics
    fact_recall: float = Field(default=0.0, description="Overall fact recall")
    critical_fact_recall: float = Field(default=0.0)
    external_fact_recall: float = Field(
        default=0.0, description="Recall on discoverable=False facts only"
    )
    discoverable_fact_recall: float = Field(
        default=0.0, description="Recall on discoverable=True facts only"
    )
    noise_rejection: float = Field(
        default=0.0, description="1 - distractor acceptance rate"
    )
    qa_accuracy: float = Field(
        default=0.0, description="Fraction of QA pairs answered correctly"
    )
    category_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-category recall {api_behavior: 0.8, ...}",
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Diagnostics (ingest call count, bytes dropped, etc.)",
    )

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()


class QAVerificationResult(BaseModel):
    """Per-QA-pair verification result from the verify-qa stage."""
    qa_id: str
    question: str = ""
    requires_memory: bool = False
    # Check A: solvability (facts → answer)
    positive_score: float = Field(default=0.0, description="Fact recall from facts-only answer")
    positive_answer: str = Field(default="", description="LLM answer from facts alone")
    positive_facts_used: List[str] = Field(default_factory=list, description="Fact IDs cited by LLM")
    positive_passed: bool = False
    # Check B: distractor confusion (distractors → answer)
    distractor_score: float = Field(default=0.0, description="Fact recall from distractor-only answer")
    distractor_answer: str = Field(default="", description="LLM answer from distractors alone")
    distractor_passed: bool = Field(default=True, description="True = distractors did NOT help (good)")
    # Check C: question leakage (question alone → answer)
    leakage_score: float = Field(default=0.0, description="Fact recall from question-only answer")
    leakage_answer: str = Field(default="", description="LLM answer from question alone")
    leakage_passed: bool = Field(default=True, description="True = question does NOT leak (good)")
    # Verdict
    verdict: Literal["valid", "broken", "confusable", "leaky", "confusable_and_leaky"] = "broken"


class InstanceVerificationResult(BaseModel):
    """Per-instance verification summary from the verify-qa stage."""
    instance_id: str
    image_name: str = ""
    num_qa_total: int = 0
    num_valid: int = 0
    num_broken: int = 0
    num_confusable: int = 0
    num_leaky: int = 0
    num_confusable_and_leaky: int = 0
    num_after_dedup: int = Field(default=0, description="Valid QA pairs remaining after dedup+cap")
    num_removed_by_dedup: int = Field(default=0, description="Valid QA pairs removed by dedup or cap")
    qa_results: List[QAVerificationResult] = Field(default_factory=list)
    kept: bool = False
    kept_qa_ids: List[str] = Field(default_factory=list)

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# JSON schemas for Coding QA LLM structured output
# ---------------------------------------------------------------------------

FACT_QA_GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_qa_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["question", "category"],
            "additionalProperties": False,
        },
    },
}

CODING_QA_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "coding_qa_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "sources_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "confidence", "sources_used"],
            "additionalProperties": False,
        },
    },
}

FACT_JUDGE_QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contains_fact": {"type": "boolean"},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["contains_fact", "confidence", "explanation"],
            "additionalProperties": False,
        },
    },
}

NOISE_JUDGE_QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "noise_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "misled": {"type": "boolean"},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["misled", "confidence", "explanation"],
            "additionalProperties": False,
        },
    },
}

VERIFY_QA_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verify_qa_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "facts_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sources_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "confidence", "facts_used", "sources_used"],
            "additionalProperties": False,
        },
    },
}

GOLD_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "gold_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

NOTE_TAKING_QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "note_taking",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
            },
            "required": ["notes"],
            "additionalProperties": False,
        },
    },
}

FACT_RETRIEVAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_retrieval",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "source_file": {"type": "string"},
                            "relevance": {"type": "string"},
                        },
                        "required": ["content", "source_file", "relevance"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}

MULTI_FACT_QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "multi_fact_qa",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["question", "category"],
            "additionalProperties": False,
        },
    },
}

ADVERSARIAL_QA_DISTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "adversarial_qa_distractors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["id", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
    },
}
