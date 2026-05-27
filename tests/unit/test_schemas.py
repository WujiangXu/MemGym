"""Tests for v2 Pydantic schema models."""

import pytest

pytest.importorskip(
    "datasets",
    reason=(
        "coding-synth pipeline eagerly imports `datasets` via its package "
        "__init__. Install with `uv pip install -e .[swe]` to run this suite."
    ),
)

from memgym.pipelines.coding_synthetic.types.schemas import (
    GroundingFact,
    DistractorFact,
    ContextTokens,
    MemGymInstance,
    FilteredInstance,
    PatchMetrics,
)


class TestGroundingFact:
    """Tests for GroundingFact model."""

    def test_create_valid_fact(self):
        fact = GroundingFact(
            id="F1",
            content="The OAuth scope parameter must be returned as a list",
            category="user_requirement",
            discoverable=False,
            relevance="critical",
            hunks_enabled=[0, 1],
            evidence="Only mentioned in external RFC docs",
        )
        assert fact.id == "F1"
        assert fact.category == "user_requirement"
        assert fact.discoverable is False
        assert fact.relevance == "critical"
        assert fact.hunks_enabled == [0, 1]

    def test_fact_defaults(self):
        fact = GroundingFact(
            id="F2",
            content="Test content",
            category="bug_description",
            discoverable=True,
            relevance="helpful",
        )
        assert fact.hunks_enabled == []
        assert fact.evidence == ""

    def test_all_categories_valid(self):
        categories = [
            "user_requirement", "bug_description", "repro_condition",
            "error_detail", "debug_finding", "api_behavior",
            "design_decision", "constraint",
        ]
        for cat in categories:
            fact = GroundingFact(
                id="F1", content="c", category=cat,
                discoverable=True, relevance="background",
            )
            assert fact.category == cat

    def test_invalid_category_raises(self):
        with pytest.raises(Exception):
            GroundingFact(
                id="F1", content="c", category="invalid_cat",
                discoverable=True, relevance="critical",
            )

    def test_invalid_relevance_raises(self):
        with pytest.raises(Exception):
            GroundingFact(
                id="F1", content="c", category="bug_description",
                discoverable=True, relevance="very_important",
            )

    def test_all_relevance_levels(self):
        for rel in ["critical", "helpful", "background"]:
            fact = GroundingFact(
                id="F1", content="c", category="bug_description",
                discoverable=True, relevance=rel,
            )
            assert fact.relevance == rel


class TestDistractorFact:
    """Tests for DistractorFact model."""

    def test_create_distractor(self):
        d = DistractorFact(
            id="D1",
            content="Irrelevant API method documentation",
            source="cross_instance",
        )
        assert d.id == "D1"
        assert d.source == "cross_instance"

    def test_all_sources_valid(self):
        for src in ["cross_instance", "same_repo", "adversarial", "same_function"]:
            d = DistractorFact(id="D1", content="c", source=src)
            assert d.source == src

    def test_invalid_source_raises(self):
        with pytest.raises(Exception):
            DistractorFact(id="D1", content="c", source="unknown_source")


class TestPatchMetrics:
    """Tests for PatchMetrics model."""

    def test_create_patch_metrics(self):
        metrics = PatchMetrics(
            lines_added=10, lines_removed=5, lines_changed=15,
            num_hunks=3, files_changed=["a.py", "b.py"],
        )
        assert metrics.lines_changed == 15
        assert len(metrics.files_changed) == 2


class TestFilteredInstance:
    """Tests for FilteredInstance model."""

    def test_create_filtered_instance(self):
        instance = FilteredInstance(
            instance_id="test__repo-001",
            problem_statement="Bug description here",
            patch="--- a/file.py\n+++ b/file.py",
            patch_metrics=PatchMetrics(
                lines_added=5, lines_removed=3, lines_changed=8,
                num_hunks=2, files_changed=["file.py"],
            ),
            FAIL_TO_PASS=["test_1", "test_2"],
            PASS_TO_PASS=["test_3"],
            image_name="docker.io/test:latest",
            repo="owner/repo",
            base_commit="abc123",
        )
        assert instance.instance_id == "test__repo-001"
        assert len(instance.FAIL_TO_PASS) == 2

    def test_extra_fields_allowed(self):
        """FilteredInstance has extra='allow' for SWE-smith fields."""
        instance = FilteredInstance(
            instance_id="test",
            problem_statement="p",
            patch="p",
            patch_metrics=PatchMetrics(
                lines_added=1, lines_removed=0, lines_changed=1,
                num_hunks=1, files_changed=["f.py"],
            ),
            FAIL_TO_PASS=[],
            PASS_TO_PASS=[],
            image_name="img",
            repo="r",
            base_commit="c",
            some_extra_field="allowed",
        )
        assert instance.instance_id == "test"


class TestMemGymInstance:
    """Tests for MemGymInstance v2 model."""

    @pytest.fixture
    def sample_instance(self):
        return MemGymInstance(
            instance_id="memgym__test-001",
            base_instance="test-001",
            task_prompt="Something is broken in the OAuth handling",
            memory_files={
                "debug_notes.md": "During debugging we found scope issues...",
                "code_review.md": "Review noted sorting was inconsistent...",
            },
            grounding_facts=[
                GroundingFact(
                    id="F1",
                    content="scope_to_list must return sorted list",
                    category="user_requirement",
                    discoverable=False,
                    relevance="critical",
                    hunks_enabled=[0],
                ),
                GroundingFact(
                    id="F2",
                    content="Function raises ValueError on empty input",
                    category="bug_description",
                    discoverable=True,
                    relevance="helpful",
                    hunks_enabled=[0, 1],
                ),
            ],
            distractors=[
                DistractorFact(id="D1", content="Unrelated fact", source="cross_instance"),
                DistractorFact(id="D2", content="Wrong behavior", source="same_function"),
            ],
            gold_fix="--- b/oauth/utils.py\n+++ a/oauth/utils.py\n@@ -10,3 +10,1 @@\n-new\n+old",
            transforms_applied=["prompt_fuzz", "indirection"],
            difficulty_preset="medium",
            patch="--- a/oauth/utils.py\n+++ b/oauth/utils.py\n@@ -10,1 +10,3 @@\n-old\n+new",
            FAIL_TO_PASS=["test_scope"],
            PASS_TO_PASS=["test_other"],
            image_name="docker.io/test:latest",
            referenced_files=["oauth/utils.py"],
        )

    def test_create_instance(self, sample_instance):
        assert sample_instance.instance_id == "memgym__test-001"
        assert len(sample_instance.grounding_facts) == 2
        assert len(sample_instance.distractors) == 2
        assert len(sample_instance.memory_files) == 2

    def test_defaults(self):
        instance = MemGymInstance(
            instance_id="memgym__min",
            base_instance="min",
            task_prompt="Fix the bug",
            patch="--- a/f.py\n+++ b/f.py",
        )
        assert instance.memory_files == {}
        assert instance.grounding_facts == []
        assert instance.distractors == []
        assert instance.transforms_applied == []
        assert instance.difficulty_preset == "medium"
        assert instance.gold_fix == ""
        assert instance.referenced_files == []
        assert instance.context_tokens is None
        assert instance.verification is None

    def test_to_jsonl_dict(self, sample_instance):
        result = sample_instance.to_jsonl_dict()
        assert isinstance(result, dict)
        assert result["instance_id"] == "memgym__test-001"
        assert "grounding_facts" in result
        assert "distractors" in result
        assert "memory_files" in result
        assert len(result["grounding_facts"]) == 2

    def test_model_copy_update(self, sample_instance):
        updated = sample_instance.model_copy(
            update={"difficulty_preset": "hard", "transforms_applied": ["prompt_fuzz"]}
        )
        assert updated.difficulty_preset == "hard"
        assert updated.transforms_applied == ["prompt_fuzz"]
        # Original unchanged
        assert sample_instance.difficulty_preset == "medium"

    def test_with_verification(self, sample_instance):
        updated = sample_instance.model_copy(
            update={
                "verification": {
                    "cond_a_llm_score": 0.8,
                    "cond_b_llm_score": 0.3,
                    "is_accepted": True,
                }
            }
        )
        assert updated.verification["cond_a_llm_score"] == 0.8

    def test_with_context_tokens(self):
        tokens = ContextTokens(
            task_prompt=100, memory_files=500, repo_context=2000,
            grounding_facts=300, distractors=200, total=3100,
        )
        instance = MemGymInstance(
            instance_id="t", base_instance="t", task_prompt="p",
            patch="p", context_tokens=tokens,
        )
        assert instance.context_tokens.total == 3100
