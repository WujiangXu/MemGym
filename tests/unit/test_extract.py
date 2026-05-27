"""Tests for v2 extraction stage functions."""

import pytest

pytest.importorskip(
    "datasets",
    reason=(
        "coding-synth pipeline eagerly imports `datasets` via its package "
        "__init__. Install with `uv pip install -e .[swe]` to run this suite."
    ),
)

from memgym.pipelines.coding_synthetic.stages.extract import (
    passes_quality_gate,
    analyze_repo_context,
    compute_patch_coverage,
    _filter_leaky_seeds,
)


class TestPassesQualityGate:
    """Tests for passes_quality_gate()."""

    def test_passes_with_enough_external_critical(self):
        facts = [
            {"id": "F1", "discoverable": False, "relevance": "critical"},
            {"id": "F2", "discoverable": False, "relevance": "critical"},
            {"id": "F3", "discoverable": True, "relevance": "helpful"},
        ]
        assert passes_quality_gate(facts) is True

    def test_fails_with_one_external_critical(self):
        facts = [
            {"id": "F1", "discoverable": False, "relevance": "critical"},
            {"id": "F2", "discoverable": True, "relevance": "critical"},
        ]
        assert passes_quality_gate(facts) is False

    def test_fails_with_no_facts(self):
        assert passes_quality_gate([]) is False

    def test_fails_with_only_discoverable(self):
        facts = [
            {"id": "F1", "discoverable": True, "relevance": "critical"},
            {"id": "F2", "discoverable": True, "relevance": "critical"},
        ]
        assert passes_quality_gate(facts) is False

    def test_fails_with_only_helpful(self):
        facts = [
            {"id": "F1", "discoverable": False, "relevance": "helpful"},
            {"id": "F2", "discoverable": False, "relevance": "helpful"},
        ]
        assert passes_quality_gate(facts) is False

    def test_custom_threshold(self):
        facts = [
            {"id": "F1", "discoverable": False, "relevance": "critical"},
        ]
        assert passes_quality_gate(facts, min_external_critical=1) is True
        assert passes_quality_gate(facts, min_external_critical=2) is False

    def test_missing_keys_treated_as_discoverable(self):
        """Facts missing 'discoverable' key default to True (discoverable)."""
        facts = [
            {"id": "F1", "relevance": "critical"},
            {"id": "F2", "relevance": "critical"},
        ]
        assert passes_quality_gate(facts) is False


class TestAnalyzeRepoContext:
    """Tests for analyze_repo_context()."""

    @pytest.fixture
    def sample_patch(self):
        return """\
--- a/oauth/utils.py
+++ b/oauth/utils.py
@@ -10,5 +10,8 @@ def scope_to_list(scope_str):
-    return scope_str.split()
+    return sorted(scope_str.split())
"""

    @pytest.fixture
    def sample_repo_context(self):
        return {
            "oauth/utils.py": (
                "import re\n"
                "\n"
                "def scope_to_list(scope_str):\n"
                '    """Convert scope string to sorted list."""\n'
                "    return scope_str.split()\n"
                "\n"
                "def list_to_scope(scope_list):\n"
                "    return ' '.join(scope_list)\n"
            ),
            "tests/test_oauth.py": (
                "from oauth.utils import scope_to_list\n"
                "\n"
                "def test_scope_to_list():\n"
                "    assert scope_to_list('read write') == ['read', 'write']\n"
            ),
            "oauth/client.py": (
                "from .utils import scope_to_list\n"
                "\n"
                "def get_scopes(raw):\n"
                "    return scope_to_list(raw)\n"
            ),
        }

    def test_extracts_function_info(self, sample_patch, sample_repo_context):
        analysis = analyze_repo_context(sample_patch, sample_repo_context)
        assert "scope_to_list" in analysis["functions"]
        func = analysis["functions"]["scope_to_list"]
        assert func["file"] == "oauth/utils.py"
        assert "signature" in func

    def test_finds_callers(self, sample_patch, sample_repo_context):
        analysis = analyze_repo_context(sample_patch, sample_repo_context)
        assert "scope_to_list" in analysis["callers"]
        callers = analysis["callers"]["scope_to_list"]
        assert any("client.py" in c for c in callers)

    def test_finds_test_references(self, sample_patch, sample_repo_context):
        analysis = analyze_repo_context(sample_patch, sample_repo_context)
        assert "scope_to_list" in analysis["test_references"]

    def test_extracts_imports(self, sample_patch, sample_repo_context):
        analysis = analyze_repo_context(sample_patch, sample_repo_context)
        assert "oauth/utils.py" in analysis["imports"]
        assert any("import re" in imp for imp in analysis["imports"]["oauth/utils.py"])

    def test_empty_repo_context(self, sample_patch):
        analysis = analyze_repo_context(sample_patch, {})
        assert analysis["functions"] == {}
        assert analysis["callers"] == {}

    def test_no_match_in_patch(self):
        patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1,1 +1,1 @@\n-x=1\n+x=2\n"
        analysis = analyze_repo_context(patch, {})
        assert analysis["functions"] == {}

    def test_extracts_docstring(self, sample_patch, sample_repo_context):
        analysis = analyze_repo_context(sample_patch, sample_repo_context)
        func = analysis["functions"].get("scope_to_list", {})
        assert "docstring" in func


class TestComputePatchCoverage:
    """Tests for compute_patch_coverage()."""

    @pytest.fixture
    def two_hunk_patch(self):
        return (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old1\n+new1\n"
            "@@ -10,3 +10,3 @@\n"
            "-old2\n+new2\n"
        )

    def test_full_coverage(self, two_hunk_patch):
        facts = [
            {"hunks_enabled": [0]},
            {"hunks_enabled": [1]},
        ]
        result = compute_patch_coverage(facts, two_hunk_patch)
        assert result["coverage_ratio"] == 1.0
        assert result["uncovered_hunks"] == []

    def test_partial_coverage(self, two_hunk_patch):
        facts = [{"hunks_enabled": [0]}]
        result = compute_patch_coverage(facts, two_hunk_patch)
        assert result["coverage_ratio"] == 0.5
        assert result["uncovered_hunks"] == [1]

    def test_no_coverage(self, two_hunk_patch):
        facts = [{"hunks_enabled": []}]
        result = compute_patch_coverage(facts, two_hunk_patch)
        assert result["coverage_ratio"] == 0.0
        assert sorted(result["uncovered_hunks"]) == [0, 1]

    def test_empty_patch(self):
        result = compute_patch_coverage([], "no hunks here")
        assert result["coverage_ratio"] == 1.0

    def test_overlapping_hunk_indices(self, two_hunk_patch):
        facts = [
            {"hunks_enabled": [0, 1]},
            {"hunks_enabled": [1]},
        ]
        result = compute_patch_coverage(facts, two_hunk_patch)
        assert result["coverage_ratio"] == 1.0

    def test_out_of_range_index_counted(self, two_hunk_patch):
        """Out-of-range indices are included in covered set (no filtering)."""
        facts = [{"hunks_enabled": [0, 1, 99]}]
        result = compute_patch_coverage(facts, two_hunk_patch)
        # coverage_ratio can exceed 1.0 since covered set includes out-of-range
        assert result["coverage_ratio"] >= 1.0
        assert 99 in result["covered_hunks"]

    def test_missing_hunks_enabled_key(self, two_hunk_patch):
        facts = [{"id": "F1"}]
        result = compute_patch_coverage(facts, two_hunk_patch)
        assert result["coverage_ratio"] == 0.0


class TestFilterLeakySeeds:
    """Tests for _filter_leaky_seeds()."""

    def test_removes_code_change_language(self):
        seeds = [
            {"question": "Why was sorted() added to the function?"},
            {"question": "What ordering constraints exist on scope values?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 1
        assert "ordering" in result[0]["question"]

    def test_removes_line_references(self):
        seeds = [
            {"question": "What changed on line 42?"},
            {"question": "What should the return type be?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 1

    def test_removes_patch_references(self):
        seeds = [
            {"question": "What was the purpose of the patch?"},
            {"question": "What behavior does the user expect?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 1
        assert "user expect" in result[0]["question"]

    def test_removes_diff_markers(self):
        seeds = [
            {"question": "Why does +++ b/file.py show this change?"},
            {"question": "What does the function return on empty input?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 1

    def test_keeps_all_clean_seeds(self):
        seeds = [
            {"question": "What should scope_to_list return for single items?"},
            {"question": "What ordering constraints exist?"},
            {"question": "Under what conditions does validation fail?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 3

    def test_empty_input(self):
        assert _filter_leaky_seeds([]) == []

    def test_removes_deleted_keyword(self):
        seeds = [
            {"question": "Why was the null check deleted?"},
            {"question": "What constraints apply to the API?"},
        ]
        result = _filter_leaky_seeds(seeds)
        assert len(result) == 1
        assert "constraints" in result[0]["question"]
