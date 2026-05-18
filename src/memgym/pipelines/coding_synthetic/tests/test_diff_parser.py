"""Tests for diff parser module."""

import pytest
from ..utils.diff_parser import parse_patch, meets_filter_criteria, PatchMetrics


class TestParsePatch:
    """Tests for parse_patch function."""

    def test_parse_valid_patch(self):
        """Test parsing a valid unified diff."""
        patch = '''diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 def foo():
-    pass
+    return 1
+    # comment
'''
        metrics = parse_patch(patch)
        assert metrics.lines_added == 2
        assert metrics.lines_removed == 1
        assert metrics.lines_changed == 3
        assert metrics.num_hunks == 1
        assert "test.py" in metrics.files_changed

    def test_parse_empty_patch(self):
        """Test parsing empty patch."""
        metrics = parse_patch("")
        assert metrics.lines_changed == 0
        assert metrics.num_hunks == 0
        assert metrics.files_changed == []

    def test_parse_none_patch(self):
        """Test parsing None-like input."""
        metrics = parse_patch(None)
        assert metrics.lines_changed == 0

    def test_parse_malformed_patch_fallback(self):
        """Test that malformed patches use fallback parser."""
        # This patch has wrong line counts but should still be parsed
        patch = '''--- a/src/api.py
+++ b/src/api.py
@@ -10,5 +10,8 @@ def get_data():
-    return None
+    if not authenticated:
+        raise AuthError()
+    return fetch_data()
'''
        metrics = parse_patch(patch)
        # Fallback parser should count the + and - lines
        assert metrics.lines_added >= 3
        assert metrics.lines_removed >= 1
        assert metrics.lines_changed >= 4
        assert metrics.num_hunks >= 1

    def test_parse_multiple_files(self):
        """Test parsing patch with multiple files."""
        patch = '''diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
'''
        metrics = parse_patch(patch)
        assert len(metrics.files_changed) >= 2

    def test_parse_patch_with_context(self):
        """Test patch with context lines."""
        patch = '''--- a/test.py
+++ b/test.py
@@ -5,7 +5,8 @@ class Test:
     def __init__(self):
         self.x = 1
         self.y = 2
-        self.z = 3
+        self.z = 4
+        self.w = 5

     def method(self):
         pass
'''
        metrics = parse_patch(patch)
        assert metrics.lines_added >= 2
        assert metrics.lines_removed >= 1


class TestMeetsFilterCriteria:
    """Tests for filter criteria checking."""

    def test_meets_all_criteria(self):
        """Test instance that meets all criteria."""
        metrics = PatchMetrics(
            lines_added=5,
            lines_removed=5,
            lines_changed=10,
            num_hunks=3,
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="A" * 250,
            fail_to_pass_tests=["test1", "test2", "test3"],
            min_lines_changed=8,
            min_fail_to_pass=2,
            min_hunks=2,
            min_problem_length=200
        )
        assert result is True

    def test_fails_lines_changed(self):
        """Test instance that fails lines_changed criteria."""
        metrics = PatchMetrics(
            lines_added=2,
            lines_removed=2,
            lines_changed=4,  # Less than 8
            num_hunks=3,
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="A" * 250,
            fail_to_pass_tests=["test1", "test2"],
            min_lines_changed=8
        )
        assert result is False

    def test_fails_hunks(self):
        """Test instance that fails num_hunks criteria."""
        metrics = PatchMetrics(
            lines_added=10,
            lines_removed=5,
            lines_changed=15,
            num_hunks=1,  # Less than 2
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="A" * 250,
            fail_to_pass_tests=["test1", "test2"],
            min_hunks=2
        )
        assert result is False

    def test_fails_tests(self):
        """Test instance that fails test count criteria."""
        metrics = PatchMetrics(
            lines_added=10,
            lines_removed=5,
            lines_changed=15,
            num_hunks=3,
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="A" * 250,
            fail_to_pass_tests=["test1"],  # Only 1 test
            min_fail_to_pass=2
        )
        assert result is False

    def test_fails_problem_length(self):
        """Test instance that fails problem length criteria."""
        metrics = PatchMetrics(
            lines_added=10,
            lines_removed=5,
            lines_changed=15,
            num_hunks=3,
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="Short",  # Less than 200 chars
            fail_to_pass_tests=["test1", "test2"],
            min_problem_length=200
        )
        assert result is False

    def test_edge_case_exact_threshold(self):
        """Test instance at exact threshold values."""
        metrics = PatchMetrics(
            lines_added=4,
            lines_removed=4,
            lines_changed=8,  # Exactly 8
            num_hunks=2,      # Exactly 2
            files_changed=["file.py"]
        )
        result = meets_filter_criteria(
            metrics=metrics,
            problem_statement="A" * 200,  # Exactly 200
            fail_to_pass_tests=["test1", "test2"],  # Exactly 2
            min_lines_changed=8,
            min_fail_to_pass=2,
            min_hunks=2,
            min_problem_length=200
        )
        assert result is True
