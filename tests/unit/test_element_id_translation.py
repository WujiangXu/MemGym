"""Tests for element-ID translation across browser sessions.

Validates that _parse_a11y_elements, _build_id_translation, and
_translate_action correctly map old element IDs to new ones using
semantic (role + name) matching.
"""
import pytest

from memgym.runners.webarena_runner import (
    _parse_a11y_elements,
    _build_id_translation,
    _translate_action,
    _normalize,
)


# -- Fixtures: realistic a11y tree snippets --------------------------------

OLD_TREE = """\
INTERACTIVE ELEMENTS (use these IDs with click/type/select):
[1] link "Home"
[2] button "Search"
[3] textbox "Search query"
[4] link "Settings"
[5] button "Create Label"
[6] checkbox "Remember me"
[7] button "Submit"
[8] link "Help"
[9] button "Edit"
[10] button "Edit"
"""

# Same page, different session → same elements, shuffled IDs
NEW_TREE = """\
INTERACTIVE ELEMENTS (use these IDs with click/type/select):
[1] link "Home"
[2] link "Settings"
[3] button "Search"
[4] textbox "Search query"
[5] checkbox "Remember me"
[6] button "Create Label"
[7] link "Help"
[8] button "Submit"
[9] button "Edit"
[10] button "Edit"
"""


class TestParseA11yElements:
    def test_parse_string(self):
        elems = _parse_a11y_elements(OLD_TREE)
        assert len(elems) == 10
        assert elems[0] == (1, "link", "Home")
        assert elems[4] == (5, "button", "Create Label")
        assert elems[5] == (6, "checkbox", "Remember me")

    def test_parse_dict(self):
        obs = {"url": "http://localhost:8000", "title": "Test", "accessibility_tree": OLD_TREE}
        elems = _parse_a11y_elements(obs)
        assert len(elems) == 10
        assert elems[2] == (3, "textbox", "Search query")

    def test_parse_empty(self):
        assert _parse_a11y_elements("") == []
        assert _parse_a11y_elements({}) == []
        assert _parse_a11y_elements(None) == []

    def test_element_without_name(self):
        tree = '[1] button\n[2] link "Home"'
        elems = _parse_a11y_elements(tree)
        assert elems[0] == (1, "button", "")
        assert elems[1] == (2, "link", "Home")

    def test_element_with_value(self):
        tree = '[3] textbox "Query" value="hello"'
        elems = _parse_a11y_elements(tree)
        assert elems[0] == (3, "textbox", "Query")


class TestNormalize:
    def test_basic(self):
        assert _normalize("Create Label") == "create label"

    def test_extra_whitespace(self):
        assert _normalize("  Create   Label  ") == "create label"

    def test_empty(self):
        assert _normalize("") == ""


class TestBuildIdTranslation:
    def test_basic_translation(self):
        id_map = _build_id_translation(OLD_TREE, NEW_TREE)

        # link "Home" → 1→1 (same)
        assert id_map[1] == 1
        # button "Search" → 2→3
        assert id_map[2] == 3
        # textbox "Search query" → 3→4
        assert id_map[3] == 4
        # link "Settings" → 4→2
        assert id_map[4] == 2
        # button "Create Label" → 5→6
        assert id_map[5] == 6
        # checkbox "Remember me" → 6→5
        assert id_map[6] == 5
        # button "Submit" → 7→8
        assert id_map[7] == 8
        # link "Help" → 8→7
        assert id_map[8] == 7

    def test_duplicate_role_name_uses_occurrence(self):
        """Two 'button Edit' elements map by occurrence order."""
        id_map = _build_id_translation(OLD_TREE, NEW_TREE)
        # First "Edit" button: old=9 → new=9
        assert id_map[9] == 9
        # Second "Edit" button: old=10 → new=10
        assert id_map[10] == 10

    def test_dict_observations(self):
        old_obs = {"accessibility_tree": OLD_TREE}
        new_obs = {"accessibility_tree": NEW_TREE}
        id_map = _build_id_translation(old_obs, new_obs)
        assert id_map[5] == 6  # button "Create Label"

    def test_missing_element(self):
        """Element in old tree but not in new → omitted from map."""
        new_tree_missing = '[1] link "Home"\n[2] button "Search"'
        id_map = _build_id_translation(OLD_TREE, new_tree_missing)
        assert 1 in id_map  # link "Home" exists
        assert 2 in id_map  # button "Search" exists
        assert 5 not in id_map  # button "Create Label" missing

    def test_empty_trees(self):
        assert _build_id_translation("", "") == {}
        assert _build_id_translation(OLD_TREE, "") == {}


class TestTranslateAction:
    def setup_method(self):
        self.id_map = _build_id_translation(OLD_TREE, NEW_TREE)

    def test_click(self):
        action, ok = _translate_action("click [5]", self.id_map)
        assert ok is True
        assert action == "click [6]"  # button "Create Label" 5→6

    def test_type(self):
        action, ok = _translate_action("type [3] hello world", self.id_map)
        assert ok is True
        assert action == "type [4] hello world"  # textbox "Search query" 3→4

    def test_type_with_brackets(self):
        action, ok = _translate_action("type [3] [hello world]", self.id_map)
        assert ok is True
        assert action == "type [4] [hello world]"

    def test_select(self):
        action, ok = _translate_action("select [6] [Yes]", self.id_map)
        assert ok is True
        assert action == "select [5] [Yes]"  # checkbox "Remember me" 6→5

    def test_goto_passthrough(self):
        action, ok = _translate_action("goto [http://example.com]", self.id_map)
        assert ok is True
        assert action == "goto [http://example.com]"

    def test_key_passthrough(self):
        action, ok = _translate_action("key [Enter]", self.id_map)
        assert ok is True
        assert action == "key [Enter]"

    def test_scroll_passthrough(self):
        action, ok = _translate_action("scroll [down] [500]", self.id_map)
        assert ok is True
        assert action == "scroll [down] [500]"

    def test_stop_passthrough(self):
        action, ok = _translate_action("stop [N/A]", self.id_map)
        assert ok is True
        assert action == "stop [N/A]"

    def test_unmapped_id(self):
        # ID 99 doesn't exist in old tree
        action, ok = _translate_action("click [99]", {})
        assert ok is False
        assert action == "click [99]"  # returned unchanged

    def test_empty_action(self):
        action, ok = _translate_action("", self.id_map)
        assert ok is True
        assert action == ""

    def test_none_map(self):
        action, ok = _translate_action("click [5]", {})
        assert ok is False
