"""Lock the M1 fix: backticks regex must accept bash/sh/shell/no-tag.

Loads the regex directly via ``importlib`` from the source file so the
``memgym`` package's heavy ML imports (sentence-transformers etc.) don't
have to be installed to run this unit test. The real
``BackticksToolParser.extract_tool_calls`` only adds tokenizer decoding
around the same regex match — covering the regex covers the parser's
contract.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "memgym" / "training" / "rl" / "way_a_verl" / "agent_loop.py"
_spec = importlib.util.spec_from_file_location("_way_a_agent_loop_iso", _SRC)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
# Stub the verl-import side effects so module exec doesn't pull verl in.
import sys
sys.modules.setdefault("verl", type(sys)("verl"))
try:
    _spec.loader.exec_module(_mod)
except Exception:  # pragma: no cover — regex is module-level so it loaded already
    pass
_BASH_BLOCK_RE = _mod._BASH_BLOCK_RE


def _extract(text: str) -> list[str]:
    return [m.group(1).strip() for m in _BASH_BLOCK_RE.finditer(text)]


def test_bash_tag_matches():
    assert _extract("THOUGHT: x\n```bash\nls\n```\n") == ["ls"]


def test_sh_tag_matches():
    assert _extract("```sh\necho hi\n```") == ["echo hi"]


def test_shell_tag_matches():
    assert _extract("```shell\npwd\n```") == ["pwd"]


def test_no_tag_matches():
    assert _extract("```\nls -la\n```") == ["ls -la"]


def test_multiline_command_kept_intact():
    body = "for i in 1 2 3; do\n  echo $i\ndone"
    assert _extract(f"```bash\n{body}\n```") == [body]


def test_multiple_blocks_each_extracted():
    text = "```bash\nls\n```\nmid\n```sh\npwd\n```"
    assert _extract(text) == ["ls", "pwd"]
