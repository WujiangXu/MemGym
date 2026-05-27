"""Resume mechanism tests for coding-synth eval.

Companion to the resume bug fix in
`src/memgym/pipelines/coding_synthetic/stages/qa/eval.py`.

Two failure modes the previous code had:

1. **Misleading "Resuming: N" banner.** The message printed
   ``len(done_ids)`` without verifying that those IDs actually appeared
   in the input file. A checkpoint written for a different bucket would
   silently produce a from-scratch run with the banner still saying
   "726 done."
2. **`tqdm` total never adjusted.** Even when the skip filter worked,
   the bar showed ``0 / total_hint`` with an ETA computed against the
   full input, hiding the resume from the user.

These tests pin the new behavior: peek-based stale-checkpoint detection,
adjusted total, and a post-run summary that reports the *actual* skip
count.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

pytest.importorskip(
    "unidiff",
    reason="coding-synth pipeline pulls in `unidiff` (under the `swe` extras). "
           "Install with `uv pip install -e .[swe]` to run this suite.",
)

from memgym.pipelines.coding_synthetic.stages.qa import eval as eval_mod
from memgym.pipelines.coding_synthetic.types.schemas import (
    CodingQAInstance,
    QAEvalResult,
)


def _make_instances(ids: List[str]) -> List[CodingQAInstance]:
    return [
        CodingQAInstance(instance_id=i, base_instance=i, task_prompt="t")
        for i in ids
    ]


def _write_checkpoint(path: Path, ids: List[str]) -> None:
    with path.open("w") as f:
        for i in ids:
            f.write(json.dumps(QAEvalResult(instance_id=i).to_jsonl_dict()) + "\n")


def _stub_evaluate_instance(inst, *args, **kwargs):
    """Bypass the LLM path; return a trivial result."""
    return QAEvalResult(instance_id=inst.instance_id)


class TestEvaluateAllResume(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, instances, ckpt_ids):
        ckpt = self.tmp_path / "ckpt.jsonl"
        _write_checkpoint(ckpt, ckpt_ids)
        # Use a fresh checkpoint for the run output so we don't double-count
        # against the priming `_load_checkpoint_full` reload at the end.
        run_ckpt = self.tmp_path / "run_ckpt.jsonl"
        _write_checkpoint(run_ckpt, ckpt_ids)
        buf = io.StringIO()
        with patch.object(eval_mod, "evaluate_instance", _stub_evaluate_instance):
            with redirect_stdout(buf):
                results = eval_mod.evaluate_all(
                    instances=iter(instances),
                    client=None,
                    show_progress=False,
                    num_workers=1,
                    checkpoint_path=str(run_ckpt),
                    total_hint=len(instances),
                )
        return results, buf.getvalue()

    def test_matching_checkpoint_skips_done_instances(self):
        instances = _make_instances(["a", "b", "c", "d"])
        results, log = self._run(instances, ckpt_ids=["a", "b"])
        new_ids = {r.instance_id for r in results} - {"a", "b"}
        # `c` and `d` are the only fresh items the run actually processed.
        self.assertEqual(new_ids, {"c", "d"})
        self.assertIn("Resume summary", log)
        self.assertIn("skipped 2", log)

    def test_stale_checkpoint_warns_and_processes_all(self):
        instances = _make_instances(["x", "y", "z"])
        results, log = self._run(
            instances, ckpt_ids=["unrelated-1", "unrelated-2"]
        )
        # No overlap → all three are processed; the stale warning fires
        # before the run starts.
        processed_ids = {r.instance_id for r in results} - {
            "unrelated-1", "unrelated-2"
        }
        self.assertEqual(processed_ids, {"x", "y", "z"})
        self.assertIn("checkpoint is likely stale", log)

    def test_empty_checkpoint_quiet_run(self):
        instances = _make_instances(["a", "b"])
        results, log = self._run(instances, ckpt_ids=[])
        processed_ids = {r.instance_id for r in results}
        self.assertEqual(processed_ids, {"a", "b"})
        # No "Resuming" / "Resume summary" noise when there's nothing to skip.
        self.assertNotIn("Resume summary", log)
        self.assertNotIn("Resuming", log)

    def test_skip_count_box_tracks_actual_matches(self):
        """Even when total_hint is None, the post-run summary should
        report the true skip count (not just len(done_ids))."""
        instances = _make_instances(["a", "b", "c"])
        ckpt = self.tmp_path / "ckpt.jsonl"
        _write_checkpoint(ckpt, ["a", "ghost-1", "ghost-2"])
        buf = io.StringIO()
        with patch.object(eval_mod, "evaluate_instance", _stub_evaluate_instance):
            with redirect_stdout(buf):
                results = eval_mod.evaluate_all(
                    instances=iter(instances),
                    client=None,
                    show_progress=False,
                    num_workers=1,
                    checkpoint_path=str(ckpt),
                    total_hint=None,
                )
        log = buf.getvalue()
        new_ids = {r.instance_id for r in results} - {"a", "ghost-1", "ghost-2"}
        self.assertEqual(new_ids, {"b", "c"})
        self.assertIn("skipped 1", log)
        self.assertIn("checkpoint had 3 total entries", log)


if __name__ == "__main__":
    unittest.main()
