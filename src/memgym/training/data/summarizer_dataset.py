"""Loader + validator for summarizer-SFT pairs.

Works with rows produced by `summarizer_pairs.py` — schema documented
at `the internal handoff doc`. The important contract
for reuse of `training.models.sft._train_sft` is:

* Every row has non-empty `prompt` and `completion` strings.
* Every row has a `split` field (`"train"` or `"eval"`).
* No `label` field (we intentionally don't carry the classifier
  label into summarizer SFT — all rows are pre-filtered to SAFE).

This module also surfaces a small helper for the H.2 eval step:
`load_eval_rows` pulls just the held-out split for a quick perplexity
sanity check before spending compute on the end-to-end resolve-rate
smoke run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List

logger = logging.getLogger(__name__)


REQUIRED_KEYS = ("prompt", "completion", "trajectory_id", "split")


def _iter_rows(pairs_path: Path) -> Iterable[dict]:
    with pairs_path.open() as fh:
        for line_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{pairs_path}:{line_idx} bad JSON: {exc}"
                ) from exc


def validate_summarizer_pairs(pairs_path: Path, limit: int = 0) -> List[dict]:
    """Load + validate rows, raising on schema or emptiness problems.

    `limit` is honored for smoke tests. Rows whose `completion` is
    empty after stripping are dropped with a warning — they would
    contribute zero loss under completion-only training and only waste
    a gradient step.
    """
    rows: List[dict] = []
    n_empty = 0
    for line_idx, record in enumerate(_iter_rows(pairs_path)):
        if limit and len(rows) >= limit:
            break
        missing = [k for k in REQUIRED_KEYS if k not in record]
        if missing:
            raise ValueError(
                f"{pairs_path}:{line_idx} missing required keys {missing!r}; "
                f"expected {list(REQUIRED_KEYS)}"
            )
        completion = (record.get("completion") or "").strip()
        if not completion:
            n_empty += 1
            continue
        if record["split"] not in {"train", "eval"}:
            raise ValueError(
                f"{pairs_path}:{line_idx} unexpected split {record['split']!r}; "
                f"expected 'train' or 'eval'"
            )
        rows.append(record)
    if not rows:
        raise ValueError(f"summarizer pairs at {pairs_path} is empty after filter")
    if n_empty:
        logger.warning(
            "dropped %d rows with empty completion during validation", n_empty,
        )
    n_train = sum(1 for r in rows if r["split"] == "train")
    logger.info(
        "summarizer pairs validated: total=%d train=%d eval=%d from %s",
        len(rows), n_train, len(rows) - n_train, pairs_path,
    )
    return rows


def load_eval_rows(pairs_path: Path) -> List[dict]:
    """Return just the held-out split — used by the H.2 perplexity check."""
    return [r for r in validate_summarizer_pairs(pairs_path) if r["split"] == "eval"]


__all__ = ["REQUIRED_KEYS", "validate_summarizer_pairs", "load_eval_rows"]
