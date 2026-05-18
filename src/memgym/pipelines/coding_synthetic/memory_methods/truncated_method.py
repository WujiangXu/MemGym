"""Truncated-input baseline: no LLM, just keep the last N chars of raw text.

This is the simplest possible "memory" — a sliding window over raw document
text with no summarization or indexing.  Useful as a lower-bound reference:
any real memory method should beat raw truncation.
"""
from __future__ import annotations


class TruncatedMethod:
    """Keep a fixed-size tail of raw document text."""

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars
        self._buffer: str = ""

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        self._buffer += f"\n## {doc_name}\n{doc_content}"
        if len(self._buffer) > self.max_chars:
            self._buffer = self._buffer[-self.max_chars :]

    def retrieve(self, question: str, task_prompt: str) -> str:
        return self._buffer if self._buffer else "(no documents ingested)"

    def reset(self) -> None:
        self._buffer = ""
