"""LLMLingua-2 summarizer backend — (see paper) literature baseline.

Drops into `NaiveSummarizationMemory.summarizer_backend` as a
`SummarizerBackend`-compatible object, but unlike the generative
backends it does **token pruning**, not LLM re-generation:

- Input: the same `system + user` strings the LLM-based backends see.
- Output: a character-compressed version of `user` produced by
  `llmlingua.PromptCompressor` at a target compression `rate`. The
  `system` prompt is ignored — it is instructions for an LLM, not
  load-bearing context to compress.

Why this matters for the paper: H.4' claims **Δ-resolve(our SFT vs
LLMLingua-2) ≥ +0.02 at matched compression ratio on Verified**. That
framing requires a same-pipeline baseline where the only changed
component is the summarizer itself; wrapping LLMLingua-2 as a backend
is the cleanest way to hold everything else (memory trigger, keep_first
/ keep_recent, agent model) constant.

Tuning `rate`:

- Default 0.33 is chosen to roughly match the H.1.1 datacard's
  `completion_chars / num_input_chars` p50 = 0.322 on the 582 SAFE
  SFT pairs. We want H.4' to compare **at matched compression**, not
  at LLMLingua's default 0.5.
- LLMLingua's `rate` is token-based, our datacard measures chars;
  they correlate but aren't identical. If the empirical p50 char
  compression on a smoke run differs by > 10 %, tune `--llmlingua2-rate`
  on eval_memory_ab.py until char-p50 matches the SFT summarizer's.

Model choice:

- `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` is the default
  LLMLingua-2 compressor checkpoint (small RoBERTa, CPU-friendly). The
  BERT-base variant is cheaper but less accurate; override via
  `model_name` if latency becomes an issue on CI smoke runs.

Installation:

- `pip install llmlingua` — not pinned in `setup.py` because
  LLMLingua-2 is us-owned (H.4' literature baseline); the collaborator
  never needs it. Import is deferred so `from memgym.memory import ...`
  does not explode without the package installed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class LLMLingua2SummarizerBackend:
    """Token-pruning "summarizer" backed by `llmlingua.PromptCompressor`.

    Not a drop-in replacement for LLM-based summarization in the
    narrative sense — LLMLingua-2 produces compressed-but-disfluent
    text, not a coherent USER_CONTEXT / TASK_TRACKING block. It is
    exactly the right comparison for the paper's matched-compression
    claim, though.

    `force_tokens` defaults to preserving newline + question-mark
    characters (LLMLingua-2 README recommendation for dialog/code
    contexts). We extend with `=== Step`, `Thought:`, `Action:`,
    `Observation:` markers the SFT prompt uses, so structural anchors
    the agent's downstream tail relies on don't get pruned.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        rate: float = 0.33,
        device_map: str = "cpu",
        force_tokens: Optional[List[str]] = None,
        preserve_sections: bool = True,
    ):
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise RuntimeError(
                "LLMLingua-2 backend requires `pip install llmlingua` "
                "(not pinned in setup.py — literature baseline for H.4' only)."
            ) from e

        self._compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )
        self.rate = rate
        # Default force list from the LLMLingua-2 README, extended with
        # our step-triple markers. LLMLingua matches on lowercase token
        # surface form after its tokenizer; passing short strings works
        # because the XLM-RoBERTa vocab covers ASCII punctuation and
        # common English words as single tokens.
        self.force_tokens = list(force_tokens) if force_tokens is not None else [
            "\n", "?", ":", ",",
        ]
        if preserve_sections:
            self.force_tokens.extend([
                "Step", "Problem", "Thought", "Action", "Observation",
                "Previous", "summary", "Summary",
            ])
        # Last-call telemetry (read by callers that want to plot the
        # matched-compression curve without re-running the compressor).
        self.last_stats: Dict[str, Any] = {}

    def summarize(self, system: str, user: str, max_tokens: int = 1024) -> str:  # noqa: ARG002
        """Compress `user` at the configured rate. `system` is ignored
        (LLMLingua-2 is not LLM-generated — there is no instruction
        channel) and `max_tokens` is unused: LLMLingua-2's output size
        is controlled by `rate`, not by a separate length cap.

        Returns the compressed text. Empty string on failure — matches
        the other backends' best-effort contract, so the caller keeps
        the previous summary rather than advancing `_summarized_upto`.
        """
        if not user.strip():
            self.last_stats = {"reason": "empty input"}
            return ""

        try:
            out = self._compressor.compress_prompt(
                user,
                rate=self.rate,
                force_tokens=self.force_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — summarization is best-effort
            logger.warning("llmlingua compression failed: %s", exc)
            self.last_stats = {"error": str(exc)}
            return ""

        # LLMLingua returns dict with compressed_prompt + origin/compressed
        # token counts; we record everything for matched-ratio tuning.
        compressed = (out.get("compressed_prompt") or "").strip()
        self.last_stats = {
            "rate_requested": self.rate,
            "origin_tokens": out.get("origin_tokens"),
            "compressed_tokens": out.get("compressed_tokens"),
            "input_chars": len(user),
            "output_chars": len(compressed),
            "char_ratio": len(compressed) / max(1, len(user)),
        }
        return compressed


__all__ = ["LLMLingua2SummarizerBackend"]
