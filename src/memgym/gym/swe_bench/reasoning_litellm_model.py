"""Reasoning-aware LitellmTextbasedModel for Qwen3-thinking-family agents.

Extends `minisweagent.models.litellm_textbased_model.LitellmTextbasedModel`
with two fixes needed to make vLLM-hosted Qwen3 thinking-mode models usable
under `--reasoning-parser qwen3`:

  B2 (content-empty → reasoning_content fallback): with `--reasoning-parser
      qwen3` enabled, vLLM routes the whole turn's output into
      `message.reasoning_content` and leaves `message.content` empty when
      the model emits its bash action *inside* the `<think>…</think>` block.
      The base class's regex then sees no text, raises FormatError, and the
      scaffold loops until step-limit. Fall back to `reasoning_content`
      whenever `content` is empty.

  B6 (defense-in-depth `<think>` stripping): vLLM issue #35221 (truncated
      `</think>` closing tags) can dump raw `<think>…` into `content`.
      Strip `<think>…</think>` spans (and any trailing unclosed `<think>…`)
      from the parsed text before the action regex runs, so stray backticks
      inside the reasoning block don't pollute the match.
"""
from __future__ import annotations

import re
from typing import List

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.actions_text import parse_regex_actions

# DOTALL: span newlines. Non-greedy: stop at the first `</think>` so real
# post-reasoning content survives.
_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Fallback for the vLLM #35221 case: unclosed `<think>` trailing to end-of-string.
_THINK_OPEN_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL)


def _strip_thinking(text: str) -> str:
    text = _THINK_CLOSED.sub("", text)
    return _THINK_OPEN_UNCLOSED.sub("", text)


class ReasoningAwareLitellmTextbasedModel(LitellmTextbasedModel):
    """Qwen3-thinking-compatible textbased model."""

    def _parse_actions(self, response) -> List[dict]:
        msg = response.choices[0].message
        content = msg.content or ""
        if not content.strip():
            content = getattr(msg, "reasoning_content", "") or ""
        content = _strip_thinking(content)
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )
