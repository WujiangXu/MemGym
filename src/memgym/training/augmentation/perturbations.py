"""Perturbation strategies for counterfactual replay augmentation.

Two levels of perturbation:
  1. Config-level: change compression thresholds (AggressiveCompression, etc.)
  2. Content-level: modify the actual messages/summary at a compaction step
     (SummaryRedaction, AttributeDeletion, ContextTruncation, SummaryNoise)

Content-level perturbations test *what specific information matters* —
they surgically modify the context to see if the agent's action changes.

Usage:
    from memgym.training.augmentation.perturbations import DEFAULT_PERTURBATIONS

    for p in DEFAULT_PERTURBATIONS:
        perturbed = p.perturb_messages(messages, step)
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerturbedConfig:
    """Result of applying a perturbation to memory config."""

    name: str
    description: str
    original_config: Dict[str, Any]
    perturbed_config: Dict[str, Any]


class PerturbationStrategy(ABC):
    """Base class for summarization perturbations."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        """Create a perturbed memory configuration.

        Args:
            config: Original memory config (max_size, condensation_ratio, etc.)

        Returns:
            PerturbedConfig with modified parameters.
        """
        ...

    @abstractmethod
    def perturb_messages(
        self,
        messages: List[Dict[str, Any]],
        compaction_step_idx: int,
    ) -> List[Dict[str, Any]]:
        """Optionally perturb the message history directly.

        Default implementations return messages unchanged (config-only perturbation).
        """
        ...


class AggressiveCompression(PerturbationStrategy):
    """Halve max_size / max_tokens to force heavier summarization.

    This tests whether more aggressive compression would lose critical info.
    If the agent's action changes after more aggressive compression,
    the original compression was near the "safe" boundary.
    """

    def __init__(self, factor: float = 0.5):
        self.factor = factor

    @property
    def name(self) -> str:
        return f"aggressive_{self.factor}"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        new_config = copy.deepcopy(config)

        if "max_size" in new_config:
            new_config["max_size"] = max(10, int(new_config["max_size"] * self.factor))
        if "max_tokens" in new_config:
            new_config["max_tokens"] = max(500, int(new_config["max_tokens"] * self.factor))
        if "condensation_ratio" in new_config:
            new_config["condensation_ratio"] = max(0.3, new_config["condensation_ratio"] * self.factor)

        return PerturbedConfig(
            name=self.name,
            description=f"Multiply compression thresholds by {self.factor}",
            original_config=config,
            perturbed_config=new_config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        return messages


class LenientCompression(PerturbationStrategy):
    """Double max_size / max_tokens to keep more context.

    This tests whether skipping compression would be safe.
    If the agent's action stays the same without compression,
    the compression was unnecessary at this step.
    """

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    @property
    def name(self) -> str:
        return f"lenient_{self.factor}"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        new_config = copy.deepcopy(config)

        if "max_size" in new_config:
            new_config["max_size"] = int(new_config["max_size"] * self.factor)
        if "max_tokens" in new_config:
            new_config["max_tokens"] = int(new_config["max_tokens"] * self.factor)
        if "condensation_ratio" in new_config:
            new_config["condensation_ratio"] = min(0.95, new_config["condensation_ratio"] * 1.2)

        return PerturbedConfig(
            name=self.name,
            description=f"Multiply compression thresholds by {self.factor}",
            original_config=config,
            perturbed_config=new_config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        return messages


class RandomDrop(PerturbationStrategy):
    """Randomly drop N messages from the context to simulate info loss.

    Unlike config-based perturbations, this directly modifies the message
    history. Tests whether specific messages are critical.
    """

    def __init__(self, drop_fraction: float = 0.2, seed: int = 42):
        self.drop_fraction = drop_fraction
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"random_drop_{self.drop_fraction}"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description=f"Drop {self.drop_fraction*100:.0f}% of messages randomly",
            original_config=config,
            perturbed_config=config,  # no config change
        )

    def perturb_messages(
        self,
        messages: List[Dict[str, Any]],
        compaction_step_idx: int,
    ) -> List[Dict[str, Any]]:
        if len(messages) <= 3:
            return messages

        # Never drop first message (system) or last 2 (recent context)
        droppable = list(range(1, max(1, len(messages) - 2)))
        n_drop = max(1, int(len(droppable) * self.drop_fraction))
        to_drop = set(self.rng.sample(droppable, min(n_drop, len(droppable))))

        return [m for i, m in enumerate(messages) if i not in to_drop]


class SkipCompression(PerturbationStrategy):
    """Skip compression entirely at this step — pass full context through.

    The strongest test: if the agent acts identically with the full
    uncompressed context, then compression was safe at this step.
    """

    @property
    def name(self) -> str:
        return "skip_compression"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        new_config = copy.deepcopy(config)
        # Set thresholds very high so compression never triggers
        new_config["max_size"] = 99999
        new_config["max_tokens"] = 999999
        return PerturbedConfig(
            name=self.name,
            description="Skip compression entirely (full context)",
            original_config=config,
            perturbed_config=new_config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        return messages


# ─────────────────────────────────────────────────────────────────────
# Content-level perturbations
# ─────────────────────────────────────────────────────────────────────

# Regex patterns for extractable attributes in messages
_FILE_PATH_RE = re.compile(r"/(?:testbed|home|opt|usr|tmp)[\w/\-\.]+\.(?:py|js|ts|json|yaml|yml|cfg|ini|txt|md|rst)")
_ERROR_MSG_RE = re.compile(r"(?:Error|Exception|Traceback|FAILED|AssertionError|TypeError|ValueError|KeyError|AttributeError|ImportError)[^\n]{0,200}")
_FUNCTION_RE = re.compile(r"(?:def |class |function )\w+")
_LINE_NUM_RE = re.compile(r"(?:line |:)\d+")


class SummaryRedaction(PerturbationStrategy):
    """Remove key facts from the summary message in the context.

    Finds the summary (condensation record or system-like summary message)
    and strips file paths, function names, and error messages from it.
    Tests whether the summary's specific details matter or just its gist.
    """

    def __init__(self, redact_paths: bool = True, redact_errors: bool = True,
                 redact_functions: bool = True):
        self._redact_paths = redact_paths
        self._redact_errors = redact_errors
        self._redact_functions = redact_functions

    @property
    def name(self) -> str:
        return "summary_redaction"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description="Redact file paths, errors, and function names from summary",
            original_config=config,
            perturbed_config=config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        messages = copy.deepcopy(messages)
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            # Target summary-like messages (contain condensation markers)
            if any(kw in content for kw in [
                "USER_CONTEXT", "TASK_TRACKING", "CURRENT_STATE",
                "Summary of", "summary of", "condensed",
            ]):
                if self._redact_paths:
                    content = _FILE_PATH_RE.sub("<REDACTED_PATH>", content)
                if self._redact_errors:
                    content = _ERROR_MSG_RE.sub("<REDACTED_ERROR>", content)
                if self._redact_functions:
                    content = _FUNCTION_RE.sub("<REDACTED_FUNC>", content)
                m["content"] = content
        return messages


class AttributeDeletion(PerturbationStrategy):
    """Strip specific attribute types from ALL messages in context.

    Removes file paths, line numbers, or error messages throughout the
    entire message history. Tests whether those specific details are
    needed for the agent's next action.
    """

    def __init__(self, delete_type: str = "paths"):
        """Args:
            delete_type: "paths", "errors", "line_numbers", or "all"
        """
        self.delete_type = delete_type

    @property
    def name(self) -> str:
        return f"attr_delete_{self.delete_type}"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description=f"Delete {self.delete_type} from all messages",
            original_config=config,
            perturbed_config=config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        messages = copy.deepcopy(messages)
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if self.delete_type in ("paths", "all"):
                content = _FILE_PATH_RE.sub("___", content)
            if self.delete_type in ("errors", "all"):
                content = _ERROR_MSG_RE.sub("___", content)
            if self.delete_type in ("line_numbers", "all"):
                content = _LINE_NUM_RE.sub("", content)
            m["content"] = content
        return messages


class ContextTruncation(PerturbationStrategy):
    """Keep only the last N messages, discarding older history.

    Tests how much historical context the agent actually needs.
    Different from RandomDrop — this is a clean truncation, keeping
    the system prompt and the most recent messages.
    """

    def __init__(self, keep_last: int = 10):
        self.keep_last = keep_last

    @property
    def name(self) -> str:
        return f"truncate_last_{self.keep_last}"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description=f"Keep system prompt + last {self.keep_last} messages only",
            original_config=config,
            perturbed_config=config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        if len(messages) <= self.keep_last + 1:
            return messages
        # Keep first message (system prompt) + last N
        return [messages[0]] + messages[-(self.keep_last):]


class SummaryNoise(PerturbationStrategy):
    """Inject misleading details into summary messages.

    Adds plausible but wrong file paths, function names, or status info.
    Tests whether the agent is robust to noisy summaries or gets confused.
    """

    NOISE_SNIPPETS = [
        "\n\nNote: Also investigated /testbed/src/utils/deprecated_helper.py which may be related.",
        "\n\nPreviously found potential issue in validate_input() around line 342.",
        "\n\nAlso checked /testbed/tests/test_regression.py — tests were passing before this change.",
        "\n\nThe config parser in /testbed/src/config/loader.py may need updating as well.",
        "\n\nCOMPLETED: Fixed the type annotation in base_handler.py (unrelated to current task).",
    ]

    def __init__(self, n_snippets: int = 2, seed: int = 42):
        self.n_snippets = n_snippets
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "summary_noise"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description=f"Inject {self.n_snippets} misleading details into summary",
            original_config=config,
            perturbed_config=config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        messages = copy.deepcopy(messages)
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if any(kw in content for kw in [
                "USER_CONTEXT", "TASK_TRACKING", "CURRENT_STATE",
                "Summary of", "condensed",
            ]):
                snippets = self.rng.sample(
                    self.NOISE_SNIPPETS,
                    min(self.n_snippets, len(self.NOISE_SNIPPETS)),
                )
                m["content"] = content + "".join(snippets)
                break  # only modify one summary
        return messages


class LLMSummaryRewrite(PerturbationStrategy):
    """Use an LLM to paraphrase the summary, potentially losing details.

    This is the most realistic perturbation — it simulates what would
    happen if a different summarization prompt or model was used.
    Requires an LLM call per perturbation (budget-aware).
    """

    REWRITE_PROMPT = (
        "Rewrite this summary more concisely. Keep the main points but "
        "you may omit minor details. Do NOT add new information.\n\n"
        "Original summary:\n{summary}\n\n"
        "Rewritten summary:"
    )

    def __init__(self, model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        self._model = model

    @property
    def name(self) -> str:
        return "llm_summary_rewrite"

    def perturb_config(self, config: Dict[str, Any]) -> PerturbedConfig:
        return PerturbedConfig(
            name=self.name,
            description="LLM paraphrases the summary (may lose details)",
            original_config=config,
            perturbed_config=config,
        )

    def perturb_messages(self, messages, compaction_step_idx):
        messages = copy.deepcopy(messages)
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if any(kw in content for kw in [
                "USER_CONTEXT", "TASK_TRACKING", "CURRENT_STATE",
                "Summary of", "condensed",
            ]):
                rewritten = self._rewrite(content)
                if rewritten:
                    m["content"] = rewritten
                break
        return messages

    def _rewrite(self, summary: str) -> str:
        try:
            import litellm

            response = litellm.completion(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": self.REWRITE_PROMPT.format(summary=summary[:3000]),
                }],
                max_tokens=1500,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM rewrite failed: %s", e)
            return ""


# ─────────────────────────────────────────────────────────────────────
# Perturbation sets
# ─────────────────────────────────────────────────────────────────────

# Config-level only (no LLM calls, cheap)
CONFIG_PERTURBATIONS = [
    AggressiveCompression(factor=0.5),
    LenientCompression(factor=2.0),
    SkipCompression(),
]

# Content-level only (no LLM calls, cheap)
CONTENT_PERTURBATIONS = [
    RandomDrop(drop_fraction=0.2),
    SummaryRedaction(),
    AttributeDeletion(delete_type="paths"),
    AttributeDeletion(delete_type="errors"),
    ContextTruncation(keep_last=10),
    ContextTruncation(keep_last=20),
    SummaryNoise(),
]

# Default set: balanced mix (no LLM calls)
DEFAULT_PERTURBATIONS = [
    AggressiveCompression(factor=0.5),
    RandomDrop(drop_fraction=0.2),
    SummaryRedaction(),
    AttributeDeletion(delete_type="paths"),
    ContextTruncation(keep_last=10),
    SummaryNoise(),
]

# Full set including LLM-based (more expensive)
ALL_PERTURBATIONS = DEFAULT_PERTURBATIONS + [
    LenientCompression(factor=2.0),
    SkipCompression(),
    AttributeDeletion(delete_type="errors"),
    ContextTruncation(keep_last=20),
    LLMSummaryRewrite(),
]
