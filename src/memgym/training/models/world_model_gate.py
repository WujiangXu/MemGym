"""WorldModelGate — runtime wrapper for the the training phase SFT classifier.

Wraps the `aug_sft_8b_yn_cw3_v4` champion (Qwen3-8B-Base + LoRA,
class-weighted CE cap 3.0, `Y`/`N` labels, t*=0.78) behind the
deployment contract from `the internal handoff doc`:

    gate(agent_model, full_history, compressed_history) -> {
        "decision": bool,          # prob_safe >= threshold
        "prob_safe": float,
        "recorded_action": str,    # agent_model on full_history
        "proposed_action": str,    # agent_model on compressed_history
        "proposed_response": dict, # reusable — agent skips its own call
        "latency_ms": {...},
    }

The `recorded_action` / `proposed_action` terminology is the exact
training-time contract from `aug_pairs.PROMPT_TEMPLATE`. Keeping it
verbatim is the whole point of Option-A deployment — the gate sees
the same prompt shape at inference that the classifier was trained on.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol

from memgym.training.data.aug_pairs import RESPONSE_PREFIX, build_prompt_template
from memgym.training.models.world_model import MemoryWorldModel, ModelConfig

logger = logging.getLogger(__name__)


class _AgentModel(Protocol):
    """Minimal structural type the gate needs from mini-swe-agent's model."""

    def query(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]: ...


_ACTION_RE = re.compile(r"```(?:bash)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_action(content: str) -> str:
    """Pull the bash action out of a mini-swe-agent response.

    Kept local instead of imported from `swe_bench.agent` to keep
    `training/models/` free of gym-layer dependencies. The regex matches
    the first ```bash ... ``` block; empty string if none found.
    """
    match = _ACTION_RE.search(content or "")
    return match.group(1).strip() if match else ""


class WorldModelGate:
    """the online smoke phase gate. Loads the classifier once; cheap per-call."""

    def __init__(
        self,
        checkpoint: str,
        threshold: float = 0.78,
        base_model: str = "Qwen/Qwen3-8B-Base",
        safe_completion: str = " Y",
        harmful_completion: str = " N",
        load_in_4bit: bool = True,
    ):
        config = ModelConfig(
            base_model=base_model,
            load_in_4bit=load_in_4bit,
            safe_completion=safe_completion,
            harmful_completion=harmful_completion,
        )
        self._wm = MemoryWorldModel.from_checkpoint(checkpoint, config)
        self.threshold = threshold
        self._template = build_prompt_template(safe_completion, harmful_completion)
        self._response_prefix = RESPONSE_PREFIX
        logger.info(
            "WorldModelGate ready: checkpoint=%s threshold=%.3f base=%s labels=%r/%r",
            checkpoint, threshold, base_model, safe_completion, harmful_completion,
        )

    def score(
        self,
        recorded_action: str,
        proposed_action: str,
        step: int = 0,
        perturbation: str = "unknown",
    ) -> float:
        """Classifier-only path: score pre-materialized actions.

        Used by the baseline-train phase prompt-fidelity audit and by `gate()` after the
        two agent calls. `perturbation="unknown"` is safe — the baseline-train phase-pre
        audit (perturbation_sensitivity.json) confirmed Δ-SAFE-F1 = +0.004
        when the perturbation tag is rewritten to `"unknown"`, so the
        classifier ignores this field at prediction time.
        """
        text = self._template.format(
            step=step,
            perturbation=perturbation,
            recorded=recorded_action,
            predicted=proposed_action,
        ) + self._response_prefix
        _, prob_safe = self._wm.predict_logits_text(text, safe_threshold=self.threshold)
        return prob_safe

    def gate(
        self,
        agent_model: _AgentModel,
        full_history: List[Dict[str, Any]],
        compressed_history: List[Dict[str, Any]],
        step: int = 0,
        perturbation: str = "unknown",
    ) -> Dict[str, Any]:
        """Deployment path: double-call + classifier + threshold.

        `full_history` = agent.messages (pre-compression, full length).
        `compressed_history` = memory_manager output (post-compression).
        Returns `proposed_response` so the caller can skip its own
        agent-query on the compressed side — the gate already did that
        call and there's no reason to repeat it.
        """
        t0 = time.perf_counter()
        recorded_response = agent_model.query(full_history)
        recorded_action = _extract_action(recorded_response.get("content", ""))
        t1 = time.perf_counter()

        proposed_response = agent_model.query(compressed_history)
        proposed_action = _extract_action(proposed_response.get("content", ""))
        t2 = time.perf_counter()

        prob_safe = self.score(
            recorded_action=recorded_action,
            proposed_action=proposed_action,
            step=step,
            perturbation=perturbation,
        )
        t3 = time.perf_counter()

        return {
            "decision": prob_safe >= self.threshold,
            "prob_safe": prob_safe,
            "recorded_action": recorded_action,
            "proposed_action": proposed_action,
            "proposed_response": proposed_response,
            "latency_ms": {
                "agent_full": (t1 - t0) * 1000.0,
                "agent_compressed": (t2 - t1) * 1000.0,
                "classifier": (t3 - t2) * 1000.0,
                "total": (t3 - t0) * 1000.0,
            },
        }
