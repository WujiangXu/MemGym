"""Prompt Fuzzing transform: make the task prompt vaguer."""

from typing import Dict, Optional

from .base import Transform
from ..types.schemas import MemGymInstance, PROMPT_FUZZ_SCHEMA
from ..llm.client import LLMClient
from ..llm.prompts import PROMPT_FUZZ_PROMPT


class PromptFuzzTransform(Transform):
    """Make the task prompt vaguer at a specified intensity level."""

    name = "prompt_fuzz"
    requires_llm = True

    def apply(
        self,
        instance: MemGymInstance,
        worker_client: Optional[LLMClient] = None,
        repo_context: Optional[Dict[str, str]] = None,
        intensity: str = "medium",
        **kwargs,
    ) -> MemGymInstance:
        if not worker_client:
            return instance

        prompt = PROMPT_FUZZ_PROMPT.format(
            original_prompt=instance.task_prompt,
            intensity=intensity,
        )

        try:
            result = worker_client.get_json_completion(
                prompt=prompt,
                response_format=PROMPT_FUZZ_SCHEMA,
                temperature=0.7,
            )
            fuzzed = result.get("fuzzed_prompt", "")
            if fuzzed and len(fuzzed) > 20:
                updated = instance.model_copy(
                    update={
                        "task_prompt": fuzzed,
                        "transforms_applied": instance.transforms_applied
                        + [f"prompt_fuzz_{intensity}"],
                    }
                )
                return updated
        except Exception:
            pass

        return instance

    def can_apply(self, instance: MemGymInstance) -> bool:
        return bool(instance.task_prompt)
