"""Distractor Scaling transform: increase distractor count/similarity."""

from typing import Dict, List, Optional

from .base import Transform
from ..types.schemas import (
    MemGymInstance,
    DistractorFact,
    ADVERSARIAL_DISTRACTOR_SCHEMA,
)
from ..llm.client import LLMClient
from ..llm.prompts import ADVERSARIAL_DISTRACTOR_PROMPT
from ..utils.expansion import inject_distractors_into_files


class DistractorScaleTransform(Transform):
    """Increase the number of distractors in memory files."""

    name = "distractor_scale"
    requires_llm = True

    def apply(
        self,
        instance: MemGymInstance,
        worker_client: Optional[LLMClient] = None,
        repo_context: Optional[Dict[str, str]] = None,
        target_count: int = 5,
        **kwargs,
    ) -> MemGymInstance:
        current_count = len(instance.distractors)
        if current_count >= target_count:
            return instance

        needed = target_count - current_count

        new_distractors = list(instance.distractors)

        # Try LLM-generated adversarial distractors
        if worker_client and needed > 0:
            facts_str = "\n".join(
                f"- {f.content}" for f in instance.grounding_facts
            )
            code_ctx = ""
            if repo_context:
                code_ctx = "\n".join(
                    f"=== {fp} (first 30 lines) ===\n"
                    + "\n".join(content.split("\n")[:30])
                    for fp, content in list(repo_context.items())[:3]
                )

            prompt = ADVERSARIAL_DISTRACTOR_PROMPT.format(
                num_distractors=needed,
                real_facts=facts_str,
                code_context=code_ctx[:3000],
            )

            try:
                result = worker_client.get_json_completion(
                    prompt=prompt,
                    response_format=ADVERSARIAL_DISTRACTOR_SCHEMA,
                    temperature=0.7,
                )
                for i, d in enumerate(result.get("distractors", [])[:needed]):
                    new_distractors.append(
                        DistractorFact(
                            id=f"D{current_count + i + 1}",
                            content=d.get("content", ""),
                            source="adversarial",
                        )
                    )
            except Exception:
                pass

        if len(new_distractors) == current_count:
            return instance  # nothing added

        # Inject new distractors into existing memory files (no LLM needed)
        memory_files = dict(instance.memory_files)
        if memory_files:
            added_distractors = new_distractors[current_count:]
            memory_files = inject_distractors_into_files(memory_files, added_distractors)

        return instance.model_copy(
            update={
                "distractors": new_distractors,
                "memory_files": memory_files,
                "transforms_applied": instance.transforms_applied
                + [f"distractor_scale_{target_count}"],
            }
        )

    def can_apply(self, instance: MemGymInstance) -> bool:
        return bool(instance.memory_files)
