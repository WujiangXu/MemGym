"""Indirection transform: replace direct references with multi-hop descriptions."""

from typing import Dict, Optional

from .base import Transform
from ..types.schemas import MemGymInstance, INDIRECTION_TRANSFORM_SCHEMA
from ..llm.client import LLMClient
from ..llm.prompts import INDIRECTION_TRANSFORM_PROMPT


class IndirectionTransform(Transform):
    """Replace direct code references in memory files with indirect descriptions."""

    name = "indirection"
    requires_llm = True

    def apply(
        self,
        instance: MemGymInstance,
        worker_client: Optional[LLMClient] = None,
        repo_context: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> MemGymInstance:
        if not worker_client or not instance.memory_files or not repo_context:
            return instance

        # Build repo context summary for the prompt
        repo_summary_parts = []
        for fp, content in list(repo_context.items())[:5]:
            lines = content.split("\n")[:50]
            repo_summary_parts.append(f"=== {fp} ===\n" + "\n".join(lines))
        repo_summary = "\n\n".join(repo_summary_parts)[:6000]

        # Apply indirection to ALL memory files (not just the largest)
        files = dict(instance.memory_files)
        all_indirections = []

        for target_file in list(files.keys()):
            if len(files[target_file]) < 50:
                continue  # Skip very short files

            prompt = INDIRECTION_TRANSFORM_PROMPT.format(
                file_content=files[target_file][:4000],
                repo_context_summary=repo_summary,
            )

            try:
                result = worker_client.get_json_completion(
                    prompt=prompt,
                    response_format=INDIRECTION_TRANSFORM_SCHEMA,
                    temperature=0.7,
                )
                rewritten = result.get("rewritten_content", "")
                indirections = result.get("indirections_added", [])

                if rewritten and len(rewritten) > 50 and indirections:
                    files[target_file] = rewritten
                    all_indirections.extend(indirections)
            except Exception:
                continue

        if all_indirections:
            return instance.model_copy(
                update={
                    "memory_files": files,
                    "transforms_applied": instance.transforms_applied
                    + ["indirection"],
                }
            )

        return instance

    def can_apply(self, instance: MemGymInstance) -> bool:
        return bool(instance.memory_files) and len(instance.memory_files) > 0
