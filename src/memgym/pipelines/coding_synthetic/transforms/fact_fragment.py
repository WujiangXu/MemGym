"""Fact Fragmentation transform: split critical facts across multiple memory files."""

import random
from typing import Dict, Optional

from .base import Transform
from ..types.schemas import MemGymInstance, FACT_FRAGMENT_SCHEMA
from ..llm.client import LLMClient
from ..llm.prompts import FACT_FRAGMENT_PROMPT


class FactFragmentTransform(Transform):
    """Split critical facts into partial hints across different memory files.

    This requires multi-hop reasoning: the agent must combine information
    from multiple documents to reconstruct the full fact needed for the fix.
    """

    name = "fact_fragment"
    requires_llm = True

    def apply(
        self,
        instance: MemGymInstance,
        worker_client: Optional[LLMClient] = None,
        repo_context: Optional[Dict[str, str]] = None,
        num_fragments: int = 2,
        **kwargs,
    ) -> MemGymInstance:
        if not worker_client or not instance.memory_files:
            return instance
        if len(instance.memory_files) < 2:
            return instance  # Need at least 2 files for fragmentation

        # Identify critical non-discoverable facts
        critical_facts = [
            f for f in instance.grounding_facts
            if not f.discoverable and f.relevance == "critical"
        ]
        if not critical_facts:
            return instance

        # Select facts to fragment (up to num_fragments)
        to_fragment = random.sample(
            critical_facts, min(num_fragments, len(critical_facts))
        )

        facts_str = "\n".join(
            f"- [{f.id}] {f.content}" for f in to_fragment
        )
        file_names = list(instance.memory_files.keys())

        prompt = FACT_FRAGMENT_PROMPT.format(
            facts_to_fragment=facts_str,
            file_names=", ".join(file_names),
            num_files=len(file_names),
        )

        try:
            result = worker_client.get_json_completion(
                prompt=prompt,
                response_format=FACT_FRAGMENT_SCHEMA,
                temperature=0.7,
            )
            fragments = result.get("fragments", [])
            if not fragments:
                return instance

            # Apply fragments: append partial hints to target files
            files = dict(instance.memory_files)
            applied = 0
            for frag in fragments:
                target_file = frag.get("target_file", "")
                partial_text = frag.get("partial_content", "")
                if target_file in files and partial_text:
                    files[target_file] += f"\n\n{partial_text}"
                    applied += 1

            if applied > 0:
                return instance.model_copy(
                    update={
                        "memory_files": files,
                        "transforms_applied": instance.transforms_applied
                        + ["fact_fragment"],
                    }
                )
        except Exception:
            pass

        return instance

    def can_apply(self, instance: MemGymInstance) -> bool:
        return (
            bool(instance.memory_files)
            and len(instance.memory_files) >= 2
            and any(
                not f.discoverable and f.relevance == "critical"
                for f in instance.grounding_facts
            )
        )
