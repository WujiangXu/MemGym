"""Base class and presets for difficulty transforms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..types.schemas import MemGymInstance
from ..llm.client import LLMClient


class Transform(ABC):
    """Base class for difficulty transforms."""

    name: str = "base"
    requires_llm: bool = False

    @abstractmethod
    def apply(
        self,
        instance: MemGymInstance,
        worker_client: Optional[LLMClient] = None,
        repo_context: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> MemGymInstance:
        """Apply transform, return modified instance."""
        ...

    def can_apply(self, instance: MemGymInstance) -> bool:
        """Check if transform is applicable to this instance."""
        return True


# Difficulty presets: each preset is a list of (transform_name, kwargs) tuples.
PRESETS: Dict[str, List[tuple]] = {
    "easy": [
        ("prompt_fuzz", {"intensity": "light"}),
        ("distractor_scale", {"target_count": 3}),
    ],
    "medium": [
        ("prompt_fuzz", {"intensity": "medium"}),
        ("distractor_scale", {"target_count": 5}),
        ("indirection", {}),
    ],
    "hard": [
        ("prompt_fuzz", {"intensity": "heavy"}),
        ("distractor_scale", {"target_count": 8}),
        ("indirection", {}),
        ("fact_fragment", {"num_fragments": 2}),
    ],
}
