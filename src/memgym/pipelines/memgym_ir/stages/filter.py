"""Stage 0: Mine -- Load dataset via adapter, filter by complexity criteria.

Supports multiple datasets via adapter pattern:
- musique: MuSiQue multi-hop QA (HuggingFace)
- hotpotqa: HotpotQA multi-hop QA (HuggingFace)
- synthetic: Fully LLM-generated instances
"""

import json
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..types.schemas import FilteredIRInstance
from ..llm.client import LLMClient
from .adapters import ADAPTERS


def filter_instance(
    instance: FilteredIRInstance,
    min_hops: int = 2,
    min_supporting: int = 2,
    min_answer_length: int = 2,
    min_question_length: int = 30,
) -> bool:
    """Check if an instance meets filter criteria."""
    answerable = getattr(instance, "answerable", True)
    if not answerable:
        return False

    if instance.num_hops < min_hops:
        return False

    if len(instance.supporting_paragraphs) < min_supporting:
        return False

    if len(instance.answer) < min_answer_length:
        return False

    if len(instance.question) < min_question_length:
        return False

    return True


def filter_instances(
    dataset: str = "musique",
    limit: Optional[int] = None,
    min_hops: int = 2,
    min_supporting: int = 2,
    min_answer_length: int = 2,
    min_question_length: int = 30,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
    worker_client: Optional[LLMClient] = None,
    dry_run: bool = False,
) -> List[FilteredIRInstance]:
    """Load dataset and filter by complexity criteria.

    Args:
        dataset: Dataset name ("musique", "hotpotqa", "synthetic").
        worker_client: LLM client for adapters that need it (HotpotQA decomposition, synthetic generation).
        dry_run: Skip LLM calls in adapters.

    Returns:
        List of FilteredIRInstance that meet all criteria.
    """
    if dataset not in ADAPTERS:
        available = ", ".join(ADAPTERS.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")

    adapter = ADAPTERS[dataset]()
    raw_instances = adapter.load_raw(limit=limit)

    filtered = []
    iterator = tqdm(raw_instances, desc=f"Filtering {dataset}") if show_progress else raw_instances

    for raw in iterator:
        parsed = adapter.parse(raw, worker_client=worker_client, dry_run=dry_run)
        if parsed is None:
            continue
        if filter_instance(
            parsed,
            min_hops=min_hops,
            min_supporting=min_supporting,
            min_answer_length=min_answer_length,
            min_question_length=min_question_length,
        ):
            filtered.append(parsed)

    print(f"Filtered to {len(filtered)} instances "
          f"({len(filtered)/max(len(raw_instances),1)*100:.1f}%)")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for inst in filtered:
                f.write(json.dumps(inst.model_dump()) + "\n")
        print(f"Saved to {output_path}")

    return filtered


def load_filtered_instances(path: Path) -> List[FilteredIRInstance]:
    """Load filtered instances from JSONL checkpoint."""
    instances = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            instances.append(FilteredIRInstance(**data))
    return instances
