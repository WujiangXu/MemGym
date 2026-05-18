"""Cross-instance distractor pool — recycle adversarial facts to skip LLM calls.

Each verified instance already carries 5-30 high-quality adversarial distractors
(from `distractors` field, source ∈ {"adversarial_qa", "adversarial_expansion",
...}). When expanding to large token budgets we usually need 30-50 *more* of the
same type. The current pipeline pays an LLM call per instance per phase to
regenerate them — but those generated distractors are interchangeable across
instances within the same project / topic area.

This module builds a pool once, then `sample()` draws from it instead of calling
the LLM. The exclude_ids guard prevents an instance from receiving its own
original distractors back (which would make the noise trivially detectable).
"""

import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence

from ...types.schemas import CodingQAInstance, DistractorFact


# Mapping from sample(kind=...) to distractor source tags written by upstream
# generators. Source values must match DistractorFact.source Literal in
# types/schemas.py. Keep in sync with stages/qa/convert.py:_generate_qa_distractors
# and stages/expand.py:_generate_more_distractors.
_KIND_TO_SOURCES = {
    "qa":      {"adversarial_qa"},
    "generic": {"adversarial_expansion", "adversarial", "same_function",
                "cross_instance", "same_repo"},
    "any":     None,  # accept everything
}


class DistractorPool:
    """In-memory pool of recyclable distractors keyed by source tag and origin."""

    def __init__(self) -> None:
        # source_tag -> list of (origin_instance_id, DistractorFact)
        self._by_source: Dict[str, List[tuple]] = defaultdict(list)
        self._n_total = 0

    def __len__(self) -> int:
        return self._n_total

    def add(self, instance_id: str, distractor: DistractorFact) -> None:
        src = (distractor.source or "unknown").strip()
        self._by_source[src].append((instance_id, distractor))
        self._n_total += 1

    def add_instance(self, inst: CodingQAInstance) -> int:
        """Add all distractors from one instance. Returns how many were added."""
        n = 0
        for d in inst.distractors:
            self.add(inst.instance_id, d)
            n += 1
        return n

    def sample(
        self,
        kind: str,
        count: int,
        exclude_ids: Optional[str] = None,
        rng: Optional[Random] = None,
    ) -> List[DistractorFact]:
        """Draw `count` distractors of the given kind, excluding origin overlap.

        Returns fewer than `count` (possibly empty) if the pool is depleted.
        Caller should check `len(result) < count` and decide whether to
        fall back to LLM generation for the remainder.
        """
        if count <= 0:
            return []
        rng = rng or Random()
        sources = _KIND_TO_SOURCES.get(kind)
        if sources is None and kind != "any":
            raise ValueError(f"Unknown kind {kind!r}; expected one of {list(_KIND_TO_SOURCES)}")

        candidates: List[tuple] = []
        if sources is None:
            for items in self._by_source.values():
                candidates.extend(items)
        else:
            for src in sources:
                candidates.extend(self._by_source.get(src, ()))

        excluded = {exclude_ids} if exclude_ids else set()
        candidates = [(iid, d) for iid, d in candidates if iid not in excluded]
        if not candidates:
            return []

        # Sample without replacement (or with, if pool is too small).
        if len(candidates) >= count:
            picked = rng.sample(candidates, count)
        else:
            picked = list(candidates)
            picked.extend(rng.choices(candidates, k=count - len(candidates)))

        # Re-id the borrowed distractors so injectors don't collide; preserve
        # the original source tag (DistractorFact.source is a Literal so we
        # can't add a "pool:" prefix without expanding the schema).
        out: List[DistractorFact] = []
        for i, (origin_iid, d) in enumerate(picked):
            out.append(DistractorFact(
                id=f"DP{i+1}",
                content=d.content,
                source=d.source,
            ))
        return out


def build_pool(instances: Iterable[CodingQAInstance]) -> DistractorPool:
    """Build a pool by harvesting `distractors` from every input instance."""
    pool = DistractorPool()
    n_inst = 0
    for inst in instances:
        n_inst += 1
        pool.add_instance(inst)
    print(f"  Built distractor pool: {len(pool)} entries from {n_inst} instances "
          f"({sorted(pool._by_source.keys())})")
    return pool


def load_pool_from_jsonl(path: Path) -> DistractorPool:
    """Stream a jsonl of CodingQAInstance and build the pool. Memory-light."""
    pool = DistractorPool()
    n_inst = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = CodingQAInstance(**json.loads(line))
            n_inst += 1
            pool.add_instance(inst)
    print(f"  Loaded distractor pool: {len(pool)} entries from {n_inst} instances")
    return pool
