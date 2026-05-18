"""Combined search backend — cascading search across multiple backends.

Queries backends in order (best first). Each backend adds unique results.
Stops early once num_results is reached. Deduplicates by title.

Suggested order: semantic_scholar → openalex → arxiv → pubmed → core → wikipedia → ddg
(best relevance → broadest coverage → domain-specific → full text → general)
"""

import logging
from typing import List

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)


class CombinedBackend(SearchBackend):
    """Cascade search across multiple backends in priority order."""

    def __init__(self, backends: List[SearchBackend]):
        if not backends:
            raise ValueError("CombinedBackend requires at least one backend")
        self.backends = backends

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        all_results = []
        seen_titles = set()

        for backend in self.backends:
            if len(all_results) >= num_results:
                break  # already have enough

            # Ask each backend for enough to fill remaining slots
            remaining = num_results - len(all_results)
            try:
                results = backend.search(query, num_results=remaining + 2)  # +2 for dedup headroom
            except Exception as e:
                logger.warning(f"Backend {type(backend).__name__} failed: {e}")
                continue

            for r in results:
                if len(all_results) >= num_results:
                    break
                key = r.title.lower().strip()
                if key and key not in seen_titles:
                    all_results.append(r)
                    seen_titles.add(key)

            logger.debug(
                f"  {type(backend).__name__}: +{len(results)} raw, "
                f"{len(all_results)} total unique so far"
            )

        return all_results
