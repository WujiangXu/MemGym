"""arXiv search backend (free, full paper abstracts).

Uses the arxiv Python package for structured search.
Returns title, abstract, authors, categories, and PDF URL.

Rate limit: arXiv export API returns 429 under heavy concurrent load.
We use page_size=num_results (not default 100) and a global rate limiter.
"""

import logging
import threading
import time
from typing import List

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_ARXIV_RATE_DELAY = 1.0  # 1s between requests (lighter than Semantic Scholar's 3s)
_arxiv_lock = threading.Lock()
_arxiv_last_request = 0.0


class ArxivBackend(SearchBackend):
    """Search arXiv papers via the arxiv Python package."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv not installed — pip install arxiv")
            return []

        self._rate_limit()
        try:
            client = arxiv.Client(page_size=num_results, delay_seconds=3.0, num_retries=3)
            search = arxiv.Search(
                query=query,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = list(client.results(search))
        except Exception as e:
            logger.warning(f"arXiv search failed for '{query}': {e}")
            return []

        results = []
        for paper in papers:
            abstract = paper.summary or ""
            authors = [a.name for a in paper.authors]

            aid = paper.entry_id.split("/")[-1] if paper.entry_id else ""
            from .paper_id import make_paper_id
            pid = make_paper_id(arxiv_id=aid, source="arxiv", title=paper.title or "")

            results.append(SearchResult(
                title=paper.title or "",
                url=paper.entry_id or "",
                snippet=abstract[:200],
                full_text=abstract,
                source="arxiv",
                paper_id=pid,
                metadata={
                    "authors": authors,
                    "published": paper.published.isoformat() if paper.published else "",
                    "categories": paper.categories or [],
                    "pdf_url": paper.pdf_url or "",
                    "arxiv_id": aid,
                },
            ))

        logger.info(f"arXiv: {len(results)} papers for '{query}'")
        return results

    def _rate_limit(self):
        """Global rate limiter across all ArxivBackend instances/threads."""
        global _arxiv_last_request
        with _arxiv_lock:
            elapsed = time.time() - _arxiv_last_request
            if elapsed < _ARXIV_RATE_DELAY:
                time.sleep(_ARXIV_RATE_DELAY - elapsed)
            _arxiv_last_request = time.time()
