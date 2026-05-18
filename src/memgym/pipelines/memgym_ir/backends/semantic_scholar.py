"""Semantic Scholar search backend (free API, no key needed).

Returns academic papers with title, abstract, TLDR, authors, year.
Papers on the same topic but different findings = ideal natural distractors.

API docs: https://api.semanticscholar.org/api-docs/graph
Rate limit: 1 request/second for unauthenticated requests.
"""

import logging
import os
import threading
import time
from typing import List

import requests

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,tldr,authors,year,citationCount,externalIds,url"
_RATE_DELAY = 1.1  # seconds between requests (1 req/sec with API key, slight buffer)

# Global rate limiter shared across ALL instances and threads.
# Without this, each SemanticScholarBackend() has its own timer and
# concurrent threads all fire at t=0, causing 429 storms.
_global_lock = threading.Lock()
_global_last_request = 0.0


class SemanticScholarBackend(SearchBackend):
    """Search academic papers via Semantic Scholar's free API."""

    def __init__(self, api_key: str = ""):
        self._headers = {"User-Agent": "MemGym-IR/1.0 (research project)"}
        key = api_key or os.environ.get("S2_API_KEY", "")
        if key:
            self._headers["x-api-key"] = key
            logger.info("Semantic Scholar: using API key (1 req/sec dedicated)")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        data = self._search_with_retry(query, num_results)
        if not data:
            return []

        results = []
        for paper in data:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "") or ""
            tldr = ""
            if paper.get("tldr"):
                tldr = paper["tldr"].get("text", "")

            # Build full text: abstract + TLDR
            full_text_parts = []
            if abstract:
                full_text_parts.append(abstract)
            if tldr:
                full_text_parts.append(f"\nTL;DR: {tldr}")
            full_text = "\n\n".join(full_text_parts)

            # Build URL
            paper_id = paper.get("paperId", "")
            url = paper.get("url", "") or f"https://www.semanticscholar.org/paper/{paper_id}"

            # Authors
            authors = [a.get("name", "") for a in paper.get("authors", [])]

            ext = paper.get("externalIds", {}) or {}
            from .paper_id import make_paper_id
            pid = make_paper_id(
                doi=ext.get("DOI", ""),
                arxiv_id=ext.get("ArXiv", ""),
                backend_id=paper_id,
                source="s2",
                title=title,
            )

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=tldr or abstract[:200],
                full_text=full_text,
                source="semantic_scholar",
                paper_id=pid,
                metadata={
                    "authors": authors,
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                    "paper_id": paper_id,
                    "external_ids": ext,
                },
            ))

        logger.info(f"Semantic Scholar: {len(results)} papers for '{query}'")
        return results

    def _search_with_retry(self, query: str, num_results: int, max_retries: int = 3) -> list:
        """Search with retry on 429 rate limit errors."""
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(
                    f"{_API_BASE}/paper/search",
                    params={
                        "query": query,
                        "limit": num_results,
                        "fields": _FIELDS,
                    },
                    headers=self._headers,
                    timeout=15,
                )
                if resp.status_code == 429:
                    delay = _RATE_DELAY * (2 ** attempt)
                    logger.info(f"Semantic Scholar 429 rate limit, retrying in {delay}s")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json().get("data", [])
            except requests.exceptions.HTTPError:
                raise  # re-raise non-429 HTTP errors after raise_for_status
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed for '{query}' (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(_RATE_DELAY * (2 ** attempt))
        return []

    def _rate_limit(self):
        """Enforce API rate limit globally across all instances/threads."""
        global _global_last_request
        with _global_lock:
            elapsed = time.time() - _global_last_request
            if elapsed < _RATE_DELAY:
                time.sleep(_RATE_DELAY - elapsed)
            _global_last_request = time.time()
