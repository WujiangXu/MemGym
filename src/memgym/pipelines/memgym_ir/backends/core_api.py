"""CORE API search backend (core.ac.uk) — open access full text.

The only free API that returns actual full text of papers (not just abstracts).
431M records, 46M full texts. Rate limit: ~30 requests/minute.

API docs: https://api.core.ac.uk/docs/v3
"""

import logging
import time
from typing import List

import requests

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.core.ac.uk/v3"
_RATE_DELAY = 2.5  # seconds between requests (~30 req/min for search)


class COREBackend(SearchBackend):
    """Search open access papers via CORE API with full text."""

    def __init__(self, api_key: str = ""):
        self._headers = {"User-Agent": "MemGym-IR/1.0 (research project)"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._last_request_time = 0.0

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        self._rate_limit()
        try:
            resp = requests.post(
                f"{_API_BASE}/search/works",
                json={"q": query, "limit": num_results},
                headers=self._headers,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            works = data.get("results", [])
        except Exception as e:
            logger.warning(f"CORE search failed for '{query}': {e}")
            return []

        results = []
        for work in works:
            title = work.get("title", "") or ""
            abstract = work.get("abstract", "") or ""
            full_text = work.get("fullText", "") or ""

            # Use full text if available, otherwise abstract
            content = full_text if full_text else abstract

            # Truncate very long full texts to ~3000 words
            if content and len(content.split()) > 3000:
                content = " ".join(content.split()[:3000])

            # Authors
            authors = []
            for author in work.get("authors", [])[:10]:
                name = author.get("name", "")
                if name:
                    authors.append(name)

            # URLs
            doi = work.get("doi", "") or ""
            download_url = work.get("downloadUrl", "") or ""
            url = doi if doi else download_url

            from .paper_id import make_paper_id
            pid = make_paper_id(doi=doi, source="core", title=title)

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=abstract[:200] if abstract else content[:200],
                full_text=content,
                source="core",
                paper_id=pid,
                metadata={
                    "authors": authors,
                    "year": work.get("yearPublished"),
                    "doi": doi,
                    "download_url": download_url,
                    "has_full_text": bool(full_text),
                    "language": work.get("language", {}).get("code", ""),
                },
            ))

        num_full = sum(1 for r in results if r.metadata.get("has_full_text"))
        logger.info(f"CORE: {len(results)} works for '{query}' ({num_full} with full text)")
        return results

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_DELAY:
            time.sleep(_RATE_DELAY - elapsed)
        self._last_request_time = time.time()
