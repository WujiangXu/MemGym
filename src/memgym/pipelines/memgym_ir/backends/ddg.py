"""DuckDuckGo search backend with retry logic and optional URL scraping.

Retries with exponential backoff on empty results (common on datacenter IPs).
Optionally scrapes full text from result URLs via scraper module.
"""

import logging
import time
from typing import List

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8


class DDGBackend(SearchBackend):
    """DuckDuckGo search with retry and optional full-text scraping."""

    def __init__(self, scrape: bool = True, max_scrape_tokens: int = 2000):
        self.scrape = scrape
        self.max_scrape_tokens = max_scrape_tokens

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed — pip install duckduckgo-search")
            return []

        raw_results = self._search_with_retry(DDGS, query, num_results)
        if not raw_results:
            return []

        results = []
        for r in raw_results:
            url = r.get("href", "")
            snippet = r.get("body", "")
            full_text = ""

            if self.scrape and url:
                full_text = self._scrape_url(url)

            from .paper_id import make_paper_id
            pid = make_paper_id(url=url, source="ddg", title=r.get("title", ""))

            results.append(SearchResult(
                title=r.get("title", ""),
                url=url,
                snippet=snippet,
                full_text=full_text,
                source="ddg",
                paper_id=pid,
            ))

        logger.info(f"DDG: {len(results)} results for '{query}'"
                     f" ({sum(1 for r in results if r.full_text)} scraped)")
        return results

    def _search_with_retry(self, DDGS, query: str, num_results: int) -> list:
        for attempt in range(_MAX_RETRIES):
            try:
                results = DDGS().text(query, max_results=num_results)
                if results:
                    return results
                delay = _BACKOFF_BASE * (2 ** attempt)
                logger.info(
                    f"DDG returned 0 results for '{query}' "
                    f"(attempt {attempt+1}/{_MAX_RETRIES}), retrying in {delay}s"
                )
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"DDG search failed for '{query}' (attempt {attempt+1}): {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))

        logger.warning(f"DDG returned 0 results for '{query}' after {_MAX_RETRIES} retries")
        return []

    def _scrape_url(self, url: str) -> str:
        """Scrape full text from a URL. Returns empty string on failure."""
        try:
            from ..scraper import scrape_url
            return scrape_url(url, max_tokens=self.max_scrape_tokens)
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return ""
