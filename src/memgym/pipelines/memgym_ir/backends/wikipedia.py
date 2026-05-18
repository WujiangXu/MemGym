"""Wikipedia search backend with full article intro text.

Uses the MediaWiki API to search and retrieve full introduction sections
(not just HTML snippets). Provides 200-800 word introductions.
"""

import logging
import re
from typing import List

import requests

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_API_URL = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "MemGym-IR/1.0 (research project)"}


class WikipediaBackend(SearchBackend):
    """Wikipedia search with full intro text via the extracts API."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        # Step 1: Search for page titles
        try:
            search_resp = requests.get(
                _API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": num_results,
                    "srprop": "snippet|titlesnippet",
                },
                headers=_HEADERS,
                timeout=10,
            )
            search_resp.raise_for_status()
            hits = search_resp.json().get("query", {}).get("search", [])
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
            return []

        if not hits:
            return []

        # Step 2: Fetch full intro text for each page via extracts API
        titles = [h["title"] for h in hits]
        try:
            extract_resp = requests.get(
                _API_URL,
                params={
                    "action": "query",
                    "titles": "|".join(titles),
                    "prop": "extracts",
                    "exintro": True,        # Intro section only
                    "explaintext": True,    # Plain text, not HTML
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=15,
            )
            extract_resp.raise_for_status()
            pages = extract_resp.json().get("query", {}).get("pages", {})
            extracts_by_title = {
                p.get("title", ""): p.get("extract", "")
                for p in pages.values()
            }
        except Exception as e:
            logger.warning(f"Wikipedia extracts failed: {e}")
            extracts_by_title = {}

        results = []
        for hit in hits:
            title = hit["title"]
            snippet = _strip_html(hit.get("snippet", ""))
            full_text = extracts_by_title.get(title, "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            from .paper_id import make_paper_id
            pid = make_paper_id(url=url, source="wikipedia", title=title)

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                full_text=full_text,
                source="wikipedia",
                paper_id=pid,
            ))

        logger.info(f"Wikipedia: {len(results)} articles for '{query}'")
        return results


def _strip_html(text: str) -> str:
    """Remove HTML tags from Wikipedia snippets."""
    return re.sub(r"<[^>]+>", "", text)
