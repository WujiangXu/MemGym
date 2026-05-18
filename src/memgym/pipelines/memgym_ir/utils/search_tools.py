"""Search tool wrappers for the deep research trajectory generator.

Provides DuckDuckGo (free, no API key) and Wikipedia (MediaWiki API)
search backends. Both return a uniform list of {title, url, text} dicts.
"""

import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# DDG retry settings (EC2 IPs frequently get rate-limited/captcha'd)
_DDG_MAX_RETRIES = 3
_DDG_BACKOFF_BASE = 2  # seconds: 2, 4, 8


def search_ddg(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo text search via the duckduckgo-search library.

    Retries with exponential backoff on empty results (common on EC2).
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed — pip install duckduckgo-search")
        return []

    for attempt in range(_DDG_MAX_RETRIES):
        try:
            results = DDGS().text(query, max_results=num_results)
            if results:
                return [
                    {"title": r.get("title", ""), "url": r.get("href", ""), "text": r.get("body", "")}
                    for r in results
                ]
            # Empty results (rate-limited / captcha) — retry with backoff
            delay = _DDG_BACKOFF_BASE * (2 ** attempt)
            logger.info(f"DDG returned 0 results for '{query}' (attempt {attempt+1}/{_DDG_MAX_RETRIES}), retrying in {delay}s")
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"DDG search failed for '{query}' (attempt {attempt+1}): {e}")
            if attempt < _DDG_MAX_RETRIES - 1:
                time.sleep(_DDG_BACKOFF_BASE * (2 ** attempt))

    logger.warning(f"DDG returned 0 results for '{query}' after {_DDG_MAX_RETRIES} retries")
    return []


def search_wikipedia(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Wikipedia search via the MediaWiki API (no key needed)."""
    try:
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": num_results,
                "srprop": "snippet|titlesnippet",
            },
            headers={"User-Agent": "MemGym-IR/1.0 (research project)"},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        return [
            {
                "title": r["title"],
                "url": f"https://en.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
                "text": _strip_html(r.get("snippet", "")),
            }
            for r in hits
        ]
    except ImportError:
        logger.warning("requests not installed — pip install requests")
        return []
    except Exception as e:
        logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return []


def combined_search(
    query: str,
    num_results: int = 5,
    ddg_results: int = 3,
    wiki_results: int = 2,
) -> List[Dict[str, str]]:
    """Combined DDG + Wikipedia search. Deduplicates by title."""
    results = search_ddg(query, ddg_results)
    seen_titles = {r["title"].lower() for r in results}
    for r in search_wikipedia(query, wiki_results):
        if r["title"].lower() not in seen_titles:
            results.append(r)
            seen_titles.add(r["title"].lower())
    return results[:num_results]


def get_search_fn(search_tool: str = "ddg+wikipedia"):
    """Return a search function based on the search_tool string."""
    if search_tool == "ddg":
        return search_ddg
    elif search_tool == "wikipedia":
        return search_wikipedia
    elif search_tool == "ddg+wikipedia":
        return combined_search
    else:
        raise ValueError(f"Unknown search tool: {search_tool}. Use ddg, wikipedia, or ddg+wikipedia")


def _strip_html(text: str) -> str:
    """Remove HTML tags from Wikipedia snippets."""
    import re
    return re.sub(r"<[^>]+>", "", text)
