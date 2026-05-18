"""URL content scraper using trafilatura.

Extracts main article text from web pages. Gracefully returns empty string
on any failure (paywall, JS-rendered, timeout, etc.).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-memory cache to avoid re-scraping the same URL
_cache: dict = {}


def scrape_url(url: str, max_tokens: int = 2000) -> str:
    """Scrape full text content from a URL.

    Returns extracted text truncated to max_tokens words, or empty string on failure.
    Results are cached in memory.
    """
    if url in _cache:
        return _cache[url]

    text = _fetch_and_extract(url)
    if text and max_tokens > 0:
        words = text.split()
        if len(words) > max_tokens:
            text = " ".join(words[:max_tokens])

    _cache[url] = text
    return text


def _fetch_and_extract(url: str) -> str:
    """Fetch URL and extract main content. Returns empty string on failure."""
    try:
        import trafilatura
    except ImportError:
        logger.debug("trafilatura not installed — pip install trafilatura")
        return ""

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded) or ""
        return text.strip()
    except Exception as e:
        logger.debug(f"Scrape failed for {url}: {e}")
        return ""
