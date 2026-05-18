"""OpenAlex search backend (free tier: $1/day, 1K searches/day).

Broadest academic coverage (297M works). Abstracts stored as inverted
index — reconstructed to plaintext here. Requires free API key since Feb 2026.

API docs: https://docs.openalex.org/
"""

import logging
import os
from typing import List

import requests

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.openalex.org"


class OpenAlexBackend(SearchBackend):
    """Search academic papers via OpenAlex API."""

    def __init__(self, email: str = "", api_key: str = ""):
        self._params = {}
        # API key (required since Feb 2026) — from arg, env, or anonymous
        self._api_key = api_key or os.environ.get("OPENALEX_API_KEY", "")
        if email:
            self._params["mailto"] = email  # polite pool

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        headers = {"User-Agent": "MemGym-IR/1.0 (research project)"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = requests.get(
                f"{_API_BASE}/works",
                params={
                    "search": query,
                    "per_page": num_results,
                    "select": "id,doi,title,abstract_inverted_index,authorships,"
                              "publication_year,cited_by_count,primary_location,"
                              "open_access",
                    **self._params,
                },
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            works = resp.json().get("results", [])
        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

        results = []
        for work in works:
            title = work.get("title", "") or ""
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

            # Build URL
            doi = work.get("doi", "")
            openalex_id = work.get("id", "")
            url = doi or openalex_id

            # Authors
            authors = []
            for authorship in work.get("authorships", [])[:10]:
                author = authorship.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            # Open access PDF
            oa = work.get("open_access", {})
            pdf_url = oa.get("oa_url", "") if oa else ""

            from .paper_id import make_paper_id
            pid = make_paper_id(
                doi=doi,
                backend_id=openalex_id,
                source="openalex",
                title=title,
            )

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=abstract[:200] if abstract else "",
                full_text=abstract,
                source="openalex",
                paper_id=pid,
                metadata={
                    "authors": authors,
                    "year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "openalex_id": openalex_id,
                    "doi": doi,
                    "pdf_url": pdf_url,
                },
            ))

        logger.info(f"OpenAlex: {len(results)} works for '{query}'")
        return results


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct plaintext abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as {"word": [pos1, pos2, ...], ...}.
    We invert this to get the original text.
    """
    if not inverted_index:
        return ""
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(word for _, word in word_positions)
    except Exception:
        return ""
