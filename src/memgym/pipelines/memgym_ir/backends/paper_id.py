"""Canonical paper identity for cross-backend deduplication.

Each backend has its own ID scheme (arxiv_id, S2 paperId, OpenAlex ID, DOI, PMID).
This module normalizes them into a single canonical ``paper_id`` string so the
FictionalizedBackend (Invariant A) can cache per-paper, not per-retrieval.

Fallback chain:  DOI → arxiv_id → backend-specific ID → sha1(normalized_title).
"""

import hashlib
import re
import unicodedata


def normalize_title(title: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_hash(title: str) -> str:
    """SHA-1 hex digest of the normalized title (first 16 chars)."""
    norm = normalize_title(title)
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def make_paper_id(
    *,
    doi: str = "",
    arxiv_id: str = "",
    backend_id: str = "",
    source: str = "",
    title: str = "",
    url: str = "",
) -> str:
    """Build a canonical paper_id from available identifiers.

    Priority:
      1. DOI (globally unique, cross-backend)
      2. arXiv ID (unique within arxiv, but S2/OpenAlex often carry it too)
      3. Backend-specific ID (S2 paperId, OpenAlex ID, PMID, etc.)
      4. URL (for non-paper sources like Wikipedia, DDG)
      5. sha1(normalized_title) as last resort
    """
    if doi:
        return f"doi:{doi.lower().strip()}"
    if arxiv_id:
        # Strip version suffix for dedup (2301.12345v1 → 2301.12345)
        clean = re.sub(r"v\d+$", "", arxiv_id.strip())
        return f"arxiv:{clean}"
    if backend_id and source:
        return f"{source}:{backend_id}"
    if url:
        return f"url:{url}"
    if title:
        return f"title:{title_hash(title)}"
    return "unknown"
