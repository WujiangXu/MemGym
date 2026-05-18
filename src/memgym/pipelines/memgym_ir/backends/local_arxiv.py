"""Local arXiv search backend — SQLite FTS5, zero rate limits.

Searches a pre-built SQLite database of ~2.7M arXiv paper metadata
(title + abstract). No API calls, no rate limits, ~5-20ms per query.

Build the index first:
    python scripts/build_local_arxiv_index.py

DB path resolution (priority order):
    1. Constructor db_path argument
    2. LOCAL_ARXIV_DB environment variable
    3. ~/.memgym/arxiv_papers.db
"""

import logging
import os
import re
import sqlite3
from typing import List

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.expanduser("~/.memgym/arxiv_papers.db")

# Whitelist approach: FTS5 has a growing list of syntax-significant characters
# (`"()*+-:^{}~`) and the tokenizer can choke on others like `/` ("STM/STS ..."
# crashed with `fts5: syntax error near "/"` in production). Rather than
# enumerate every special char, strip anything that isn't a word char or
# whitespace. Unicode word chars stay in (Greek letters, accents, etc.).
_FTS5_NONWORD = re.compile(r'[^\w\s]', re.UNICODE)
_FTS5_OPERATORS = re.compile(r'\b(AND|OR|NOT|NEAR)\b', re.IGNORECASE)


class LocalArxivBackend(SearchBackend):
    """Search local arXiv SQLite FTS5 index."""

    def __init__(self, db_path: str = ""):
        path = db_path or os.environ.get("LOCAL_ARXIV_DB", _DEFAULT_DB_PATH)
        self._db_path = path

        if not os.path.exists(path):
            logger.warning(
                f"Local arXiv DB not found at {path}. "
                f"Run: python scripts/build_local_arxiv_index.py"
            )
            self._available = False
            self._conn = None
            return

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._available = True

        count = self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        logger.info(f"Local arXiv backend: {count:,} papers from {path}")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not self._available:
            return []

        clean = self._sanitize(query)
        if not clean.strip():
            return []

        # First attempt: FTS5's default implicit AND. High precision —
        # every token must appear in the paper. Good when the query is
        # 2-3 focused keywords.
        rows = self._fts_match(clean, num_results)

        # Fallback: when AND returns nothing, broaden to OR. BM25 still
        # ranks the top-k, so results concentrate on papers matching the
        # most query terms. Only kicks in for ≥3-token queries — a
        # 1-2-token empty result is a genuine miss, not over-specificity.
        if not rows and len(clean.split()) >= 3:
            or_query = " OR ".join(clean.split())
            rows = self._fts_match(or_query, num_results)
            if rows:
                logger.info(f"local_arxiv: AND→OR fallback for '{query[:60]}'")

        results = [self._to_result(r) for r in rows]
        logger.info(f"local_arxiv: {len(results)} papers for '{query[:60]}'")
        return results

    def _fts_match(self, fts_query: str, num_results: int) -> List[sqlite3.Row]:
        """Run one FTS5 MATCH query. Returns [] on OperationalError so
        the caller can decide whether to fall back to a broader query."""
        try:
            return self._conn.execute(
                """SELECT p.arxiv_id, p.title, p.abstract, p.authors,
                          p.categories, p.year
                   FROM papers p
                   JOIN papers_fts f ON p.rowid = f.rowid
                   WHERE papers_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, num_results),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Local arXiv FTS5 error on '{fts_query[:60]}': {e}")
            return []

    def _sanitize(self, query: str) -> str:
        """Strip anything that isn't a word char or whitespace, then drop
        FTS5 boolean operators (AND/OR/NOT/NEAR) which would otherwise be
        interpreted as query logic instead of search terms."""
        q = _FTS5_NONWORD.sub(" ", query)
        q = _FTS5_OPERATORS.sub(" ", q)
        # Collapse whitespace
        return " ".join(q.split())

    def _to_result(self, row: sqlite3.Row) -> SearchResult:
        arxiv_id = row["arxiv_id"]
        authors_str = row["authors"] or ""
        categories_str = row["categories"] or ""

        from .paper_id import make_paper_id
        pid = make_paper_id(arxiv_id=arxiv_id, source="local_arxiv", title=row["title"])

        return SearchResult(
            title=row["title"],
            url=f"https://arxiv.org/abs/{arxiv_id}",
            snippet=row["abstract"][:200],
            full_text=row["abstract"],
            source="local_arxiv",
            paper_id=pid,
            metadata={
                "authors": [a.strip() for a in authors_str.split(",") if a.strip()][:10],
                "year": row["year"],
                "categories": categories_str.split() if categories_str else [],
                "arxiv_id": arxiv_id,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            },
        )
