"""Pluggable search backends for the MemGym-IR research pipeline.

Each backend implements the SearchBackend protocol — just a search() method
returning List[SearchResult]. Add new backends by creating a new file in this
directory and registering it in BACKENDS below.

Usage:
    from memgym.pipelines.memgym_ir.backends import get_backend

    backend = get_backend("semantic_scholar+wikipedia")
    results = backend.search("transformer attention mechanism", num_results=5)
    for r in results:
        print(r.title, len(r.full_text), r.source)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from any backend."""

    title: str
    url: str
    snippet: str = ""       # Short text from search engine (always available)
    full_text: str = ""     # Full article/paper text (may be empty if scraping failed)
    source: str = ""        # Backend name: "ddg", "wikipedia", "semantic_scholar", "arxiv"
    metadata: Dict = field(default_factory=dict)  # Source-specific (authors, year, etc.)
    paper_id: str = ""      # Canonical identity for cross-backend dedup (Enabler D)

    @property
    def text(self) -> str:
        """Best available text content (full_text if available, else snippet)."""
        return self.full_text if self.full_text else self.snippet


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for search backends. Implement search() to create a new backend."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]: ...


# Backend registry — maps short names to lazy-loaded classes
def get_backend(spec: str) -> "SearchBackend":
    """Create a search backend from a spec string.

    Spec can be a single backend name or a '+'-separated combination:
        "semantic_scholar"              → SemanticScholarBackend
        "ddg+wikipedia"                 → CombinedBackend([DDG, Wikipedia])
        "semantic_scholar+arxiv+wikipedia" → CombinedBackend([SS, arXiv, Wiki])
    """
    parts = [p.strip() for p in spec.split("+")]

    if len(parts) == 1:
        return _load_backend(parts[0])

    from .combined import CombinedBackend
    backends = [_load_backend(p) for p in parts]
    return CombinedBackend(backends)


# Default search order: local (no limits) → API no-rate-limit → rate-limited → general
SUGGESTED_ORDER = "local_arxiv+arxiv+semantic_scholar+openalex+pubmed+core+wikipedia+ddg"


def _load_backend(name: str) -> "SearchBackend":
    """Lazy-load a single backend by name."""
    if name == "local_arxiv":
        from .local_arxiv import LocalArxivBackend
        return LocalArxivBackend()
    elif name == "ddg":
        from .ddg import DDGBackend
        return DDGBackend()
    elif name == "wikipedia":
        from .wikipedia import WikipediaBackend
        return WikipediaBackend()
    elif name == "semantic_scholar":
        from .semantic_scholar import SemanticScholarBackend
        return SemanticScholarBackend()
    elif name == "arxiv":
        from .arxiv_backend import ArxivBackend
        return ArxivBackend()
    elif name == "openalex":
        from .openalex import OpenAlexBackend
        return OpenAlexBackend()
    elif name == "core":
        from .core_api import COREBackend
        return COREBackend()
    elif name == "pubmed":
        from .pubmed import PubMedBackend
        return PubMedBackend()
    else:
        raise ValueError(
            f"Unknown backend: {name}. "
            f"Available: local_arxiv, semantic_scholar, openalex, arxiv, pubmed, core, wikipedia, ddg"
        )
