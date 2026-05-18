"""PubMed / NCBI E-utilities search backend (free, biomedical focus).

37M+ citations with structured MeSH-based search. Abstracts always available.
Rate limit: 3 req/sec (no key), 10 req/sec (free key).

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

import logging
import os
import time
from typing import List
from xml.etree import ElementTree

import requests

from . import SearchBackend, SearchResult

logger = logging.getLogger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_RATE_DELAY = 0.35  # ~3 req/sec without key


class PubMedBackend(SearchBackend):
    """Search biomedical papers via PubMed E-utilities."""

    def __init__(self, api_key: str = "", email: str = ""):
        self._api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        self._email = email
        self._last_request_time = 0.0

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        # Step 1: Search for PMIDs
        pmids = self._esearch(query, num_results)
        if not pmids:
            return []

        # Step 2: Fetch article details
        return self._efetch(pmids)

    def _esearch(self, query: str, num_results: int) -> List[str]:
        """Search PubMed and return PMIDs."""
        self._rate_limit()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": num_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email

        try:
            resp = requests.get(_ESEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.warning(f"PubMed esearch failed for '{query}': {e}")
            return []

    def _efetch(self, pmids: List[str]) -> List[SearchResult]:
        """Fetch article details for given PMIDs."""
        self._rate_limit()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            resp = requests.get(_EFETCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.warning(f"PubMed efetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> List[SearchResult]:
        """Parse PubMed XML response into SearchResults."""
        results = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as e:
            logger.warning(f"PubMed XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                # Title
                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None and title_el.text else ""

                # Abstract
                abstract_parts = []
                for abs_text in article.findall(".//AbstractText"):
                    label = abs_text.get("Label", "")
                    text = abs_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                # Authors
                authors = []
                for author in article.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    first = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{first} {last}".strip())

                # PMID
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                # Year
                year_el = article.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None

                # MeSH terms
                mesh_terms = []
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    if mesh.text:
                        mesh_terms.append(mesh.text)

                from .paper_id import make_paper_id
                pid = make_paper_id(backend_id=pmid, source="pubmed", title=title)

                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=abstract[:200] if abstract else "",
                    full_text=abstract,
                    source="pubmed",
                    paper_id=pid,
                    metadata={
                        "authors": authors[:10],
                        "year": year,
                        "pmid": pmid,
                        "mesh_terms": mesh_terms[:10],
                    },
                ))
            except Exception:
                continue

        logger.info(f"PubMed: {len(results)} articles fetched")
        return results

    def _rate_limit(self):
        delay = 0.11 if self._api_key else _RATE_DELAY
        elapsed = time.time() - self._last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()
