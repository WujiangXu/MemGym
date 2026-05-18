"""Fictional-first search backend wrapper (Invariant A).

Wraps any SearchBackend and fictionalizes every paper on first retrieval.
The wrapper maintains a global substitution registry keyed on paper_id
(Enabler D) so the same paper always gets the same fictional text, and
entity mappings are consistent across the entire pipeline run.

Downstream stages (Grow, Craft, Scale) never see real entity names in
search results. Anti-leak preambles on generation prompts prevent LLMs
from reintroducing real entities from pretraining knowledge.

Usage:
    real_backend = get_backend("local_arxiv+semantic_scholar")
    fictional = FictionalizedBackend(real_backend, llm_client)
    results = fictional.search("attention mechanism", num_results=5)
    # results have fictional entity names
"""

import json
import logging
import re
import threading
from typing import Dict, List, Optional, Set, Tuple

from . import SearchBackend, SearchResult
from .paper_id import make_paper_id

logger = logging.getLogger(__name__)

# Common English function words that LLMs sometimes extract as "entities"
# when the extraction prompt pressure is high. These must never enter the
# substitution registry — replacing "is" with "QS" breaks every sentence.
ENGLISH_STOP_WORDS = frozenset({
    "a", "an", "the",                                    # articles
    "is", "are", "was", "were", "be", "been", "am",      # copula
    "do", "does", "did",                                  # aux verbs
    "has", "have", "had",                                 # aux verbs
    "can", "may", "will", "shall", "could", "would",      # modals
    "should", "might", "must",
    "in", "on", "at", "of", "for", "with", "by", "to",   # prepositions
    "from", "into", "onto", "upon", "about", "through",
    "and", "or", "but", "not", "no", "nor", "so", "yet",  # conjunctions
    "if", "as", "than", "that", "this", "then", "when",   # subordinators
    "it", "we", "he", "she", "they", "us", "me", "him",   # pronouns
    "her", "them", "its", "our", "their", "my", "your",
    "who", "what", "how", "which", "where", "why",        # wh-words
    "all", "each", "both", "more", "most", "some", "any", # determiners
    "also", "very", "just", "only", "even", "still",      # adverbs
    "via", "per", "vs",                                    # misc short
})


# Prompt for per-paper entity extraction (simpler than FICTIONALIZE_EXTRACT_PROMPT
# which requires full instance context). Works on title + abstract only.
PAPER_ENTITY_EXTRACT_PROMPT = """\
You are anonymizing a research paper for a benchmark. Extract ALL named entities \
that a pretrained language model could recognize from its training data.

Paper title: {title}
Paper text:
{text}

For each entity, produce a fictional replacement:
- People: fictional but plausible names (e.g., "Vaswani" → "Dr. Kenta Morisaki")
- Models/systems: fictional names (e.g., "BERT" → "STRATUM", "Transformer" → "Nexavar")
- Organizations: fictional names (e.g., "Google" → "Helios Labs")
- Datasets: fictional names (e.g., "ImageNet" → "VistaSet")
- Method names: plausible fictional names (e.g., "attention" → stay generic terms unchanged, \
but "ProbSparse attention" → "StochasticFocus attention")
- Specific numbers: modify slightly, keep plausible (e.g., "94.78%" → "91.32%")
- Years: shift by random offset (e.g., "2017" → "2021")

RULES:
1. Replace EVERY entity that could be recognized from pretraining data
2. Keep generic technical terms (e.g., "neural network", "gradient", "loss function") unchanged
3. Do NOT extract common English words: articles (a, an, the), prepositions (in, on, at, of, \
for, with, by, to, from), auxiliary verbs (is, are, was, were, does, do, did, has, have, had), \
modals (can, may, will, could, would, should), pronouns (it, we, they, he, she), or conjunctions \
(and, or, but, if, as, so). These are NEVER entities.
4. Replacements must be same TYPE as original (person→person, model→model)
5. Numbers stay in same rough magnitude
6. Output 10-30 substitutions (more for entity-rich papers, fewer for abstract ones)

Respond in JSON:
{{"substitutions": [
    {{"original": "<entity>", "replacement": "<fictional>", "type": "<person|model|org|number|year|dataset|method|other>"}},
    ...
]}}
"""

PAPER_ENTITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_entity_extract",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "substitutions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "replacement": {"type": "string"},
                            "type": {"type": "string"},
                        },
                        "required": ["original", "replacement", "type"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["substitutions"],
            "additionalProperties": False,
        },
    },
}


def _compile_subs(subs: List[Dict]) -> Tuple[Optional[re.Pattern], Dict[str, str]]:
    """Compile substitutions into a single alternation regex + lookup dict.

    Returns (compiled_pattern, lookup) where lookup maps lowercased originals
    to their replacements. A single regex pass is O(m) vs O(n*m) for n
    sequential re.sub calls — critical when n > 1000.
    """
    # Filter and sort by length descending (longer phrases take priority)
    valid = [(s["original"], s["replacement"])
             for s in subs
             if s.get("original", "").strip() and len(s["original"]) >= 2
             and s["original"].lower() != s.get("replacement", "").lower()
             and s["original"].lower() not in ENGLISH_STOP_WORDS]
    if not valid:
        return None, {}
    valid.sort(key=lambda x: len(x[0]), reverse=True)
    lookup = {orig.lower(): repl for orig, repl in valid}
    # Build alternation: \b(entity1|entity2|...)s?\b
    escaped = [re.escape(orig) for orig, _ in valid]
    pattern = re.compile(
        r"\b(" + "|".join(escaped) + r")s?\b",
        re.IGNORECASE,
    )
    return pattern, lookup


def _apply_subs(text: str, subs: List[Dict]) -> str:
    """Apply substitutions via a single compiled alternation regex."""
    pattern, lookup = _compile_subs(subs)
    if not pattern:
        return text
    return pattern.sub(lambda m: lookup.get(m.group(1).lower(), m.group(0)), text)


class FictionalizedBackend:
    """Wraps a real SearchBackend, fictionalizing results on first sight.

    Thread-safe: the substitution registry uses a lock for concurrent access
    during parallel searches in Grow and Scale stages.
    """

    def __init__(
        self,
        real_backend: SearchBackend,
        llm_client: "LLMClient",
        *,
        enabled: bool = True,
    ):
        self.real_backend = real_backend
        self.llm = llm_client
        self.enabled = enabled

        # Global substitution registry: original → replacement
        # Accumulated across all papers seen in this pipeline run.
        self._registry: Dict[str, Dict] = {}  # original → {replacement, type}
        # Per-paper cache: paper_id → True (already processed)
        self._seen_papers: Set[str] = set()
        # All real entities ever seen (for post-sanitizer)
        self._real_entities: Set[str] = set()
        self._lock = threading.Lock()

    @property
    def substitutions(self) -> List[Dict]:
        """Return the full substitution list (for post-sanitizer)."""
        with self._lock:
            return [
                {"original": orig, "replacement": info["replacement"], "type": info["type"]}
                for orig, info in self._registry.items()
            ]

    @property
    def real_entities(self) -> Set[str]:
        """All real entity strings ever seen (for leak detection)."""
        with self._lock:
            return set(self._real_entities)

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search via real backend, then fictionalize each result."""
        results = self.real_backend.search(query, num_results=num_results)
        if not self.enabled:
            return results
        return [self._fictionalize_result(r) for r in results]

    def _fictionalize_result(self, result: SearchResult) -> SearchResult:
        """Fictionalize a single SearchResult, using cache if available."""
        pid = result.paper_id
        if not pid:
            # No paper_id — generate from title as fallback
            pid = make_paper_id(title=result.title, url=result.url, source=result.source)

        with self._lock:
            already_seen = pid in self._seen_papers

        if already_seen:
            # Already extracted entities for this paper — just apply registry
            return self._apply_registry(result)

        # First time seeing this paper — extract entities
        new_subs = self._extract_entities(result)

        # Merge into global registry (existing mappings take priority for consistency)
        with self._lock:
            for sub in new_subs:
                orig = sub["original"]
                self._real_entities.add(orig)
                if orig not in self._registry:
                    self._registry[orig] = {
                        "replacement": sub["replacement"],
                        "type": sub["type"],
                    }
            self._seen_papers.add(pid)

        return self._apply_registry(result)

    def _extract_entities(self, result: SearchResult) -> List[Dict]:
        """Call LLM to extract entities from a paper's text."""
        # Use first 2000 chars of text to keep the extraction focused
        text = result.text[:2000] if result.text else result.snippet[:500]
        if not text.strip():
            return []

        prompt = PAPER_ENTITY_EXTRACT_PROMPT.format(
            title=result.title,
            text=text,
        )
        try:
            data = self.llm.get_json_completion(
                prompt=prompt,
                response_format=PAPER_ENTITY_SCHEMA,
                temperature=0.5,
            )
            subs = data.get("substitutions", [])
            # Filter out empty/trivial substitutions
            subs = [
                s for s in subs
                if s.get("original", "").strip()
                and s.get("replacement", "").strip()
                and len(s["original"]) >= 2
                and s["original"].lower() != s["replacement"].lower()
                and s["original"].lower() not in ENGLISH_STOP_WORDS
            ]
            return subs
        except Exception as e:
            logger.warning(f"Entity extraction failed for '{result.title[:50]}': {e}")
            return []

    def _apply_registry(self, result: SearchResult) -> SearchResult:
        """Apply the global substitution registry to a SearchResult."""
        with self._lock:
            subs = [
                {"original": orig, "replacement": info["replacement"], "type": info["type"]}
                for orig, info in self._registry.items()
            ]

        if not subs:
            return result

        # Compile once, apply to all fields
        pattern, lookup = _compile_subs(subs)
        if not pattern:
            return result

        def _sub(text: str) -> str:
            if not text:
                return text
            return pattern.sub(
                lambda m: lookup.get(m.group(1).lower(), m.group(0)), text
            )

        return SearchResult(
            title=_sub(result.title),
            url=result.url,  # URL stays real (not visible to downstream)
            snippet=_sub(result.snippet),
            full_text=_sub(result.full_text),
            source=result.source,
            metadata=result.metadata,  # metadata stays real (paper_id, etc.)
            paper_id=result.paper_id,
        )
