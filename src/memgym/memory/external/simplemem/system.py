"""Shared SimpleMem core (`pip install simplemem`, aiming-lab).

SimpleMem expects a dialogue stream `(speaker, text, timestamp)` rather than
arbitrary documents. We adapt by stuffing each "doc" in as one
`add_dialogue("system", text, ts)` with a synthetic monotonic timestamp
(same trick the LightMEM adapter uses at `lightmem_method.py:161-165`).

Retrieval bypasses SimpleMem's `ask()` (which calls its own answerer LLM)
and goes through the underlying `HybridRetriever.retrieve(query)` to return
the raw retrieved memory entries — our pipeline owns the answerer.

**Domain-neutral extraction prompt patch (Apr-2026).**
Upstream SimpleMem (`simplemem.core.memory_builder.MemoryBuilder`) hard-
codes a chat-only extractor: the system role says
*"... extracting structured information from conversations ..."* and the
user prompt example asks the LLM to populate ``persons``, ``location``
and ``timestamp`` from an Alice/Bob/Starbucks meeting transcript. Feeding
this Python source code (or any non-conversational content) yields empty
or hallucinated ``MemoryEntry`` records, which is why SimpleMem scored
*below* even ``truncated`` on the coding_synthetic Phase-5 sweep.

We monkey-patch ``MemoryBuilder._build_extraction_prompt`` with a
domain-neutral version that:
  1. Explicitly overrides the system-message bias by stating in the user
     prompt that the content may be of ANY type (documents, code,
     dialogues, articles).
  2. Preserves the entire 7-field ``MemoryEntry`` schema
     (``lossless_restatement / keywords / topic / persons / location /
     timestamp / entities``) so the downstream three-layer hybrid
     retriever stays byte-compatible.
  3. Tells the LLM to leave LOCOMO-shaped fields (``persons``,
     ``location``, ``timestamp``) empty/null when the content does not
     mention them, instead of fabricating values. The retriever's
     symbolic layer (``hybrid_retriever.py:343-345``) short-circuits
     to an empty result when none of those fields are populated by the
     query analyser, so empty values cleanly degrade SimpleMem to a
     semantic + lexical hybrid retriever on code-domain content (which
     is the sensible behaviour, given code lacks persons/locations).
  4. Keeps ``entities`` open-ended so code identifiers / file paths /
     symbol names land in the symbolic layer naturally.

The patch is installed once, lazily, the first time
``SimpleMemBackend._create_system`` runs in a given process; subsequent
calls are no-ops. Because this lives in the shared core, the IR-side
``ir_simplemem`` adapter inherits the fix automatically.
"""
from __future__ import annotations

import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from typing import List, Optional


_NEUTRAL_EXTRACTION_PATCH_LOCK = threading.Lock()
_NEUTRAL_EXTRACTION_PATCH_INSTALLED = False


def _neutral_build_extraction_prompt(
    self, dialogue_text: str, dialogue_ids, context: str
) -> str:
    """Drop-in replacement for ``MemoryBuilder._build_extraction_prompt``.

    Intentionally keeps the same signature so the patch is transparent to
    ``MemoryBuilder._generate_memory_entries``. The leading override block
    is required because the system role string in the upstream method is
    hard-coded inside the same function (``memory_builder.py:200-209`` and
    ``:425-434``) and cannot be replaced without rewriting both methods;
    a strong user-prompt override is the minimum-surface fix.
    """
    return f"""IMPORTANT OVERRIDE — read first:
The system role above mentions "conversations" for legacy reasons. The
content below MAY OR MAY NOT be a conversation. Treat it according to its
actual format — documents, source code, articles, technical text, or
dialogues. Do NOT assume there are speakers or that the content is a
transcript.

Your task is to extract all valuable information from the following
content and convert it into structured memory entries.

{context}

[Current Window Content]
{dialogue_text}

[Requirements]
1. Complete Coverage: generate enough memory entries to capture ALL
   meaningful information in the content.
2. Force Disambiguation: ABSOLUTELY no pronouns (he, she, it, they, this,
   that) and no relative time (yesterday, today, last week, tomorrow).
3. Lossless Information: each entry's `lossless_restatement` must be a
   complete, independent, understandable sentence.
4. Domain-aware structured fields — fill ONLY when they actually apply
   to the content; do NOT fabricate values:
   - keywords: salient nouns / verbs / identifiers / topic words.
   - timestamp: explicit ISO 8601 time IN THE CONTENT, else null.
   - location: physical/geographical place ACTUALLY mentioned, else null.
   - persons: human names ACTUALLY mentioned, else empty list [].
   - entities: named entities (companies, products, organisations,
     identifiers, file paths, function/class/module names, symbols).
     This field is open-ended; use it for code identifiers when present.
   - topic: short topic phrase summarising this entry.
   For non-conversational content (e.g. source code), it is CORRECT and
   EXPECTED for `persons`, `location`, and `timestamp` to be empty/null.

[Output Format]
Return a JSON array; each element is one memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc., when present)",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", ...],
    "entities": ["entity1", ...],
    "topic": "topic phrase"
  }},
  ...
]
```

Now process the above content. Return ONLY the JSON array, no other
explanations.
"""


def _install_neutral_extraction_prompt() -> None:
    """Install the neutral extraction-prompt patch once per process.

    Idempotent and thread-safe; subsequent calls are no-ops. We import
    ``MemoryBuilder`` lazily so that ``import memgym.memory.external.simplemem``
    does not require the optional ``simplemem`` dependency at import time.
    """
    global _NEUTRAL_EXTRACTION_PATCH_INSTALLED
    if _NEUTRAL_EXTRACTION_PATCH_INSTALLED:
        return
    with _NEUTRAL_EXTRACTION_PATCH_LOCK:
        if _NEUTRAL_EXTRACTION_PATCH_INSTALLED:
            return
        from simplemem.core.memory_builder import MemoryBuilder

        MemoryBuilder._build_extraction_prompt = _neutral_build_extraction_prompt
        _NEUTRAL_EXTRACTION_PATCH_INSTALLED = True


class SimpleMemBackend:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        retrieve_k: int = 5,
        storage_dir: Optional[str] = None,
        enable_planning: bool = True,
        enable_reflection: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ) -> None:
        # SimpleMem also instantiates `openai.OpenAI` directly
        # (simplemem/utils/llm_client.py:8); strip the litellm-style
        # "openai/<model>" prefix so the OpenAI SDK gets a real model name.
        # Mirrors the fix in HippoRAGSystem.__init__.
        if isinstance(llm_model, str) and llm_model.startswith("openai/"):
            llm_model = llm_model[len("openai/"):]
        self._llm_model = llm_model
        self._api_base = api_base
        self._api_key = api_key
        self._retrieve_k = retrieve_k
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="simplemem_")
        self._enable_planning = enable_planning
        self._enable_reflection = enable_reflection
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._dialogue_id = 0
        self._base_time = datetime(2024, 1, 1, 9, 0)
        self._committed = False
        self._system = None

    def _make_timestamp(self) -> str:
        ts = self._base_time + timedelta(minutes=self._dialogue_id)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    def _create_system(self):
        # SimpleMem's embedder is configured via the global SimpleMemConfig
        # singleton (no ctor arg). Override it before instantiation. The
        # default Qwen3-Embedding-0.6B requires transformers >= 4.50; in
        # mixed envs (hipporag pins transformers 4.45) we fall back to the
        # ubiquitous all-MiniLM-L6-v2 (384-dim).
        #
        # Also install the domain-neutral extraction-prompt patch (see
        # module docstring) before any extraction can run. Idempotent.
        _install_neutral_extraction_prompt()
        from simplemem import SimpleMemSystem
        from simplemem.config import get_config, set_config

        cfg = get_config()
        cfg.embedding_model = self._embedding_model
        cfg.embedding_dimension = self._embedding_dimension
        set_config(cfg)

        db_path = os.path.join(
            self._storage_dir, f"simplemem_{uuid.uuid4().hex[:8]}.db"
        )
        return SimpleMemSystem(
            api_key=self._api_key,
            model=self._llm_model,
            base_url=self._api_base,
            db_path=db_path,
            clear_db=True,
            enable_planning=self._enable_planning,
            enable_reflection=self._enable_reflection,
            # Disable streaming — our pipeline expects sync returns.
            use_streaming=False,
            # We never call .ask(), so the answerer's parallel knobs don't
            # matter; leave at defaults.
        )

    def add_doc(self, doc_id: str, text: str) -> None:
        if not text or not text.strip():
            return
        if self._system is None:
            self._system = self._create_system()
        self._system.add_dialogue(
            speaker="system",
            content=f"[{doc_id}]\n{text}",
            timestamp=self._make_timestamp(),
        )
        self._dialogue_id += 1
        self._committed = False

    def commit(self) -> None:
        """Flush buffered dialogues into structured memory units."""
        if self._committed or self._system is None:
            return
        self._system.finalize()
        self._committed = True

    def retrieve(self, question: str, k: Optional[int] = None) -> List[str]:
        if self._system is None:
            return []
        self.commit()
        retriever = getattr(self._system, "hybrid_retriever", None)
        if retriever is None:
            return []
        entries = retriever.retrieve(question)
        top_k = k or self._retrieve_k
        return [
            getattr(e, "content", str(e))
            for e in (entries or [])[:top_k]
        ]

    def reset(self) -> None:
        self._dialogue_id = 0
        self._committed = False
        self._system = None
