"""Shared prompts for LightMem adapters (coding-synth + IR).

Single source of truth for the domain-neutral replacement of
LightMem upstream's ``METADATA_GENERATE_PROMPT``
(``third_party/LightMem/src/lightmem/memory/prompts.py:1-52``), which
is hard-coded as a *Personal Information Extractor* operating on
User-tagged chat dialogues with Alice/Paris-style examples. Feeding
that template source code or generic-document content silently
destroys signal — the LLM has been primed to look for biographical
facts about a "user" — and was the root cause of LightMem scoring
below ``truncated``/``prompt`` on coding-synth the training phase and on the
IR multi-hop slice.

Both adapters pass this prompt via the official per-call
``METADATA_GENERATE_PROMPT`` argument of ``LightMemory.add_memory``
(``lightmem.py:204-211``); the output schema (``{source_id, fact}``)
is preserved unchanged so the downstream Qdrant indexer is byte-
compatible. The structure mirrors A-MEM's ``METADATA_PROMPT`` (see
``src/memgym/memory/amem/note.py:48-75``): no domain-shaped framing,
no domain-shaped examples, no fields outside the original schema.

Lives outside both ``pipelines.coding_synthetic`` and ``memory.ir`` so
neither side has to depend on the other (avoids a layering inversion).
"""

NEUTRAL_METADATA_PROMPT = """
You are a fact extractor. Your job is to extract self-contained,
unambiguous factual statements from the input content. The content may
be of any type — documents, source code, articles, dialogues, technical
text, etc. Adapt to the actual format; do NOT assume the input is a
conversation, do NOT assume there is a "user".

Important Instructions:
0. You MUST process content units **strictly in ascending sequence_number
   order** (lowest -> highest). Treat units one-by-one; do NOT reorder,
   batch-skip, or skip ahead.
1. For each unit, decide whether it contains any factual information.
   - If yes -> extract it and rephrase as a complete, self-contained
     sentence.
   - If no (pure boilerplate, empty, or completely meaningless) -> skip it.
   - Do NOT skip just because the information looks minor or trivial;
     small details (e.g. "Function `foo` calls `bar`.") must be kept.
     Only skip if the unit is *completely* meaningless.
2. Each fact must stand on its own — ABSOLUTELY no pronouns (he, she, it,
   they, this, that) and no relative time (yesterday, today, last week);
   resolve all referents inline.
3. Use the "sequence_number" (the integer prefix before each unit) as
   the ``source_id``.
4. Output format (this schema is fixed — do NOT add or rename fields):
   ```
   {
     "data": [
       {"source_id": <source_id>, "fact": "<complete fact with all specifics>"}
     ]
   }
   ```

Reminder: Be exhaustive. Unless a unit is purely meaningless, extract it
as a fact.
"""
