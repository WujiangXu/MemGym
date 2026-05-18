"""Prompt templates for MemGym-IR v2 pipeline."""

# Stage 1: Pre-screen -- parametric test
PARAMETRIC_TEST_PROMPT = """\
Answer the following question using ONLY your general knowledge. Do NOT make up information.
If you are unsure, say "I don't know".

Question: {question}

Provide your answer in JSON format:
{{"answer": "<your answer>", "confidence": <0.0-1.0>}}
"""

# Stage 1: Pre-screen -- partial context test
PARTIAL_CONTEXT_TEST_PROMPT = """\
Answer the following question using ONLY the provided context.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"answer": "<your answer>", "confidence": <0.0-1.0>}}
"""

# Stage 1/4: Answer judge
ANSWER_JUDGE_PROMPT = """\
You are an answer judge. Compare the predicted answer to the gold answer.
Score 1.0 if the predicted answer is correct (semantically equivalent), 0.0 if wrong.
Partial credit (0.3-0.7) for partially correct answers.

Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
Question: {question}

Respond in JSON format:
{{"score": <0.0-1.0>, "explanation": "<brief explanation>"}}
"""

# Stage 2: Fact extraction from supporting paragraphs
FACT_EXTRACTION_PROMPT = """\
You are extracting grounding facts from a multi-hop QA decomposition.

Question: {question}
Answer: {answer}

Decomposition (ordered sub-questions and their answers):
{decomposition}

For each sub-question, extract the specific factual claim from its supporting paragraph
that is needed to answer it. Classify each fact:

- bridge_fact: An intermediate answer that feeds into the next hop
- terminal_fact: Directly contributes to the final answer
- temporal_fact: A date/time-specific fact
- numerical_fact: A number that could be confused

For each fact, specify:
- needed_at_hop: Which hop number needs this fact (0-based)
- first_available_at_hop: Which hop's search results first surface this paragraph (0-based)

Respond in JSON:
{{
  "facts": [
    {{
      "fact_id": "F1",
      "content": "<factual statement>",
      "fact_type": "bridge_fact|terminal_fact|temporal_fact|numerical_fact",
      "source_paragraph_title": "<paragraph title>",
      "needed_at_hop": 0,
      "first_available_at_hop": 0,
      "intermediate_answer": "<intermediate answer or empty>"
    }}
  ]
}}
"""

# Stage 2: Fact verification by verifier
FACT_VERIFICATION_PROMPT = """\
Verify whether each extracted fact is actually supported by its source paragraph,
and whether the fact could be inferred WITHOUT the paragraph (discoverable).

Facts to verify:
{facts}

Source paragraphs:
{paragraphs}

For each fact, respond:
{{
  "verifications": [
    {{
      "fact_id": "F1",
      "supported_by_paragraph": true,
      "is_discoverable": false,
      "explanation": "<brief explanation>"
    }}
  ]
}}
"""

# Stage 3: Adversarial distractor generation
ADVERSARIAL_DISTRACTOR_PROMPT = """\
Generate a plausible but INCORRECT passage about the same entity.
The passage should mention the same entity names but contain wrong facts
that could mislead a reader.

Entity: {entity}
Correct fact: {correct_fact}
Target confusion: The reader should believe a wrong relationship/number/date.

Write a single paragraph (100-200 words) that sounds authoritative but contains
the wrong information. Do NOT include any disclaimers.

Respond in JSON:
{{"distractor_text": "<the misleading paragraph>", "wrong_claim": "<what's wrong>"}}
"""

# Stage 3: Same-entity contradiction generation
SAME_ENTITY_CONTRADICTION_PROMPT = """\
Generate a passage about {entity} that contradicts the following fact:
"{fact}"

The contradiction should:
1. Be about the SAME entity and time period
2. Sound equally authoritative and well-sourced
3. Contain a specific wrong number/date/relationship
4. Be 100-200 words

Respond in JSON:
{{"contradiction_text": "<the contradicting paragraph>", "contradicted_claim": "<what it contradicts>"}}
"""

# Stage 5: Temporal shift distractor generation
TEMPORAL_SHIFT_PROMPT = """\
Generate a passage about {entity} that contains DIFFERENT dates, numbers, or
time periods than the following fact:
"{fact}"

The passage should:
1. Be about the SAME entity
2. Sound equally authoritative and well-sourced
3. Use different but plausible dates/numbers
4. Be 80-150 words

Respond in JSON:
{{"distractor_text": "<the time-shifted paragraph>", "wrong_claim": "<what's different>"}}
"""

# Stage 5: Bridge/terminal fact shift distractor generation
BRIDGE_SHIFT_PROMPT = """\
Generate a passage about {entity} that contains DIFFERENT facts, relationships,
or details than the following {fact_type}:
"{fact}"

The passage should:
1. Be about the SAME entity
2. Sound equally authoritative and well-sourced
3. Change the key relationship, entity connection, or conclusion
4. Be 80-150 words
5. For bridge facts: change the intermediate entity (e.g., "born in X" → "born in Y")
6. For terminal facts: change the final detail (e.g., "population 50K" → "population 120K")

Respond in JSON:
{{"distractor_text": "<the shifted paragraph>", "wrong_claim": "<what's different>"}}
"""

# Stage 5: Filler noise generation (topic-related but irrelevant)
FILLER_NOISE_PROMPT = """\
Generate a short paragraph (80-120 words) that is loosely related to the topic
of "{topic}" but contains NO information useful for answering the following question:

Question: {question}

Requirements:
1. Write about a tangentially related subject (same domain, different specifics)
2. Include realistic-sounding facts, dates, or statistics
3. Sound like an excerpt from an encyclopedia or news article
4. Do NOT mention any of these entities: {excluded_entities}
5. Do NOT contain information that helps answer the question

Respond in JSON:
{{"filler_text": "<the filler paragraph>", "topic": "<what it's about>"}}
"""

FILLER_NOISE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "filler_noise",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "filler_text": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["filler_text", "topic"],
            "additionalProperties": False,
        },
    },
}

# Stage 4: Ablation QA prompt
ABLATION_QA_PROMPT = """\
Answer the following question using ONLY the provided documents.
If the documents don't contain enough information, say "insufficient information".

Documents:
{documents}

Question: {question}

Provide your answer in JSON:
{{"answer": "<your answer>", "confidence": <0.0-1.0>}}
"""

# Stage 5: Query fuzz prompt
QUERY_FUZZ_PROMPT = """\
Rephrase the following question to make it vaguer and harder.
Remove specific entity names where possible, use pronouns or descriptions instead.
The rephrased question must still be answerable with the same answer.

Original question: {question}
Answer (for reference, do NOT include in output): {answer}

Intensity: {intensity}
- light: Replace 1 entity with a description
- heavy: Replace all specific names with descriptions/pronouns

Respond in JSON:
{{"fuzzed_question": "<rephrased question>", "changes": ["<change1>", "<change2>"]}}
"""

# JSON response format schemas
ANSWER_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["score", "explanation"],
            "additionalProperties": False,
        },
    },
}

FACT_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "content": {"type": "string"},
                            "fact_type": {
                                "type": "string",
                                "enum": [
                                    "bridge_fact",
                                    "terminal_fact",
                                    "temporal_fact",
                                    "numerical_fact",
                                ],
                            },
                            "source_paragraph_title": {"type": "string"},
                            "needed_at_hop": {"type": "integer"},
                            "first_available_at_hop": {"type": "integer"},
                            "intermediate_answer": {"type": "string"},
                        },
                        "required": [
                            "fact_id",
                            "content",
                            "fact_type",
                            "source_paragraph_title",
                            "needed_at_hop",
                            "first_available_at_hop",
                            "intermediate_answer",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}

DISTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "distractor",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distractor_text": {"type": "string"},
                "wrong_claim": {"type": "string"},
            },
            "required": ["distractor_text", "wrong_claim"],
            "additionalProperties": False,
        },
    },
}

# HotpotQA decomposition generation
HOTPOTQA_DECOMPOSITION_PROMPT = """\
Given a multi-hop question and its supporting paragraphs, decompose the
question into sub-questions, each answerable by one paragraph.

Question: {question}
Answer: {answer}

{paragraphs}

For each supporting paragraph, create a sub-question that is answered by that
paragraph. The sub-questions should chain together to answer the main question.
Use #N to reference the answer of sub-question N in later sub-questions.

Return JSON:
{{
  "sub_questions": [
    {{"sub_question": "<sub-question text>", "sub_answer": "<answer from this paragraph>", "paragraph_title": "<title of the paragraph>"}},
    ...
  ]
}}
"""

HOTPOTQA_DECOMPOSITION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hotpotqa_decomposition",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sub_question": {"type": "string"},
                            "sub_answer": {"type": "string"},
                            "paragraph_title": {"type": "string"},
                        },
                        "required": ["sub_question", "sub_answer", "paragraph_title"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["sub_questions"],
            "additionalProperties": False,
        },
    },
}

# Synthetic generation: Step 1 -- entities + paragraphs
SYNTHETIC_ENTITIES_PROMPT = """\
Generate {num_entities} related entities with factual paragraphs suitable for
a multi-hop question in the domain of {topic}.

Requirements:
- Entities should be connected (e.g., person -> birthplace -> country)
- Each paragraph should be 80-150 words of factual content
- Facts should be specific: include dates, numbers, names, or relationships
- Entities must chain together so information from one leads to the next

Return JSON:
{{
  "entities": [
    {{"title": "<entity name>", "text": "<factual paragraph>", "key_fact": "<the critical linking fact>"}},
    ...
  ]
}}
"""

SYNTHETIC_ENTITIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthetic_entities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                            "key_fact": {"type": "string"},
                        },
                        "required": ["title", "text", "key_fact"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}

# Synthetic generation: Step 2 -- question + decomposition
SYNTHETIC_QUESTION_PROMPT = """\
Given these related entities and their paragraphs, create a {num_hops}-hop
question that requires information from ALL paragraphs to answer.

The question should:
1. Require chaining facts across entities (bridge reasoning)
2. Not be answerable from any single paragraph alone
3. Have a specific, unambiguous answer

Entities:
{entities}

Return JSON:
{{
  "question": "<the multi-hop question>",
  "answer": "<the final answer>",
  "decomposition": [
    {{"sub_question": "<sub-question for hop 1>", "sub_answer": "<intermediate answer>", "paragraph_title": "<entity title used>"}},
    ...
  ]
}}
"""

SYNTHETIC_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthetic_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "decomposition": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sub_question": {"type": "string"},
                            "sub_answer": {"type": "string"},
                            "paragraph_title": {"type": "string"},
                        },
                        "required": ["sub_question", "sub_answer", "paragraph_title"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question", "answer", "decomposition"],
            "additionalProperties": False,
        },
    },
}

# Synthetic generation: Step 3 -- distractor paragraphs
SYNTHETIC_DISTRACTORS_PROMPT = """\
Generate {count} distractor paragraphs about entities related to but different
from the following entities: {entity_titles}

Each distractor should:
1. Be about a DIFFERENT entity in a similar domain
2. Sound factual and authoritative (80-150 words)
3. Contain facts that could plausibly confuse a reader

Return JSON:
{{
  "distractors": [
    {{"title": "<entity name>", "text": "<factual paragraph>"}},
    ...
  ]
}}
"""

SYNTHETIC_DISTRACTORS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthetic_distractors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "distractors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["title", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["distractors"],
            "additionalProperties": False,
        },
    },
}

# Stage 3: Red herring sub-query generation
RED_HERRING_QUERY_PROMPT = """\
Generate a plausible search query that is topically related to the following \
question but does NOT help answer it. The query should look like a reasonable \
intermediate research step but actually leads to irrelevant information.

Main question: {question}
Topic area: {topic}

The red herring query should:
1. Be about a related but ultimately irrelevant entity or relationship
2. Sound like a genuine sub-question someone might ask while researching the main question
3. NOT contain the answer or any key bridge entities

Respond in JSON:
{{"query": "<the red herring search query>", "reasoning": "<why this is a plausible distraction>"}}
"""

RED_HERRING_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "red_herring_query",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["query", "reasoning"],
            "additionalProperties": False,
        },
    },
}

# Stage 3: Paragraph expansion -- bury key facts in realistic article context
PARAGRAPH_EXPANSION_PROMPT = """\
IMPORTANT: This is a fictional research universe. Do NOT reference any \
real-world researcher, institution, model, dataset, or event. Only use \
entity names that appear in the input text below.

You are expanding a short supporting paragraph into a realistic search result \
document. The key fact MUST be preserved verbatim, but it should be buried \
within a larger, realistic article-style document.

Original paragraph title: {title}
Original paragraph text: {text}
Key fact to preserve: {key_fact}

Requirements:
1. Write a {target_tokens}-word article-style document about the topic
2. Include the EXACT key fact somewhere in the middle (not at the start or end)
3. Add realistic surrounding context: background info, related but non-critical \
details, historical context, statistics, quotes
4. The document should read like a Wikipedia article or encyclopedia entry
5. Do NOT add false information -- the surrounding context should be plausible \
but non-essential for answering the question
6. The key fact should NOT be highlighted, bolded, or otherwise made to stand out

Respond in JSON:
{{"expanded_text": "<the full expanded document>", "key_fact_location": "early|middle|late"}}
"""

PARAGRAPH_EXPANSION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "paragraph_expansion",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "expanded_text": {"type": "string"},
                "key_fact_location": {"type": "string"},
            },
            "required": ["expanded_text", "key_fact_location"],
            "additionalProperties": False,
        },
    },
}

# Stage 3: Overlapping document paraphrase
OVERLAP_PARAPHRASE_PROMPT = """\
Paraphrase the following passage so it conveys the SAME key information but \
using completely different wording, sentence structure, and organization. \
The paraphrase should be similar in length to the original.

Original passage:
{text}

Requirements:
1. Preserve all factual claims but rephrase them
2. Use different sentence structure and vocabulary
3. Reorder information where possible
4. The reader should recognize it covers the same topic but not realize it's \
a direct paraphrase at first glance
5. Keep the same approximate length ({target_length} words)

Respond in JSON:
{{"paraphrased_text": "<the paraphrased passage>", "title": "{title} (alternative source)"}}
"""

OVERLAP_PARAPHRASE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "overlap_paraphrase",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "paraphrased_text": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["paraphrased_text", "title"],
            "additionalProperties": False,
        },
    },
}

# Stage 3: Dead-end document generation (topically related but useless)
DEAD_END_DOCUMENT_PROMPT = """\
Generate a document that is topically related to the following question but \
contains NO information useful for answering it. The document should look like \
a legitimate search result that an agent would retrieve and read, wasting time.

Question: {question}
Topic entities: {entities}

Requirements:
1. Write about the SAME entities or closely related ones
2. Include specific facts, dates, and details (make them plausible)
3. The content must be factual-sounding but completely irrelevant to answering \
the question
4. Length: approximately {target_tokens} words
5. Format like a search result snippet or encyclopedia excerpt
6. Do NOT include any information that helps answer the question

Respond in JSON:
{{"title": "<document title>", "text": "<the dead-end document>"}}
"""

DEAD_END_DOCUMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "dead_end_document",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["title", "text"],
            "additionalProperties": False,
        },
    },
}

# Stage 3: Conflicting authority source generation
CONFLICTING_SOURCES_PROMPT = """\
Generate a pair of authoritative-sounding documents that present DIFFERENT \
wrong answers to the following question. These documents should create confusion \
by contradicting each other AND the correct answer.

Question: {question}
Correct answer: {correct_answer}
Key supporting fact: {supporting_fact}

Requirements:
1. Document A should assert wrong answer X with confidence and citations
2. Document B should assert DIFFERENT wrong answer Y
3. Both should sound equally authoritative (like expert sources)
4. Neither should contain the correct answer ({correct_answer})
5. Each document should be {target_tokens} words
6. Include plausible-sounding sources, dates, or expert opinions
7. The wrong answers should be plausible alternatives

Respond in JSON:
{{
  "doc_a": {{"title": "<title>", "text": "<authoritative doc with wrong answer X>", "wrong_answer": "<X>"}},
  "doc_b": {{"title": "<title>", "text": "<authoritative doc with wrong answer Y>", "wrong_answer": "<Y>"}}
}}
"""

CONFLICTING_SOURCES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "conflicting_sources",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "doc_a": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                        "wrong_answer": {"type": "string"},
                    },
                    "required": ["title", "text", "wrong_answer"],
                    "additionalProperties": False,
                },
                "doc_b": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                        "wrong_answer": {"type": "string"},
                    },
                    "required": ["title", "text", "wrong_answer"],
                    "additionalProperties": False,
                },
            },
            "required": ["doc_a", "doc_b"],
            "additionalProperties": False,
        },
    },
}

# =========================================================================
# Memory Eviction Evaluation Prompts
# =========================================================================

NOTE_TAKING_PROMPT = """\
You are an AI research assistant conducting a multi-step search. You are reading \
search results one batch at a time. After each batch, earlier results will be \
REMOVED from your context — you will NOT be able to see them again.

Your task: Write concise notes capturing the KEY FACTS from the current search \
results that might be needed later to answer a research question.

Question you are trying to answer: {question}

Current search results (Turn {turn_index}):
{documents}

{previous_notes_section}

Write your updated notes below. You have a budget of approximately {budget} tokens.
Focus on:
- Specific names, dates, numbers, and relationships
- Facts that connect entities across different documents
- Information that directly helps answer the question
- Bridge facts that link one entity to another

Do NOT include:
- General background information
- Obvious or well-known facts
- Redundant information already in your notes

Respond in JSON:
{{"notes": "<your concise notes>"}}
"""

NOTE_TAKING_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "note_taking",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
            },
            "required": ["notes"],
            "additionalProperties": False,
        },
    },
}

EVICTED_ANSWER_PROMPT = """\
You are an AI research assistant. You have been conducting a multi-step search \
to answer a question. Earlier search results have been removed from your context, \
but you took notes along the way.

Question: {question}

Your accumulated notes from previous turns:
{notes}

{visible_documents_section}

Using your notes and any visible documents above, answer the question.
If your notes don't contain enough information, say "insufficient information".

Provide your answer in JSON:
{{"answer": "<your answer>", "confidence": <0.0-1.0>}}
"""

QUERY_FUZZ_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_fuzz",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fuzzed_question": {"type": "string"},
                "changes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["fuzzed_question", "changes"],
            "additionalProperties": False,
        },
    },
}

# =========================================================================
# Quality Review Prompts
# =========================================================================

QUALITY_MULTIHOP_PROMPT = """\
Evaluate whether this question genuinely requires multi-step retrieval and reasoning \
from an AGENT's perspective. The agent sees ONLY the question below — it does NOT \
have the decomposition or supporting paragraphs. It must search a document collection \
to find the answer.

Note: entity names in this benchmark are deliberately FICTIONALIZED (e.g. "Nexavar",
"MemChain", "FlexiGate") to defeat memorization. Unfamiliar names are expected; do
not treat them as a defect.

IMPORTANT CONTEXT: This is a MEMORY benchmark. The agent processes a stream of ~300
search-result documents (~100K tokens) delivered across multiple turns. The challenge
is RETAINING relevant facts from earlier turns while processing later ones. Even if
a question names a specific entity, the agent must:
  (a) identify the relevant facts across multiple documents in different turns,
  (b) remember facts from earlier turns when combining them with later discoveries,
  (c) synthesize information scattered across the document stream.
A question that names an entity but requires combining facts from 4 different papers
appearing in 4 different turns IS genuinely multi-hop for a memory benchmark — the
agent cannot simply "search" within a 100K-token stream efficiently.

Question (as the agent sees it): {question}
Answer: {answer}

Ground-truth reasoning chain (hidden from agent):
{decomposition}

Supporting paragraphs (hidden from agent):
{paragraphs}

Evaluate FROM THE AGENT'S PERSPECTIVE:
1. Does answering require combining facts from MULTIPLE supporting paragraphs/documents?
2. Are the needed facts spread across the document stream (different turns), requiring the agent to retain earlier information?
3. Does the chain of reasoning require information from one document to correctly interpret another?

Rate 1-5:
- 5: Answer requires synthesizing 4+ facts from different documents/turns; no single document suffices
- 4: Answer requires 3+ facts from different documents; strong memory retention needed
- 3: Answer requires 2-3 facts; moderate retention across turns
- 2: Answer can mostly be found in a single document with minor supplementation
- 1: Answer is fully contained in a single document/paragraph

Respond in JSON:
{{"score": <1-5>, "could_be_single_hop": <true/false>, "reasoning": "<brief explanation>", "weakest_hop": <hop index or null>}}
"""

QUALITY_MULTIHOP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "multihop_eval",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "could_be_single_hop": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "weakest_hop": {"type": ["integer", "null"]},
            },
            "required": ["score", "could_be_single_hop", "reasoning", "weakest_hop"],
            "additionalProperties": False,
        },
    },
}

QUALITY_DISTRACTOR_PROMPT = """\
Evaluate the quality of distractors in this multi-hop QA instance.

Note: entity names in this benchmark are deliberately FICTIONALIZED to defeat
parametric memorization. Unfamiliar names like "Nexavar" or "FlexiGate" are
expected and should not be treated as defects.

DISTRACTOR DESIGN: This benchmark uses two distractor tiers:
1. **Sibling near-miss distractors** — generated from search results that shared
   the same query as the gold document at each hop, then minimally modified to
   create plausible but incorrect alternatives. These share vocabulary with the
   gold facts and are the hardest to distinguish.
2. **Search filler** — real (fictionalized) papers retrieved from the same topic
   domain, providing ~300 documents of topical noise across ~100K tokens. These
   create the "needle in a haystack" challenge for memory retention.

Question: {question}
Answer: {answer}

Supporting paragraphs (gold):
{supporting}

Sibling near-miss distractors (share vocabulary with gold facts):
{cross_instance_sample}

Search filler samples (topical noise from same domain):
{adversarial_sample}

Distractor composition: {composition}

Evaluate:
1. Are sibling near-miss distractors plausible alternatives that share vocabulary with the gold facts?
2. Is the search filler from the same topic domain (not obviously off-topic)?
3. Would an agent without memory notes struggle to distinguish gold facts from distractors?

Rate 1-5:
- 5: Near-miss distractors are highly confusing; filler is topically dense; strong memory needed
- 4: Near-miss distractors share significant vocabulary; filler is mostly on-topic
- 3: Some near-miss quality; filler is mixed relevance
- 2: Near-miss distractors are weak; filler is mostly off-topic
- 1: All distractors are trivially distinguishable from gold facts

Respond in JSON:
{{"score": <1-5>, "cross_instance_relevance": "high|medium|low", "adversarial_quality": "high|medium|low|none", "reasoning": "<brief explanation>"}}
"""

QUALITY_DISTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "distractor_eval",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "cross_instance_relevance": {"type": "string"},
                "adversarial_quality": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "cross_instance_relevance", "adversarial_quality", "reasoning"],
            "additionalProperties": False,
        },
    },
}

QUALITY_ANSWER_PROMPT = """\
Verify the answer correctness for this multi-hop question.

IMPORTANT — fictionalization context:
This benchmark deliberately replaces real-world entities (papers, methods, models,
datasets, people, places) with FICTIONAL names to prevent pretrained models from
answering via memorization. Names like "Nexavar", "MemChain", "FlexiGate",
"BioScale-FM", "DRPG", "APEX" etc. are EXPECTED — they are not hallucinations or
fabrications. Your job is to verify INTERNAL consistency: does the answer follow
from the supporting paragraphs and the reasoning chain provided here? Do NOT try
to verify entities against real-world knowledge, and do NOT penalize an instance
because the named entities are unfamiliar — that is the whole point of the design.

Question: {question}
Gold answer: {answer}

Reasoning chain:
{decomposition}

Supporting paragraphs:
{paragraphs}

For each hop, verify (using ONLY the provided paragraphs as ground truth):
1. Is the sub-answer actually supported by its paragraph?
2. Does the reasoning chain logically lead to the final answer?
3. Is the final answer entailed by the chain + paragraphs?

Rate 1-5:
- 5: All hops correct, chain is logically sound, answer follows from the docs
- 4: Chain is correct but minor ambiguity in one hop
- 3: One intermediate answer is debatable
- 2: Reasoning chain has a logical gap
- 1: Answer is not supported by the paragraphs or chain is broken

Respond in JSON:
{{"score": <1-5>, "chain_valid": <true/false>, "reasoning": "<brief explanation>", "issues": ["<issue if any>"]}}
"""

QUALITY_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_eval",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "chain_valid": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "chain_valid", "reasoning", "issues"],
            "additionalProperties": False,
        },
    },
}

QUALITY_OVERALL_PROMPT = """\
Provide an overall quality assessment for this multi-hop QA benchmark instance.

IMPORTANT — fictionalization context:
This benchmark deliberately replaces real-world entities with FICTIONAL names
(e.g. "Nexavar", "MemChain", "FlexiGate") to defeat parametric memorization.
Unfamiliar entity names are EXPECTED and DESIRED — do not flag them as
hallucinations or fabrications. Judge the instance on its internal consistency,
the genuineness of the multi-hop chain, and the difficulty of the distractors.

Question: {question}
Answer: {answer}
Hops: {num_hops}
Grounding facts: {num_facts}
Verification scores: score_all_memory={score_a}, score_no_memory={score_b}, score_long_context={score_c}, memory_gap={gap_ab}
Distractor composition: {composition}
Transforms applied: {transforms}

Sub-scores from other evaluations:
- Multi-hop genuineness: {multihop_score}/5
- Distractor difficulty: {distractor_score}/5
- Answer correctness: {answer_score}/5

IMPORTANT notes on verification scores:
- score_no_memory=0.0 and memory_gap≈1.0 are IDEAL outcomes (question truly requires memory/retrieval to answer)
- Penalize LOW memory_gap (<0.3) and HIGH score_no_memory (>0.3), as these mean the question can be answered without the supporting documents
- Do NOT penalize score_no_memory=0 or memory_gap close to 1.0 — these indicate a well-constructed benchmark instance
- score_long_context measures how well a model does with the FULL unfiltered document dump; if score_all_memory ≥ score_long_context, curated memory is at least as good as raw context

Rate overall quality 1-5 and list specific issues.
Recommend: keep (usable as-is), review (needs human check), or reject (not usable).

Respond in JSON:
{{"score": <1-5>, "issues": ["<issue>", ...], "recommendation": "keep|review|reject", "reasoning": "<brief explanation>"}}
"""

QUALITY_OVERALL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "overall_eval",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "recommendation": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "issues", "recommendation", "reasoning"],
            "additionalProperties": False,
        },
    },
}

# =========================================================================
# Hop Upgrade: Entity Fictionalization + Chain Extension
# =========================================================================

FICTIONALIZE_ENTITIES_PROMPT = """\
You are given a multi-hop question with supporting paragraphs that use real-world entities.
Your task is to replace ALL real entity names with plausible-sounding but entirely FICTIONAL names,
and perturb all specific facts (dates, numbers, populations, etc.) so the information cannot be
verified from real-world knowledge.

Original question: {question}
Original answer: {answer}

Entities and their paragraphs:
{entities_json}

Decomposition (sub-questions):
{decomposition_json}

Instructions:
1. Create a fictional name for EVERY entity (people, places, organizations, events, etc.)
2. Perturb all dates by ±5-50 years, all numbers by ±20%, rename all locations
3. Rewrite each paragraph using the fictional names and perturbed facts
4. Rewrite the question and all sub-questions/answers using fictional names
5. Maintain logical consistency — if "France" becomes "Valdoria", use "Valdoria" everywhere
6. Keep paragraph length and style similar to originals
7. The answer must ONLY be derivable from the rewritten paragraphs, not from real-world knowledge

Return JSON:
{{
  "entity_mapping": [{{"real_name": "<real>", "fictional_name": "<fictional>"}}, ...],
  "question": "<rewritten question>",
  "answer": "<rewritten answer>",
  "decomposition": [
    {{"sub_question": "...", "sub_answer": "...", "paragraph_title": "<fictional title>", "paragraph_text": "...", "hop_index": 0}},
    ...
  ],
  "supporting_paragraphs": [
    {{"title": "<fictional title>", "text": "<rewritten paragraph>"}},
    ...
  ]
}}
"""

FICTIONALIZE_ENTITIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fictionalized_instance",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entity_mapping": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "real_name": {"type": "string"},
                            "fictional_name": {"type": "string"},
                        },
                        "required": ["real_name", "fictional_name"],
                        "additionalProperties": False,
                    },
                },
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "decomposition": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sub_question": {"type": "string"},
                            "sub_answer": {"type": "string"},
                            "paragraph_title": {"type": "string"},
                            "paragraph_text": {"type": "string"},
                            "hop_index": {"type": "integer"},
                        },
                        "required": ["sub_question", "sub_answer", "paragraph_title", "paragraph_text", "hop_index"],
                        "additionalProperties": False,
                    },
                },
                "supporting_paragraphs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["title", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entity_mapping", "question", "answer", "decomposition", "supporting_paragraphs"],
            "additionalProperties": False,
        },
    },
}

HOP_UPGRADE_CHAIN_PROMPT = """\
You are given a {current_hops}-hop multi-hop question with fictional entities.
Your task is to EXTEND the reasoning chain to {target_hops} hops by inserting
{new_hops} intermediate bridge steps.

Current chain:
{chain_json}

Current question: {question}
Current answer: {answer}

Instructions:
1. Insert {new_hops} new intermediate hops BETWEEN the existing hops
2. Each new hop must:
   - Introduce a NEW fictional entity with a factual paragraph (80-150 words)
   - Have a sub-question that uses #N reference to the previous hop's answer
   - Produce a sub-answer that the NEXT hop depends on
3. The chain must be logically coherent — each hop naturally leads to the next
4. Use ONLY fictional entities — do NOT use real-world names, places, or facts
5. Renumber all hops sequentially (0, 1, 2, ..., {last_hop_index})
6. Rewrite the final question so it spans the full {target_hops}-hop chain
7. The answer should remain derivable ONLY from the full chain of paragraphs

Return JSON:
{{
  "question": "<extended question spanning all hops>",
  "answer": "<final answer>",
  "decomposition": [
    {{"sub_question": "...", "sub_answer": "...", "paragraph_title": "...", "paragraph_text": "...", "hop_index": 0}},
    ...
  ],
  "supporting_paragraphs": [
    {{"title": "...", "text": "..."}},
    ...
  ]
}}
"""

# =========================================================================
# Deep Research Trajectory Generator Prompts
# =========================================================================

RESEARCH_PLAN_PROMPT = """\
You are a research agent investigating a complex question. Based on the question \
and your research so far, decide what to search next.

Question: {question}

Research history so far:
{history}

Choose ONE action:
1. "search" — issue a web search query to find more information
2. "answer" — you have enough information to answer the question

If searching, write a specific, targeted query that addresses a gap in your knowledge.
Do NOT repeat a query you've already tried.

Respond in JSON:
{{"action": "search|answer", "query": "<search query if action=search>", "reasoning": "<why this query or why you're ready to answer>"}}
"""

RESEARCH_PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "research_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "answer"]},
                "query": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "query", "reasoning"],
            "additionalProperties": False,
        },
    },
}

RESEARCH_ANALYZE_PROMPT = """\
You are a research agent analyzing search results to extract key facts relevant \
to answering a question.

Question: {question}

Search query: {query}

Search results:
{results}

Research so far:
{history}

Extract the KEY FACTS from these results that help answer the question. For each fact:
- State the fact clearly and specifically
- Note which search result it came from (by title)
- Indicate whether this fact bridges to finding more information (bridge) or \
directly contributes to the final answer (terminal)

Also note any contradictions with previously found facts.

Respond in JSON:
{{
  "facts": [
    {{"content": "<factual statement>", "source": "<result title>", "type": "bridge|terminal"}}
  ],
  "reasoning": "<how these facts advance your understanding>",
  "contradictions": ["<any contradictions found>"],
  "confidence": <0.0-1.0 confidence you can now answer the question>
}}
"""

RESEARCH_ANALYZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "research_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "source": {"type": "string"},
                            "type": {"type": "string", "enum": ["bridge", "terminal"]},
                        },
                        "required": ["content", "source", "type"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["facts", "reasoning", "contradictions", "confidence"],
            "additionalProperties": False,
        },
    },
}

RESEARCH_FINAL_ANSWER_PROMPT = """\
You are a research agent. Based on your complete research trajectory, answer the question.

Question: {question}

Complete research trajectory:
{trajectory_summary}

Provide your final answer based ONLY on the facts you discovered during research.

Respond in JSON:
{{"answer": "<your answer>", "confidence": <0.0-1.0>, "supporting_facts": ["<fact1>", "<fact2>", ...]}}
"""

RESEARCH_FINAL_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "research_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "supporting_facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence", "supporting_facts"],
            "additionalProperties": False,
        },
    },
}

QUESTION_GENERATION_PROMPT = """\
Generate a complex factual question that requires 3-5 web searches to answer.
The question should chain entities: finding X leads to Y, which leads to Z.

Topic: {topic}
Difficulty: {difficulty}

Requirements:
1. The question must require at least 3 distinct facts from different sources
2. The answer should be a specific, verifiable fact (not an opinion)
3. The chain of reasoning should not be obvious from the question alone
4. Each intermediate step should require looking up a different entity or relationship

Respond in JSON:
{{"question": "<your question>", "expected_steps": <estimated number of search steps>, "topic_chain": "<brief description of the entity chain>"}}
"""

QUESTION_GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "question_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "expected_steps": {"type": "integer"},
                "topic_chain": {"type": "string"},
            },
            "required": ["question", "expected_steps", "topic_chain"],
            "additionalProperties": False,
        },
    },
}


HOP_UPGRADE_CHAIN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "upgraded_chain",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "decomposition": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sub_question": {"type": "string"},
                            "sub_answer": {"type": "string"},
                            "paragraph_title": {"type": "string"},
                            "paragraph_text": {"type": "string"},
                            "hop_index": {"type": "integer"},
                        },
                        "required": ["sub_question", "sub_answer", "paragraph_title", "paragraph_text", "hop_index"],
                        "additionalProperties": False,
                    },
                },
                "supporting_paragraphs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["title", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question", "answer", "decomposition", "supporting_paragraphs"],
            "additionalProperties": False,
        },
    },
}


# =========================================================================
# Grow-by-search prompts (v4 unified pipeline)
# =========================================================================

SEED_QUESTION_PROMPT = """\
IMPORTANT: This is a fictional research universe. Do NOT reference any \
real-world researcher, institution, model, dataset, or event. Only use \
entity names that appear in the input text below.

You are starting a multi-step research investigation in the area of \
{topic}. Generate a SPECIFIC starting question that:

1. Targets a concrete entity: a specific method, model, dataset, \
technique, finding, or phenomenon in this research area
2. Is answerable from a single academic paper or authoritative source
3. Will naturally lead to follow-up questions — the answer should \
reveal connections to other methods, datasets, or findings

The question should feel like the first step a researcher takes when \
investigating a new area: "What exactly is X?" or "What does Y use \
for Z?" — specific enough to get precise results, but connected \
enough to open doors for deeper investigation.

Do NOT generate:
- Broad survey questions ("What are the main approaches to X?")
- Opinion questions ("Is X better than Y?")
- Questions with obvious/well-known answers

Also provide a SHORT search query (3-7 words) suitable for academic search \
engines — not the full question, just the key terms to search.

Respond in JSON:
{{"question": "<your specific starting question>", \
"search_query": "<3-7 word search query for academic search engines>", \
"target_entity": "<the specific method/model/dataset this question is about>", \
"expected_answer_type": "<what kind of answer: a name, a number, a technique, a property>", \
"why_leads_deeper": "<what follow-up questions would the answer naturally lead to>"}}
"""

SEED_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "seed_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "search_query": {"type": "string"},
                "target_entity": {"type": "string"},
                "expected_answer_type": {"type": "string"},
                "why_leads_deeper": {"type": "string"},
            },
            "required": ["question", "search_query", "target_entity", "expected_answer_type", "why_leads_deeper"],
            "additionalProperties": False,
        },
    },
}


HOP_ANALYZE_PROMPT = """\
IMPORTANT: This is a fictional research universe. Do NOT reference any \
real-world researcher, institution, model, dataset, or event. Only use \
entity names that appear in the input text below.

You are conducting a step-by-step research investigation. You just \
searched for information and received results. Your task is to identify \
the single most important NEW fact from these results.

Research investigation topic: {topic}

Current question you searched for: {question}

Search results:
{results}

Facts you already know from previous steps:
{previous_facts}

Analyze the search results and identify:

1. KEY FACT: The single most important factual claim from these results \
that you did not already know. This should be:
- Specific: include exact names, numbers, dates, or technical terms
- Verifiable: stated in one or more of the search results
- Novel: not a restatement of something in your previous facts
- Relevant: directly helps answer the overall research topic

2. SOURCE DOCUMENTS: Which search results contain this fact? List them \
by their index (0-based). A document is "supporting" if it contains \
or directly states the key fact. All other documents are distractors.

3. RELEVANCE: How does this fact advance your understanding? What does \
it tell you that you didn't know before?

4. CONFIDENCE: How confident are you that this fact is accurate, based \
on the source quality and consistency across results? (0.0-1.0)

Important:
- Extract only ONE primary fact per search step. If multiple \
important facts exist, choose the one most relevant to the overall \
research topic. Additional interesting facts can be noted separately.
- The key fact MUST be a positive factual claim found IN the results. \
Do NOT extract "the results do not contain X" or "no information about Y" \
as a fact. If the results don't answer the question directly, extract \
the most relevant fact that IS present — what the results DO tell you.

Respond in JSON:
{{"key_fact": "<the single most important new factual claim>", \
"supporting_doc_indices": [0], \
"relevance": "<how this advances the investigation>", \
"confidence": 0.9, \
"additional_facts": ["<other interesting facts from these results, if any>"]}}
"""

HOP_ANALYZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hop_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "key_fact": {"type": "string"},
                "supporting_doc_indices": {"type": "array", "items": {"type": "integer"}},
                "relevance": {"type": "string"},
                "confidence": {"type": "number"},
                "additional_facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key_fact", "supporting_doc_indices", "relevance", "confidence", "additional_facts"],
            "additionalProperties": False,
        },
    },
}


_DEPTH_GUIDANCE = {
    2: "You are building the foundation. Look for related methods, "
       "alternative approaches, or the evaluation setup used with this entity.",
    3: "You have multiple pieces now. Look for comparisons, performance "
       "differences, or connections between the facts you've gathered.",
    4: "You are nearing the synthesis stage. Look for quantitative results, "
       "limitations, or conclusions that tie your earlier findings together.",
}


HOP_EXTEND_PROMPT = """\
You are deepening a research investigation step by step. You just \
discovered a new fact. Now generate the next question that BUILDS ON \
this fact — a question you could NOT have asked without knowing it.

Research area: {topic}

Fact you just discovered (hop {hop_number} of {target_hops}):
  "{current_fact}"

Facts from earlier hops:
{previous_facts}

Generate the next research question that:

1. DEPENDS on the current fact: The question must reference or require \
knowledge of the fact you just discovered. Someone who has not read \
the previous search results could NOT formulate this question.

2. GOES DEEPER: The question should move the investigation forward — \
toward a comparison, a quantitative result, a limitation, or a \
synthesis that requires additional sources.

3. IS SEARCHABLE: The question should be specific enough to return \
useful results from academic search engines. Avoid questions that \
are too abstract or philosophical.

{depth_guidance}

Also provide a SHORT search query (3-7 words) that you would type into \
an academic search engine to find the answer. This should be concise \
key terms, not the full question.

Respond in JSON:
{{"next_question": "<the next research question that builds on the current fact>", \
"search_query": "<3-7 word search query for academic search engines>", \
"dependency": "<explain exactly how this question depends on the current fact>", \
"what_this_will_reveal": "<what kind of information the answer to this question will provide>"}}
"""

HOP_EXTEND_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hop_extension",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "next_question": {"type": "string"},
                "search_query": {"type": "string"},
                "dependency": {"type": "string"},
                "what_this_will_reveal": {"type": "string"},
            },
            "required": ["next_question", "search_query", "dependency", "what_this_will_reveal"],
            "additionalProperties": False,
        },
    },
}


SYNTHESIZE_QUESTION_PROMPT = """\
You have completed a multi-step research investigation. You discovered \
the following facts through sequential search steps:

Research area: {topic}

Facts discovered (in order):
{facts_with_hops}

Search trajectory:
{trajectory_summary}

Now synthesize a SINGLE research question that:

1. REQUIRES ALL FACTS: Answering this question correctly requires \
knowing every fact in the chain above. If any fact is missing, \
the question cannot be fully answered.

2. SOUNDS NATURAL: The question should read like something a \
researcher would genuinely ask — not like a quiz or trivia \
question. It should feel like a question that motivates the \
investigation you just completed.

3. IS NOT DECOMPOSABLE: The question should NOT be easily breakable \
into independent sub-questions. The facts should be CHAINED — \
each fact builds on the previous one to reach the answer.

4. HAS A SPECIFIC ANSWER: The answer should be a concrete statement \
that can be verified — not an open-ended discussion.

Also provide:
- The GOLD ANSWER: derived from chaining the facts together
- A REASONING TRACE: show how fact F1 + F2 + ... = answer

Respond in JSON:
{{"question": "<the synthesized multi-hop research question>", \
"answer": "<the specific, verifiable answer>", \
"reasoning_trace": "<F1 tells us X, which combined with F2 tells us Y, ...>", \
"facts_used": ["F1", "F2", "F3"], \
"why_all_facts_needed": "<explain why removing any single fact makes the question unanswerable>"}}
"""

SYNTHESIZE_QUESTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthesized_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "reasoning_trace": {"type": "string"},
                "facts_used": {"type": "array", "items": {"type": "string"}},
                "why_all_facts_needed": {"type": "string"},
            },
            "required": ["question", "answer", "reasoning_trace", "facts_used", "why_all_facts_needed"],
            "additionalProperties": False,
        },
    },
}


DISTRACTOR_ENHANCE_PROMPT = """\
You are creating a challenging reading comprehension distractor. You \
have a real document and a set of specific facts that a reader needs \
to remember accurately. Rewrite the document so it becomes confusingly \
similar to the actual facts — close enough to cause doubt, but \
subtly different in key details.

Memory facts the reader needs to retain:
{facts_to_mimic}

Style reference (excerpt from a supporting document):
{supporting_doc_excerpt}

Original distractor document to rewrite:
{distractor_text}

Rewrite the distractor document following these principles:

1. SURFACE SIMILARITY: Use similar terminology, phrasing patterns, \
and technical vocabulary as the memory facts and supporting \
document. The rewritten text should feel like it's discussing \
the same research area.

2. SUBTLE DIFFERENCES: Change key details that matter for the answer:
- Numbers that are close but not identical (87.2% vs 88.9%)
- Method names that sound similar but are different
- Dataset names that are related but not the same benchmark
- Temporal claims that are slightly shifted
- Causal relationships that are reversed or weakened

3. INTERNAL CONSISTENCY: The rewritten document must be self-consistent \
and factually plausible on its own. It should read like a real \
academic paper or article — not like something intentionally wrong.

4. PRESERVE STRUCTURE: Keep the original document's overall structure, \
length, and format. Only modify content details, not the document \
skeleton.

Do NOT:
- Make the differences obvious or exaggerated
- Add clearly false or absurd claims
- Change the general topic of the document
- Make the document significantly shorter or longer

Respond in JSON:
{{"rewritten_text": "<the rewritten distractor document>", \
"changes_made": ["<specific change 1: 'Changed X to Y to create confusion about fact F2'>"]}}
"""

DISTRACTOR_ENHANCE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "enhanced_distractor",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_text": {"type": "string"},
                "changes_made": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["rewritten_text", "changes_made"],
            "additionalProperties": False,
        },
    },
}


NEAR_MISS_FACT_PROMPT = """\
You are building a distractor document for a long-context memory benchmark. \
A reader will try to answer a specific question using only their memory of \
a real grounding fact. Your distractor will appear among many other papers \
in the reader's search results WITHOUT the grounding fact being shown.

The test is:
- The reader's MEMORY contains the grounding fact.
- The reader's VISIBLE DOCUMENTS include your distractor (but NOT the \
grounding fact).
- The reader must answer the question correctly only when they use memory. \
If your distractor contains the answer, the test is broken.

Question the reader will be asked:
  "{question}"

Correct answer (the reader must recall this from memory, NOT infer it from \
your distractor):
  "{answer}"

Real grounding fact (in memory, hidden from visible docs):
  "{real_fact}"

From supporting document: "{source_title}"

A natural sibling paper retrieved by the same search query (use this as the \
structural base so your distractor looks like a legitimate retrieval hit):
  "{distractor_text}"

Your task: write a distractor paragraph that is stylistically and topically \
similar to the sibling paper and could plausibly appear next to the \
grounding fact in a literature search, but that DOES NOT contain the answer \
to the question above in any form.

Hard rules (each one is a test you must pass):
1. NO ANSWER LEAKAGE — The answer "{answer}" must not appear verbatim, as a \
synonym, as a paraphrase, as a partial match, as a list containing it, or \
as a value the reader could reconstruct by reading your distractor alone. \
Actively scrub any phrase that would let the reader answer without memory.
2. WRONG-BUT-PLAUSIBLE — Where the grounding fact states the answer, your \
distractor must state a DIFFERENT, plausibly wrong alternative (different \
value, different entity, different mechanism). It should make the reader \
feel tempted but conflict with what memory says.
3. DIFFERENT ENTITY OR ROLE — Ideally your distractor discusses a different \
method, paper, or project than the one in the grounding fact. If the \
grounding fact names an entity, you may reuse that entity only in a \
secondary role (cited, compared against), not as the subject of the claim \
the question asks about.
4. LOOKS REAL — The distractor must read like a real paper abstract. Same \
register, comparable length, specific language. No generic filler.
5. DO NOT COPY the grounding fact's surface form. Avoid repeating more than \
a 5-word span from the grounding fact.

Before you return, silently ask yourself: "If someone read ONLY this \
distractor text and the question, could they answer the question correctly \
or even partially?" If yes, rewrite.

Respond in JSON:
{{"near_miss_text": "<the distractor paragraph>", \
"changed_detail": "<what the distractor claims instead of the answer — e.g. 'claims MCMC sampling instead of Bayesian inference'>", \
"targets_fact": "<the fact id or description this distractor competes with>"}}
"""

NEAR_MISS_FACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "near_miss_distractor",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "near_miss_text": {"type": "string"},
                "changed_detail": {"type": "string"},
                "targets_fact": {"type": "string"},
            },
            "required": ["near_miss_text", "changed_detail", "targets_fact"],
            "additionalProperties": False,
        },
    },
}


BULK_CONFUSED_CONTENT_PROMPT = """\
IMPORTANT: This is a fictional research universe. Do NOT reference any \
real-world researcher, institution, model, dataset, or event. Only use \
entity names that appear in the input text below.

You are generating background content for a research reading comprehension \
test. The content should be topically related to the research area but \
contain plausible-sounding factual claims that are SUBTLY INCORRECT.

Research area: {topic}
Key entities in this research: {entities}

The reader will encounter this content while trying to remember specific \
facts about {topic}. Your content should create BACKGROUND NOISE — it \
discusses the same research area with authority and confidence, but its \
specific claims are slightly wrong or describe fictional-but-plausible \
findings.

Generate {num_paragraphs} paragraphs, each approximately {target_words} \
words. Each paragraph should:

1. Discuss a real sub-topic within {topic} using correct terminology
2. Include 1-2 specific quantitative claims that sound real but are \
fabricated (e.g., citing plausible accuracy numbers, dataset sizes, \
or publication years that are slightly off)
3. Reference real-sounding method names, dataset names, or author names \
that may or may not exist
4. Read like an excerpt from a real academic survey or textbook chapter
5. NOT directly address the same specific questions the reader is trying \
to answer — this is background noise, not targeted near-misses

Do NOT:
- Use obviously fake names or absurd claims
- Directly copy real facts (the reader should not find correct answers here)
- Make the paragraphs about completely unrelated topics

Respond in JSON:
{{"paragraphs": [ \
{{"title": "<paragraph topic>", \
"text": "<the paragraph content with plausible-but-wrong claims>", \
"fake_claims": ["<list the specific wrong claims embedded>"]}} \
]}}
"""

BULK_CONFUSED_CONTENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "bulk_confused_content",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                            "fake_claims": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "text", "fake_claims"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["paragraphs"],
            "additionalProperties": False,
        },
    },
}


MEMORY_ABLATION_QA_PROMPT = """\
You are answering a research question. You have two sources of \
information:

1. NOTES from earlier research — facts you retained from documents \
you read previously but can no longer see
2. DOCUMENTS from the most recent search step — the only documents \
currently visible to you

Use both your notes and the visible documents to answer the question. \
If the information is insufficient, state what is missing.

Research question:
{question}

Your retained notes:
{notes}

Currently visible documents:
{visible_documents}

Respond in JSON:
{{"answer": "<your answer, or 'insufficient information' if you cannot answer>", \
"reasoning": "<how you used your notes and/or documents to reach this answer>", \
"notes_used": "<which specific notes were critical for your answer>", \
"confidence": 0.9}}
"""

MEMORY_ABLATION_QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ablation_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "notes_used": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "reasoning", "notes_used", "confidence"],
            "additionalProperties": False,
        },
    },
}


# =========================================================================
# Deep Research Pipeline — additional prompts
# =========================================================================


ACADEMIC_SURVEY_FILLER_PROMPT = """\
IMPORTANT: This is a fictional research universe. Do NOT reference any \
real-world researcher, institution, model, dataset, or event. Only use \
entity names that appear in the input text below.

You are writing a section of an academic survey paper about {topic}. \
Generate a detailed, realistic survey-style passage that discusses \
{subtopic} in depth.

The passage should:
1. Read like a real academic survey (cite plausible-sounding works with years)
2. Cover related but DISTINCT findings from the actual research question
3. Include specific numbers, method names, and dataset references
4. Be approximately {target_words} words long
5. Discuss {num_subsections} sub-points with concrete technical details
6. Sound authoritative but contain NO correct answers to: "{question}"

Key entities in this research area: {entities}

CRITICAL: The passage must NOT directly answer or hint at the answer to \
the research question above. It should discuss the same broad area but \
cover different aspects, methods, or findings.

Respond in JSON:
{{"title": "<section title>", \
"text": "<the academic survey passage>", \
"fake_references": ["<Author et al. (Year). Title. Venue.>"]}}
"""

ACADEMIC_SURVEY_FILLER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "academic_survey_filler",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "text": {"type": "string"},
                "fake_references": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "text", "fake_references"],
            "additionalProperties": False,
        },
    },
}


ADVERSARIAL_HACK_CHECK_PROMPT = """\
You are testing whether a research question can be answered by "hacking" — \
using reasoning shortcuts, common patterns, or educated guessing WITHOUT \
actually knowing the specific facts.

Research question:
{question}

Distractors and context available (these are the ONLY documents visible — \
no notes, no memory):
{context}

Try your HARDEST to answer the question using:
1. Pattern matching (common answers in this field)
2. Statistical guessing (most frequent values, typical findings)
3. Process of elimination from the distractors
4. World knowledge about the topic
5. Inferring from document structure or emphasis

If you can confidently guess the answer, the question is HACKABLE and \
needs harder distractors.

Respond in JSON:
{{"answer": "<your best guess, or 'cannot determine'>", \
"confidence": <0.0-1.0>, \
"hack_method": "<which shortcut you used: pattern/guess/elimination/knowledge/none>", \
"reasoning": "<how you arrived at this answer without memory>"}}
"""

ADVERSARIAL_HACK_CHECK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hack_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "hack_method": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "confidence", "hack_method", "reasoning"],
            "additionalProperties": False,
        },
    },
}


SUBTOPIC_GENERATOR_PROMPT = """\
Generate {count} related but distinct sub-topics within the research \
area of "{topic}" that are suitable for writing academic survey passages.

Each sub-topic should:
- Be related to {topic} but cover a DIFFERENT aspect
- Be specific enough to write a detailed 500+ word passage about
- NOT directly address: "{question}"

Respond in JSON:
{{"subtopics": ["<sub-topic 1>", "<sub-topic 2>", ...]}}
"""

SUBTOPIC_GENERATOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtopics",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subtopics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["subtopics"],
            "additionalProperties": False,
        },
    },
}


# =========================================================================
# Deep Research Pipeline — Fictionalization prompts
# =========================================================================


FICTIONALIZE_EXTRACT_PROMPT = """\
You are analyzing a set of grounding facts from a research benchmark. \
Your job is to identify ALL named entities, specific numbers, dates, \
and technical terms that a pretrained language model could recognize \
from its training data.

Facts:
{facts}

Question: {question}
Answer: {answer}

For each entity/value found, produce a fictional replacement:
- People: Replace with plausible but fictional names (e.g., "Einstein" → "Dr. Yuki Harashima")
- Models/Systems: Replace with fictional but plausible names (e.g., "Transformer" → "Nexavar", "BERT" → "STRATUM")
- Organizations: Replace with fictional names (e.g., "Google" → "Helios Labs")
- Specific numbers: Modify slightly but keep plausible (e.g., "94.78%" → "91.32%", "O(n²)" → "O(n√n)")
- Years: Shift by a random offset (e.g., "2017" → "2021")
- Dataset names: Replace with fictional names (e.g., "ImageNet" → "VistaSet")
- Method names: Replace with plausible fictional names (e.g., "ProbSparse attention" → "StochasticFocus attention")

RULES:
1. Every entity that could be recognized from pretraining data MUST be replaced
2. Replacements must be internally consistent (if "Transformer" → "Nexavar", use "Nexavar" everywhere)
3. Keep the same TYPE of entity (a person stays a person, a model stays a model)
4. Numbers should stay in the same rough magnitude
5. The fictional world must be self-consistent — relationships between entities should still make sense
6. Do NOT extract common English words: articles (a, an, the), prepositions (in, on, at, of, \
for, with, by, to, from), auxiliary verbs (is, are, was, were, does, do, did, has, have, had), \
modals (can, may, will, could, would, should), pronouns (it, we, they, he, she), or conjunctions \
(and, or, but, if, as, so). These are NEVER entities.

Respond in JSON with a list of substitutions and the rewritten question/answer:
{{"substitutions": [
    {{"original": "<original entity>", "replacement": "<fictional replacement>", "type": "<person|model|org|number|year|dataset|method|other>"}},
    ...
],
"new_question": "<question with all substitutions applied>",
"new_answer": "<answer with all substitutions applied>"}}
"""

FICTIONALIZE_EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fictionalize_extract",
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
                "new_question": {"type": "string"},
                "new_answer": {"type": "string"},
            },
            "required": ["substitutions", "new_question", "new_answer"],
            "additionalProperties": False,
        },
    },
}


FICTIONALIZE_REWRITE_PROMPT = """\
You are rewriting a document passage by applying entity substitutions. \
Replace ALL occurrences of the original entities with their fictional \
replacements. Keep the document structure, style, and flow identical — \
only change the specific entities/numbers listed.

Substitutions to apply:
{substitutions}

Original passage:
{passage}

RULES:
1. Apply ALL substitutions — do not miss any occurrence
2. Also catch partial matches and variations (e.g., "the Transformer model" should become "the Nexavar model")
3. Keep surrounding text exactly the same
4. If an entity appears in possessive form ("Einstein's") or plural ("Transformers"), apply the substitution naturally
5. Do NOT add or remove any other content

Return the rewritten passage as JSON:
{{"rewritten_text": "<passage with all substitutions applied>"}}
"""

FICTIONALIZE_REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fictionalize_rewrite",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_text": {"type": "string"},
            },
            "required": ["rewritten_text"],
            "additionalProperties": False,
        },
    },
}
