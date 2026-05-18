"""Prompt templates for Coding-Synthetic pipeline v2.

V2 prompts cover:
- Split stage: grounding fact extraction + verification
- Craft stage: task prompt generation, adversarial distractors, memory file assembly
- Difficulty stage: prompt fuzzing, indirection transforms
- Verify stage: patch generation, LLM judge
"""

# ---------------------------------------------------------------------------
# Stage 2 — Split
# ---------------------------------------------------------------------------

FACT_SEED_PROMPT = '''\
You are analyzing a code patch to identify what information a developer would
need to fix this bug. You will produce **behavioral questions** — NOT answers,
NOT descriptions of code changes.

## Patch (bug-introducing diff — reverting this is the fix)
```diff
{patch}
```

## Files modified in the patch
{patch_files}

## Static analysis of changed code
{repo_analysis}

## Instructions

For each distinct hunk in the patch, generate 1-2 behavioral questions that
capture what a developer would need to KNOW (from outside the repo) to produce
the correct fix. Questions must be about:
- Expected behavior, constraints, or specifications
- Runtime conditions or user-visible symptoms
- External API contracts or protocol requirements
- Design intent or architectural context

### CRITICAL RULES
- Questions must be phrased as BEHAVIORAL/SPECIFICATION questions
- Do NOT describe code changes (no "why was X changed to Y", "what was added/removed")
- Do NOT reference specific line numbers, diff markers, or patch content
- Do NOT mention "the patch", "the diff", or "the change"
- Frame questions as if you are a developer investigating the bug, NOT reading a patch
- Each question should be answerable from the problem statement, tests, or external docs

### Examples of GOOD questions
- "What should `scope_to_list` return when given a single-item string?"
- "What ordering constraints exist on the return value of the scope conversion?"
- "Under what conditions does the OAuth token validation raise an error?"
- "What backwards-compatibility requirements apply to this API endpoint?"

### Examples of BAD questions (DO NOT generate these)
- "Why was `sorted()` removed from line 42?" (references code change)
- "What was the purpose of adding the null check?" (describes change)
- "Why did the patch change the return type?" (references patch)

Generate 6-12 seed questions. Each must specify which patch hunk indices
(0-based) it covers.

Return a JSON object with a "seeds" array.
'''


FACT_EXTRACTION_PROMPT = '''\
You are an expert at analyzing bugs and identifying the key information
needed to understand and fix them.

Given a problem statement, behavioral questions about the bug, the failing
tests, and a static analysis summary of the changed code, extract
**grounding facts** — specific pieces of information that a developer
would need to produce the fix.

## Problem Statement
{problem_statement}

## Key questions to answer about this bug
{fact_seeds}

## Failing tests (FAIL_TO_PASS)
{fail_to_pass_tests}

## Files modified
{patch_files}

## Static analysis of changed code
{repo_analysis}

## Instructions

Extract 10-18 grounding facts. Each fact must be a *specific, falsifiable*
statement (not vague). Answer the seed questions above using information
from the problem statement, tests, and code analysis.

### Categories (with examples)

- **user_requirement**: Expected behavior or constraint only a user would know
  Example: "The OAuth scope parameter must be returned as a list, not a string"

- **bug_description**: What exactly is wrong / how the bug manifests
  Example: "The function returns None instead of raising ValueError on empty input"

- **repro_condition**: Steps or conditions to trigger the bug
  Example: "Bug only triggers when scope string contains a single item with no spaces"

- **error_detail**: Error messages, stack traces, assertion failures
  Example: "AssertionError in test_scope_parsing: expected ['read'] but got 'read'"

- **debug_finding**: Insight gained by debugging (e.g., variable state at crash)
  Example: "At the crash point, `scope` is a string 'read write' but the callee expects a list"

- **api_behavior**: External library behavior or protocol detail
  Example: "RFC 6749 Section 3.3 specifies scope values are space-delimited strings"

- **design_decision**: Architectural choice that informs the fix
  Example: "Scope parsing was extracted into a utility for reuse by both OAuth1 and OAuth2"

- **constraint**: Performance, backwards-compatibility, or other constraint
  Example: "The fix must preserve backward compatibility with callers passing scope as a list"

### Category distribution requirements

You MUST include facts from **at least 4 different categories**. Aim for:
- At least 1 fact in `repro_condition` or `error_detail`
- At least 1 fact in `debug_finding` or `api_behavior`
- At least 1 fact in `design_decision` or `constraint`
- No more than 40% of facts in any single category

If a category genuinely does not apply, explain why in the evidence field,
but still try to find at least one fact for it.

### Topic / Content separation (CRITICAL for QA leakage prevention)

Each fact has TWO fields:

- **content** — the full factual statement WITH the specific value/detail.
  This is what goes into memory.
  Example: "The SYM parser uses 0x7FF as the frame-type threshold, treating
  IDs above this as extended frames."

- **topic** — WHAT the fact is about, stripped of the specific answer.
  This seeds question synthesis. It must NOT contain the value, identifier,
  or behavior that the content reveals.
  Example (paired with above): "the frame-type threshold the SYM parser uses
  when distinguishing standard vs. extended CAN IDs".

Rules for writing `topic`:
1. Strip the numeric value, enum name, literal string, or specific behavior
   that answers "what is it". Keep the subject + the question it answers.
2. Must be a noun phrase, not a statement. Starts with "the ...", "what ...",
   or "how ...". Never "X is Y" or "X does Z".
3. Must identify a unique fact (don't say "the parser's behavior" — say
   "the parser's behavior when the Subtype attribute is missing").
4. Self-check: if I showed you only the `topic` and asked "what's the value
   or behavior?", could you guess? If yes, the topic still leaks the content.

### Classification

For each fact, decide:

- **discoverable** = true if the fact can be found by reading the repo source
  code, docstrings, comments, or test assertions. Set to false if the info
  can ONLY come from the user or from running the code (e.g., expected behavior,
  user-specific constraints, runtime error messages not in source).

- **relevance** =
  - "critical": without this fact the correct fix CANNOT be generated
  - "helpful": makes fixing easier but a skilled dev could figure it out
  - "background": nice to know but not needed for the fix

### Quality requirements

- At least 2 facts must be discoverable=false AND relevance=critical
- All patch hunks must be covered (hunks_enabled must collectively span all hunks)
- Express facts in terms of BEHAVIOR and SPECIFICATIONS, not code changes
- Only extract project-specific information, NOT general programming knowledge
- Provide concrete evidence for each discoverable/non-discoverable classification

### Self-check before returning

1. Count how many facts are in each category
2. Verify at least 4 different categories are represented
3. Verify no single category exceeds 40% of total facts
4. If a category is missing, reconsider whether the problem/tests imply
   any facts of that type

Return your response as a JSON object with "facts" and "referenced_files".
'''


FACT_VERIFICATION_PROMPT = '''\
You are a senior engineer verifying fact classifications against actual source code.

Below are grounding facts that were classified as **not discoverable from the
repo** (discoverable=false). For each one, check the provided repo context to
see whether the information IS actually present in the codebase (in source code,
comments, docstrings, test assertions, config files, etc.).

## Facts to verify
{facts_to_verify}

## Full repo context
{repo_context}

## Instructions

For each fact, output:
- fact_id: the fact's ID
- new_discoverable: true if you found the information in the repo, false if not
- evidence: quote the repo line/section that supports your decision, or explain
  why the information cannot be found

Be strict: if the fact says "function X should return Y" and the repo has a
docstring or test assertion confirming this, mark it discoverable=true.
Only mark discoverable=false if there is genuinely no trace in the codebase.

Return a JSON object with a "reclassifications" array.
'''


# ---------------------------------------------------------------------------
# Stage 3 — Craft
# ---------------------------------------------------------------------------

TASK_PROMPT_CRAFT_PROMPT = '''\
You are an expert at writing realistic bug reports and task descriptions.

Given the original problem statement and the critical grounding facts, rewrite
the problem as a **vague, user-style task prompt** that an engineer would see
in an issue tracker.

## Original Problem Statement
{problem_statement}

## Critical external facts (NOT in repo — must be omitted from prompt)
{critical_external_facts}

## Rules
1. Describe the SYMPTOM, not the root cause
2. Reference the general code area, not exact function names
3. Omit ALL critical external facts — the task prompt should NOT contain
   enough information to solve the problem without the memory files
4. Sound like a real user filing a bug: casual, possibly imprecise
5. Do NOT mention specific variable names, exact line numbers, or fix logic
6. Keep it 2-5 sentences
7. You may reference file paths vaguely (e.g., "the OAuth utilities" not
   "oauthlib/oauth2/rfc6749/utils.py")

Return a JSON object with a "task_prompt" field.
'''


ADVERSARIAL_DISTRACTOR_PROMPT = '''\
You are generating adversarial near-miss facts for a coding task.

Given the real grounding facts for a bug fix, generate {num_distractors}
distractor facts that are **plausible but subtly wrong or irrelevant**.
These will be mixed with real facts to test whether a memory system can
distinguish signal from noise.

## Real grounding facts (do NOT repeat these)
{real_facts}

## Code context
{code_context}

## Requirements for each distractor
- Must be technically plausible and domain-relevant
- Must NOT be useful for generating the correct fix
- Should be tricky: similar enough to real facts to be confusing
- Types to generate:
  - Slightly wrong behavior descriptions (e.g., wrong return type, wrong parameter)
  - Red-herring functions/files (related module but wrong target)
  - Outdated information (was true in older version but not now)
  - Correct facts about unrelated parts of the same module

Return a JSON object with a "distractors" array, each having "id" and "content".
'''


SAME_FUNCTION_DISTRACTOR_PROMPT = '''\
You are generating adversarial distractor facts that CONTRADICT real facts
about the SAME functions and code areas.

Given the real grounding facts for a bug fix, generate {num_distractors}
distractor facts. Each distractor must:
1. Reference the SAME function, class, or code area as a real fact
2. Describe DIFFERENT (incorrect) behavior for that same code
3. Be plausible enough that someone unfamiliar with the code might believe it
4. NOT be useful for generating the correct fix

## Real grounding facts
{real_facts}

## Code context
{code_context}

## Examples of good contradicting distractors
If real fact says: "scope_to_list should return a sorted list"
Good distractor: "scope_to_list preserves insertion order for backward compatibility"

If real fact says: "The function raises ValueError on empty input"
Good distractor: "The function returns an empty list on empty input, which is the expected behavior"

Return a JSON object with a "distractors" array, each having "id" and "content".
'''


MEMORY_FILE_ASSEMBLY_PROMPT = '''\
You are packaging technical facts into natural-looking documents that would
plausibly exist alongside a code repository.

Given a list of facts (some real, some distractors) and optional code snippets,
create 4-8 document files that contain these facts embedded in realistic prose.
The documents should look like they were written by developers — NOT like
structured data dumps.

## Facts to include
{all_facts}

## Code snippets found during investigation
{code_snippets}

## Document types to choose from
- `debug_notes.md` — Developer's debugging journal with timestamps
- `issue_discussion.md` — Back-and-forth discussion thread about the bug
- `code_review_comments.md` — Code review feedback on a related PR
- `incident_report.md` — Post-incident analysis document
- `design_doc.md` — Technical design document for the feature area
- `testing_notes.md` — QA notes about test failures and reproduction
- `standup_notes.md` — Daily standup meeting notes with multiple topics
- `sprint_retro.md` — Sprint retrospective notes covering many items
- `slack_thread.md` — Exported Slack conversation with mixed topics

## Rules
1. Each document should contain 2-6 facts, naturally woven into the prose
2. Facts should NOT be presented as bullet lists — embed them in paragraphs,
   dialogue, or narrative
3. Mix real facts with distractors within the same document
4. Use realistic formatting (headers, timestamps, @mentions, code snippets)
5. Documents should reference each other occasionally for realism
6. Total output across all files: {total_words_range} words
7. Do NOT label which facts are real vs distractors
8. Include code snippets in appropriate documents (paste them in code blocks
   within debug notes or code review comments)
9. IMPORTANT: At least half of each document's content should be UNRELATED to
   the bug — include realistic off-topic discussions such as:
   - Team chatter about other features, deployments, or sprint planning
   - Code review comments on unrelated PRs or refactoring efforts
   - Mentions of other bugs, tech debt, performance concerns
   - Discussions about testing strategies, CI/CD, code style
   - References to meetings, deadlines, or team processes
   This off-topic content makes the documents realistic and forces careful reading.

Return a JSON object with a "files" array, each having "filename" and "content".
'''


# ---------------------------------------------------------------------------
# Stage 4 — Difficulty transforms
# ---------------------------------------------------------------------------

PROMPT_FUZZ_PROMPT = '''\
You are an expert at making technical descriptions vaguer while preserving
their essential meaning.

Given a task prompt for a coding bug, rewrite it to be MORE vague at the
specified intensity level.

## Original task prompt
{original_prompt}

## Intensity: {intensity}
- light: Replace specific function/variable names with behavioral descriptions.
  Keep file references vague.
- medium: Also remove error message specifics, replace with general symptoms.
  Remove most technical identifiers.
- heavy: Maximum vagueness. Only describe the general area and user-visible
  symptom. No technical details at all.

## Rules
1. The rewritten prompt must still point to the SAME bug (not a different one)
2. An engineer with repo access should still be able to find the right code area
3. Do NOT introduce new information that wasn't in the original
4. Keep the natural, user-like tone

Return a JSON object with "fuzzed_prompt" and "changes_made" (list of strings
describing what was changed).
'''


FACT_FRAGMENT_PROMPT = '''\
You are splitting complete technical facts into partial hints that must be
combined to reconstruct the full information.

## Facts to fragment
{facts_to_fragment}

## Available target files
{file_names} ({num_files} files)

## Instructions
For each fact, split it into 2-3 partial hints. Each partial hint should:
1. Contain only PART of the information (not enough to reconstruct the fix alone)
2. Be placed in a DIFFERENT file from the other parts
3. Use natural language that fits a developer document style
4. Reference the same concept/function but from different angles

Example:
  Full fact: "scope_to_list must split on spaces and return a sorted list"
  Fragment A (in debug_notes.md): "During debugging, noticed scope_to_list
    splits input on spaces but the ordering seemed inconsistent"
  Fragment B (in code_review.md): "Review comment: the return value of scope
    conversion should always be sorted alphabetically per RFC spec"

Return a JSON object with a "fragments" array, each having:
- "source_fact_id": the original fact ID
- "target_file": which file to place this fragment in
- "partial_content": the text to add (1-3 sentences, natural prose)
'''


INDIRECTION_TRANSFORM_PROMPT = '''\
You are adding multi-hop reasoning requirements to memory file content.

Given a memory file and repo context, rewrite specific direct references to
use indirect/behavioral descriptions that require reading the repo to resolve.

## Memory file content
{file_content}

## Repo context (for creating valid indirections)
{repo_context_summary}

## Rules
1. Replace direct function/class names with descriptions via callers, behavior,
   or relationships
   - "scope_to_list" → "the function called by params_from_uri that converts
     scope strings"
   - "test_parameters.py" → "the test module that checks URI parsing"
2. Use caller/callee relationships from the repo context
3. Preserve essential information content — a developer WITH repo access
   should still be able to resolve the indirection
4. Keep the document's natural tone and formatting

Return a JSON object with "rewritten_content" and "indirections_added" (array
of {{"original": "...", "indirect": "..."}}).
'''


# ---------------------------------------------------------------------------
# Stage 5 — Verify
# ---------------------------------------------------------------------------

PATCH_GENERATION_SYSTEM = (
    "You are an expert software engineer. Given a task description and "
    "optionally repository source files and/or memory files, generate a "
    "unified diff patch that fixes the described problem.\n\n"
    "Output ONLY the unified diff — no markdown fences, no explanation. Use "
    "standard format:\n"
    "diff --git a/file b/file\n"
    "--- a/file\n"
    "+++ b/file\n"
    "@@ -line,count +line,count @@\n"
    " context\n"
    "-removed line\n"
    "+added line"
)


LLM_JUDGE_PROMPT = (
    "You are evaluating a generated code patch against a gold-standard fix.\n\n"
    "## Generated Patch\n```\n{generated}\n```\n\n"
    "## Gold Fix (Reference)\n```\n{gold_fix}\n```\n\n"
    "## Evaluation Criteria\n"
    "Score from 0.0 to 1.0 based on:\n"
    "1. **Same files modified?** (0.2) — Does the generated patch modify the same files?\n"
    "2. **Same functions/areas changed?** (0.2) — Are the changes in the same code regions?\n"
    "3. **Behavioral equivalence?** (0.4) — Would the generated patch produce the same behavior fix?\n"
    "4. **Completeness?** (0.2) — Are all necessary changes present?\n\n"
    "Return a JSON object with \"score\" (float 0.0-1.0) and \"explanation\" (string)."
)


# ---------------------------------------------------------------------------
# Coding QA — Question generation, evaluation, and judging prompts
# ---------------------------------------------------------------------------

FACT_TOPIC_EXTRACTION_PROMPT = '''\
You are rewriting a grounding fact into a "topic" — a stripped noun phrase
describing WHAT the fact is about, without revealing the specific value,
identifier, enum, or behavior that answers "what is it".

## Fact
Category: {category}
Content: {fact_content}

## Rules
1. Output ONE noun phrase, 6-25 words. Not a sentence.
2. Start with "the", "what", or "how". Never use "is", "does", "returns".
3. Strip every specific answer: numbers, enum names, literal strings,
   specific method names of the behavior being asked about, and any
   description of the behavior itself. KEEP context needed to disambiguate
   (which module, which function, which scenario).
4. Self-check: given only your topic + no repo access, could a competent
   engineer guess the content? If yes, your topic leaks — rewrite.

## Good examples
- content: "The SYM parser uses 0x7FF as the frame-type threshold."
  topic:   "the frame-type threshold value the SYM parser applies to CAN IDs"
- content: "_async_request returns a list of strings, not a tuple, when the
            packet contains a name-list."
  topic:   "the return type and shape of _async_request when the response
            packet has a name-list field"
- content: "When keep_unknowns=True, the parser yields entries for
            unparseable lines."
  topic:   "the behavior of the parser toward unparseable log lines when
            keep_unknowns is enabled"

## Bad examples (these STILL leak — do NOT produce topics like this)
- content: "Finnish SSN generation raises UnboundLocalError: local variable
            _checksum referenced before assignment."
  bad topic:  "the variable initialization order issue in Finnish SSN
               generation causing reference errors"
  why bad:    "reference errors" hints at UnboundLocalError.
  good topic: "what happens when Finnish SSN generation is invoked under
               the documented failing conditions"

- content: "The parse_files_to_codes_mapping function accepts both string
            and sequence inputs for value_ and converts them."
  bad topic:  "how parse_files_to_codes_mapping handles multiple input
               types for the value_ parameter"
  why bad:    "multiple input types" already reveals the answer (it
               accepts more than one type).
  good topic: "the input-type contract of parse_files_to_codes_mapping
               for its value_ parameter"

- content: "When STRICT mode is disabled and a PDF font's Subtype is
            missing, the system falls back to a default font type."
  bad topic:  "what the font processor does when Subtype is missing and
               STRICT mode is disabled, falling back to defaults"
  why bad:    "falling back to defaults" IS the answer.
  good topic: "the font processor's handling of a missing Subtype
               attribute in non-STRICT mode"

General rule: if any word in your topic appears in the content's
answer/verb/object, delete it. Topic should describe the SITUATION, never
the RESPONSE.

Return a JSON object with a "topic" field.
'''


FACT_TOPIC_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_topic_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
}


PER_FACT_QA_PROMPT = '''\
You are generating a behavioral question about a software codebase. Given a
grounding fact about code behavior, write a question whose answer REQUIRES
knowing this specific fact.

## Grounding Fact
Category: {category}
Topic (what the fact is about): {fact_topic}

## Bug Context
{task_prompt}

## Rules
1. The question must be about BEHAVIOR, REQUIREMENTS, or SPECIFICATIONS
2. Do NOT reference code changes, patches, diffs, or fixes
3. Do NOT mention specific line numbers
4. Frame the question as if asked by a developer investigating the bug
5. The question should be answerable in 1-3 sentences
6. Make the question specific enough that a vague answer would be incorrect

## Anti-Leakage Rules (CRITICAL)

You have been given ONLY the TOPIC of the fact, not its content. This is
intentional: you cannot accidentally leak the answer because you don't know
it. Your job is to rephrase the topic as a question, preserving just enough
context for the question to be well-formed.

7. Do NOT invent or guess the answer. If you find yourself about to write
   "returns None" or "uses 0x7FF" or any specific value — stop, you don't
   know the answer. Just phrase the topic as a question.

8. Self-check: "Could a competent engineer who has NEVER seen this codebase
   answer this correctly by reading only the question?" If yes, the topic
   gave away too much — still, do your best to generalize further. Never
   hallucinate project specifics to fill the gap.

9. Reference CONTEXT (module, function, scenario) so the question is
   well-defined, but don't describe the answer.

Return a JSON object with "question" and "category" fields.
'''


GOLD_ANSWER_PROMPT = '''\
You are generating a gold-standard answer to a behavioral question about code.

## Question
{question}

## Relevant Facts (use ALL of these in your answer)
{facts}

## Repository Context
{repo_context}

## Rules
1. Answer in 1-3 sentences
2. Include information from ALL provided facts
3. Be specific and precise — vague answers are not acceptable
4. Reference concrete behavior, not abstract descriptions

Return a JSON object with an "answer" field.
'''


CODING_QA_ANSWER_PROMPT = '''\
You are a software engineer investigating a bug. Using the provided source
files, documents, and repository context, answer the following question.

## Bug Description
{task_prompt}

## Repository Source Files
{repo_files_text}

## Repository Context
{repo_context}

## Developer Documents
{memory_files_text}

## Question
{question}

Answer using ONLY information from the source files, documents, and repo
context above. If you cannot find enough information, say "insufficient
information."

Return a JSON object with:
- "answer": your answer (1-3 sentences)
- "confidence": 0.0-1.0
- "sources_used": list of document filenames you used
'''


FACT_JUDGE_PROMPT = '''\
You are evaluating whether an answer contains information consistent with a
specific grounding fact.

## Grounding Fact
{fact_content}

## Agent's Answer
{answer}

## Question That Was Asked
{question}

Does the agent's answer demonstrate knowledge of this specific fact?

Score criteria:
- contains_fact=true if the answer reflects the core information in the fact
  (does not need to be word-for-word, but must capture the same meaning)
- contains_fact=false if the answer does not contain this information, or
  contradicts it, or is too vague to confirm

Return a JSON object with "contains_fact" (boolean), "confidence" (0.0-1.0),
and "explanation" (string).
'''


NOISE_JUDGE_PROMPT = '''\
You are checking whether an agent was misled by a distractor fact.

## Distractor (INCORRECT information)
{distractor_content}

## Agent's Answer
{answer}

## Question That Was Asked
{question}

Did the agent incorporate this incorrect distractor information into their
answer? Check if the answer reflects the distractor's wrong claim.

Return a JSON object with "misled" (boolean), "confidence" (0.0-1.0),
and "explanation" (string).
'''


FACT_TEST_SYNTHESIS_PROMPT = '''\
You are writing a minimal pytest test that would prove a specific factual
claim about a codebase. The test will be executed inside a Docker container
that already contains the repository, with the repo installed in editable
mode. Do NOT add pip installs, fixtures, or network calls.

## Grounding Fact to verify
{fact_content}

## Relevant repo file snippets
{evidence_snippets}

Constraints:
- The test must be self-contained in one .py file, ≤ 30 lines.
- Import only from the repo under test (and stdlib / pytest).
- Use `assert` with a clear message; pytest exit code 0 == fact proved.
- Prefer behavioral tests over reading source string — call the
  function/class/method and assert observable behavior.
- If the fact cannot be proven behaviorally (e.g. it describes a
  design decision), return `testable=false` and an empty test.

Return JSON:
{{"testable": bool,
  "test_name": "<short snake_case name>",
  "test_code": "<full .py file contents, including imports and a single test_* function>",
  "why_it_proves_fact": "<1 sentence>"}}
'''


VERIFY_POSITIVE_PROMPT = '''\
You are a technical expert answering a question about software code behavior.
You are given a set of verified facts about the codebase and a specific question.

## Verified Technical Facts
{facts_text}

## Question
{question}

## Instructions
1. Answer the question using ONLY the facts provided above
2. Your answer must be specific and concrete — reference exact behaviors,
   method names, return values, or error types mentioned in the facts
3. If the facts do not contain enough information to answer the question,
   respond with EXACTLY: "INSUFFICIENT_FACTS"
4. Do NOT guess, speculate, or use general programming knowledge
5. Do NOT add information that is not explicitly stated in the facts
6. Keep your answer to 1-3 sentences

Return a JSON object with:
- "answer": your answer (or "INSUFFICIENT_FACTS")
- "confidence": 0.0-1.0 (how confident you are the facts support this answer)
- "facts_used": list of fact IDs you relied on (e.g., ["F1", "F3"])
- "sources_used": [] (empty for this check)
'''


VERIFY_DISTRACTOR_PROMPT = '''\
You are a technical expert answering a question about software code behavior.
You are given a set of technical facts about the codebase and a specific question.

## Technical Facts
{distractor_facts_text}

## Question
{question}

## Instructions
1. Answer the question using ONLY the facts provided above
2. Your answer must be specific and concrete — reference exact behaviors,
   method names, return values, or error types mentioned in the facts
3. If the facts do not contain enough information to answer the question,
   respond with EXACTLY: "INSUFFICIENT_FACTS"
4. Do NOT guess, speculate, or use general programming knowledge
5. Do NOT add information that is not explicitly stated in the facts
6. Keep your answer to 1-3 sentences

Return a JSON object with:
- "answer": your answer (or "INSUFFICIENT_FACTS")
- "confidence": 0.0-1.0
- "facts_used": list of fact IDs you relied on (e.g., ["D1", "D3"])
- "sources_used": [] (empty for this check)
'''


VERIFY_LEAKAGE_PROMPT = '''\
You are a software engineer. Answer the following technical question based
solely on your general programming knowledge. You have NO access to any
specific codebase, documentation, or technical notes about this project.

## Question
{question}

## Instructions
1. Answer based ONLY on general software engineering knowledge
2. If the question references specific functions, classes, or behaviors
   that you cannot determine without project-specific context, respond
   with EXACTLY: "NEED_PROJECT_CONTEXT"
3. Do NOT guess or make assumptions about project-specific implementation
4. If you can partially answer, provide only what you are confident about
5. Keep your answer to 1-3 sentences

Return a JSON object with:
- "answer": your answer (or "NEED_PROJECT_CONTEXT")
- "confidence": 0.0-1.0
- "facts_used": [] (empty for this check)
- "sources_used": [] (empty for this check)
'''


NOTE_TAKING_QA_PROMPT = '''\
You are a software engineer reading developer documents to investigate a bug.
After reading each document, previous documents will be REMOVED from your
context permanently. You must take notes to remember key information.

## Bug Description
{task_prompt}

## Current Document: {filename} (Document {doc_index}/{total_docs})
{document_content}

{previous_notes_section}

## Note-Taking Strategy
{strategy_instruction}

Maximum {budget} tokens for your notes.
Do NOT include general observations or summaries of the document structure.

Return a JSON object with a "notes" field.
'''


EVICTED_ANSWER_QA_PROMPT = '''\
You are a software engineer who has been reading developer documents about a
bug. Earlier documents have been removed from your context. You only have
your notes, repository source files, and the repository context.

## Bug Description
{task_prompt}

## Your Notes from Previous Documents
{notes}

## Repository Source Files
{repo_files_text}

## Repository Context
{repo_context}

## Question
{question}

Answer using your notes, the repository source files, and the repository
context. If your notes don't contain enough information, say "insufficient
information."

Return a JSON object with:
- "answer": your answer (1-3 sentences)
- "confidence": 0.0-1.0
- "sources_used": list of document filenames you referenced in your notes
'''


EVICTED_ANSWER_QA_PROMPT_MEMORY_ONLY = '''\
You are a software engineer who has been reading developer documents about a
bug. Earlier documents have been removed from your context. You only have
your notes and the repository structure.

## Bug Description
{task_prompt}

## Your Notes from Previous Documents
{notes}

## Repository Structure
{repo_context}

## Question
{question}

Answer using your notes and the repository structure. If your notes don't
contain enough information, say "insufficient information."

Return a JSON object with:
- "answer": your answer (1-3 sentences)
- "confidence": 0.0-1.0
- "sources_used": list of document filenames you referenced in your notes
'''


FACT_RETRIEVAL_PROMPT = '''\
You are a software engineer investigating a bug. Read all the provided
documents and repository context, then list ALL specific facts that are
relevant to understanding and fixing this bug.

## Bug Description
{task_prompt}

## Repository Context
{repo_context}

## Developer Documents
{memory_files_text}

## Instructions
List every specific, actionable fact you found that would help fix this bug.
For each fact, note which document file it came from and how relevant it is
(critical / helpful / background).

Do NOT include:
- General observations about document structure
- Vague statements without specific information
- Information about unrelated bugs or features discussed in the documents

Return a JSON object with a "facts" array, each having "content",
"source_file", and "relevance" fields.
'''


# ---------------------------------------------------------------------------
# Coding QA v2 — Multi-fact questions + adversarial distractors
# ---------------------------------------------------------------------------

MULTI_FACT_QA_PROMPT = '''\
Generate ONE behavioral question about a software codebase whose answer
REQUIRES knowing ALL of the following facts together (not just one).

## Facts (ALL required to answer correctly)
{facts_list}

## Bug Context
{task_prompt}

## Other questions already generated (DO NOT repeat or rephrase these)
{existing_questions}

## Rules
1. Question must be about BEHAVIOR, REQUIREMENTS, or SPECIFICATIONS
2. The question should be answerable ONLY if you know ALL the listed facts
3. If only some facts are known, the answer would be incomplete or wrong
4. Make the question DIFFERENT from the existing questions listed above
5. Do NOT reference code changes, patches, diffs, or line numbers
6. Frame the question as if asked by a developer investigating the bug

Return a JSON object with "question" and "category" fields.
'''


ADVERSARIAL_QA_DISTRACTOR_PROMPT = '''\
Generate {num_distractors} plausible but INCORRECT distractor claims for this
question about a software codebase. Each distractor should be:
- Related to the same code area and topic as the question
- Technically plausible (a developer might believe it)
- Specifically WRONG in a way that would lead to an incorrect answer
- Different from the correct answer

## Question
{question}

## Correct Answer Summary
{gold_answer_summary}

## Correct Facts (generate claims that CONTRADICT or MISLEAD from these)
{correct_facts}

## Rules
1. Each distractor should sound like a real developer note or observation
2. Do NOT make distractors obviously wrong (no typos, no absurd claims)
3. Each distractor should suggest a different wrong approach or understanding
4. Vary the type: wrong return values, wrong error handling, wrong API contract,
   wrong design intent
5. Do NOT repeat any correct fact — each distractor must be factually incorrect

Return a JSON object with "distractors" array, each having "id" and "content"
fields.
'''
