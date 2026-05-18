"""LLM-based synthesis of per-fact pytest tests.

Given a GroundingFact whose `discoverable=True`, call an LLM with the
fact content + a small slice of the repo and produce a single-file
pytest test. Synthesis is best-effort: if the fact can't be proved
behaviorally, the LLM returns `testable=false` and we skip it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...llm.client import LLMClient
from ...llm.prompts import FACT_TEST_SYNTHESIS_PROMPT
from ...types.schemas import GroundingFact


FACT_TEST_SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "fact_test",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "testable": {"type": "boolean"},
                "test_name": {"type": "string"},
                "test_code": {"type": "string"},
                "why_it_proves_fact": {"type": "string"},
            },
            "required": [
                "testable", "test_name", "test_code", "why_it_proves_fact",
            ],
            "additionalProperties": False,
        },
    },
}


def _format_evidence(
    repo_files: Dict[str, str],
    evidence_files: Optional[List[str]],
    max_chars_per_file: int = 6000,
) -> str:
    """Stitch a snippet for each evidence file (fallback: first 2 files)."""
    if not repo_files:
        return "(no repo files available)"
    candidates = evidence_files or []
    if not candidates:
        candidates = list(repo_files)[:2]
    parts: List[str] = []
    for fp in candidates:
        content = repo_files.get(fp)
        if not content:
            continue
        parts.append(f"### {fp}\n```python\n{content[:max_chars_per_file]}\n```")
    return "\n\n".join(parts) or "(no matching evidence files)"


def synthesize_fact_test(
    fact: GroundingFact,
    repo_files: Dict[str, str],
    client: LLMClient,
    evidence_files: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a dict with test_name / test_code / testable, or None on failure.

    Returns None only on LLM call failure — a non-testable fact returns
    a dict with `testable=False` and the caller should skip it.
    """
    prompt = FACT_TEST_SYNTHESIS_PROMPT.format(
        fact_content=fact.content,
        evidence_snippets=_format_evidence(repo_files, evidence_files),
    )
    try:
        return client.get_json_completion(
            prompt=prompt,
            response_format=FACT_TEST_SYNTHESIS_SCHEMA,
            temperature=0.0,
        )
    except Exception:
        return None
