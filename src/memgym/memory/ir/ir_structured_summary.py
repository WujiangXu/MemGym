"""IR-specific Structured Summary Memory for research trajectories.

Uses function calling to produce a structured state summary with
IR-specific fields: research question, entities, bridge connections,
key evidence, candidate answer, contradictions, and unresolved gaps.
"""

from typing import Any, Dict, List, Optional
import json

import litellm

from ..base import BaseMemoryManager, FilteredContext, register_memory_model

# IR-specific structured summary fields
IR_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "create_research_summary",
        "description": "Create a structured summary of research progress",
        "parameters": {
            "type": "object",
            "properties": {
                "research_question": {
                    "type": "string",
                    "description": "The main question being investigated",
                },
                "entities_discovered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific entities (people, places, orgs, dates) found so far",
                },
                "bridge_connections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Facts connecting one entity to another (e.g., 'X was born in Y')",
                },
                "key_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Facts that directly help answer the question",
                },
                "candidate_answer": {
                    "type": "string",
                    "description": "Best current answer based on evidence so far",
                },
                "contradictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Conflicting information found across sources",
                },
                "unresolved_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What still needs to be searched or verified",
                },
            },
            "required": [
                "research_question",
                "entities_discovered",
                "bridge_connections",
                "key_evidence",
                "candidate_answer",
                "contradictions",
                "unresolved_gaps",
            ],
        },
    },
}


def _format_research_summary(data: Dict) -> str:
    """Format structured summary data into readable text."""
    lines = [f"RESEARCH QUESTION: {data.get('research_question', '?')}"]

    entities = data.get("entities_discovered", [])
    if entities:
        lines.append(f"\nENTITIES ({len(entities)}):")
        for e in entities:
            lines.append(f"  - {e}")

    bridges = data.get("bridge_connections", [])
    if bridges:
        lines.append(f"\nBRIDGE CONNECTIONS ({len(bridges)}):")
        for b in bridges:
            lines.append(f"  - {b}")

    evidence = data.get("key_evidence", [])
    if evidence:
        lines.append(f"\nKEY EVIDENCE ({len(evidence)}):")
        for e in evidence:
            lines.append(f"  - {e}")

    answer = data.get("candidate_answer", "")
    if answer:
        lines.append(f"\nCANDIDATE ANSWER: {answer}")

    contradictions = data.get("contradictions", [])
    if contradictions:
        lines.append(f"\nCONTRADICTIONS ({len(contradictions)}):")
        for c in contradictions:
            lines.append(f"  - {c}")

    gaps = data.get("unresolved_gaps", [])
    if gaps:
        lines.append(f"\nUNRESOLVED ({len(gaps)}):")
        for g in gaps:
            lines.append(f"  - {g}")

    return "\n".join(lines)


class IRStructuredSummary(BaseMemoryManager):
    """Structured summary memory adapted for IR research trajectories.

    Uses function calling to produce a summary with specific IR fields.
    This forces the LLM to organize information into actionable categories.
    """

    def __init__(
        self,
        max_size: int = 10,
        keep_first: int = 1,
        summarization_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        condensation_ratio: float = 0.75,
        max_tokens: int = 4000,
        question: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.max_size = max_size
        self.keep_first = keep_first
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.question = question

        self._history: List[Any] = []
        self._summary = ""
        self._summary_data: Dict = {}
        self._condensation_count = 0

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        if current_observation:
            self._history.append(current_observation)

        if len(self._history) > self.max_size:
            self._condense()

        view = []
        if self._summary:
            view.append(f"[STRUCTURED RESEARCH STATE]\n{self._summary}")
        view.extend(self._history)

        tokens = self.count_tokens(view)
        return FilteredContext(
            content=view,
            metadata={
                "tokens": tokens,
                "was_compacted": self._condensation_count > 0,
                "condensation_count": self._condensation_count,
                "strategy": "ir_structured",
                "history_size": len(self._history),
            },
        )

    def _condense(self):
        """Summarize older entries using function calling for structured output."""
        target = int(self.max_size * self.condensation_ratio)
        keep_count = max(self.keep_first, target)

        forgotten = self._history[:len(self._history) - keep_count]
        kept = self._history[len(self._history) - keep_count:]

        if not forgotten:
            return

        forgotten_text = "\n\n".join(str(item) for item in forgotten)

        prompt = (
            f"Summarize this research progress into a structured format.\n\n"
            f"Research question: {self.question}\n\n"
            f"Previous state:\n{self._summary or '(none)'}\n\n"
            f"New information being summarized:\n{forgotten_text}\n\n"
            f"Call the create_research_summary function with the updated state."
        )

        try:
            response = litellm.completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": "You are a research summarizer. Extract structured facts."},
                    {"role": "user", "content": prompt},
                ],
                tools=[IR_SUMMARY_TOOL],
                tool_choice={"type": "function", "function": {"name": "create_research_summary"}},
                temperature=0.0,
                max_tokens=1000,
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                args_json = tool_calls[0].function.arguments
                self._summary_data = json.loads(args_json)
                self._summary = _format_research_summary(self._summary_data)
            else:
                # Fallback to plain text
                self._summary = response.choices[0].message.content or forgotten_text[:1000]

        except Exception as e:
            self._summary = (self._summary + "\n" + forgotten_text)[:2000]

        self._history = kept
        self._condensation_count += 1

    def reset(self) -> None:
        self._history = []
        self._summary = ""
        self._summary_data = {}
        self._condensation_count = 0

    def get_notes_text(self) -> str:
        """Return current memory state as text."""
        parts = []
        if self._summary:
            parts.append(f"[STRUCTURED RESEARCH STATE]\n{self._summary}")
        parts.extend(str(item) for item in self._history)
        return "\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "condensation_count": self._condensation_count,
            "history_size": len(self._history),
            "has_summary": bool(self._summary),
            "summary_fields": list(self._summary_data.keys()) if self._summary_data else [],
        }


register_memory_model("ir_structured", IRStructuredSummary)
