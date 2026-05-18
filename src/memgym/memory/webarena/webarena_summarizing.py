"""
LLM summarizing memory manager for WebArena (Web GUI) trajectories.

Architecture mirrors `memory/llm_summarizing.py` (the proven SWE-bench port of
OpenHands LLMSummarizingCondenser) line-for-line:

- CondensationRecord dataclass persists condensation decisions
- build_view() rebuilds the condensed view from full history + stored records
- should_condense checks VIEW length, not raw history length
- manage_context() triggers condensation when len(view) > max_size OR
  condensation_requested
- keep_first always-preserved prefix
- condensation_ratio controls eviction aggressiveness
- Persistence: stored records survive across manage_context() calls

**Only the prompt changes**: the SWE-bench prompt is tailored for code edit
state (CODE_STATE / TESTS / CHANGES / DEPS / VERSION_CONTROL_STATUS). Those
fields are useless for browsing trajectories — a web GUI agent's critical
state is navigation, forms, DOM elements interacted with, and data extracted
from pages. `WEBARENA_SUMMARIZATION_PROMPT` replaces the SWE schema with the
7-section web schema documented in the `ethereal-meandering-bee` plan.

This adapter also adds two constructor fields (following the IR precedent at
`memory/ir/ir_summarizing.py`):

- task_instruction: the task the agent was asked to complete (analogous to
  IR's `question` field)
- app_name: which webarena-infinity app this trajectory runs in (e.g.
  `gmail`, `gitlab`), surfaced to the prompt so the summarizer can use
  app-specific hints in the future
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from ..base import BaseMemoryManager, FilteredContext, register_memory_model


# =============================================================================
# Web-GUI-tailored summarization prompt
# =============================================================================

WEBARENA_SUMMARIZATION_PROMPT = """You are maintaining a context-aware state summary for a web GUI agent.
The agent is controlling a browser (via Playwright/Chromium) to complete a task.
You will be given a list of events corresponding to observations and actions taken by the agent,
and the most recent previous summary if one exists.

Some of the earlier observations and actions are being EVICTED from the agent's context window —
your summary is the only record the agent will have of them going forward. The summary must
preserve the critical browsing state so the agent can continue seamlessly.

TASK: {task_instruction}
APP: {app_name}

Track the following sections. Preserve exact URLs, element IDs, field names, and values.
Do NOT hallucinate elements, pages, or actions that were not in the events.

CURRENT_LOCATION: (Current page URL and title; how we got here — the entry path taken)

NAVIGATION_HISTORY: (Pages visited so far in causal order, one line each. URL + brief purpose)

INTERACTED_ELEMENTS: (What the agent has already done on each page — forms filled, buttons
clicked, links followed, text typed, items selected. Include enough detail that the agent
does NOT redo them. Format as: element_id or selector → action → value/result)

EXTRACTED_DATA: (Facts the agent observed from page content that are needed downstream —
search results, account state, table rows, error messages, prices, dates, names, etc.)

BLOCKERS: (Errors, missing elements, timeouts, captchas, permission denials, 404s, dead-ends
the agent hit. Include what was tried and what failed so the agent can avoid repeating.)

TASK_PROGRESS: (Break the original task into subtasks and mark each as DONE / IN_PROGRESS / TODO.
Start with a completion counter: "X of Y items/steps complete".
For batch operations on multiple items, track EACH item individually — e.g. if the task
is "delete emails from A, B, and C", list each sender's status separately.
This section is CRITICAL — the agent uses it to decide whether to continue or stop.)

NEXT_STEP: (What the agent should do next based on the TASK_PROGRESS above.
Must be a concrete action — "click [ID]", "type [ID] value", etc.)

PRIORITIZE:
1. Exact identifiers over prose ("clicked #submit-btn" not "submitted the form")
2. Task progress and remaining items — the agent MUST know what is not yet done
3. State that cannot be re-derived by re-visiting the current page
4. Blockers encountered — avoid repeating failed approaches

SKIP: Irrelevant page chrome (nav menus, footers) and decorative content

Example:

TASK: Post a new issue titled "login broken on mobile" to the gitlab project "myorg/webapp"
APP: gitlab

CURRENT_LOCATION: /myorg/webapp/-/issues/new, title "New Issue · myorg/webapp · GitLab"
 — reached from /myorg/webapp → "Issues" tab → "New issue" button

NAVIGATION_HISTORY:
  /dashboard → landing after login
  /myorg/webapp → project page, confirmed project exists
  /myorg/webapp/-/issues → issues list, 12 open issues
  /myorg/webapp/-/issues/new → new-issue form

INTERACTED_ELEMENTS:
  #issue_title → type → "login broken on mobile"
  #issue_description → type → (empty so far, cursor placed)

EXTRACTED_DATA:
  project_id: myorg/webapp
  default_assignee: @current_user
  label_options: ["bug","enhancement","question","critical","blocked"]

BLOCKERS: none

TASK_PROGRESS: 2 of 5 steps complete
  DONE: navigated to new-issue form, filled title
  IN_PROGRESS: filling description
  TODO: add "bug" label, click "Submit issue", verify created

NEXT_STEP: type description, apply "bug" label, click #new_issue submit button"""


# Summary marker used to identify summary messages in the view
SUMMARY_MARKER = "[Summary]"


# =============================================================================
# CondensationRecord — mirrors llm_summarizing.CondensationRecord
# =============================================================================

@dataclass
class CondensationRecord:
    """Record of a condensation event, stored in memory manager state."""
    forgotten_events_start: int
    forgotten_events_end: int
    summary: str
    summary_offset: int


# =============================================================================
# build_view — mirrors llm_summarizing.build_view
# =============================================================================

def build_view(
    messages: List[Any],
    condensations: List[CondensationRecord]
) -> Tuple[List[Any], List[int]]:
    """Build condensed view from full message history + stored condensation records."""
    forgotten = set()
    for record in condensations:
        forgotten.update(range(record.forgotten_events_start, record.forgotten_events_end))

    kept_indices = [i for i in range(len(messages)) if i not in forgotten]
    kept_messages = [messages[i] for i in kept_indices]

    if condensations:
        last = condensations[-1]
        summary_msg = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {last.summary}"
        }
        offset = last.summary_offset
        view_messages = kept_messages[:offset] + [summary_msg] + kept_messages[offset:]
        view_indices = kept_indices[:offset] + [-1] + kept_indices[offset:]
        return view_messages, view_indices

    return kept_messages, kept_indices


# =============================================================================
# Helper functions
# =============================================================================

def _truncate_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]


def _is_summary_message(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    content = msg.get("content", "")
    return isinstance(content, str) and SUMMARY_MARKER in content


def _get_message_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content", "")
        return str(content) if content else ""
    return str(msg)


# =============================================================================
# WebArenaSummarizingMemory — web GUI variant of LLMSummarizingMemory
# =============================================================================

class WebArenaSummarizingMemory(BaseMemoryManager):
    """
    LLM-summarizing memory manager for web GUI agent trajectories.

    Architecture identical to `LLMSummarizingMemory` (the proven SWE-bench
    port of OpenHands LLMSummarizingCondenser). Only the summarization prompt
    is web-task-tailored, plus two extra constructor fields (task_instruction,
    app_name) used to format the prompt header.

    Args:
        task_instruction: The user's task description, formatted into the
            summarization prompt header. Analogous to IR's `question` field.
        app_name: Which webarena-infinity app (gmail/gitlab/etc.). Surfaced
            to the summarizer so it can use app-specific hints.
        max_size: Maximum number of events in view before condensation.
        keep_first: Number of initial events to always keep.
        max_event_length: Maximum character length per event in prompt.
        summarization_model: LLM model for summarization.
        condensation_ratio: Fraction of max_size the view is compressed to.
        max_tokens: Token budget for the final view (inherited from base).
        max_token_size: If set, trigger condensation by token count instead
            of message count.
        tokenizer_name: Tokenizer name for counting.
    """

    def __init__(
        self,
        task_instruction: str = "",
        app_name: str = "",
        max_size: int = 100,
        keep_first: int = 1,
        max_event_length: int = 20_000,
        summarization_model: str = "gpt-4o-mini",
        condensation_ratio: float = 0.75,
        max_tokens: int = 4000,
        max_token_size: Optional[int] = None,
        tokenizer_name: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ):
        if keep_first >= int(max_size * condensation_ratio):
            raise ValueError(
                f"keep_first ({keep_first}) must be less than half of max_size ({max_size})"
            )
        if keep_first < 0:
            raise ValueError(f"keep_first ({keep_first}) cannot be negative")
        if max_size < 1:
            raise ValueError(f"max_size ({max_size}) cannot be non-positive")

        super().__init__(max_tokens, tokenizer_name, config)

        self.task_instruction = task_instruction
        self.app_name = app_name
        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.max_token_size = max_token_size

        self._condensations: List[CondensationRecord] = []
        self._condensation_requested: bool = False

        self._summarizer_prompt_tokens: int = 0
        self._summarizer_completion_tokens: int = 0
        self._last_summarizer_prompt_tokens: int = 0
        self._last_summarizer_completion_tokens: int = 0

    def request_condensation(self) -> None:
        """Force condensation on next manage_context() call."""
        self._condensation_requested = True

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilteredContext:
        """Manage context with rolling LLM summarization (web-tailored prompt)."""
        messages = list(original_context) + [current_observation]

        view, view_to_original = build_view(messages, self._condensations)
        view_tokens = self.count_tokens(view)

        if self.max_token_size is not None:
            needs_condensation = (
                view_tokens > self.max_token_size
                or self._condensation_requested
            )
        else:
            needs_condensation = (
                len(view) > self.max_size
                or self._condensation_requested
            )

        if not needs_condensation:
            self._update_token_tracking(view_tokens)
            return FilteredContext(
                content=view,
                metadata={
                    "tokens": view_tokens,
                    "original_tokens": self.count_tokens(messages),
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": self.count_tokens(messages) / max(1, view_tokens),
                    "strategy": "webarena_summarizing",
                    "view_size": len(view),
                    "raw_size": len(messages),
                    "condensation_count": len(self._condensations),
                    "step": self._step_count
                }
            )

        self._condensation_requested = False

        head = view[:self.keep_first]
        target_size = int(self.max_size * self.condensation_ratio)

        events_from_tail = target_size - len(head) - 1
        if events_from_tail < 1:
            events_from_tail = 1
        tail = view[-events_from_tail:]

        summary_event_content = ""
        if self.keep_first < len(view) and _is_summary_message(view[self.keep_first]):
            raw = _get_message_content(view[self.keep_first])
            summary_event_content = raw.replace(SUMMARY_MARKER, "").strip()

        middle_start = self.keep_first
        middle_end = len(view) - events_from_tail
        forgotten_events = []
        forgotten_view_indices = []
        for vi in range(middle_start, middle_end):
            if not _is_summary_message(view[vi]):
                forgotten_events.append(view[vi])
                forgotten_view_indices.append(vi)

        if not forgotten_events:
            self._update_token_tracking(view_tokens)
            return FilteredContext(
                content=view,
                metadata={
                    "tokens": view_tokens,
                    "original_tokens": self.count_tokens(messages),
                    "was_compacted": len(self._condensations) > 0,
                    "direct_messages": True,
                    "compression_ratio": 1.0,
                    "strategy": "webarena_summarizing",
                    "step": self._step_count
                }
            )

        summary = self._call_llm_summarize(forgotten_events, summary_event_content)

        original_forgotten_indices = [
            view_to_original[vi] for vi in forgotten_view_indices
            if vi < len(view_to_original) and view_to_original[vi] >= 0
        ]

        if original_forgotten_indices:
            self._condensations.append(CondensationRecord(
                forgotten_events_start=min(original_forgotten_indices),
                forgotten_events_end=max(original_forgotten_indices) + 1,
                summary=summary,
                summary_offset=self.keep_first,
            ))

        self._compaction_count += 1

        summary_message = {
            "role": "user",
            "content": f"{SUMMARY_MARKER} {summary}"
        }
        result = head + [summary_message] + tail

        filtered_tokens = self.count_tokens(result)
        original_tokens = self.count_tokens(messages)
        self._update_token_tracking(filtered_tokens)

        return FilteredContext(
            content=result,
            metadata={
                "tokens": filtered_tokens,
                "original_tokens": original_tokens,
                "was_compacted": True,
                "direct_messages": True,
                "compression_ratio": original_tokens / max(1, filtered_tokens),
                "strategy": "webarena_summarizing",
                "summarization_count": self._compaction_count,
                "condensation_count": len(self._condensations),
                "messages_before": len(view),
                "messages_after": len(result),
                "forgotten_events": len(forgotten_events),
                "view_size": len(result),
                "raw_size": len(messages),
                "step": self._step_count,
                "condensation_event": {
                    "summary_text": summary,
                    "forgotten_count": len(forgotten_events),
                    "summarization_model": self.summarization_model,
                    "summarization_time_seconds": getattr(self, '_last_summarization_time', 0),
                    "summarizer_prompt_tokens": self._last_summarizer_prompt_tokens,
                    "summarizer_completion_tokens": self._last_summarizer_completion_tokens,
                }
            }
        )

    def _call_llm_summarize(
        self,
        forgotten_events: List[Any],
        previous_summary: str
    ) -> str:
        """Call LLM to summarize forgotten events with web-task-tailored prompt."""
        # Format prompt header with task instruction + app name
        prompt = WEBARENA_SUMMARIZATION_PROMPT.format(
            task_instruction=self.task_instruction or "(not provided)",
            app_name=self.app_name or "(unknown)",
        ) + "\n\n"

        truncated_summary = _truncate_content(previous_summary, self.max_event_length)
        prompt += f"<PREVIOUS SUMMARY>\n{truncated_summary}\n</PREVIOUS SUMMARY>\n"
        prompt += "\n\n"

        for i, msg in enumerate(forgotten_events):
            event_content = _get_message_content(msg)
            truncated = _truncate_content(event_content, self.max_event_length)
            prompt += f"<EVENT id={i}>\n{truncated}\n</EVENT>\n"

        prompt += "Now summarize the events using the rules above."

        try:
            t0 = time.time()
            response = completion(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}]
            )
            self._last_summarization_time = time.time() - t0

            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                self._last_summarizer_prompt_tokens = (
                    getattr(response.usage, 'prompt_tokens', 0)
                    or getattr(response.usage, 'input_tokens', 0) or 0
                )
                self._last_summarizer_completion_tokens = (
                    getattr(response.usage, 'completion_tokens', 0)
                    or getattr(response.usage, 'output_tokens', 0) or 0
                )
                self._summarizer_prompt_tokens += self._last_summarizer_prompt_tokens
                self._summarizer_completion_tokens += self._last_summarizer_completion_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._last_summarization_time = 0
            self._last_summarizer_prompt_tokens = 0
            self._last_summarizer_completion_tokens = 0
            print(f"Warning: WebArena LLM summarization failed: {e}")
            return previous_summary or "Summary unavailable due to error."

    def reset(self) -> None:
        """Reset memory state for new episode."""
        self._condensations.clear()
        self._condensation_requested = False
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0
        self._summarizer_prompt_tokens = 0
        self._summarizer_completion_tokens = 0
        self._last_summarizer_prompt_tokens = 0
        self._last_summarizer_completion_tokens = 0

    def get_condensation_history(self) -> List[Dict[str, Any]]:
        """Export condensation records for trajectory logging."""
        return [
            {
                "forgotten_events_start": r.forgotten_events_start,
                "forgotten_events_end": r.forgotten_events_end,
                "summary": r.summary,
                "summary_offset": r.summary_offset,
            }
            for r in self._condensations
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            "summarization_model": self.summarization_model,
            "max_size": self.max_size,
            "keep_first": self.keep_first,
            "max_event_length": self.max_event_length,
            "condensation_count": len(self._condensations),
            "summarizer_total_prompt_tokens": self._summarizer_prompt_tokens,
            "summarizer_total_completion_tokens": self._summarizer_completion_tokens,
            "summarizer_total_tokens": self._summarizer_prompt_tokens + self._summarizer_completion_tokens,
            "task_instruction": self.task_instruction,
            "app_name": self.app_name,
        })
        return base_stats


# Register in memory model registry
register_memory_model("webarena_summarizing", WebArenaSummarizingMemory)
