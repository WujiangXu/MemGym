"""CUA-specific Structured Summary Memory for GUI agent trajectories.

Uses function calling to produce a structured state summary with
CUA-specific fields: goal, screen state, applications visited,
actions completed/failed, errors, and progress toward the task.
"""

from typing import Any, Dict, List, Optional
import json

import litellm

from ...base import BaseMemoryManager, FilteredContext, register_memory_model

# CUA-specific structured summary fields (replaces SWE's code-centric 17 fields)
CUA_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "create_gui_state_summary",
        "description": "Create a structured summary of GUI agent progress on a desktop task",
        "parameters": {
            "type": "object",
            "properties": {
                "task_goal": {
                    "type": "string",
                    "description": "The user's original task goal and success criteria",
                },
                "current_screen_state": {
                    "type": "string",
                    "description": "Current visible application(s), window layout, and key UI elements",
                },
                "applications_visited": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Applications opened/used (e.g., 'Chrome', 'LibreOffice Calc', 'File Manager')",
                },
                "actions_completed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Successful actions taken (e.g., 'Clicked Settings > Search Engine > Changed to Bing')",
                },
                "actions_failed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actions that failed or had unexpected outcomes",
                },
                "error_messages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Error dialogs, warnings, or unexpected UI states encountered",
                },
                "progress_toward_goal": {
                    "type": "string",
                    "description": "Assessment of how much of the task is complete and what remains",
                },
                "pending_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Next steps or actions still needed to complete the task",
                },
                "key_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important visual observations about the screen state or UI behavior",
                },
            },
            "required": [
                "task_goal",
                "current_screen_state",
                "applications_visited",
                "actions_completed",
                "actions_failed",
                "progress_toward_goal",
                "pending_actions",
            ],
        },
    },
}


def _format_gui_summary(data: Dict) -> str:
    """Format structured GUI summary data into readable text."""
    lines = [f"TASK GOAL: {data.get('task_goal', '?')}"]

    screen = data.get("current_screen_state", "")
    if screen:
        lines.append(f"\nSCREEN STATE: {screen}")

    apps = data.get("applications_visited", [])
    if apps:
        lines.append(f"\nAPPLICATIONS ({len(apps)}):")
        for a in apps:
            lines.append(f"  - {a}")

    completed = data.get("actions_completed", [])
    if completed:
        lines.append(f"\nACTIONS COMPLETED ({len(completed)}):")
        for c in completed:
            lines.append(f"  - {c}")

    failed = data.get("actions_failed", [])
    if failed:
        lines.append(f"\nACTIONS FAILED ({len(failed)}):")
        for f_item in failed:
            lines.append(f"  - {f_item}")

    errors = data.get("error_messages", [])
    if errors:
        lines.append(f"\nERRORS ({len(errors)}):")
        for e in errors:
            lines.append(f"  - {e}")

    progress = data.get("progress_toward_goal", "")
    if progress:
        lines.append(f"\nPROGRESS: {progress}")

    pending = data.get("pending_actions", [])
    if pending:
        lines.append(f"\nPENDING ({len(pending)}):")
        for p in pending:
            lines.append(f"  - {p}")

    observations = data.get("key_observations", [])
    if observations:
        lines.append(f"\nOBSERVATIONS ({len(observations)}):")
        for o in observations:
            lines.append(f"  - {o}")

    return "\n".join(lines)


class CUAStructuredSummaryMemory(BaseMemoryManager):
    """Structured summary memory adapted for CUA GUI agent trajectories.

    Uses function calling to produce a summary with GUI-specific fields.
    This forces the LLM to organize interaction history into actionable
    categories relevant to desktop task completion.
    """

    def __init__(
        self,
        max_size: int = 20,
        keep_first: int = 1,
        summarization_model: str = "gpt-4o-mini",
        condensation_ratio: float = 0.75,
        max_tokens: int = 16000,
        goal: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.max_size = max_size
        self.keep_first = keep_first
        self.summarization_model = summarization_model
        self.condensation_ratio = condensation_ratio
        self.goal = goal

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
            view.append(f"[STRUCTURED GUI STATE]\n{self._summary}")
        view.extend(self._history)

        tokens = self.count_tokens(view)
        return FilteredContext(
            content=view,
            metadata={
                "tokens": tokens,
                "was_compacted": self._condensation_count > 0,
                "condensation_count": self._condensation_count,
                "strategy": "cua_structured_summary",
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
            f"Summarize this GUI agent's interaction history into a structured format.\n\n"
            f"Task goal: {self.goal or '(not specified)'}\n\n"
            f"Previous state:\n{self._summary or '(none)'}\n\n"
            f"New interaction steps being summarized:\n{forgotten_text}\n\n"
            f"Call the create_gui_state_summary function with the updated state."
        )

        try:
            response = litellm.completion(
                model=self.summarization_model,
                messages=[
                    {"role": "system", "content": "You are a GUI interaction summarizer. Extract structured facts about desktop task progress."},
                    {"role": "user", "content": prompt},
                ],
                tools=[CUA_SUMMARY_TOOL],
                tool_choice={"type": "function", "function": {"name": "create_gui_state_summary"}},
                temperature=0.0,
                max_tokens=1000,
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                args_json = tool_calls[0].function.arguments
                self._summary_data = json.loads(args_json)
                self._summary = _format_gui_summary(self._summary_data)
            else:
                # Fallback to plain text
                self._summary = response.choices[0].message.content or forgotten_text[:1000]

        except Exception:
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
            parts.append(f"[STRUCTURED GUI STATE]\n{self._summary}")
        parts.extend(str(item) for item in self._history)
        return "\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "condensation_count": self._condensation_count,
            "history_size": len(self._history),
            "has_summary": bool(self._summary),
            "summary_fields": list(self._summary_data.keys()) if self._summary_data else [],
            "compaction_count": self._condensation_count,
        }


register_memory_model("cua_structured_summary", CUAStructuredSummaryMemory)
