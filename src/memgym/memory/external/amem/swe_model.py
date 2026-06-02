"""
SWEAMemModel: A-mem adapted for SWE-bench code tasks.

SWE-bench involves software engineering tasks where agents:
- Read and analyze code files
- Make edits to fix bugs or add features
- Run commands and tests
- Debug errors

This model is optimized for:
- Code-focused observation processing
- File operation tracking
- Error and debugging context
- Code-aware query generation
"""

from typing import Any, Dict, List, Optional

# Package-relative import; see model.py for the rationale (the previous
# sys.path / `from base import ...` form only worked in dev venvs and broke
# in a clean `pip install -e .[memory-eval]` install).
from ...base import Context, MemoryAction, register_memory_model

from .model import AMemMemoryModel
from .system import AgenticMemorySystem


# SWE-specific query generation prompt
SWE_QUERY_PROMPT = """You are helping a software engineering AI agent retrieve relevant context for debugging/coding.

Current task: {task_description}

Recent actions:
{recent_observations}

The agent needs to find relevant past information to continue working on the code task.
Generate a retrieval query focusing on:
1. File contents and code structures that were examined
2. Error messages and stack traces encountered
3. Previous edit attempts and their outcomes
4. Test results and debugging information
5. Repository structure and file locations

Return a JSON object with:
- query: A search query to find relevant code context (1-3 sentences)
- reasoning: Brief explanation of your query choice
"""


class SWEAMemModel(AMemMemoryModel):
    """
    A-mem model adapted for SWE-bench code tasks.

    Extends AMemMemoryModel with:
    - Code operation tracking (read, edit, command, test)
    - File-aware memory tagging
    - Error-focused retrieval
    - Code structure understanding

    Config options (in addition to AMemMemoryModel):
        track_files: Track file operations (default: True)
        track_errors: Prioritize error tracking (default: True)
        max_file_content: Max chars of file content to store (default: 2000)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SWE A-mem model."""
        config = config or {}

        # SWE-specific defaults
        config.setdefault("recent_turns", 3)   # Fewer recent turns (steps are longer)
        config.setdefault("retrieve_k", 5)     # Focused retrieval
        config.setdefault("max_context_tokens", 6000)  # Lower threshold for code

        super().__init__(config)

        # SWE-specific config
        self._track_files = self.config.get("track_files", True)
        self._track_errors = self.config.get("track_errors", True)
        self._max_file_content = self.config.get("max_file_content", 2000)

        # Track code operations
        self._step_count = 0
        self._operation_counts = {
            "read": 0,
            "edit": 0,
            "command": 0,
            "test": 0,
            "error": 0,
            "other": 0
        }
        self._files_touched: Dict[str, int] = {}  # file -> access count
        self._errors_encountered: List[str] = []

    def process_observation(
        self,
        observation: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryAction:
        """
        Process a code operation step.

        Args:
            observation: Step content (file content, command output, etc.)
            metadata: Should include 'operation' (read/edit/command/test/error)
                     and optionally 'file_path'

        Returns:
            MemoryAction with code tracking info
        """
        metadata = metadata or {}
        operation = metadata.get("operation", "other")
        file_path = metadata.get("file_path")

        # Track operation counts
        if operation in self._operation_counts:
            self._operation_counts[operation] += 1
        else:
            self._operation_counts["other"] += 1

        # Track files touched
        if file_path and self._track_files:
            self._files_touched[file_path] = self._files_touched.get(file_path, 0) + 1

        # Process content
        content = str(observation)

        # Truncate file content if too long
        if operation == "read" and len(content) > self._max_file_content:
            content = content[:self._max_file_content] + "\n...(truncated)"

        # Tag content with operation type
        if operation == "read" and file_path:
            tagged_content = f"[FILE READ: {file_path}]\n{content}"
        elif operation == "edit" and file_path:
            tagged_content = f"[FILE EDIT: {file_path}]\n{content}"
        elif operation == "command":
            command = metadata.get("command", "")
            tagged_content = f"[COMMAND: {command}]\n{content}"
        elif operation == "test":
            tagged_content = f"[TEST OUTPUT]\n{content}"
        elif operation == "error":
            tagged_content = f"[ERROR]\n{content}"
            if self._track_errors:
                self._errors_encountered.append(content[:500])  # Store truncated error
        else:
            tagged_content = content

        # Increment step count
        self._step_count += 1
        metadata["step_number"] = self._step_count
        metadata["operation"] = operation

        # Store in memory
        timestamp = metadata.get("timestamp")
        note_id = self._memory_system.add_note(tagged_content, time=timestamp)

        self._observation_count += 1
        self._last_observation = content
        self._current_tokens += self._memory_system.estimate_tokens(tagged_content)

        self._last_action = MemoryAction(
            action_type="store_code_operation",
            metadata={
                "note_id": note_id,
                "operation": operation,
                "file_path": file_path,
                "step_number": self._step_count,
                "total_memories": len(self._memory_system.memories),
                "observation_count": self._observation_count,
                "current_tokens": self._current_tokens,
                **metadata
            }
        )
        return self._last_action

    def process_file_read(self, file_path: str, content: str, metadata: Optional[Dict] = None) -> MemoryAction:
        """Convenience method for file reads."""
        metadata = metadata or {}
        metadata["operation"] = "read"
        metadata["file_path"] = file_path
        return self.process_observation(content, metadata)

    def process_file_edit(self, file_path: str, edit_description: str, metadata: Optional[Dict] = None) -> MemoryAction:
        """Convenience method for file edits."""
        metadata = metadata or {}
        metadata["operation"] = "edit"
        metadata["file_path"] = file_path
        return self.process_observation(edit_description, metadata)

    def process_command(self, command: str, output: str, metadata: Optional[Dict] = None) -> MemoryAction:
        """Convenience method for command execution."""
        metadata = metadata or {}
        metadata["operation"] = "command"
        metadata["command"] = command
        return self.process_observation(output, metadata)

    def process_test_result(self, result: str, metadata: Optional[Dict] = None) -> MemoryAction:
        """Convenience method for test results."""
        metadata = metadata or {}
        metadata["operation"] = "test"
        return self.process_observation(result, metadata)

    def process_error(self, error: str, metadata: Optional[Dict] = None) -> MemoryAction:
        """Convenience method for errors."""
        metadata = metadata or {}
        metadata["operation"] = "error"
        return self.process_observation(error, metadata)

    def get_files_touched(self) -> Dict[str, int]:
        """Get dict of files touched and their access counts."""
        return self._files_touched.copy()

    def get_recent_errors(self, n: int = 5) -> List[str]:
        """Get most recent errors encountered."""
        return self._errors_encountered[-n:]

    def get_code_summary(self) -> Dict[str, Any]:
        """Get summary of code operations."""
        return {
            "step_count": self._step_count,
            "operation_counts": self._operation_counts.copy(),
            "files_touched": len(self._files_touched),
            "errors_encountered": len(self._errors_encountered),
            "total_memories": len(self._memory_system.memories),
            "current_tokens": self._current_tokens,
            "compressed": self._current_tokens >= self._max_context_tokens
        }

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self._step_count = 0
        self._operation_counts = {
            "read": 0,
            "edit": 0,
            "command": 0,
            "test": 0,
            "error": 0,
            "other": 0
        }
        self._files_touched = {}
        self._errors_encountered = []

    def get_stats(self) -> Dict[str, Any]:
        """Get SWE-specific statistics."""
        stats = super().get_stats()
        stats.update({
            "step_count": self._step_count,
            "operation_counts": self._operation_counts.copy(),
            "files_touched": len(self._files_touched),
            "errors_encountered": len(self._errors_encountered),
            "track_files": self._track_files,
            "track_errors": self._track_errors,
            "max_file_content": self._max_file_content
        })
        return stats


# Register with MemGym
register_memory_model("swe-amem", SWEAMemModel)
register_memory_model("swe_amem", SWEAMemModel)
