"""
SWE-bench specific agents.

Contains:
- SWEAgent: LLM-based agent for patch generation
- SWEAgentTracker: Tracks patch submissions and code generation actions
"""

from typing import Any, Dict, Optional


from memgym.agents.base import BaseAgent, AgentAction, register_agent


class SWEAgentTracker(BaseAgent):
    """
    Tracker agent for SWE-bench coding tasks.

    Tracks:
    - Number of patches submitted
    - Total patch size
    - Patch classification (valid diff vs other)

    This agent records patch submissions for trajectory analysis and training.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._patches_submitted = 0
        self._total_patch_size = 0

    def act(self, context: Any) -> AgentAction:
        """
        Return last recorded action (patches generated externally by LLM).

        Args:
            context: FilteredContext from memory manager

        Returns:
            Last recorded action or placeholder
        """
        if self._last_action is not None:
            return self._last_action

        return AgentAction(
            action="",
            metadata={
                "action_type": "pending",
                "context_length": len(str(context)) if context else 0,
                "patches_submitted": self._patches_submitted
            }
        )

    def set_action_from_external(
        self,
        action: Any,
        context: Any = None
    ) -> AgentAction:
        """
        Record an externally-generated patch action.

        Args:
            action: The patch or action that was taken
            context: The context that was provided

        Returns:
            AgentAction with the action and metadata
        """
        action_str = str(action)
        is_patch = self._is_patch(action_str)

        if is_patch:
            self._patches_submitted += 1
            self._total_patch_size += len(action_str)

        self._last_action = AgentAction(
            action=action_str,
            metadata={
                "action_type": "patch" if is_patch else "other",
                "context_length": len(str(context)) if context else 0,
                "patch_size": len(action_str) if is_patch else 0,
                "patches_submitted": self._patches_submitted
            }
        )

        self._record_action(self._last_action)
        return self._last_action

    def _is_patch(self, action: str) -> bool:
        """Check if action looks like a patch."""
        action_lower = action.lower()
        return "patch" in action_lower or "diff" in action_lower or "---" in action

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": self.__class__.__name__,
            "patches_submitted": self._patches_submitted,
            "total_patch_size": self._total_patch_size,
            "avg_patch_size": self._total_patch_size / max(1, self._patches_submitted)
        }

    def reset(self) -> None:
        """Reset agent state."""
        self._patches_submitted = 0
        self._total_patch_size = 0
        self._last_action = None
        self._action_history.clear()


# Legacy alias
SWEReasoningModel = SWEAgentTracker


class SWEAgent:
    """
    Agent for SWE-bench that generates patches using LLM.

    This provides a clean interface that can be wrapped with
    memory-aware filtering for context management.

    Tracks patch statistics:
    - Number of patches submitted
    - Total patch size
    - Average patch size
    """

    def __init__(
        self,
        model: str,
        model_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SWE agent.

        Args:
            model: LLM model name (LiteLLM format)
            model_args: Additional arguments for LLM calls
        """
        self.model = model
        self.model_args = model_args or {}

        # Statistics tracking
        self._patches_generated = 0
        self._total_patch_size = 0

    def generate_patch(
        self,
        problem_statement: str,
        repo: str,
        hints: str = "",
        base_commit: str = "",
        version: str = "",
        context: Optional[str] = None
    ) -> str:
        """
        Generate a patch for the given problem.

        Args:
            problem_statement: The issue description
            repo: Repository name
            hints: Optional hints for fixing the issue
            base_commit: Base commit hash
            version: Repository version
            context: Optional context (from memory manager)

        Returns:
            Generated patch as string
        """
        try:
            import litellm

            # Build prompt
            prompt = self._build_prompt(
                problem_statement=problem_statement,
                repo=repo,
                hints=hints,
                base_commit=base_commit,
                version=version
            )

            # If context provided (from memory), prepend it
            if context:
                final_prompt = f"{context}\n\n{prompt}"
            else:
                final_prompt = prompt

            # Call LLM
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert software engineer who generates git patches to fix bugs."},
                    {"role": "user", "content": final_prompt}
                ],
                **self.model_args
            )

            patch = response.choices[0].message.content.strip()
            return patch

        except Exception as e:
            return f"# Error generating patch: {str(e)}"

    def _build_prompt(
        self,
        problem_statement: str,
        repo: str,
        hints: str = "",
        base_commit: str = "",
        version: str = ""
    ) -> str:
        """Build prompt for patch generation."""
        prompt = f"""You are an expert software engineer tasked with fixing a bug in the {repo} repository.

**Problem Statement:**
{problem_statement}

**Hints:**
{hints if hints else 'No hints provided.'}

**Repository Information:**
- Base commit: {base_commit}
- Version: {version}

**Task:**
Generate a git patch (unified diff format) that fixes the issue described above.
The patch should:
1. Be in valid unified diff format (with --- and +++ headers)
2. Only modify the necessary lines to fix the issue
3. Follow the existing code style
4. Be minimal and focused

Please provide ONLY the patch in your response, starting with the diff headers."""

        return prompt


# Register in agent registry
register_agent("swe", SWEAgentTracker)
register_agent("swe-bench", SWEAgentTracker)
register_agent("coding", SWEAgentTracker)
