"""
Memory-aware wrapper for OpenHands CodeAct agent.

Runs the CodeAct agent via OpenHands' own controller loop, using OpenHands'
native condenser system for memory management. MemGym strategy names are
mapped to equivalent OpenHands condenser configs.

Follows the same pattern as OpenHands' own IssueResolver:
  1. Create runtime externally (so it survives after run_controller)
  2. Run the controller with runtime=runtime
  3. Extract patch via `git diff` in the still-alive container
  4. Extract condenser stats from event history
  5. Close runtime

Requires: pip install -e third_party/OpenHands (see install.sh --openhands)
"""

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# OpenHands imports (optional — guarded by try/except at module level)
_OPENHANDS_AVAILABLE = False
try:
    from openhands.core.config import OpenHandsConfig
    from openhands.core.config.agent_config import AgentConfig
    from openhands.core.config.condenser_config import (
        CondenserConfig,
        LLMSummarizingCondenserConfig,
        NoOpCondenserConfig,
        ObservationMaskingCondenserConfig,
        StructuredSummaryCondenserConfig,
        CondenserPipelineConfig,
    )
    from openhands.core.config.llm_config import LLMConfig
    from openhands.core.config.sandbox_config import SandboxConfig
    from openhands.core.main import run_controller
    from openhands.core.setup import create_runtime, generate_sid
    from openhands.events.action import CmdRunAction, MessageAction
    from openhands.events.action.agent import AgentFinishAction, CondensationAction
    from openhands.events.observation.commands import CmdOutputObservation
    from openhands.utils.async_utils import call_async_from_sync
    _OPENHANDS_AVAILABLE = True
except ImportError:
    pass


def check_openhands_available():
    """Raise ImportError if OpenHands is not installed."""
    if not _OPENHANDS_AVAILABLE:
        raise ImportError(
            "OpenHands not installed. Run: ./install.sh --openhands\n"
            "Or: git submodule update --init third_party/OpenHands && "
            "pip install -e third_party/OpenHands"
        )


# =============================================================================
# Strategy-to-condenser mapping
# =============================================================================

def memgym_strategy_to_condenser_config(
    strategy: str,
    summarization_model: str = "gpt-4o-mini",
    max_size: int = 100,
    keep_first: int = 1,
    attention_window: int = 100,
    max_event_length: int = 10_000,
) -> "CondenserConfig":
    """Map a MemGym memory strategy name to an OpenHands CondenserConfig.

    OpenHands has native equivalents for most MemGym strategies. This function
    maps between them so the CodeAct agent uses the right condenser.

    Supported mappings:
        none/passthrough        -> NoOpCondenserConfig
        observation_masking     -> ObservationMaskingCondenserConfig
        llm_summarizing         -> LLMSummarizingCondenserConfig
        structured_summary      -> StructuredSummaryCondenserConfig
        pipeline_masking_*      -> CondenserPipelineConfig

    Not yet supported (no native OpenHands equivalent):
        naive, sliding_window   -> Falls back to NoOpCondenserConfig with warning

    Args:
        strategy: MemGym strategy name
        summarization_model: LLM for summarization (for llm_summarizing/structured_summary)
        max_size: Max events before condensation
        keep_first: Initial events to preserve
        attention_window: Observation masking window size
        max_event_length: Max chars per event in summarization prompt

    Returns:
        OpenHands CondenserConfig instance
    """
    check_openhands_available()

    # LLM config for strategies that need summarization
    def _make_llm_config():
        return LLMConfig(model=summarization_model)

    if strategy in ("none", "passthrough"):
        return NoOpCondenserConfig()

    elif strategy == "observation_masking":
        return ObservationMaskingCondenserConfig(
            attention_window=attention_window,
        )

    elif strategy == "llm_summarizing":
        return LLMSummarizingCondenserConfig(
            llm_config=_make_llm_config(),
            max_size=max_size,
            keep_first=keep_first,
            max_event_length=max_event_length,
        )

    elif strategy == "structured_summary":
        return StructuredSummaryCondenserConfig(
            llm_config=_make_llm_config(),
            max_size=max_size,
            keep_first=keep_first,
            max_event_length=max_event_length,
        )

    elif strategy == "pipeline_masking_summarizing":
        return CondenserPipelineConfig(
            condensers=[
                ObservationMaskingCondenserConfig(attention_window=attention_window),
                LLMSummarizingCondenserConfig(
                    llm_config=_make_llm_config(),
                    max_size=max_size,
                    keep_first=keep_first,
                    max_event_length=max_event_length,
                ),
            ]
        )

    elif strategy == "pipeline_masking_structured":
        return CondenserPipelineConfig(
            condensers=[
                ObservationMaskingCondenserConfig(attention_window=attention_window),
                StructuredSummaryCondenserConfig(
                    llm_config=_make_llm_config(),
                    max_size=max_size,
                    keep_first=keep_first,
                    max_event_length=max_event_length,
                ),
            ]
        )

    else:
        # naive, sliding_window, etc. — no native OpenHands equivalent yet
        print(
            f"Warning: MemGym strategy '{strategy}' has no native OpenHands condenser "
            f"equivalent. Using NoOp (no memory filtering) for CodeAct agent. "
            f"Use llm_summarizing or observation_masking for CodeAct memory experiments."
        )
        return NoOpCondenserConfig()


# =============================================================================
# OpenHands config builder
# =============================================================================

def create_openhands_config(
    model: str,
    condenser_config: "CondenserConfig",
    max_iterations: int = 100,
    max_budget: float = 3.0,
    runtime: str = "docker",
    sandbox_image: Optional[str] = None,
    workspace_base: Optional[str] = None,
    save_trajectory_path: Optional[str] = None,
) -> "OpenHandsConfig":
    """Build an OpenHandsConfig for running CodeAct programmatically.

    Args:
        model: LLM model name (e.g., "openai/gpt-5", "anthropic/claude-sonnet-4-5-20250929")
        condenser_config: Condenser configuration (from memgym_strategy_to_condenser_config)
        max_iterations: Max agent steps per instance
        max_budget: Max USD per instance
        runtime: Runtime type ("docker", "local")
        sandbox_image: Docker image for SWE-bench instance (e.g., "swebench/sweb.eval.x86_64....")
        workspace_base: Base workspace directory
        save_trajectory_path: Path to save trajectories

    Returns:
        OpenHandsConfig instance
    """
    check_openhands_available()

    # Fix native_tool_calling for models that OpenHands' model_features.py
    # doesn't recognize correctly:
    #
    # 1. Llama/Nemotron on Bedrock: return tool calls as plain text instead
    #    of structured toolUse blocks. Force text mode (fn_call_converter).
    #
    # 2. GPT-OSS on Bedrock: supports native function calling via Converse
    #    API, but OpenHands' FUNCTION_CALLING_PATTERNS matches "gpt-*" which
    #    doesn't match "gpt-oss*" (fnmatch). Force native mode so tools are
    #    passed through to litellm instead of being stripped.
    model_lower = model.lower()
    # All Bedrock models except Claude/Anthropic need text mode with CodeAct.
    # OpenHands' 6+ complex tool schemas cause validation errors on Bedrock's
    # Converse API for non-Anthropic models (GPT-OSS rejects nested schemas,
    # Llama/Nemotron return tool calls as plain text).
    is_bedrock = "bedrock" in model_lower
    is_anthropic = "anthropic" in model_lower or "claude" in model_lower
    needs_text_mode = is_bedrock and not is_anthropic
    if needs_text_mode:
        native_tool_calling = False
    else:
        native_tool_calling = None  # auto-detect via model_features.py

    llm_config = LLMConfig(model=model, native_tool_calling=native_tool_calling)

    agent_config = AgentConfig(
        condenser=condenser_config,
        enable_browsing=False,  # SWE-bench doesn't need browser
        enable_jupyter=True,
        enable_mcp=False,
    )

    sandbox_config = SandboxConfig()
    if sandbox_image:
        sandbox_config.base_container_image = sandbox_image

    config = OpenHandsConfig(
        llms={"llm": llm_config},
        agents={"agent": agent_config},
        default_agent="CodeActAgent",
        sandbox=sandbox_config,
        runtime=runtime,
        max_iterations=max_iterations,
        max_budget_per_task=max_budget,
        save_trajectory_path=save_trajectory_path,
    )

    if workspace_base:
        config.workspace_base = workspace_base

    return config


# =============================================================================
# Post-run extraction helpers
# =============================================================================

def _extract_patch_from_runtime(runtime, verbose: bool = False) -> str:
    """Extract git diff from a still-alive OpenHands runtime.

    Follows the same approach as OpenHands' IssueResolver.complete_runtime():
    git add -A, then git diff --cached to capture all changes.
    """
    try:
        if verbose:
            print(f"  [CodeAct] Attempting patch extraction from runtime...")

        # Find the git repo — could be /workspace or /testbed
        repo_dir = "/workspace"
        obs = runtime.run_action(CmdRunAction(command="cd /workspace && git rev-parse --show-toplevel 2>/dev/null || cd /testbed && git rev-parse --show-toplevel 2>/dev/null || echo NOTFOUND"))
        if isinstance(obs, CmdOutputObservation) and obs.content.strip() and "NOTFOUND" not in obs.content:
            repo_dir = obs.content.strip().split("\n")[-1]
        if verbose:
            print(f"  [CodeAct] Git repo found at: {repo_dir}")

        # Stage all changes
        obs = runtime.run_action(CmdRunAction(command=f"cd {repo_dir} && git add -A"))

        # Extract diff — only .py files that existed at HEAD to avoid junk
        action = CmdRunAction(
            command=(
                f"cd {repo_dir} && "
                "git diff --no-color --cached -- "
                "$(git ls-tree -r HEAD --name-only | grep '\\.py$' | tr '\\n' ' ')"
            )
        )
        action.set_hard_timeout(120)
        obs = runtime.run_action(action)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            patch = obs.content.strip()
            if patch:
                if verbose:
                    print(f"  [CodeAct] Extracted patch from runtime: {len(patch)} chars")
                return patch

        # Fallback: include all cached changes
        action = CmdRunAction(command=f"cd {repo_dir} && git diff --no-color --cached")
        action.set_hard_timeout(120)
        obs = runtime.run_action(action)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            patch = obs.content.strip()
            if verbose and patch:
                print(f"  [CodeAct] Extracted patch (fallback) from runtime: {len(patch)} chars")
            return patch

    except Exception as e:
        if verbose:
            print(f"  [CodeAct] Patch extraction failed: {e}")
    return ""


def _extract_condenser_stats(state) -> Dict[str, Any]:
    """Extract memory/condenser statistics from the event history."""
    if state is None or not hasattr(state, 'history'):
        return {}

    condensations = [e for e in state.history if isinstance(e, CondensationAction)]
    total_forgotten = sum(
        len(c.forgotten_event_ids or [])
        for c in condensations
    )
    return {
        "total_queries": getattr(state, 'iteration', 0) or len([
            e for e in state.history
            if hasattr(e, 'source') and str(getattr(e, 'source', '')) == 'EventSource.AGENT'
        ]),
        "times_filtered": len(condensations),
        "total_events": len(state.history),
        "total_forgotten_events": total_forgotten,
    }


def _extract_training_trajectory(state, memory_strategy: str) -> Optional[Dict[str, Any]]:
    """Extract per-step training trajectory from the event history.

    Walks the event history and pairs agent actions with their observations,
    matching the format produced by MemoryAwareSWEAgent for mini-swe-agent.
    """
    if state is None or not hasattr(state, 'history'):
        return None

    steps = []
    pending_action = None

    for event in state.history:
        # Agent actions (CmdRunAction, etc.) have a 'thought' field
        if hasattr(event, 'action') and hasattr(event, 'thought'):
            # Flush previous pending action without observation
            if pending_action is not None:
                steps.append(pending_action)

            action_str = ""
            if hasattr(event, 'command'):
                action_str = event.command
            elif hasattr(event, 'code'):
                action_str = event.code
            else:
                action_str = str(event)[:500]

            pending_action = {
                "step": len(steps) + 1,
                "thought": getattr(event, 'thought', '') or '',
                "action": action_str[:500],
                "observation": "",
                "memory": {
                    "strategy": memory_strategy,
                    "is_condensation": isinstance(event, CondensationAction),
                },
            }

        # Observations fill the pending action
        elif pending_action is not None and hasattr(event, 'content'):
            pending_action["observation"] = str(event.content)[:500]
            steps.append(pending_action)
            pending_action = None

    # Flush last pending
    if pending_action is not None:
        steps.append(pending_action)

    if not steps:
        return None

    return {
        "memory_strategy": memory_strategy,
        "num_steps": len(steps),
        "steps": steps,
    }


# =============================================================================
# Episode runner
# =============================================================================

async def _run_codeact_episode_async(
    task: str,
    config: "OpenHandsConfig",
    condenser_config: "CondenserConfig",
    memory_strategy: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Async inner loop: create runtime, run controller, extract patch."""
    from openhands.utils.utils import create_registry_and_conversation_stats

    t0 = time.time()

    sid = generate_sid(config)
    llm_registry, conversation_stats, config = create_registry_and_conversation_stats(
        config, sid, None,
    )

    # Create runtime EXTERNALLY so it survives after run_controller.
    # This is the same pattern as OpenHands' IssueResolver.process_issue().
    runtime = create_runtime(config, llm_registry, sid=sid, headless_mode=True)
    call_async_from_sync(runtime.connect)

    initial_action = MessageAction(content=task)

    def fake_user_response(state, **kwargs):
        return "Please continue working on the task."

    state = None
    try:
        state = await run_controller(
            config=config,
            initial_user_action=initial_action,
            runtime=runtime,
            fake_user_response_fn=fake_user_response,
            headless_mode=True,
        )
    except Exception as e:
        if verbose:
            print(f"  [CodeAct] Controller error: {e}")
        # Don't return early — still try to extract patch below

    # Extract patch from the runtime.
    # The runtime may still be alive (normal exit) or dead (crash).
    # Try extraction either way — _extract_patch_from_runtime handles errors.
    patch = _extract_patch_from_runtime(runtime, verbose=verbose)

    # If runtime-based extraction failed, try Docker exec as last resort
    # (the container may have exited but we can start it briefly)
    if not patch and hasattr(runtime, 'container_name'):
        try:
            import subprocess
            container = getattr(runtime, 'container_name', None) or getattr(runtime, 'container_id', None)
            if container:
                # Try to get diff from stopped container by starting it briefly
                result = subprocess.run(
                    ["docker", "start", container],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    result = subprocess.run(
                        ["docker", "exec", container, "bash", "-c",
                         "cd /workspace 2>/dev/null || cd /testbed && git add -A && git diff --no-color --cached"],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        patch = result.stdout.strip()
                        if verbose:
                            print(f"  [CodeAct] Extracted patch from stopped container: {len(patch)} chars")
        except Exception as docker_err:
            if verbose:
                print(f"  [CodeAct] Docker fallback failed: {docker_err}")

    # Done with the runtime
    try:
        runtime.close()
    except Exception:
        pass

    elapsed = time.time() - t0
    status = str(getattr(state, 'agent_state', 'Unknown')) if state else "Unknown"

    # Extract condenser stats and training trajectory
    condenser_stats = _extract_condenser_stats(state)
    training_trajectory = _extract_training_trajectory(state, memory_strategy)

    # Serialize event history for replay
    history_dicts = []
    if state and hasattr(state, 'history'):
        for event in state.history:
            try:
                history_dicts.append(dataclasses.asdict(event))
            except Exception:
                history_dicts.append({"type": type(event).__name__, "str": str(event)[:200]})

    if verbose:
        print(f"  [CodeAct] Finished in {elapsed:.1f}s, status={status}")
        print(f"  [CodeAct] Patch size: {len(patch)} chars")
        if condenser_stats.get("times_filtered"):
            print(f"  [CodeAct] Condensations: {condenser_stats['times_filtered']}, "
                  f"forgotten: {condenser_stats.get('total_forgotten_events', 0)} events")

    return {
        "patch": patch,
        "status": status,
        "reward": 0.0,
        "elapsed_seconds": elapsed,
        "agent_type": "codeact",
        "memory_strategy": memory_strategy,
        "condenser_type": type(condenser_config).__name__,
        "condenser_stats": condenser_stats,
        "training_trajectory": training_trajectory,
        "history": history_dicts,
        "num_iterations": getattr(state, 'iteration', 0) if state else 0,
    }


def run_codeact_episode(
    task: str,
    model: str,
    memory_strategy: str = "none",
    sandbox_image: Optional[str] = None,
    max_iterations: int = 100,
    max_budget: float = 3.0,
    runtime: str = "docker",
    save_trajectory_path: Optional[str] = None,
    verbose: bool = False,
    # Strategy-specific params
    summarization_model: str = "gpt-4o-mini",
    max_size: int = 100,
    keep_first: int = 1,
    attention_window: int = 100,
) -> Dict[str, Any]:
    """Run a single CodeAct episode with memory management.

    Uses OpenHands' own controller loop with its native condenser system.
    MemGym strategy names are mapped to OpenHands condenser configs.

    The runtime is created externally and kept alive after the controller
    finishes, allowing proper patch extraction via git diff — the same
    pattern used by OpenHands' own IssueResolver.

    Args:
        task: Problem statement / task description
        model: LLM model name
        memory_strategy: MemGym strategy name ("none", "llm_summarizing", etc.)
        sandbox_image: Docker image for the SWE-bench instance
        max_iterations: Max agent steps
        max_budget: Max USD budget
        runtime: Runtime type
        save_trajectory_path: Path to save trajectory
        verbose: Print progress
        summarization_model: LLM for summarization
        max_size: Max events before condensation
        keep_first: Initial events to preserve
        attention_window: Observation masking window

    Returns:
        Dict with: patch, status, condenser_stats, training_trajectory, history
    """
    check_openhands_available()

    # Map strategy to condenser config
    condenser_config = memgym_strategy_to_condenser_config(
        strategy=memory_strategy,
        summarization_model=summarization_model,
        max_size=max_size,
        keep_first=keep_first,
        attention_window=attention_window,
    )

    # Build OpenHands config
    config = create_openhands_config(
        model=model,
        condenser_config=condenser_config,
        max_iterations=max_iterations,
        max_budget=max_budget,
        runtime=runtime,
        sandbox_image=sandbox_image,
        save_trajectory_path=save_trajectory_path,
    )

    if verbose:
        print(f"  [CodeAct] Model: {model}")
        print(f"  [CodeAct] Memory: {memory_strategy} -> {type(condenser_config).__name__}")
        print(f"  [CodeAct] Max iterations: {max_iterations}")
        if sandbox_image:
            print(f"  [CodeAct] Sandbox image: {sandbox_image}")

    return asyncio.run(
        _run_codeact_episode_async(
            task=task,
            config=config,
            condenser_config=condenser_config,
            memory_strategy=memory_strategy,
            verbose=verbose,
        )
    )
