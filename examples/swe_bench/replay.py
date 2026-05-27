#!/usr/bin/env python
"""
SWE-bench Trajectory Replay.

Replay recorded trajectories through different memory strategies without
re-running full episodes. Two modes:

  analyze: Offline replay — see what context each memory strategy would
           show the LLM at each step. No Docker or agent LLM needed.

  fork:    Fork from step N — replay commands 1..N in Docker, then continue
           with a different memory strategy + real LLM. Gets actual different
           rollout and outcome.

Usage:
    # Analyze: What would llm_summarizing show the agent?
    python scripts/replay_swe_bench.py analyze \
        --replay results/baseline/trajectories/astropy__astropy-12907_replay.json \
        --memory llm_summarizing --max-size 10 \
        -o results/replay_analysis/ -v

    # Fork from step 5 with different memory
    python scripts/replay_swe_bench.py fork \
        --replay results/baseline/trajectories/astropy__astropy-12907_replay.json \
        --fork-step 5 --memory llm_summarizing --max-size 50 \
        --model openai/gpt-5-mini \
        -o results/forked/ -v
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# MemGym imports


from memgym.gym.swe_bench.evaluate import create_memory_model


def _extract_action(content: str) -> str:
    """Extract bash command from assistant response."""
    match = re.search(r'```(?:bash)?\s*\n(.*?)\n```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def load_replay(replay_path: str) -> Dict[str, Any]:
    """Load a _replay.json file."""
    with open(replay_path) as f:
        data = json.load(f)
    if "messages" not in data or "steps" not in data:
        raise ValueError(f"Invalid replay file: missing 'messages' or 'steps' in {replay_path}")
    return data


def find_replay_files(path: str) -> List[str]:
    """Find all _replay.json files in a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        # Prefer trajectories/ subdir if it exists (matches evaluate.py layout)
        traj_dir = p / "trajectories"
        search_dir = traj_dir if traj_dir.is_dir() else p
        files = sorted(search_dir.glob("*_replay.json"))
        return [str(f) for f in files]
    else:
        raise FileNotFoundError(f"Replay path not found: {path}")


def compute_auto_fork_step(replay_data: Dict, max_size: int) -> Optional[int]:
    """Find the earliest step where the memory strategy would first trigger.

    llm_summarizing and similar threshold-based strategies condense when the
    pre-query message count exceeds `max_size`. Below that threshold, the
    memory-augmented run is byte-identical to the baseline, so we can replay
    that prefix in Docker for free and only invoke the LLM from fork_step on.

    Returns:
        The step index at which memory would first fire, or None if the
        baseline trajectory never grew past `max_size` messages (nothing to
        learn from this instance — skip it).
    """
    steps = replay_data.get("steps", [])
    for step_info in steps:
        # pre_query_messages = messages[:assistant_idx], so its length is assistant_idx
        assistant_idx = step_info.get("assistant_msg_index")
        if assistant_idx is None:
            continue
        if assistant_idx > max_size:
            return step_info.get("step", 0)
    return None


# =============================================================================
# ANALYZE MODE
# =============================================================================

def analyze_replay(replay_data: Dict, memory_manager, verbose: bool = False) -> Dict:
    """Replay recorded messages through a memory strategy.

    Mimics MemoryAwareSWEAgent.query() at each step:
      history = messages[:-1]
      current_obs = latest message
      filtered = memory_manager.manage_context(history, current_obs)

    Returns per-step analysis of what the memory strategy would produce.
    """
    messages = replay_data["messages"]
    steps = replay_data["steps"]
    memory_manager.reset()

    analysis_steps = []

    for step_info in steps:
        step_num = step_info["step"]
        assistant_idx = step_info["assistant_msg_index"]

        # Reconstruct what manage_context sees before each query.
        # At query time, agent has messages[0:assistant_idx] (assistant hasn't been added yet).
        pre_query_messages = messages[:assistant_idx]
        if len(pre_query_messages) < 2:
            continue

        history = pre_query_messages[:-1]
        current_obs = pre_query_messages[-1]

        filtered = memory_manager.manage_context(
            original_context=history,
            current_observation=current_obs,
            metadata={"step": step_num}
        )

        filtered_msgs = filtered.content if isinstance(filtered.content, list) else []
        was_compacted = filtered.metadata.get("was_compacted", False)
        cond_event = filtered.metadata.get("condensation_event")

        step_analysis = {
            "step": step_num,
            "original_msgs": len(pre_query_messages),
            "filtered_msgs": len(filtered_msgs),
            "original_tokens": filtered.metadata.get("original_tokens", 0),
            "filtered_tokens": filtered.metadata.get("tokens", 0),
            "compacted": was_compacted,
            "compression_ratio": filtered.metadata.get("compression_ratio", 1.0),
        }

        if was_compacted and cond_event:
            step_analysis["condensation"] = {
                "summary": cond_event.get("summary_text", "")[:500],
                "forgotten": cond_event.get("forgotten_count", 0),
                "summarizer_prompt_tokens": cond_event.get("summarizer_prompt_tokens", 0),
                "summarizer_completion_tokens": cond_event.get("summarizer_completion_tokens", 0),
            }

        analysis_steps.append(step_analysis)

        if verbose:
            status = "CONDENSED" if was_compacted else "pass"
            print(f"  Step {step_num}: {len(pre_query_messages)} → {len(filtered_msgs)} msgs "
                  f"({filtered.metadata.get('tokens', '?')} tokens) [{status}]")

    # Summary
    compactions = [s for s in analysis_steps if s["compacted"]]
    all_tokens = [s["filtered_tokens"] for s in analysis_steps if s["filtered_tokens"] > 0]
    all_ratios = [s["compression_ratio"] for s in analysis_steps if s["compression_ratio"] > 1.0]
    total_original = sum(s["original_tokens"] for s in analysis_steps)
    total_filtered = sum(s["filtered_tokens"] for s in analysis_steps)

    # Summarizer token totals from condensation events
    total_summarizer_prompt = sum(
        s.get("condensation", {}).get("summarizer_prompt_tokens", 0) for s in analysis_steps
    )
    total_summarizer_completion = sum(
        s.get("condensation", {}).get("summarizer_completion_tokens", 0) for s in analysis_steps
    )
    total_summarizer = total_summarizer_prompt + total_summarizer_completion
    total_cost = total_filtered + total_summarizer

    summary = {
        "total_steps": len(analysis_steps),
        "total_compactions": len(compactions),
        "first_compaction_step": compactions[0]["step"] if compactions else None,
        "peak_original_tokens": max((s["original_tokens"] for s in analysis_steps), default=0),
        "peak_filtered_tokens": max(all_tokens, default=0),
        "total_original_tokens": total_original,
        "total_filtered_tokens": total_filtered,
        "total_summarizer_prompt_tokens": total_summarizer_prompt,
        "total_summarizer_completion_tokens": total_summarizer_completion,
        "total_summarizer_tokens": total_summarizer,
        "naive_compression_ratio": round(total_original / total_filtered, 2) if total_filtered > 0 else 1.0,
        "true_compression_ratio": round(total_original / total_cost, 2) if total_cost > 0 else 1.0,
        "memory_overhead_pct": round(total_summarizer / total_cost * 100, 1) if total_cost > 0 else 0,
        "avg_compression_ratio": sum(all_ratios) / len(all_ratios) if all_ratios else 1.0,
    }

    # Carry forward pass/fail labels from source _replay.json
    resolved = None
    if "reward" in replay_data:
        resolved = replay_data["reward"] > 0

    return {
        "instance_id": replay_data.get("instance_id", "unknown"),
        "original_strategy": replay_data.get("memory_strategy", "unknown"),
        "replay_strategy": memory_manager.__class__.__name__,
        "reward": replay_data.get("reward"),
        "resolved": resolved,
        "status": replay_data.get("status"),
        "num_steps": len(analysis_steps),
        "steps": analysis_steps,
        "summary": summary,
    }


def run_analyze(args):
    """Run analyze mode on one or more replay files."""
    replay_files = find_replay_files(args.replay)
    if not replay_files:
        print("No replay files found.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse reeval for resolved labels
    resolved_map: Dict[str, bool] = {}
    reeval_path = getattr(args, 'reeval', None)
    if not reeval_path:
        # Auto-discover: look for *reeval*.json in replay's parent/grandparent dir
        replay_dir = Path(args.replay)
        for search_dir in [replay_dir.parent, replay_dir.parent.parent, replay_dir]:
            if search_dir.is_dir():
                reeval_files = sorted(search_dir.glob("*reeval*.json"))
                if reeval_files:
                    reeval_path = str(reeval_files[0])
                    break
    if reeval_path:
        from memgym.gym.swe_bench.enrich_trajectories import parse_reeval
        resolved_map = parse_reeval([Path(reeval_path)])
        print(f"Reeval: {sum(resolved_map.values())}/{len(resolved_map)} resolved")

    print(f"\nAnalyzing {len(replay_files)} replay file(s)")
    print(f"Strategy: {args.memory}")
    print()

    all_results = []
    skipped = 0

    for replay_path in replay_files:
        replay_data = load_replay(replay_path)
        instance_id = replay_data.get("instance_id", Path(replay_path).stem)

        # Resume: skip if analysis already exists
        analysis_path = output_dir / f"{instance_id}_analysis_{args.memory}.json"
        if analysis_path.exists():
            try:
                with open(analysis_path) as f:
                    result = json.load(f)
                # Patch resolved label from reeval (even for resumed files)
                if resolved_map:
                    new_resolved = resolved_map.get(instance_id)
                    if result.get("resolved") != new_resolved:
                        result["resolved"] = new_resolved
                        # Write back patched label to disk
                        with open(analysis_path, "w") as f:
                            json.dump(result, f, indent=2)
                all_results.append(result)
                skipped += 1
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted file, re-analyze

        if args.verbose:
            print(f"[{instance_id}]")

        # Fresh memory model per instance
        memory_manager = create_memory_model(args.memory, args)

        result = analyze_replay(replay_data, memory_manager, verbose=args.verbose)

        # Add resolved label
        if resolved_map:
            result["resolved"] = resolved_map.get(instance_id)

        all_results.append(result)

        # Save per-instance analysis (includes resolved label)
        with open(analysis_path, "w") as f:
            json.dump(result, f, indent=2)

        if args.verbose:
            s = result["summary"]
            resolved_str = " PASS" if result.get("resolved") else " FAIL" if result.get("resolved") is False else ""
            print(f"  Summary: {s['total_compactions']} compactions, "
                  f"avg ratio {s['avg_compression_ratio']:.1f}x{resolved_str}\n")

    if skipped:
        print(f"Resumed: skipped {skipped} already-analyzed instances")

    # Print summary table
    resolved_count = sum(1 for r in all_results if r.get("resolved") is True)
    has_reeval = any(r.get("resolved") is not None for r in all_results)

    print("=" * 80)
    header = f"{'Instance':<40} | {'Steps':>5} | {'Compact':>7} | {'Peak Tok':>8} | {'Avg Ratio':>9}"
    if has_reeval:
        header += " | Result"
    print(header)
    print("-" * 80)
    for r in all_results:
        s = r["summary"]
        line = (f"{r['instance_id']:<40} | {s['total_steps']:>5} | {s['total_compactions']:>7} | "
                f"{s['peak_filtered_tokens']:>8} | {s['avg_compression_ratio']:>8.1f}x")
        if has_reeval:
            label = "PASS" if r.get("resolved") else "FAIL" if r.get("resolved") is False else "?"
            line += f" | {label}"
        print(line)
    print("=" * 80)

    if has_reeval:
        print(f"\nResolved: {resolved_count}/{len(all_results)}")

    # Save combined summary
    summary_path = output_dir / "replay_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "mode": "analyze",
            "strategy": args.memory,
            "timestamp": datetime.now().isoformat(),
            "resolved_count": resolved_count if has_reeval else None,
            "resolve_rate": resolved_count / len(all_results) if has_reeval and all_results else None,
            "total_instances": len(all_results),
            "results": [
                {"instance_id": r["instance_id"], "resolved": r.get("resolved"), **r["summary"]}
                for r in all_results
            ],
        }, f, indent=2)
    print(f"Saved: {summary_path}")


# =============================================================================
# FORK MODE
# =============================================================================

def _drop_baseline_format_errors(msg: Dict) -> bool:
    """Policy A: drop baseline format-error messages when translating to backticks.

    The baseline was recorded with Bedrock function-calling. When the baseline
    model failed to emit a tool_use block, mini-swe-agent injected a user-role
    message like "Tool call error: No tool calls found...". That vocabulary is
    specific to the tool-calling format and doesn't apply once the fork runs in
    backticks mode — keeping those messages trains the new agent against a
    failure it can no longer reproduce. Returns True to keep, False to drop.
    """
    if msg.get("role") != "user":
        return True
    content = str(msg.get("content", "") or "")
    return not content.lstrip().startswith("Tool call error:")


def _translate_messages_to_textbased(
    messages, system_template: str, instance_template: str, task: str, repo: str
):
    """Reshape tool-calling baseline messages into the backticks text format.

    The baseline's system/task/tool_use/tool_result messages carry formatting
    conventions a LitellmTextbasedModel cannot continue from. Rebuild the
    history in backticks shape: regenerated system + task from the text-based
    templates, tool_use blocks collapsed into THOUGHT+```mswea_bash_command
    blocks, tool-role observations flipped to user-role. Information is
    preserved; only the encoding changes.
    """
    from jinja2 import Template

    out = [
        {"role": "system", "content": Template(system_template).render()},
        {
            "role": "user",
            "content": Template(instance_template).render(task=task, repo=repo),
        },
    ]

    for msg in messages[2:]:  # skip baseline system + task (just rebuilt)
        if not _drop_baseline_format_errors(msg):
            continue
        role = msg.get("role")
        content = str(msg.get("content", "") or "")

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Pure-text assistant turn in a tool-calling baseline is almost
                # always a failed format attempt — drop to keep the translated
                # transcript coherent for the backticks agent.
                continue
            cmd = ""
            for tc in tool_calls:
                fn = (tc.get("function") or {})
                if fn.get("name") != "bash":
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                cmd = args.get("command", "") or ""
                if cmd:
                    break
            if not cmd:
                continue
            thought = content.strip() or "(continuing from baseline)"
            out.append({
                "role": "assistant",
                "content": f"THOUGHT: {thought}\n\n```mswea_bash_command\n{cmd}\n```",
            })
        elif role == "tool":
            # Baseline already wraps Docker output in <returncode>/<output>
            # tags (same observation_template across both configs), so a role
            # flip is enough — no re-templating needed.
            out.append({"role": "user", "content": content})
        elif role == "user":
            out.append({"role": "user", "content": content})
        # any other role: drop

    return out


def fork_replay(replay_data: Dict, fork_step: int, memory_manager,
                agent_llm: str, dataset: str, env_type: str,
                step_limit: int = 250, split: Optional[str] = None,
                verbose: bool = False) -> Dict:
    """Fork from step N of a recorded trajectory, continue with new memory.

    Steps 1..fork_step: re-execute recorded commands in Docker.
    Steps fork_step+1..end: real LLM agent with new memory strategy.

    `split` defaults to "train" for SWE-Gym datasets and "test" otherwise —
    SWE-Gym's HF dataset only ships a train split. `docker_image_source` is
    auto-selected the same way (SWE-Gym uses xingyaoww/ images with the
    `_s_` instance-id convention, vs. SWE-bench's swebench/ images with
    `_1776_`).
    """
    is_swe_gym = ("swe-gym" in dataset.lower() or "sweg" in dataset.lower())
    if split is None:
        split = "train" if is_swe_gym else "test"
    docker_image_source = "swe-gym" if is_swe_gym else "swebench"
    from memgym.envs import SWEMemoryEnv
    from minisweagent.models import get_model
    from minisweagent.agents.default import AgentConfig
    from memgym.gym.swe_bench.agent import MemoryAwareSWEAgent

    instance_id = replay_data["instance_id"]
    recorded_messages = replay_data["messages"]
    recorded_steps = replay_data["steps"]

    if fork_step < 0 or fork_step > len(recorded_steps):
        raise ValueError(f"fork_step must be 0..{len(recorded_steps)}, got {fork_step}")

    if verbose:
        print(f"\nForking {instance_id} at step {fork_step}")
        print(f"  Original: {len(recorded_steps)} steps, strategy={replay_data.get('memory_strategy')}")
        print(f"  New strategy: {memory_manager.__class__.__name__}")
        print(f"  Docker image source: {docker_image_source}, split: {split}")

    # Create SWE env for Docker + evaluation
    swe_env = SWEMemoryEnv(
        dataset=dataset,
        split=split,
        instance_id=instance_id,
        agent_llm=agent_llm,
        agent_llm_args={},
        memory_model=memory_manager,
        use_context_management=True,
        use_docker=True,
        environment_class=env_type,
        max_steps=step_limit,
        docker_image_source=docker_image_source,
    )
    swe_env.reset()

    docker_env = swe_env._create_environment(verbose=verbose)

    # Build model with proper config (text-based mode for GPT-OSS/Llama/Nemotron,
    # Bedrock compat fixes, etc.) — mirrors SWEMemoryEnv._run_episode() logic.
    model_config = swe_env._get_model_config()
    if agent_llm and agent_llm.startswith("bedrock/"):
        import litellm as _litellm
        _litellm.modify_params = True
        os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        mkw = model_config.get("model_kwargs", {})
        mkw.pop("parallel_tool_calls", None)
        mkw.pop("tool_choice", None)
        model_config["model_kwargs"] = mkw
    if swe_env._is_textbased_model():
        model_config["model_class"] = "litellm_textbased"
        if verbose:
            print(f"  Using text-based mode (Bedrock tool calling unsupported for {agent_llm})")
    model = get_model(input_model_name=agent_llm, config=model_config)

    # --- Phase 1: Replay steps 1..fork_step ---
    # Strategy: re-execute commands in Docker for container-state side effects,
    # but copy the recorded assistant + all its tool response messages verbatim
    # into replayed_messages. This preserves the tool_use/tool_result pairing
    # that Bedrock requires (otherwise it rejects unmatched `tool_use` ids).
    def _extract_bash_commands_from_tool_calls(msg):
        cmds = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if fn.get("name") == "bash":
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    cmd = args.get("command", "")
                    if cmd:
                        cmds.append(cmd)
                except (json.JSONDecodeError, TypeError):
                    pass
        # Fallback to triple-backtick in content (older agent format)
        if not cmds:
            match = re.search(r"```(?:bash)?\s*\n(.*?)\n```", str(msg.get("content", "")), re.DOTALL)
            if match:
                cmds.append(match.group(1).strip())
        return cmds

    replayed_messages = []
    replayed_messages.append(recorded_messages[0])  # system
    replayed_messages.append(recorded_messages[1])  # user task

    steps_to_replay = recorded_steps[:fork_step]
    for i, step_info in enumerate(steps_to_replay):
        assistant_idx = step_info["assistant_msg_index"]
        # Next assistant-index caps this step's message range; beyond the last
        # step we fall back to the message array end.
        if i + 1 < len(steps_to_replay):
            next_assistant_idx = steps_to_replay[i + 1]["assistant_msg_index"]
        elif fork_step < len(recorded_steps):
            next_assistant_idx = recorded_steps[fork_step]["assistant_msg_index"]
        else:
            next_assistant_idx = len(recorded_messages)

        assistant_msg = recorded_messages[assistant_idx]
        # Append assistant + ALL follow-up messages (tool responses, etc.)
        replayed_messages.extend(recorded_messages[assistant_idx:next_assistant_idx])

        # Re-execute each bash tool_call for Docker side effects
        for action in _extract_bash_commands_from_tool_calls(assistant_msg):
            try:
                docker_env.execute({"command": action})
            except Exception as e:
                if verbose:
                    print(f"  Warning: replay step {step_info['step']} cmd failed: {e}")

    if verbose:
        print(f"  Replayed {fork_step} steps ({len(replayed_messages)} messages)")
        print(f"  Continuing with LLM from step {fork_step + 1}...")

    # --- Phase 2: Continue with real agent ---
    mini_cfg = swe_env._get_mini_swe_config()

    # Baseline was recorded in tool-calling mode; a text-based model (GPT-OSS,
    # Llama, Nemotron) can't continue from tool_use/tool_result blocks. Rewrite
    # the history into backticks shape before injecting it.
    if swe_env._is_textbased_model():
        obs_for_translate = swe_env._build_observation()
        replayed_messages = _translate_messages_to_textbased(
            replayed_messages,
            system_template=mini_cfg.get("system_template", "") or "",
            instance_template=mini_cfg.get("instance_template", "") or "",
            task=obs_for_translate["problem_statement"],
            repo=obs_for_translate.get("repo", ""),
        )
        if verbose:
            print(f"  Translated baseline history to backticks format: {len(replayed_messages)} messages")

    agent = MemoryAwareSWEAgent(
        model=model,
        env=docker_env,
        memory_manager=memory_manager,
        verbose=verbose,
        config_class=AgentConfig,
        **mini_cfg
    )

    # Inject replayed messages and set up state.
    # _processed_msg_count = len(replayed_messages) - 1 so:
    #   - The last replayed message (obs from step `fork_step`) becomes
    #     `current_observation` on the first agent.query() — semantically
    #     correct, and avoids `current_obs = None` which crashes litellm
    #     (`'NoneType' object has no attribute 'items'`).
    #   - `_pending_step=None` on first call means this obs is NOT
    #     double-recorded in training_steps.
    agent.messages = replayed_messages
    agent._processed_msg_count = max(0, len(replayed_messages) - 1)
    agent._training_steps = []
    agent._pending_step = None
    agent._task_description = recorded_messages[1].get("content", "")[:2000]

    # Set template vars for mini-swe-agent
    obs = swe_env._build_observation()
    agent.extra_template_vars["task"] = obs["problem_statement"]
    agent.extra_template_vars["repo"] = obs.get("repo", "")

    # Run agent loop from fork point.
    # minisweagent v2.2.4 uses a single InterruptAgentFlow base class with
    # subclasses like LimitsExceeded and Submitted — all terminal. The
    # TerminatingException / NonTerminatingException split only exists in
    # newer/older forks.
    from minisweagent.exceptions import InterruptAgentFlow

    status = "LimitsExceeded"
    message = ""
    try:
        while True:
            try:
                agent.step()
            except InterruptAgentFlow as e:
                # Mirror DefaultAgent.run(): append the interrupt messages and
                # only break when the last message is role="exit". FormatError
                # appends role="user" and is meant to RETRY, not terminate.
                if getattr(e, "messages", None):
                    agent.add_messages(*e.messages)

            # Terminate only on explicit exit message (Submitted, LimitsExceeded).
            if agent.messages and agent.messages[-1].get("role") == "exit":
                extra = agent.messages[-1].get("extra", {}) or {}
                status = extra.get("exit_status", "Unknown")
                message = extra.get("submission", "") or agent.messages[-1].get("content", "") or ""
                break
    except Exception as e:
        if verbose:
            traceback.print_exc()
        status = "Error"
        message = str(e)

    # Extract patch.
    # IMPORTANT: Must mirror env.py:_run_episode's safe extraction. A naive
    # `git add -A && git diff --cached` stages every untracked file the agent
    # created as scratch work (patch.txt, *.bak, debug_*.py, NOTES.md, ...)
    # which swe-bench's `git apply` then rejects as "Patch Apply Failed".
    patch = message

    def _is_valid_patch(p: str) -> bool:
        if not p or not p.strip():
            return False
        stripped = p.strip()
        return stripped.startswith("diff --git") or stripped.startswith("---")

    if not _is_valid_patch(patch) or status == "LimitsExceeded":
        try:
            # Only diff .py files that existed at HEAD (original repo state).
            extract_cmd = (
                "cd /testbed && git diff HEAD -- "
                "$(git ls-tree -r HEAD --name-only | grep '\\.py$' | tr '\\n' ' ')"
            )
            diff_output = docker_env.execute({"command": extract_cmd})
            extracted = diff_output.get("output", "").strip()
            if extracted:
                patch = extracted
                if verbose:
                    print(f"  Extracted patch: {len(patch)} chars")
            else:
                # Fallback: staged .py changes only (still excludes patch.txt etc.)
                fallback_cmd = "cd /testbed && git add -A -- '*.py' && git diff --cached -- '*.py'"
                diff_output = docker_env.execute({"command": fallback_cmd})
                extracted = diff_output.get("output", "").strip()
                if extracted:
                    patch = extracted
                    if verbose:
                        print(f"  Extracted patch (fallback): {len(patch)} chars")
        except Exception:
            pass

    # Evaluate
    reward = 0.0
    evaluation = {}
    if patch and patch.strip() and not patch.startswith("# Error"):
        try:
            evaluation = swe_env._evaluate_patch(patch, env=docker_env)
            reward = 1.0 if evaluation.get("success", False) else 0.0
        except Exception as e:
            evaluation = {"error": str(e)}

    if verbose:
        result_str = "PASS" if reward > 0 else "FAIL"
        print(f"\n  Forked result: {status}, {result_str}, patch={len(patch)} chars")

    # Build step index for the forked messages
    from memgym.gym.swe_bench.env import _build_step_index
    forked_steps = _build_step_index(agent.messages)

    # Collect training trajectory
    training_traj = None
    if hasattr(agent, 'get_training_trajectory'):
        training_traj = agent.get_training_trajectory()

    result = {
        "version": 1,
        "instance_id": instance_id,
        "dataset": dataset,
        "model": agent_llm,
        "memory_strategy": memory_manager.__class__.__name__,
        "fork_step": fork_step,
        "original_strategy": replay_data.get("memory_strategy", "unknown"),
        "original_num_steps": len(recorded_steps),
        "status": status,
        "reward": reward,
        "patch": patch,
        "evaluation": evaluation,
        "messages": list(agent.messages),
        "steps": forked_steps,
        "training_trajectory": training_traj,
        "timestamp": datetime.now().isoformat(),
    }

    # Cleanup Docker container — otherwise batch mode leaks one container per instance.
    try:
        docker_env.cleanup()
    except Exception as cleanup_err:
        if verbose:
            print(f"  Warning: docker cleanup failed: {cleanup_err}")

    return result


def run_fork(args):
    """Run fork mode on a replay file."""
    replay_data = load_replay(args.replay)
    instance_id = replay_data.get("instance_id", "unknown")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset path
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    dataset = replay_data.get("dataset", dataset_map.get(args.dataset, args.dataset))

    memory_manager = create_memory_model(args.memory, args)

    print(f"\nForking {instance_id}")
    print(f"  Fork step: {args.fork_step}")
    print(f"  Memory: {args.memory}")
    print(f"  Model: {args.model}")

    result = fork_replay(
        replay_data=replay_data,
        fork_step=args.fork_step,
        memory_manager=memory_manager,
        agent_llm=args.model,
        dataset=dataset,
        env_type=args.env_type,
        step_limit=args.step_limit or 250,
        verbose=args.verbose,
    )

    # Save forked replay
    forked_path = traj_dir / f"{instance_id}_forked.json"
    with open(forked_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save predictions in SWE-bench format (disk-sourced so repeat invocations
    # against the same --output dir accumulate rather than stomp each other).
    preds_path = output_dir / "preds.json"
    _rebuild_preds_from_forks(traj_dir, preds_path, args.model)

    # Save training trajectory if available
    if result.get("training_trajectory"):
        training_path = traj_dir / f"{instance_id}_training.json"
        with open(training_path, "w") as f:
            json.dump(result["training_trajectory"], f, indent=2)

    # Print result
    result_str = "PASS" if result["reward"] > 0 else "FAIL"
    print(f"\nResult: {result['status']}, {result_str}")
    print(f"  Forked replay: {forked_path}")
    print(f"  Predictions: {preds_path}")


# =============================================================================
# FORK-BATCH MODE (parallel fork_replay across many baseline replay files)
# =============================================================================

def _rebuild_preds_from_forks(
    traj_dir: Path,
    preds_path: Path,
    default_model: str,
) -> int:
    """Build preds.json from every `*_forked.json` on disk.

    Reading the on-disk fork files (the source of truth) rather than this run's
    in-memory `summaries` makes the writer idempotent and self-healing across
    `--resume` invocations. Forks with an empty patch are excluded (they can't
    be scored by the SWE-bench harness anyway).
    """
    preds: Dict[str, dict] = {}
    for forked_file in sorted(traj_dir.glob("*_forked.json")):
        try:
            data = json.loads(forked_file.read_text())
        except Exception:
            continue
        patch = (data.get("patch") or "").strip()
        if not patch:
            continue
        iid = data.get("instance_id") or forked_file.stem.replace("_forked", "")
        preds[iid] = {
            "model_name_or_path": data.get("model") or default_model,
            "instance_id": iid,
            "model_patch": data.get("patch") or "",
        }
    preds_path.write_text(json.dumps(preds, indent=2))
    return len(preds)


def _fork_one_instance(
    replay_path: str,
    fork_step_arg: str,
    max_size: int,
    memory_name: str,
    memory_args,
    agent_llm: str,
    env_type: str,
    step_limit: int,
    traj_dir: Path,
    verbose: bool,
) -> Dict:
    """Process one replay file in a worker thread.

    Returns a small summary dict (not the full result, to avoid pickling cost
    in case we ever switch to processes).
    """
    instance_id = Path(replay_path).stem.replace("_replay", "")
    forked_path = traj_dir / f"{instance_id}_forked.json"

    try:
        replay_data = load_replay(replay_path)
        instance_id = replay_data.get("instance_id", instance_id)

        # Decide fork_step: explicit int overrides, else auto
        if fork_step_arg == "auto":
            fork_step = compute_auto_fork_step(replay_data, max_size)
            if fork_step is None:
                return {
                    "instance_id": instance_id,
                    "skipped": True,
                    "reason": f"msg_count never exceeded max_size={max_size}",
                }
        else:
            fork_step = int(fork_step_arg)

        dataset_map = {
            "lite": "princeton-nlp/SWE-bench_Lite",
            "verified": "princeton-nlp/SWE-bench_Verified",
            "full": "princeton-nlp/SWE-bench",
            "swe-gym": "SWE-Gym/SWE-Gym",
        }
        dataset = replay_data.get(
            "dataset",
            dataset_map.get(memory_args.dataset, memory_args.dataset),
        )

        # Fresh memory manager per instance — avoids leaking state across workers
        memory_manager = create_memory_model(memory_name, memory_args)

        if verbose:
            print(f"[{instance_id}] forking at step {fork_step} "
                  f"(original trajectory: {len(replay_data.get('steps', []))} steps)")

        # Split: from replay_data if recorded, else from user args, else auto
        split = replay_data.get("split") or getattr(memory_args, "split", None)

        # Pre-pull Docker image so DockerEnvironment's 120s `docker run` timeout
        # doesn't fire on first pull of a multi-GB SWE-Gym image.
        is_swe_gym = ("swe-gym" in dataset.lower() or "sweg" in dataset.lower())
        if is_swe_gym:
            docker_id = instance_id.replace("__", "_s_").lower()
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{docker_id}:latest"
        else:
            docker_id = instance_id.replace("__", "_1776_").lower()
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest"
        try:
            import subprocess
            pull = subprocess.run(
                ["docker", "pull", image_name],
                capture_output=True, timeout=1200,
            )
            if pull.returncode != 0 and verbose:
                print(f"[{instance_id}] WARN: docker pull rc={pull.returncode}: "
                      f"{pull.stderr.decode('utf-8', errors='replace')[:200]}")
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"[{instance_id}] WARN: docker pull exceeded 1200s")

        result = fork_replay(
            replay_data=replay_data,
            fork_step=fork_step,
            memory_manager=memory_manager,
            agent_llm=agent_llm,
            dataset=dataset,
            env_type=env_type,
            step_limit=step_limit,
            split=split,
            verbose=verbose,
        )

        # Save {iid}_forked.json (full record with fork_step + messages)
        with open(forked_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save {iid}_training.json for training pipeline consumption
        if result.get("training_trajectory"):
            training_path = traj_dir / f"{instance_id}_training.json"
            with open(training_path, "w") as f:
                json.dump(result["training_trajectory"], f, indent=2)

        return {
            "instance_id": instance_id,
            "skipped": False,
            "fork_step": fork_step,
            "status": result.get("status"),
            "reward": result.get("reward"),
            "patch_len": len(result.get("patch") or ""),
            "forked_steps": len(result.get("steps") or []),
        }

    except Exception as e:
        if verbose:
            traceback.print_exc()
        return {
            "instance_id": instance_id,
            "skipped": False,
            "error": str(e)[:500],
        }


def run_fork_batch(args):
    """Run fork mode on every _replay.json in a directory, in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    replay_files = find_replay_files(args.replay_dir)
    if not replay_files:
        print(f"No _replay.json files found in {args.replay_dir}")
        return

    output_dir = Path(args.output)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Optional instance filter (explicit IDs) — applied before resume filtering
    if getattr(args, "instances", None):
        wanted = set(args.instances)
        replay_files = [
            rp for rp in replay_files
            if Path(rp).stem.replace("_replay", "") in wanted
        ]

    # Resume: skip instances whose existing _forked.json has status=="Submitted".
    # Any other status (Error, FormatError, LimitsExceeded) is re-run so failed
    # forks get a second chance without manual cleanup.
    to_process = []
    resumed = 0
    retry_failed = 0
    for rp in replay_files:
        iid = Path(rp).stem.replace("_replay", "")
        forked_path = traj_dir / f"{iid}_forked.json"
        if args.resume and forked_path.exists():
            try:
                existing_status = json.loads(forked_path.read_text()).get("status")
            except Exception:
                existing_status = None
            if existing_status == "Submitted":
                resumed += 1
                continue
            retry_failed += 1
        to_process.append(rp)

    # Optional cap on how many we actually run (useful for pilot/verification
    # slices of a large directory)
    if getattr(args, "limit", 0):
        to_process = to_process[: args.limit]

    if resumed:
        print(f"Resume: skipping {resumed} instances with existing Submitted _forked.json")
    if retry_failed:
        print(f"Resume: re-running {retry_failed} instances with non-Submitted _forked.json")
    print(f"Processing {len(to_process)} instances with {args.workers} workers")
    print(f"  memory={args.memory}  max_size={args.max_size}  "
          f"fork_step={args.fork_step}  model={args.model}")

    summaries = []
    submitted = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _fork_one_instance,
                rp,
                args.fork_step,
                args.max_size,
                args.memory,
                args,
                args.model,
                args.env_type,
                args.step_limit or 250,
                traj_dir,
                args.verbose,
            ): rp
            for rp in to_process
        }
        submitted = len(futures)

        for fut in as_completed(futures):
            summary = fut.result()
            summaries.append(summary)
            completed += 1
            iid = summary["instance_id"]
            if summary.get("skipped"):
                status = f"SKIP ({summary.get('reason')})"
            elif summary.get("error"):
                status = f"ERROR: {summary['error'][:80]}"
            else:
                pf = "PASS" if summary.get("reward", 0) > 0 else "FAIL"
                status = (f"fork_step={summary['fork_step']} "
                          f"{summary['status']} {pf} "
                          f"patch={summary['patch_len']}c")
            print(f"  [{completed}/{submitted}] {iid}: {status}")

    # Aggregate preds.json from every *_forked.json on disk — includes forks
    # from prior --resume runs that aren't in this run's in-memory `summaries`.
    preds_path = output_dir / "preds.json"
    n_preds = _rebuild_preds_from_forks(traj_dir, preds_path, args.model)

    summary_path = output_dir / "fork_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "mode": "fork-batch",
            "replay_dir": args.replay_dir,
            "memory": args.memory,
            "max_size": args.max_size,
            "fork_step": args.fork_step,
            "model": args.model,
            "workers": args.workers,
            "timestamp": datetime.now().isoformat(),
            "total_replay_files": len(replay_files),
            "resumed": resumed,
            "processed": len(summaries),
            "skipped": sum(1 for s in summaries if s.get("skipped")),
            "errored": sum(1 for s in summaries if s.get("error")),
            "passed": sum(1 for s in summaries if s.get("reward", 0) > 0),
            "summaries": summaries,
        }, f, indent=2)
    print(f"\nSaved:")
    print(f"  {preds_path} ({n_preds} predictions)")
    print(f"  {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def add_memory_args(parser):
    """Add shared memory strategy arguments."""
    parser.add_argument("--memory", default="observation_masking",
                        choices=[
                            "naive", "none", "passthrough", "observation_masking",
                            "llm_summarizing", "structured_summary",
                            "pipeline_masking_summarizing", "pipeline_masking_structured",
                            "sliding_window", "adaptive_token_budget",
                        ],
                        help="Memory strategy for replay (default: observation_masking)")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-size", type=int, default=100)
    parser.add_argument("--keep-first", type=int, default=3)
    parser.add_argument("--condensation-ratio", type=float, default=0.75)
    parser.add_argument("--attention-window", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--keep-recent", type=int, default=8)
    parser.add_argument("--preserve-first-user", action="store_true", default=True)
    parser.add_argument("--no-preserve-first-user", dest="preserve_first_user", action="store_false")
    parser.add_argument("--summarization-model", default="gpt-4o-mini")


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Trajectory Replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Analyze subcommand ---
    analyze_parser = subparsers.add_parser("analyze",
        help="Offline replay through different memory strategies")
    analyze_parser.add_argument("--replay", required=True,
        help="Path to _replay.json file or directory")
    add_memory_args(analyze_parser)
    analyze_parser.add_argument("--reeval", default=None,
        help="Path to reeval report JSON (auto-discovers in --replay parent dir if omitted)")
    analyze_parser.add_argument("--output", "-o", default="results/replay_analysis/")
    analyze_parser.add_argument("--verbose", "-v", action="store_true")

    # --- Fork subcommand ---
    fork_parser = subparsers.add_parser("fork",
        help="Fork from step N with different memory + real LLM")
    fork_parser.add_argument("--replay", required=True,
        help="Path to _replay.json file")
    fork_parser.add_argument("--fork-step", type=int, required=True,
        help="Step number to fork from (0 = from beginning)")
    add_memory_args(fork_parser)
    fork_parser.add_argument("--model", default="openai/gpt-5-mini",
        help="Agent LLM for forked continuation")
    fork_parser.add_argument("--dataset", default="lite",
        help="Dataset for instance lookup (default: lite)")
    fork_parser.add_argument("--env-type", default="docker")
    fork_parser.add_argument("--step-limit", type=int, default=None)
    fork_parser.add_argument("--output", "-o", default="results/forked/")
    fork_parser.add_argument("--verbose", "-v", action="store_true")

    # --- Fork-batch subcommand ---
    batch_parser = subparsers.add_parser("fork-batch",
        help="Fork every _replay.json in a directory in parallel")
    batch_parser.add_argument("--replay-dir", required=True,
        help="Directory containing _replay.json files (or a results/ dir with a trajectories/ subfolder)")
    batch_parser.add_argument("--fork-step", default="auto",
        help="'auto' (first step where msg_count>max_size) or an integer")
    add_memory_args(batch_parser)
    batch_parser.add_argument("--model", default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="Agent LLM for forked continuation")
    batch_parser.add_argument("--dataset", default="swe-gym",
        help="Dataset name (used when replay file lacks a dataset field)")
    batch_parser.add_argument("--split", default=None,
        help="Dataset split. Auto: 'train' for SWE-Gym, 'test' for SWE-bench.")
    batch_parser.add_argument("--env-type", default="docker")
    batch_parser.add_argument("--step-limit", type=int, default=250)
    batch_parser.add_argument("--workers", type=int, default=4,
        help="Parallel workers (4 recommended for Bedrock rate limits)")
    batch_parser.add_argument("--resume", action="store_true",
        help="Skip instances whose existing _forked.json has status=Submitted; "
             "re-run any with Error/FormatError/LimitsExceeded")
    batch_parser.add_argument("--limit", type=int, default=0,
        help="If >0, only run the first N pending instances (pilot/verification slices)")
    batch_parser.add_argument("--instances", nargs="+", default=None,
        help="Specific instance IDs to fork (all others in --replay-dir are skipped)")
    batch_parser.add_argument("--output", "-o", default="results/fork_batch/")
    batch_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.mode == "analyze":
        run_analyze(args)
    elif args.mode == "fork":
        run_fork(args)
    elif args.mode == "fork-batch":
        run_fork_batch(args)


if __name__ == "__main__":
    main()
