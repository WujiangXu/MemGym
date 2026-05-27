"""
WebArenaRunner — episode loop for the WebArenaMemoryEnv.

Drives the memory → agent → env → record loop with the exact contract the
SWE-bench and IR runners use, so memory strategies can be swapped in and
compared on apples-to-apples trajectories:

    1. Reset env, agent, memory
    2. Optional PREFIX REPLAY phase — when `prefix_trajectory_dir` is set,
       deterministically re-execute the first K recorded actions from a
       baseline trajectory without calling agent.act(). The memory manager
       still runs `manage_context` at each prefix step so its view at the
       handoff is the natural continuation of a full run.
    3. LIVE phase — loop until done or max_steps:
       a. memory.manage_context(history, current_obs)
       b. agent.act(filtered_context)
       c. env.step(action)
       d. recorder.record_step(...)
       e. history.append({observation, action, reward, info})
    4. Save trajectory JSON (matching TrajectoryRecorder schema so
       TrajectoryRecorder.from_file() round-trips cleanly)
    5. Return episode result dict

The observation/action shapes are whatever `WebArenaMemoryEnv` and
`WebArenaAgent` agreed on; this runner makes no assumptions about them
beyond what `BaseRunner` does.

PREFIX-REPLAY design rationale: the replay-determinism probe (`replay_probe.py`) established that
95 baseline trajectories re-executed element-ID by element-ID reach the
same verifier reward even when the a11y tree drifts by a few elements
(60 % byte-deterministic but 100 % reward-deterministic). So "prefix
injection" is a semantically-sound experimental contract: hold the action
sequence constant for the first K steps, then diverge on what the memory
strategy + live agent do for the tail. This isolates memory's effect on
late-trajectory reasoning without forcing us to re-record the entire
action space or build a stable-selector fallback layer.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseRunner
from memgym.memory import BaseMemoryManager, PassThroughMemory
from memgym.agents.base import BaseAgent

logger = logging.getLogger("memgym.webarena.runner")


class WebArenaRunner(BaseRunner):
    """
    Episode runner for webarena-infinity tasks.

    Args:
        env: WebArenaMemoryEnv (or compatible) — must implement reset/step/close.
        agent: Any BaseAgent (typically WebArenaAgent).
        memory: Memory manager (defaults to PassThroughMemory).
        output_dir: Directory to save trajectories + per-episode artifacts.
        max_steps: Hard episode cap.
        verbose: Print per-step progress.
    """

    def __init__(
        self,
        env: Any,
        agent: BaseAgent,
        memory: Optional[BaseMemoryManager] = None,
        output_dir: str = "./results/webarena",
        max_steps: int = 30,
        verbose: bool = False,
        prefix_trajectory_dir: Optional[Path] = None,
        prefix_steps: Optional[int] = None,
        prefix_fraction: float = 0.5,
        smart_replay_trajectory_dir: Optional[Path] = None,
    ):
        super().__init__(
            env=env,
            agent=agent,
            memory=memory or PassThroughMemory(),
            output_dir=output_dir,
            verbose=verbose,
        )
        self.max_steps = max_steps
        # Prefix-injection parameters. When `prefix_trajectory_dir`
        # is set, run_episode() looks up `<dir>/<task_id>/trajectory.json`
        # and replays the first K recorded actions before handing off to
        # the live agent. `prefix_steps` is absolute; if None, K is rounded
        # from `prefix_fraction * len(recorded_steps)`. This keeps tasks
        # with wildly different lengths comparable (a 4-step paypal task
        # and a 33-step gitlab task both get the same mid-trajectory
        # handoff).
        self.prefix_trajectory_dir = (
            Path(prefix_trajectory_dir) if prefix_trajectory_dir else None
        )
        self.prefix_steps = prefix_steps
        self.prefix_fraction = prefix_fraction

        # Smart replay (live): follow baseline trajectory in the real
        # browser, applying memory at each step.  At non-compaction steps,
        # use the recorded baseline action (no policy LLM call).  At
        # compaction steps, query the policy model: if it matches baseline,
        # keep following; if it diverges, switch to fully live mode.  This
        # gives real browser-evaluated success rates while saving ~60-90 %
        # of policy calls.  Mutually exclusive with prefix_trajectory_dir.
        self.smart_replay_trajectory_dir = (
            Path(smart_replay_trajectory_dir) if smart_replay_trajectory_dir else None
        )

    def run_episode(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        example_id = self._example_id_for(task_config)
        episode_dir = self.output_dir / str(example_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Reset env / agent / memory / recorder. WebArenaMemoryEnv is built
        # with task_id/app_name in __init__, so reset uses gymnasium-style
        # (seed, options) — task_config is handed through `options` for any
        # env that wants per-episode hints, not unpacked as kwargs.
        obs, info = self.env.reset(seed=seed, options=task_config)
        self.agent.reset()
        self.memory.reset()
        self.recorder.reset()

        # Hand episode-scoped state (task instruction, app name, observation
        # mode, ...) to both the agent and the memory manager. This mirrors
        # webarena-infinity's upstream contract where the task lives on the
        # agent for the whole episode rather than being woven into per-step
        # observations. Memory adapters that summarize need it too so their
        # summaries stay task-relevant. See agents/base.py and memory/base.py
        # — both bases ship a no-op default, so non-webarena strategies are
        # unaffected.
        episode_state = {k: v for k, v in info.items() if v is not None}
        self.agent.set_episode_state(**episode_state)
        self.memory.set_episode_state(**episode_state)
        self.recorder.start_episode(metadata={
            "example_id": example_id,
            "instruction": info.get("instruction", ""),
            "app_name": info.get("app_name", ""),
            "observation_mode": info.get("observation_mode", ""),
            "memory_strategy": self.memory.__class__.__name__,
        })

        history: List[Dict[str, Any]] = []
        total_reward = 0.0
        step_idx = 0
        done = False
        truncated = False

        # --- SMART REPLAY PHASE (live — optional) ---
        # Follow baseline trajectory in the real browser.  At each step,
        # run memory.manage_context() on the REAL observation.  If memory
        # doesn't compact, use the recorded baseline action (no policy call).
        # If memory compacts, query the policy model; if the agent matches
        # baseline, keep following; if it diverges, fall through to the
        # live loop below for the remainder of the episode.
        smart_replay_info: Dict[str, Any] = {"smart_replay_applied": False}
        if self.smart_replay_trajectory_dir is not None:
            recorded_steps = _load_recorded_steps(
                self.smart_replay_trajectory_dir, str(example_id)
            )
            if recorded_steps:
                smart_replay_info = {
                    "smart_replay_applied": True,
                    "trajectory_dir": str(self.smart_replay_trajectory_dir),
                    "recorded_len": len(recorded_steps),
                    "replay_steps": 0,
                    "compaction_steps": 0,
                    "divergence_step": None,
                    "policy_calls": 0,
                    "translation_ok": 0,
                    "translation_fail": 0,
                }
                logger.info(
                    "episode %s: smart-replay %d recorded steps",
                    example_id, len(recorded_steps),
                )
                for rec_idx, rec_step in enumerate(recorded_steps):
                    step_idx += 1
                    recorded_action = rec_step["action"]
                    recorded_obs = rec_step["observation"]

                    # ---- Element-ID translation ----
                    # The recorded action uses element IDs from the old
                    # session; the current browser has different IDs for
                    # the same elements.  Build a semantic mapping
                    # (role+name) and translate before executing.
                    id_map = _build_id_translation(recorded_obs, obs)
                    translated_action, tr_ok = _translate_action(
                        recorded_action, id_map,
                    )
                    if tr_ok:
                        smart_replay_info["translation_ok"] += 1
                    else:
                        smart_replay_info["translation_fail"] += 1
                        logger.warning(
                            "smart-replay %s step %d: could not translate "
                            "action %r (unmapped element ID)",
                            example_id, step_idx, recorded_action,
                        )

                    # 1. Memory processes the REAL browser observation.
                    filtered = self.memory.manage_context(
                        original_context=history,
                        current_observation=obs,
                        metadata={"step": step_idx, "info": info,
                                  "mode": "smart_replay"},
                    )
                    was_compacted = filtered.metadata.get("was_compacted", False)

                    # 2. Decide action: baseline or agent.
                    if was_compacted:
                        smart_replay_info["compaction_steps"] += 1
                        smart_replay_info["policy_calls"] += 1
                        action_result = self.agent.act(filtered)
                        predicted_action = action_result.action

                        # Compare semantically: the agent uses CURRENT
                        # IDs, so compare against the TRANSLATED recorded
                        # action (also in current IDs).
                        if str(predicted_action).strip() != str(translated_action).strip():
                            # DIVERGENCE — use agent's action and switch to live.
                            action = predicted_action
                            smart_replay_info["divergence_step"] = step_idx
                            logger.info(
                                "smart-replay %s: DIVERGED at step %d "
                                "(predicted=%r translated_baseline=%r) → going live",
                                example_id, step_idx,
                                predicted_action, translated_action,
                            )
                        else:
                            # Match — keep following baseline.
                            action = translated_action
                            if self.verbose:
                                logger.info(
                                    "step %d: COMPACTED, agent matches baseline",
                                    step_idx,
                                )
                    else:
                        # No compaction — use translated baseline action.
                        action = translated_action

                    # 3. Execute in real browser.
                    try:
                        next_obs, reward, terminated, trunc, next_info = self.env.step(action)
                    except Exception:
                        logger.exception(
                            "smart-replay failed on %s step %d action=%r "
                            "(original=%r, translated=%s)",
                            example_id, step_idx, action,
                            recorded_action, tr_ok,
                        )
                        raise
                    total_reward += reward
                    done = terminated
                    truncated = trunc

                    # 4. Record step.
                    reasoning = {
                        "mode": "smart_replay",
                        "recorded_action": recorded_action,
                        "translated_action": translated_action,
                        "translation_ok": tr_ok,
                    }
                    if was_compacted:
                        reasoning["predicted_action"] = str(predicted_action)
                        reasoning["diverged"] = (action != translated_action)
                    self.recorder.record_step(
                        observation=_json_safe(obs),
                        action=action,
                        reward=reward,
                        terminated=done,
                        truncated=truncated,
                        info=_json_safe(next_info),
                        memory_action=filtered.metadata,
                        context=_json_safe(filtered.content) if was_compacted else None,
                        reasoning_action=reasoning,
                    )

                    # 5. Append to raw history.
                    history.append({"role": "user", "content": _history_content(obs)})
                    history.append({"role": "assistant", "content": str(action)})

                    smart_replay_info["replay_steps"] = step_idx
                    obs = next_obs
                    info = next_info

                    if done or truncated:
                        logger.info(
                            "smart-replay %s: episode ended at step %d (reward=%.3f)",
                            example_id, step_idx, total_reward,
                        )
                        break

                    # If we just diverged, break out to live loop.
                    if smart_replay_info["divergence_step"] is not None:
                        break

                logger.info(
                    "smart-replay %s: replayed %d steps, %d compactions, "
                    "%d policy calls, diverged=%s, "
                    "translations ok=%d fail=%d → %s",
                    example_id,
                    smart_replay_info["replay_steps"],
                    smart_replay_info["compaction_steps"],
                    smart_replay_info["policy_calls"],
                    smart_replay_info["divergence_step"] or "never",
                    smart_replay_info["translation_ok"],
                    smart_replay_info["translation_fail"],
                    "LIVE" if (smart_replay_info["divergence_step"] and not done) else "DONE",
                )

        # --- PREFIX REPLAY PHASE (optional) ---
        # If a recorded trajectory is available for this task, replay its
        # first K actions deterministically. memory.manage_context still
        # runs at every prefix step so its view at handoff is the natural
        # continuation of a full run, not a cold-start state.
        prefix_info: Dict[str, Any] = {"prefix_applied": False}
        if self.prefix_trajectory_dir is not None:
            recorded_steps = _load_recorded_steps(
                self.prefix_trajectory_dir, str(example_id)
            )
            if recorded_steps:
                if self.prefix_steps is not None:
                    K = min(self.prefix_steps, len(recorded_steps))
                else:
                    K = max(
                        0,
                        min(
                            len(recorded_steps),
                            math.ceil(self.prefix_fraction * len(recorded_steps)),
                        ),
                    )
                prefix_info = {
                    "prefix_applied": True,
                    "prefix_trajectory_dir": str(self.prefix_trajectory_dir),
                    "prefix_steps_planned": K,
                    "prefix_steps_executed": 0,
                    "recorded_len": len(recorded_steps),
                    "prefix_terminated_early": False,
                }
                logger.info(
                    "episode %s: prefix-replay K=%d of %d recorded steps",
                    example_id, K, len(recorded_steps),
                )
                for rec_step in recorded_steps[:K]:
                    step_idx += 1
                    recorded_action = rec_step["action"]
                    recorded_obs = rec_step["observation"]

                    # Translate element IDs from recorded → current session.
                    id_map = _build_id_translation(recorded_obs, obs)
                    action, tr_ok = _translate_action(recorded_action, id_map)
                    if not tr_ok:
                        logger.warning(
                            "prefix-replay %s step %d: untranslatable "
                            "action %r", example_id, step_idx, recorded_action,
                        )

                    # 1. Memory sees the current observation BEFORE the
                    #    recorded action — exact mirror of live mode so the
                    #    memory manager's triggers fire at the same places.
                    filtered = self.memory.manage_context(
                        original_context=history,
                        current_observation=obs,
                        metadata={"step": step_idx, "info": info,
                                  "mode": "prefix_replay"},
                    )

                    # 2. Env executes the translated action.
                    try:
                        next_obs, reward, terminated, trunc, next_info = self.env.step(action)
                    except Exception:
                        logger.exception(
                            "prefix-replay failed on %s at step %d action=%r "
                            "(original=%r, translated=%s)",
                            example_id, step_idx, action,
                            recorded_action, tr_ok,
                        )
                        raise
                    total_reward += reward
                    done = terminated
                    truncated = trunc

                    # 3. Recorder tags the step as prefix_replay via
                    #    reasoning_action.mode — prefix-injection analyzers use this
                    #    to split trajectories into "prefix" and "tail".
                    self.recorder.record_step(
                        observation=_json_safe(obs),
                        action=action,
                        reward=reward,
                        terminated=done,
                        truncated=truncated,
                        info=_json_safe(next_info),
                        memory_action=filtered.metadata,
                        context=_json_safe(filtered.content) if filtered.metadata.get("was_compacted") else None,
                        reasoning_action={
                            "mode": "prefix_replay",
                            "recorded_action": recorded_action,
                            "translated_action": action,
                            "translation_ok": tr_ok,
                        },
                    )

                    history.append({
                        "role": "user",
                        "content": _history_content(obs),
                    })
                    history.append({
                        "role": "assistant",
                        "content": str(action),
                    })

                    prefix_info["prefix_steps_executed"] = step_idx
                    obs = next_obs
                    info = next_info

                    if done or truncated:
                        prefix_info["prefix_terminated_early"] = True
                        logger.info(
                            "episode %s: prefix terminated the episode at step %d (reward=%.3f)",
                            example_id, step_idx, total_reward,
                        )
                        break

        while not done and step_idx < self.max_steps:
            step_idx += 1

            # --- 1. Memory filters context ---
            filtered = self.memory.manage_context(
                original_context=history,
                current_observation=obs,
                metadata={"step": step_idx, "info": info},
            )

            # --- 2. Agent decides action ---
            action_result = self.agent.act(filtered)
            action = action_result.action

            # --- 3. Environment executes action ---
            next_obs, reward, terminated, trunc, next_info = self.env.step(action)
            total_reward += reward
            done = terminated
            truncated = trunc

            # --- 4. Record trajectory step ---
            self.recorder.record_step(
                observation=_json_safe(obs),
                action=action,
                reward=reward,
                terminated=done,
                truncated=truncated,
                info=_json_safe(next_info),
                memory_action=filtered.metadata,
                context=_json_safe(filtered.content) if filtered.metadata.get("was_compacted") else None,
                reasoning_action=action_result.metadata,
            )

            # --- 5. Append to raw history (NOT the filtered view) ---
            history.append({
                "role": "user",
                "content": _history_content(obs),
            })
            history.append({
                "role": "assistant",
                "content": str(action),
            })

            if self.verbose:
                logger.info(
                    "step %d: action=%r reward=%.3f done=%s tokens=%s",
                    step_idx, action, reward, done,
                    filtered.metadata.get("tokens", "?"),
                )

            obs = next_obs
            info = next_info

        # --- Finalize + save ---
        elapsed = time.time() - start_time
        self.recorder.end_episode()

        memory_stats = self.memory.get_stats()
        result = {
            "example_id": example_id,
            "app_name": info.get("app_name", ""),
            "instruction": info.get("instruction", ""),
            "reward": total_reward,
            "success": total_reward >= 1.0,
            "steps": step_idx,
            "terminated": done,
            "truncated": truncated,
            "max_steps_reached": step_idx >= self.max_steps and not done,
            "memory_stats": memory_stats,
            "compaction_count": memory_stats.get("compaction_count", 0),
            "wall_time_seconds": round(elapsed, 2),
            "prefix_info": prefix_info,
            "smart_replay_info": smart_replay_info,
        }

        # Save trajectory JSON alongside the result.
        # TrajectoryRecorder.to_dict() returns {"num_episodes": ..., "episodes": [...]};
        # we wrap it so per-episode metadata lives at a predictable top-level key.
        traj_path = episode_dir / "trajectory.json"
        trajectory = self.recorder.to_dict()
        trajectory["metadata"] = result
        with traj_path.open("w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        with (episode_dir / "result.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)

        if self.verbose:
            logger.info(
                "episode %s done: reward=%.3f steps=%d success=%s",
                example_id, total_reward, step_idx, result["success"],
            )

        return result

    @staticmethod
    def _example_id_for(task_config: Optional[Dict[str, Any]]) -> str:
        if not task_config:
            return "unknown"
        return str(task_config.get("task_id") or task_config.get("id") or "unknown")


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    """Best-effort convert an arbitrary obj to JSON-serializable form.

    Playwright locators, `Popen`s, and Path objects show up in webarena info
    dicts; `default=str` in json.dump catches them but we also proactively
    strip known unserializable keys so the trajectory stays readable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _load_recorded_actions(prefix_dir: Path, task_id: str) -> List[str]:
    """Return the ordered recorded action strings for a baseline trajectory.

    Looks for `<prefix_dir>/<task_id>/trajectory.json` — the layout the
    runner writes itself, so baseline → replay is a closed loop. Returns
    an empty list (NOT raising) when the file is missing; the caller
    treats that as "no prefix, run live from step 1" so a single replay
    directory can partially cover a task set and the runner gracefully
    falls back per-task.
    """
    traj_path = prefix_dir / task_id / "trajectory.json"
    if not traj_path.exists():
        logger.warning(
            "no recorded trajectory at %s — prefix-replay skipped for this task",
            traj_path,
        )
        return []
    try:
        with traj_path.open() as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("failed to load %s: %s", traj_path, e)
        return []

    episodes = data.get("episodes") or []
    for ep in episodes:
        steps = ep.get("steps") or []
        if steps:
            return [step.get("action") for step in steps if step.get("action") is not None]
    return []


def _history_content(obs: Any) -> str:
    """Turn an observation dict into a plain string for the raw history list.

    The memory manager sees the raw history, so we keep it small and stable
    — just the essential text. Screenshots are dropped (vision-mode memory
    is handled by the multimodal adapter, which is out of scope for the
    default text track).
    """
    if not isinstance(obs, dict):
        return str(obs)
    url = obs.get("url", "")
    title = obs.get("title", "")
    tree = obs.get("accessibility_tree", "")
    if tree:
        return f"URL: {url}\nTitle: {title}\n\n{tree}"
    return f"URL: {url}\nTitle: {title}"


# =============================================================================
# Element-ID translation for replay across browser sessions
# =============================================================================
#
# The accessibility tree tagger (_TAG_INTERACTIVE_JS in env.py) assigns
# sequential numeric IDs based on DOM traversal order.  These IDs are
# session-specific: the same page in a different browser session may assign
# different numbers to the same elements.  Recorded actions like
# ``click [191]`` therefore fail in a new session.
#
# The functions below solve this by matching elements on their semantic
# identity (role + accessible name) rather than their numeric ID.  During
# replay, we build a {old_id → new_id} map from the recorded and current
# observation's a11y trees and translate actions before executing them.

_A11Y_ELEMENT_RE = re.compile(r"\[(\d+)\]\s+(\w+)(?:\s+\"([^\"]*)\")?")


def _parse_a11y_elements(obs: Any) -> List[Tuple[int, str, str]]:
    """Extract ``(id, role, name)`` tuples from an observation.

    Parses lines like ``[191] button "Create Label"`` from the
    accessibility tree text.  *name* is the empty string when absent.
    """
    if isinstance(obs, dict):
        tree = obs.get("accessibility_tree", "")
    elif isinstance(obs, str):
        tree = obs
    else:
        return []

    return [
        (int(m.group(1)), m.group(2), m.group(3) or "")
        for m in _A11Y_ELEMENT_RE.finditer(tree)
    ]


def _normalize(s: str) -> str:
    """Lower-case + collapse whitespace for fuzzy name matching."""
    return " ".join(s.lower().split())


def _build_id_translation(
    recorded_obs: Any,
    current_obs: Any,
) -> Dict[int, int]:
    """Map old element IDs → current element IDs via (role, name) matching.

    For duplicate ``(role, name)`` pairs (e.g. two "Edit" buttons), uses
    occurrence index — the *N*-th element with a given key in the recorded
    tree maps to the *N*-th element with the same key in the current tree.

    Returns ``{old_id: new_id}``.  IDs that cannot be mapped are omitted.
    """
    old_elems = _parse_a11y_elements(recorded_obs)
    new_elems = _parse_a11y_elements(current_obs)

    # current observation → {(role, name): [id, id, ...]}
    new_by_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for eid, role, name in new_elems:
        new_by_key[(_normalize(role), _normalize(name))].append(eid)

    # Walk old elements in order, consuming occurrence slots
    old_occ: Dict[Tuple[str, str], int] = defaultdict(int)
    id_map: Dict[int, int] = {}
    for eid, role, name in old_elems:
        key = (_normalize(role), _normalize(name))
        idx = old_occ[key]
        old_occ[key] += 1
        candidates = new_by_key.get(key, [])
        if idx < len(candidates):
            id_map[eid] = candidates[idx]

    return id_map


# Matches the element-ID bracket in click/type/select actions.
# Group 1 = action verb, group 2 = numeric ID.
_ACTION_ELEM_RE = re.compile(r"(click|type|select)\s*\[(\d+)\]", re.IGNORECASE)


def _translate_action(
    action_text: str,
    id_map: Dict[int, int],
) -> Tuple[str, bool]:
    """Replace the element ID in *action_text* using *id_map*.

    Returns ``(translated_action, success)``.  ``success`` is False when the
    action references an element ID that has no mapping (caller should treat
    this as an untranslatable action and handle accordingly).

    Actions without element IDs (goto, key, scroll, stop, …) pass through
    unchanged with ``success=True``.
    """
    m = _ACTION_ELEM_RE.match(action_text or "")
    if m is None:
        # No element ID to translate.
        return (action_text or ""), True

    old_id = int(m.group(2))
    new_id = id_map.get(old_id)
    if new_id is None:
        return action_text, False

    return action_text[: m.start(2)] + str(new_id) + action_text[m.end(2) :], True


def _load_recorded_steps(
    prefix_dir: Path,
    task_id: str,
) -> List[Dict[str, Any]]:
    """Load recorded steps with both action and observation for ID translation.

    Returns a list of ``{"action": str, "observation": dict}`` dicts.
    Falls back gracefully (empty list) when the file is missing.
    """
    traj_path = prefix_dir / task_id / "trajectory.json"
    if not traj_path.exists():
        logger.warning("no trajectory at %s — skipped", traj_path)
        return []
    try:
        with traj_path.open() as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("failed to load %s: %s", traj_path, e)
        return []

    for ep in data.get("episodes") or []:
        steps = ep.get("steps") or []
        if steps:
            return [
                {"action": s.get("action"), "observation": s.get("observation")}
                for s in steps
                if s.get("action") is not None
            ]
    return []
