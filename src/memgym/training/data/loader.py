"""Unified trajectory loading from ec2_trajectories/.

Handles three JSON formats transparently:
  - _training.json : per-step memory data (was_compacted, summary, etc.)
  - _replay.json   : full message history + patch + memory config
  - .traj.json     : full episode/step structure with metadata

Usage:
    loader = TrajectoryLoader("/path/to/ec2_trajectories")
    trajs = loader.load_experiment("train50_sonnet45_llmsummarizing")
    labels = loader.load_reeval_labels("path/to/reeval_report.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough chars-per-token for estimation when tiktoken is unavailable
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MemoryStepData:
    """Per-step memory metadata from _training.json.

    ``was_compacted`` is the "view is currently post-summary" signal and
    goes sticky-True after the first compaction; ``new_compaction`` is the
    per-step event signal (True only on the step where summarization ran).
    SWE-bench trajectories only ever set ``was_compacted`` (pre-dates the
    split), so ``new_compaction`` defaults False and stays a tau2-only
    signal until the SWE capture is backfilled.
    """

    original_tokens: int = 0
    filtered_tokens: int = 0
    compression_ratio: float = 1.0
    was_compacted: bool = False
    new_compaction: bool = False
    summary: Optional[str] = None
    view_size: Optional[int] = None
    forgotten: Optional[int] = None
    original_msgs: Optional[int] = None
    filtered_msgs: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryStepData":
        return cls(
            original_tokens=d.get("original_tokens", 0),
            filtered_tokens=d.get("filtered_tokens", 0),
            compression_ratio=d.get("compression_ratio", 1.0),
            was_compacted=d.get("was_compacted", False),
            new_compaction=d.get("new_compaction", False),
            summary=d.get("summary"),
            view_size=d.get("view_size"),
            forgotten=d.get("forgotten"),
            original_msgs=d.get("original_msgs"),
            filtered_msgs=d.get("filtered_msgs"),
        )


@dataclass
class StepData:
    """A single agent step with optional memory metadata."""

    step: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    memory: MemoryStepData = field(default_factory=MemoryStepData)

    @classmethod
    def from_training_dict(cls, d: Dict[str, Any]) -> "StepData":
        """Parse from _training.json step format."""
        return cls(
            step=d.get("step", 0),
            thought=d.get("thought", ""),
            action=d.get("action", ""),
            observation=d.get("observation", ""),
            memory=MemoryStepData.from_dict(d.get("memory", {})),
        )


@dataclass
class LoadedTrajectory:
    """A fully loaded trajectory with all available data."""

    instance_id: str
    model: str = ""
    memory_strategy: str = ""
    memory_config: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # "Submitted", "LimitsExceeded", etc.
    resolved: Optional[bool] = None
    patch: str = ""
    problem_statement: str = ""

    # Per-step data (from _training.json — empty for baselines)
    steps: List[StepData] = field(default_factory=list)
    condensation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Full message history (from _replay.json)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    message_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate stats
    num_steps: int = 0
    total_original_tokens: int = 0
    total_filtered_tokens: int = 0

    @property
    def has_step_data(self) -> bool:
        """Whether per-step memory data is available."""
        return len(self.steps) > 0

    @property
    def compaction_count(self) -> int:
        """Number of compaction events in the trajectory."""
        return sum(1 for s in self.steps if s.memory.was_compacted)

    @property
    def avg_compression_ratio(self) -> float:
        if self.total_filtered_tokens > 0:
            return self.total_original_tokens / self.total_filtered_tokens
        return 1.0


# ---------------------------------------------------------------------------
# Tau2 dataclasses (distinct from SWE because of two-sided memory + the
# ``(phase, domain, task_id)`` composite key). The step dataclass is
# deliberately minimal — the agent's action text isn't flattened into
# ``thought/action/observation`` strings the way SWE's ReAct loop is;
# instead, the full tau2 Message lives at ``messages[msg_index]`` in the
# replay, and event extractors dereference as needed.
# ---------------------------------------------------------------------------

@dataclass
class Tau2StepData:
    """One wrapper-produced turn in a tau2 dialogue, either agent or user side."""

    step: int
    turn_idx: int
    msg_index: int
    side: str  # "agent" | "user"
    memory: MemoryStepData = field(default_factory=MemoryStepData)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tau2StepData":
        return cls(
            step=d.get("step", 0),
            turn_idx=d.get("turn_idx", 0),
            msg_index=d.get("msg_index", -1),
            side=d.get("side", "agent"),
            memory=MemoryStepData.from_dict(d.get("memory", {})),
        )


@dataclass
class Tau2LoadedTrajectory:
    """A tau2-bench episode with per-side steps, messages, and condensation history.

    Keyed by ``(phase, domain, task_id)`` — the ``instance_key`` property
    formats it the way downstream dataset dicts expect. ``track`` is the
    SWE-vs-tau2 discriminator for any unified consumer code.
    """

    domain: str
    task_id: str
    phase: str = ""  # "baseline" | "memory"

    memory_strategy_agent: str = ""
    memory_strategy_user: str = ""
    memory_side: str = ""
    episode_outcome: str = ""
    episode_reward: float = 0.0

    # Per-step data (both sides in a single flat list — filter via ``.side``)
    steps: List[Tau2StepData] = field(default_factory=list)
    # Two-sided summaries kept by the memory manager itself; mirrors
    # ``Tau2SummarizingMemory.get_condensation_history()`` / tau2_structured's
    # implementation — shape ``{"agent": [...], "user": [...]}``.
    condensation_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Replay data
    messages: List[Dict[str, Any]] = field(default_factory=list)
    agent_step_index: List[Dict[str, Any]] = field(default_factory=list)
    user_step_index: List[Dict[str, Any]] = field(default_factory=list)

    track: str = "tau2"

    @property
    def instance_key(self) -> str:
        """Composite key used as the dict key in ``{iid: Tau2LoadedTrajectory}``."""
        parts = [p for p in (self.phase, self.domain, self.task_id) if p]
        return "/".join(parts)

    @property
    def agent_steps(self) -> List[Tau2StepData]:
        return [s for s in self.steps if s.side == "agent"]

    @property
    def user_steps(self) -> List[Tau2StepData]:
        return [s for s in self.steps if s.side == "user"]

    @property
    def agent_compaction_count(self) -> int:
        return sum(1 for s in self.agent_steps if s.memory.new_compaction)

    @property
    def user_compaction_count(self) -> int:
        return sum(1 for s in self.user_steps if s.memory.new_compaction)


# ---------------------------------------------------------------------------
# Loading functions (refactored from build_pairs.py)
# ---------------------------------------------------------------------------

def load_reeval_report(path: Path) -> Dict[str, bool]:
    """Load reeval report and return {instance_id: resolved}.

    Refactored from build_pairs.py:load_reeval_report (L25-35).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("reeval report not found at %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    resolved_ids = set(data.get("resolved", []))
    all_ids = set(data.get("resolved", [])) | set(data.get("unresolved", []))
    return {iid: iid in resolved_ids for iid in all_ids}


def _load_training_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load _training.json and parse into structured data.

    Refactored from build_pairs.py:load_training_json (L63-101).
    """
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_replay_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load _replay.json with message history and metadata.

    Refactored from build_pairs.py:load_replay (L38-60).
    """
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_traj_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load .traj.json with full episode/step structure."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def load_trajectory(
    traj_dir: Path,
    instance_id: str,
    *,
    load_messages: bool = True,
) -> Optional[LoadedTrajectory]:
    """Load a single instance trajectory from all available files.

    Merges data from _training.json (per-step), _replay.json (messages),
    and .traj.json (episode structure) into a unified LoadedTrajectory.

    Also recognizes _forked.json (fork-batch output) as a _replay.json
    variant — same schema, with extra provenance fields (fork_step,
    original_strategy, training_trajectory) that the loader ignores.
    """
    training_path = traj_dir / f"{instance_id}_training.json"
    replay_path = traj_dir / f"{instance_id}_replay.json"
    forked_path = traj_dir / f"{instance_id}_forked.json"
    traj_path = traj_dir / f"{instance_id}.traj.json"

    training_data = _load_training_json(training_path)
    replay_data = None
    if load_messages:
        # Prefer _replay.json; fall back to _forked.json (fork-batch output).
        replay_data = _load_replay_json(replay_path)
        if replay_data is None:
            replay_data = _load_replay_json(forked_path)
    traj_data = _load_traj_json(traj_path) if not training_data and not replay_data else None

    if training_data is None and replay_data is None and traj_data is None:
        return None

    traj = LoadedTrajectory(instance_id=instance_id)

    # --- Fill from _training.json (richest source for memory data) ---
    if training_data:
        traj.model = training_data.get("model", "")
        traj.memory_strategy = training_data.get("memory_strategy", "")
        traj.problem_statement = training_data.get("problem", "")
        traj.outcome = training_data.get("outcome", "")
        traj.patch = training_data.get("patch", "")
        traj.condensation_history = training_data.get("condensation_history", [])
        traj.num_steps = training_data.get("num_steps", 0)

        raw_steps = training_data.get("steps", [])
        traj.steps = [StepData.from_training_dict(s) for s in raw_steps]

        traj.total_original_tokens = sum(
            s.memory.original_tokens for s in traj.steps
        )
        traj.total_filtered_tokens = sum(
            s.memory.filtered_tokens for s in traj.steps
        )

    # --- Fill from _replay.json (message history + config) ---
    if replay_data:
        if not traj.model:
            traj.model = replay_data.get("model", "")
        if not traj.memory_strategy:
            traj.memory_strategy = replay_data.get("memory_strategy", "")
        if not traj.outcome:
            traj.outcome = replay_data.get("status", "")
        if not traj.patch:
            traj.patch = replay_data.get("patch", "")

        traj.memory_config = replay_data.get("memory_config", {})
        traj.messages = replay_data.get("messages", [])
        traj.message_steps = replay_data.get("steps", [])

        if not traj.num_steps:
            traj.num_steps = len(traj.message_steps)

        # Estimate tokens from messages if not set from training.json
        if traj.total_original_tokens == 0:
            traj.total_original_tokens = sum(
                _estimate_tokens(str(m.get("content", "")))
                for m in traj.messages
            )

    # --- Fill from .traj.json (fallback) ---
    if traj_data and not traj.model:
        episodes = traj_data.get("episodes", [])
        if episodes:
            meta = episodes[0].get("metadata", {})
            traj.model = meta.get("agent_llm", "")
            if not traj.num_steps:
                traj.num_steps = traj_data.get("total_steps", 0)

    return traj


def load_tau2_trajectory(
    task_dir: Path,
    task_id: str,
    *,
    phase: str = "",
    domain: str = "",
) -> Optional[Tau2LoadedTrajectory]:
    """Load a single tau2 episode from its ``<domain>/<task_id>/`` dir.

    Looks for ``{task_id}_training.json`` + ``{task_id}_replay.json`` and
    merges them into a ``Tau2LoadedTrajectory``. ``phase`` + ``domain``
    are optional overrides; if empty we infer them from ``task_dir`` path
    components (``.../<phase>/<domain>/<task_id>/``).
    """
    task_dir = Path(task_dir)
    training_path = task_dir / f"{task_id}_training.json"
    replay_path = task_dir / f"{task_id}_replay.json"

    training_data = _load_training_json(training_path)
    replay_data = _load_replay_json(replay_path)

    if training_data is None and replay_data is None:
        return None

    # Infer phase + domain from directory structure: .../<phase>/<domain>/<task_id>/
    parts = task_dir.parts
    if not domain and len(parts) >= 2:
        domain = parts[-2]
    if not phase and len(parts) >= 3:
        phase = parts[-3]

    traj = Tau2LoadedTrajectory(
        domain=domain,
        task_id=task_id,
        phase=phase,
    )

    if training_data:
        traj.domain = training_data.get("domain") or traj.domain
        traj.task_id = training_data.get("task_id") or traj.task_id
        traj.memory_strategy_agent = training_data.get("memory_strategy_agent", "")
        traj.memory_strategy_user = training_data.get("memory_strategy_user", "")
        traj.memory_side = training_data.get("memory_side", "")
        traj.episode_outcome = training_data.get("episode_outcome", "")
        traj.episode_reward = float(training_data.get("episode_reward") or 0.0)
        traj.steps = [Tau2StepData.from_dict(s) for s in training_data.get("steps", [])]
        cond = training_data.get("condensation_history") or {}
        # Normalize — accept either {"agent": [], "user": []} or a flat list
        # (some early captures may have stored agent-only as a list).
        if isinstance(cond, dict):
            traj.condensation_history = {
                "agent": list(cond.get("agent", []) or []),
                "user": list(cond.get("user", []) or []),
            }
        else:
            traj.condensation_history = {"agent": list(cond or []), "user": []}

    if replay_data:
        if not traj.domain:
            traj.domain = replay_data.get("domain", "")
        traj.messages = replay_data.get("messages", []) or []
        traj.agent_step_index = replay_data.get("agent_steps", []) or []
        traj.user_step_index = replay_data.get("user_steps", []) or []

    return traj


# ---------------------------------------------------------------------------
# TrajectoryLoader — main interface
# ---------------------------------------------------------------------------

class TrajectoryLoader:
    """Load trajectories from ec2_trajectories/ directory structure.

    Usage:
        loader = TrajectoryLoader("/path/to/ec2_trajectories")
        trajs = loader.load_experiment("train50_sonnet45_llmsummarizing")
        labels = loader.load_reeval_labels("path/to/reeval_report.json")
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def load_experiment(
        self,
        dir_name: str,
        *,
        load_messages: bool = True,
        max_instances: Optional[int] = None,
    ) -> Dict[str, LoadedTrajectory]:
        """Load all trajectories from an experiment directory.

        Returns {instance_id: LoadedTrajectory}.
        """
        exp_dir = self.base_dir / dir_name
        traj_dir = exp_dir / "trajectories"
        if not traj_dir.exists():
            traj_dir = exp_dir  # some dirs have files at root

        # Discover instance IDs from any available file type. _forked.json
        # is a fork-batch output variant of _replay.json (same schema, extra
        # provenance fields).
        instance_ids = set()
        for pattern in ("*_training.json", "*_replay.json", "*_forked.json",
                        "*.traj.json"):
            for f in traj_dir.glob(pattern):
                name = f.name
                for suffix in ("_training.json", "_replay.json", "_forked.json",
                               ".traj.json", ".traj_concise.json"):
                    if name.endswith(suffix):
                        instance_ids.add(name[: -len(suffix)])
                        break

        instance_ids = sorted(instance_ids)
        if max_instances:
            instance_ids = instance_ids[:max_instances]

        results = {}
        for iid in instance_ids:
            traj = load_trajectory(traj_dir, iid, load_messages=load_messages)
            if traj is not None:
                results[iid] = traj

        logger.info(
            "Loaded %d/%d trajectories from %s",
            len(results), len(instance_ids), dir_name,
        )
        return results

    def load_reeval_labels(self, path: str | Path) -> Dict[str, bool]:
        """Load resolve labels from reeval report JSON."""
        return load_reeval_report(Path(path))

    def load_tau2_experiment(
        self,
        run_dir: str | Path,
        *,
        phases: Optional[List[str]] = None,
        max_instances: Optional[int] = None,
    ) -> Dict[str, Tau2LoadedTrajectory]:
        """Load tau2 trajectories from a single run directory.

        Layout produced by ``Tau2BenchRunner``:

        ::

            results/tau2_bench/<run_id>/
            ├── args.json, summary.json
            ├── baseline/<domain>/<task_id>/{trajectory,result}.json
            └── memory/<domain>/<task_id>/{trajectory,result,<task_id>_training,<task_id>_replay}.json

        Only the ``memory/`` phase emits training artefacts (the baseline
        phase uses ``PassThroughMemory`` by design — no compaction, nothing
        to capture). ``phases`` defaults to ``("memory",)`` so callers get
        the informative subset; pass ``("baseline", "memory")`` to also
        pick up baseline dirs (which will yield empty ``steps`` but can
        still serve as reward references).

        ``run_dir`` can be passed as an absolute path OR a name relative
        to ``self.base_dir``.
        """
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = self.base_dir / run_path
        if not run_path.exists():
            logger.warning("tau2 run dir not found: %s", run_path)
            return {}

        if phases is None:
            phases = ["memory"]

        results: Dict[str, Tau2LoadedTrajectory] = {}
        for phase in phases:
            phase_dir = run_path / phase
            if not phase_dir.is_dir():
                continue
            # Walk <phase>/<domain>/<task_id>/ — only two levels deep by
            # construction, so a bounded glob keeps this fast on large runs.
            for domain_dir in sorted(phase_dir.iterdir()):
                if not domain_dir.is_dir():
                    continue
                for task_dir in sorted(domain_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    task_id = task_dir.name
                    traj = load_tau2_trajectory(
                        task_dir, task_id,
                        phase=phase, domain=domain_dir.name,
                    )
                    if traj is None:
                        continue
                    results[traj.instance_key] = traj
                    if max_instances and len(results) >= max_instances:
                        break
                if max_instances and len(results) >= max_instances:
                    break
            if max_instances and len(results) >= max_instances:
                break

        logger.info(
            "Loaded %d tau2 trajectories from %s (phases=%s)",
            len(results), run_path.name, phases,
        )
        return results

    def discover_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Scan base_dir and return metadata for all experiment directories."""
        experiments = {}
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            traj_dir = d / "trajectories"
            if not traj_dir.exists():
                # Check for replay_summary.json (analyze-mode dirs)
                if (d / "replay_summary.json").exists():
                    experiments[d.name] = {
                        "type": "replay_summary",
                        "path": str(d),
                    }
                continue

            training_count = len(list(traj_dir.glob("*_training.json")))
            replay_count = len(list(traj_dir.glob("*_replay.json")))
            forked_count = len(list(traj_dir.glob("*_forked.json")))
            traj_count = len(list(traj_dir.glob("*.traj.json")))

            experiments[d.name] = {
                "type": "full",
                "path": str(d),
                "training_json": training_count,
                "replay_json": replay_count,
                "forked_json": forked_count,
                "traj_json": traj_count,
            }

        return experiments
