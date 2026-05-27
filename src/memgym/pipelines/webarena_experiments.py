"""
WebArena experiment runner — baseline-first memory comparison pipeline.

Codifies two project principles into a single entry point:

1. **Baseline-first**: a memory-strategy comparison is only scientifically
   meaningful if the *same* agent has been measured without memory first on
   the *same* task set. This runner always launches the baseline (memory=none)
   run first, then gates subsequent memory-strategy runs on a user-defined
   validity predicate — see `should_run_memory_comparison()` below.

2. **Collision-free, trackable trajectory directories**: every experiment
   lands under `results/webarena/exp_<timestamp>_<slug>/` with one child
   directory per strategy (`baseline/`, `summarizing/`, `structured/`, ...).
   The timestamp prefix guarantees uniqueness even if the slug repeats; the
   slug keeps the path greppable. A `manifest.json` at the experiment root
   records the locked agent invariants so any sibling run that drifts from
   them can be detected at analysis time.

Typical usage (submitted to the remote job launcher, not run on the local box):

    from memgym.pipelines.webarena_experiments import ExperimentSpec, run_experiment

    spec = ExperimentSpec(
        slug="haiku_gitlab8",
        policy_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        app_name="gitlab-plan-and-track",
        task_ids=["task_1", "task_2", "task_3", "task_4",
                  "task_5", "task_6", "task_7", "task_8"],
        max_steps=25,
        observation_mode="text",
        memory_strategies=[
            {"name": "webarena_summarizing",
             "max_size": 20, "keep_first": 1, "condensation_ratio": 0.75},
            {"name": "webarena_structured",
             "max_size": 20, "keep_first": 1, "condensation_ratio": 0.75},
        ],
    )
    run_experiment(spec, base_url="http://<runner-host>:30000")

The runner does NOT execute trajectories itself — it submits sibling
`/run-webarena` jobs to the remote launcher, so parallelism, disk layout,
and job-queue etiquette all flow through the same infrastructure the
rest of the project uses.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memgym.webarena.experiments")


# =============================================================================
# Invariant locking — the set of parameters that MUST match across baseline
# and every memory-strategy sibling run. Drift on any of these contaminates
# the memory-contribution measurement.
# =============================================================================

# These field names are the parameters fair-comparison requires to stay
# identical across baseline and memory runs. `run_experiment` copies them
# verbatim from the ExperimentSpec into each per-strategy job payload, and
# the manifest records them so later analyzers can verify nothing drifted.
LOCKED_INVARIANT_FIELDS = (
    "policy_model",
    "policy_max_tokens",
    "policy_temperature",
    "max_steps",
    "observation_mode",
    "max_tokens",
    "app_name",
    "task_ids",
    "headless",
    "workers",
    "max_workers",
    "port_range",
    "seed",
)


# =============================================================================
# ExperimentSpec — declarative description of a whole experiment
# =============================================================================

@dataclass
class MemoryStrategyConfig:
    """One memory-strategy variant in an experiment.

    Every field except `name` is strategy-specific (e.g. max_size only
    applies to summarizing/structured; other strategies may ignore it).
    The runner forwards all fields as-is to `/run-webarena`.
    """
    name: str  # registered memory model name, e.g. "webarena_summarizing"
    max_size: Optional[int] = None
    keep_first: Optional[int] = None
    condensation_ratio: Optional[float] = None
    summarization_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """Declarative description of a baseline-first memory experiment.

    The fields partition into two groups:

    (1) Locked invariants — copied into every sibling run (baseline + each
        memory strategy) so the only thing that differs between them is the
        memory configuration. See `LOCKED_INVARIANT_FIELDS`.

    (2) Orchestration metadata — slug (for the dir name), strategy list,
        optional summarization model override, etc.
    """
    slug: str                              # short identifier, used in dir name
    policy_model: str                      # must be the same across all siblings
    app_name: str
    task_ids: List[str]
    memory_strategies: List[MemoryStrategyConfig]

    # Locked invariants with sensible defaults
    policy_max_tokens: int = 512
    policy_temperature: float = 0.0
    max_steps: int = 25
    observation_mode: str = "text"
    max_tokens: int = 16_000
    headless: bool = True
    workers: int = 4
    max_workers: int = 4
    port_range: str = "18000-19000"
    seed: Optional[int] = None

    # Orchestration metadata (not locked)
    default_summarization_model: Optional[str] = None
    venv: str = "${VENV_DIR}"
    results_root: str = "results/webarena"
    description: str = ""


# =============================================================================
# Hierarchical directory layout with collision avoidance
# =============================================================================

def _timestamp() -> str:
    """Compact sortable timestamp, e.g. 20260408T154750 (UTC not used on purpose:
    the project tracks times in the user's local timezone everywhere else)."""
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def build_experiment_dir(
    spec: ExperimentSpec,
    *,
    existing_dirs: Optional[List[str]] = None,
) -> str:
    """Build the experiment parent-dir name and guarantee it is new.

    Format: ``exp_<YYYYMMDDTHHMMSS>_<slug>``

    The timestamp makes the directory sortable by creation time and unique
    even when the slug repeats (two runs of the "haiku_gitlab8" slug within
    the same second are extraordinarily unlikely; within the same day they
    are differentiated by the time component).

    Args:
        spec: the experiment spec (only ``spec.slug`` is read).
        existing_dirs: optional list of existing directory names under
            results_root/. If provided, the returned name is guaranteed not
            to collide with any of them — an incrementing suffix (_v2, _v3,
            ...) is appended if necessary. When not provided, the caller is
            responsible for race-free creation (the remote runner that calls
            `run_experiment` currently holds this responsibility).

    Returns:
        The relative directory name (not a full path), e.g.
        ``"exp_20260408T154750_haiku_gitlab8"``.
    """
    base = f"exp_{_timestamp()}_{spec.slug}"
    if not existing_dirs:
        return base

    existing = set(existing_dirs)
    if base not in existing:
        return base

    # Collision (same-second re-run or recycled timestamp). Append _vN.
    for n in range(2, 1000):
        candidate = f"{base}_v{n}"
        if candidate not in existing:
            return candidate
    raise RuntimeError(f"could not find non-colliding name for {base}")


def write_manifest(exp_dir: Path, spec: ExperimentSpec) -> Path:
    """Write manifest.json at the experiment root.

    The manifest is the ground-truth record of which invariants were locked
    at submission time. A later analyzer can re-read each per-strategy
    `args.json` and verify it matches the manifest field-by-field — any
    mismatch means the comparison is contaminated and should be rejected.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": _dt.datetime.now().isoformat(),
        "slug": spec.slug,
        "description": spec.description,
        "locked_invariants": {
            field: getattr(spec, field)
            for field in LOCKED_INVARIANT_FIELDS
            if hasattr(spec, field)
        },
        "strategies": [
            {"name": "baseline", "memory_model": "none"},
        ] + [
            {"name": s.name.removeprefix("webarena_") or s.name,
             "memory_model": s.name, **{
                 k: v for k, v in asdict(s).items()
                 if k != "name" and v is not None and v != {}
             }}
            for s in spec.memory_strategies
        ],
    }
    manifest_path = exp_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote manifest: %s", manifest_path)
    return manifest_path


# =============================================================================
# Validity predicate — USER-CONTRIBUTED DECISION
# =============================================================================

def should_run_memory_comparison(baseline_summary: Dict[str, Any]) -> tuple[bool, str]:
    """Decide whether running memory strategies after this baseline is meaningful.

    This predicate is the scientific gatekeeper for the whole pipeline: given
    the baseline (memory=none) results on a task set, it decides whether a
    memory-strategy comparison on the same tasks has any chance of producing
    a meaningful signal. Returns (True, reason) to proceed with memory runs,
    (False, reason) to skip them. The `reason` string is written to the
    manifest and to the skip log so later readers know why memory was (or
    wasn't) evaluated on this baseline.

    `baseline_summary` is the dict loaded from the baseline run's
    `summary.json`. Keys you can inspect include:
      - n_tasks           : int (total tasks in the run)
      - n_success         : int (how many succeeded)
      - success_rate      : float in [0, 1]
      - avg_steps         : float (mean steps across all tasks)
      - wall_time_seconds : float (total wall clock)
      - results           : list of per-task dicts, each with:
            {example_id, success (bool), steps (int), reward (float),
             max_steps_reached (bool), terminated (bool), wall_time_seconds, ...}

    Cases to think through (there is no objectively-right answer — this
    reflects research judgment):

      * success_rate == 1.0 with short successes
            Memory has no headroom. You measure noise if you proceed.
      * success_rate == 0.0 (or nearly so)
            Usually an infrastructure or prompting bug, not a memory-testable
            scenario. Memory cannot rescue an agent that cannot act.
      * 0 < success_rate < 1.0 with at least some failures hitting max_steps
            The sweet spot — the agent *can* act, some tasks *do* need longer
            reasoning chains, and memory has headroom to help or hurt.
      * Nearly all tasks bail out early (1-2 steps each)
            Infrastructure issue (broken parser, wrong URL convention, etc).

    TODO (USER): write 5-10 lines implementing this predicate. Consider what
    minimum/maximum thresholds make sense for *your* research questions, and
    whether the presence of `max_steps_reached` failures should shift the
    decision. Return False early with a specific reason when the baseline
    clearly has no memory-testable headroom.
    """
    # Placeholder — replace with the user's decision logic.
    raise NotImplementedError(
        "should_run_memory_comparison: user-defined predicate not yet "
        "implemented. See the docstring above for the contract and the "
        "cases that need to be covered."
    )


# =============================================================================
# Job submission (remote launcher client)
# =============================================================================

def build_run_payload(
    spec: ExperimentSpec,
    strategy_name: str,
    memory_model: str,
    strategy_config: Optional[MemoryStrategyConfig],
    result_dir: str,
) -> Dict[str, Any]:
    """Construct a `/run-webarena` request body for one sibling run.

    Locked invariants are copied verbatim from `spec`. Strategy-specific
    parameters come from `strategy_config` (or are omitted for the baseline).
    """
    payload: Dict[str, Any] = {
        # Locked invariants — identical across every sibling run
        "policy_model": spec.policy_model,
        "policy_max_tokens": spec.policy_max_tokens,
        "policy_temperature": spec.policy_temperature,
        "max_steps": spec.max_steps,
        "observation_mode": spec.observation_mode,
        "max_tokens": spec.max_tokens,
        "app_name": spec.app_name,
        "task_ids": ",".join(spec.task_ids),
        "headless": spec.headless,
        "workers": spec.workers,
        "max_workers": spec.max_workers,
        "port_range": spec.port_range,
        # Strategy selector + result directory
        "memory_model": memory_model,
        "result_dir": result_dir,
        "venv": spec.venv,
    }
    if spec.seed is not None:
        payload["seed"] = spec.seed

    # Strategy-specific extras only populate when provided
    if strategy_config is not None:
        if strategy_config.max_size is not None:
            payload["max_size"] = strategy_config.max_size
        if strategy_config.keep_first is not None:
            payload["keep_first"] = strategy_config.keep_first
        if strategy_config.condensation_ratio is not None:
            payload["condensation_ratio"] = strategy_config.condensation_ratio
        summ_model = (
            strategy_config.summarization_model
            or spec.default_summarization_model
        )
        if summ_model is not None:
            payload["summarization_model"] = summ_model
        for k, v in strategy_config.extra.items():
            payload[k] = v

    return payload


# =============================================================================
# Top-level orchestration
# =============================================================================

def plan_experiment(
    spec: ExperimentSpec,
    *,
    existing_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the full submission plan without executing it.

    Returns a dict describing the experiment parent dir, the manifest
    contents, and the per-strategy job payloads (including baseline). Used
    by `run_experiment` as the inner planning step and also exported so
    callers can dry-run an experiment by inspecting the planned payloads
    before any job hits the wire.
    """
    exp_dir_name = build_experiment_dir(spec, existing_dirs=existing_dirs)
    exp_rel_root = f"{spec.results_root}/{exp_dir_name}"

    plan: Dict[str, Any] = {
        "experiment_dir": exp_rel_root,
        "baseline": {
            "result_dir": f"{exp_rel_root}/baseline",
            "payload": build_run_payload(
                spec,
                strategy_name="baseline",
                memory_model="none",
                strategy_config=None,
                result_dir=f"{exp_rel_root}/baseline",
            ),
        },
        "memory_strategies": [],
    }
    for strat in spec.memory_strategies:
        # Shorten webarena_summarizing → summarizing for directory readability
        short = strat.name.removeprefix("webarena_") or strat.name
        strat_result_dir = f"{exp_rel_root}/{short}"
        plan["memory_strategies"].append({
            "name": short,
            "memory_model": strat.name,
            "result_dir": strat_result_dir,
            "payload": build_run_payload(
                spec,
                strategy_name=short,
                memory_model=strat.name,
                strategy_config=strat,
                result_dir=strat_result_dir,
            ),
        })
    return plan
