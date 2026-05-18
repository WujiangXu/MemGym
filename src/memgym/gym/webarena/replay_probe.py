"""
Replay-determinism probe for WebArena-Infinity baseline trajectories.

the baseline-train phase of the "trajectory-prefix + memory continuation" experimental plan:
BEFORE building a runner that force-executes a recorded trajectory prefix
and then hands off to a live agent + memory strategy, we need to know
**whether raw element-ID replay is deterministic**. If it is, we can skip
stable-selector recording and go straight to prefix-injection. If it is
not, we know exactly where determinism breaks and can add a selector
fallback layer.

Given a recorded trajectory JSON:

1. Spawn a fresh server (same app) and launch Playwright.
2. For each recorded step k:
   a. Capture the current live observation (URL + accessibility tree).
   b. Compare it against the recorded observation along three axes:
      - **url**: same full URL (coarse — tells us "same page")
      - **interactive_ids**: count of tagged [N] elements in the a11y text
      - **a11y_body**: normalized accessibility-tree text (the strong check)
   c. Re-execute the recorded action (by string, e.g. "click [12]").
   d. If Playwright raises on the action, that is a hard divergence.
3. At the end, run the task verifier and compare against the recorded
   final reward. If the replay verifier gives the same result as the
   recorded run, end-to-end replay is sound.

The probe deliberately does NOT call any LLM — it is pure I/O + DOM.
Cost is just Chromium + HTTP calls to the webarena-infinity app server.

Output is a JSON report written next to the input trajectory:

    <traj_path>.replay_report.json

Top-level fields:
- status: "deterministic" | "diverged" | "replay_error"
- divergence_step: int or null (first step where any axis diverged)
- divergence_kind: "url" | "id_count" | "a11y_body" | "action_error" | null
- recorded_reward / replay_reward
- per_step: list of {step, url_match, id_count_match, a11y_body_match,
  action_ok, notes}

This intentionally produces a three-way conclusion per trajectory:

- all steps match AND final reward matches -> "deterministic" (proceed
  with raw-ID replay in Phase C without stable-selector fallback)
- at least one step diverged -> "diverged" (add stable-selector fallback
  before Phase C is safe)
- Playwright/server raised unexpectedly -> "replay_error" (infra issue,
  not a determinism statement)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memgym.webarena.replay_probe")


# =============================================================================
# Trajectory loading
# =============================================================================

def load_trajectory(traj_path: Path) -> Dict[str, Any]:
    """Load a webarena trajectory JSON written by `webarena_runner.py`.

    The on-disk format is `{"num_episodes": int, "episodes": [...], "metadata": {...}}`
    (see `webarena_runner.py` lines 192-199). We extract the single relevant
    episode — multi-episode trajectories are not produced by the runner, so
    we take the first non-empty one and error if none exist.
    """
    with traj_path.open() as f:
        data = json.load(f)

    episodes = data.get("episodes") or []
    episode = None
    for ep in episodes:
        steps = ep.get("steps") or []
        if steps:
            episode = ep
            break
    if episode is None:
        raise ValueError(f"no non-empty episode in {traj_path}")

    metadata = data.get("metadata") or episode.get("metadata") or {}
    steps = episode.get("steps") or []

    return {
        "app_name": metadata.get("app_name") or episode.get("metadata", {}).get("app_name"),
        "task_id": metadata.get("example_id") or episode.get("metadata", {}).get("example_id"),
        "recorded_reward": float(metadata.get("reward") or episode.get("total_reward") or 0.0),
        "recorded_steps": steps,
    }


# =============================================================================
# Observation comparison
# =============================================================================

# A line in the tagged accessibility-tree block starts with "[N]" — we count
# these to see if the interactive-element set is the same size. The exact
# format comes from `_format_interactive_element()` in env.py.
_ID_LINE_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)

# URL fragment + query params aren't always stable between runs (some
# webarena-infinity apps append cache-busting tokens). Also strip the
# scheme + host + port because the server_pool assigns a fresh port to
# each spawn (e.g. recorded run got 18600, probe run got 19500) — that
# is a pool artifact, not a semantic difference. We keep only the path,
# which is what actually indicates "same page" across runs.
_URL_TRIM_RE = re.compile(r"[?#].*$")
_URL_SCHEME_HOST_RE = re.compile(r"^https?://[^/]*")


def _normalize_url(url: str) -> str:
    """Return the path-only URL with query/fragment stripped.

    Dropping the scheme+host+port means `http://127.0.0.1:18600/gmail/inbox`
    and `http://127.0.0.1:19500/gmail/inbox` both normalize to `/gmail/inbox`,
    so they compare equal for the determinism probe. Apps that genuinely
    navigate cross-host will still diverge on the path prefix.
    """
    if not url:
        return ""
    trimmed = _URL_TRIM_RE.sub("", url)
    path_only = _URL_SCHEME_HOST_RE.sub("", trimmed)
    return path_only.rstrip("/") or "/"


def _count_ids(a11y: str) -> int:
    return len(_ID_LINE_RE.findall(a11y or ""))


def _normalize_a11y(a11y: str) -> str:
    """Strip cosmetic whitespace differences for byte-level equality.

    Tab/space runs collapse to a single space; blank lines drop; trailing
    whitespace stripped. This tolerates minor reformatting from browser
    version drift without losing content equality.
    """
    if not a11y:
        return ""
    lines = []
    for raw in a11y.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def compare_observations(
    recorded: Dict[str, Any],
    live: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a per-axis equality report for one (recorded, live) pair."""
    rec_url = _normalize_url(recorded.get("url", ""))
    live_url = _normalize_url(live.get("url", ""))

    rec_a11y = recorded.get("accessibility_tree", "") or ""
    live_a11y = live.get("accessibility_tree", "") or ""

    rec_id_count = _count_ids(rec_a11y)
    live_id_count = _count_ids(live_a11y)

    rec_norm = _normalize_a11y(rec_a11y)
    live_norm = _normalize_a11y(live_a11y)

    return {
        "url_match": rec_url == live_url,
        "rec_url": rec_url,
        "live_url": live_url,
        "id_count_match": rec_id_count == live_id_count,
        "rec_id_count": rec_id_count,
        "live_id_count": live_id_count,
        "a11y_body_match": rec_norm == live_norm,
        "a11y_body_len_rec": len(rec_norm),
        "a11y_body_len_live": len(live_norm),
    }


# =============================================================================
# Probe driver
# =============================================================================

def run_probe(
    traj_path: Path,
    webarena_dir: Path,
    max_steps: Optional[int] = None,
    port_range: Tuple[int, int] = (19500, 19599),
    spawn_timeout_sec: float = 30.0,
) -> Dict[str, Any]:
    """Replay `traj_path` against a fresh env and return a determinism report.

    This function imports `playwright.sync_api` lazily via the env module so
    callers can import the probe without a Playwright install — useful for
    unit tests that stub env construction.
    """
    # Lazy import — the webarena env pulls in playwright which is not
    # present in every dev environment.
    from memgym.gym.webarena.env import WebArenaMemoryEnv
    from memgym.gym.webarena.server_pool import WebArenaServerPool

    traj = load_trajectory(traj_path)
    app_name = traj["app_name"]
    task_id = traj["task_id"]
    recorded_steps = traj["recorded_steps"]
    recorded_reward = traj["recorded_reward"]

    if not app_name or not task_id:
        raise ValueError(
            f"trajectory missing app_name/task_id metadata: {traj_path}"
        )

    step_limit = len(recorded_steps)
    if max_steps is not None:
        step_limit = min(step_limit, max_steps)

    logger.info(
        "probe: %s/%s — %d recorded steps (replaying %d)",
        app_name, task_id, len(recorded_steps), step_limit,
    )

    # Fresh server pool with a private port range so the probe doesn't
    # collide with any concurrent /run-webarena jobs. `port_range` is a
    # (start, end) tuple to match the server_pool constructor contract
    # (see server_pool.py:152).
    pool = WebArenaServerPool(
        webarena_dir=webarena_dir,
        port_range=port_range,
        spawn_timeout_sec=spawn_timeout_sec,
    )

    env = WebArenaMemoryEnv(
        webarena_dir=webarena_dir,
        app_name=app_name,
        task_id=task_id,
        server_pool=pool,
        observation_mode="text",
        headless=True,
    )

    report: Dict[str, Any] = {
        "traj_path": str(traj_path),
        "app_name": app_name,
        "task_id": task_id,
        "recorded_reward": recorded_reward,
        "recorded_step_count": len(recorded_steps),
        "replayed_step_count": 0,
        "per_step": [],
        "status": "unknown",
        "divergence_step": None,
        "divergence_kind": None,
        "replay_reward": None,
        "elapsed_sec": None,
    }

    start = time.time()
    try:
        live_obs, _info = env.reset()

        total_reward = 0.0
        for step_idx in range(step_limit):
            recorded = recorded_steps[step_idx]
            rec_obs = recorded.get("observation") or {}
            rec_action = recorded.get("action")

            cmp = compare_observations(rec_obs, live_obs)
            per_step_entry = {
                "step": step_idx + 1,
                "action": rec_action,
                **cmp,
                "action_ok": None,
                "notes": "",
            }

            first_divergence = None
            if not cmp["url_match"]:
                first_divergence = "url"
            elif not cmp["id_count_match"]:
                first_divergence = "id_count"
            elif not cmp["a11y_body_match"]:
                first_divergence = "a11y_body"

            if first_divergence and report["divergence_step"] is None:
                report["divergence_step"] = step_idx + 1
                report["divergence_kind"] = first_divergence

            # Execute the recorded action regardless — we want to see if
            # Playwright actually succeeds on it, because that's another
            # layer of divergence the observation check misses.
            try:
                next_obs, reward, terminated, truncated, info = env.step(rec_action)
                per_step_entry["action_ok"] = True
                total_reward += float(reward)
                live_obs = next_obs
                report["replayed_step_count"] = step_idx + 1
                if terminated or truncated:
                    per_step_entry["notes"] = (
                        f"env terminated at replay step {step_idx + 1} "
                        f"(reward={reward}, truncated={truncated})"
                    )
                    report["per_step"].append(per_step_entry)
                    break
            except Exception as e:
                per_step_entry["action_ok"] = False
                per_step_entry["notes"] = f"action raised: {type(e).__name__}: {e}"
                if report["divergence_step"] is None:
                    report["divergence_step"] = step_idx + 1
                    report["divergence_kind"] = "action_error"
                report["per_step"].append(per_step_entry)
                break

            report["per_step"].append(per_step_entry)

        report["replay_reward"] = total_reward

        # Classify the run
        if report["divergence_step"] is None:
            if abs(total_reward - recorded_reward) < 1e-6:
                report["status"] = "deterministic"
            else:
                report["status"] = "reward_mismatch"
                report["divergence_kind"] = report["divergence_kind"] or "reward"
        else:
            report["status"] = "diverged"

    except Exception as e:
        logger.exception("replay probe crashed on %s", traj_path)
        report["status"] = "replay_error"
        report["divergence_kind"] = "replay_error"
        report["divergence_step"] = -1
        report["per_step"].append({
            "step": -1,
            "notes": f"probe crashed: {type(e).__name__}: {e}",
        })
    finally:
        try:
            env.close()
        except Exception:
            logger.debug("env.close raised", exc_info=True)
        try:
            pool.shutdown()
        except Exception:
            logger.debug("pool.shutdown raised", exc_info=True)

    report["elapsed_sec"] = round(time.time() - start, 2)
    return report


# =============================================================================
# CLI
# =============================================================================

def _find_trajectories(root: Path) -> List[Path]:
    """Walk `root` for every trajectory.json a runner might have dropped.

    Layout from `webarena_runner.py` lines 192-196:
        <result_dir>/<task_id>/trajectory.json
    """
    return sorted(root.rglob("trajectory.json"))


def _default_webarena_dir() -> Path:
    """Repo-relative path to the upstream webarena-infinity clone.

    Mirrors the fallback logic in `evaluate.py::_resolve_webarena_dir()` so
    the probe can be invoked without `--webarena-dir` in the same layouts
    the main evaluator supports.
    """
    here = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = here / "envs" / "webarena_infinity"
        if candidate.exists():
            return candidate
        candidate = here / "src" / "memgym" / "envs" / "webarena_infinity"
        if candidate.exists():
            return candidate
        here = here.parent
    raise FileNotFoundError(
        "Could not locate webarena-infinity directory. "
        "Pass --webarena-dir explicitly or run `python -m memgym.gym.webarena.install`."
    )


def _parse_port_range(spec: str) -> Tuple[int, int]:
    a, _, b = spec.partition("-")
    start, end = int(a), int(b)
    if start >= end:
        raise ValueError(f"bad port range {spec!r}")
    return start, end


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Probe replay determinism of WebArena baseline trajectories.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single trajectory.json or a result_dir to walk.",
    )
    parser.add_argument(
        "--webarena-dir",
        type=Path,
        default=None,
        help="Path to the webarena-infinity clone (defaults to memgym.gym.webarena.install.default_webarena_dir()).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap replay at N steps (default = replay every recorded step).",
    )
    parser.add_argument(
        "--port-range",
        type=str,
        default="19500-19599",
        help="Private port range for the probe's server pool.",
    )
    parser.add_argument(
        "--spawn-timeout-sec",
        type=float,
        default=30.0,
        help="Per-server spawn timeout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When walking a directory, probe at most N trajectories.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Aggregate JSON report destination (defaults to <path>/replay_probe_summary.json when walking).",
    )
    args = parser.parse_args()

    webarena_dir = args.webarena_dir or _default_webarena_dir()
    if not webarena_dir.exists():
        logger.error("webarena-infinity dir not found: %s", webarena_dir)
        return 2

    if args.path.is_file():
        traj_files = [args.path]
    else:
        traj_files = _find_trajectories(args.path)

    if args.limit is not None:
        traj_files = traj_files[: args.limit]

    if not traj_files:
        logger.error("no trajectory.json files under %s", args.path)
        return 2

    logger.info("probing %d trajectory file(s)", len(traj_files))

    reports: List[Dict[str, Any]] = []
    for i, tf in enumerate(traj_files, start=1):
        logger.info("[%d/%d] %s", i, len(traj_files), tf)
        report = run_probe(
            traj_path=tf,
            webarena_dir=webarena_dir,
            max_steps=args.max_steps,
            port_range=_parse_port_range(args.port_range),
            spawn_timeout_sec=args.spawn_timeout_sec,
        )
        # Write per-trajectory report next to the input file
        per_report = tf.with_suffix(".replay_report.json")
        with per_report.open("w") as f:
            json.dump(report, f, indent=2, default=str)
        reports.append(report)
        logger.info(
            "  -> %s (div_step=%s kind=%s reward=%.2f vs rec=%.2f, %ds)",
            report["status"],
            report["divergence_step"],
            report["divergence_kind"],
            report["replay_reward"] or 0.0,
            report["recorded_reward"],
            int(report["elapsed_sec"] or 0),
        )

    # Aggregate summary
    summary = {
        "total": len(reports),
        "deterministic": sum(1 for r in reports if r["status"] == "deterministic"),
        "diverged": sum(1 for r in reports if r["status"] == "diverged"),
        "reward_mismatch": sum(1 for r in reports if r["status"] == "reward_mismatch"),
        "replay_error": sum(1 for r in reports if r["status"] == "replay_error"),
        "by_divergence_kind": {},
        "first_divergence_step_distribution": {},
        "reports": reports,
    }
    for r in reports:
        if r["divergence_kind"]:
            summary["by_divergence_kind"].setdefault(r["divergence_kind"], 0)
            summary["by_divergence_kind"][r["divergence_kind"]] += 1
        if r["divergence_step"] is not None:
            bucket = str(r["divergence_step"])
            summary["first_divergence_step_distribution"].setdefault(bucket, 0)
            summary["first_divergence_step_distribution"][bucket] += 1

    report_out = args.report_out
    if report_out is None:
        if args.path.is_file():
            report_out = args.path.with_name("replay_probe_summary.json")
        else:
            report_out = args.path / "replay_probe_summary.json"
    with report_out.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "summary: %d deterministic / %d diverged / %d reward_mismatch / %d replay_error (out of %d) -> %s",
        summary["deterministic"],
        summary["diverged"],
        summary["reward_mismatch"],
        summary["replay_error"],
        summary["total"],
        report_out,
    )
    return 0 if summary["replay_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
