#!/usr/bin/env python3
"""Post-hoc compression-ratio aggregator for the wrapped-gyms table.

Why a separate script: SWE-Gym's `summary.json` already exposes
`avg_compression_ratio` (averaged over all per-event ratios at agent
write time, see `gym/swe_bench/agent.py:367-368`). τ2-bench and WebArena
harnesses never persisted that list, so the field is missing from their
`summary.json` files. We recover a defensible per-cohort ratio from
fields that ARE persisted.

Honest disclaimers (written into the output JSON):
  * τ2: only `wrapper_last_compression_ratio` is persisted (one value
    per episode, the LAST event's ratio). We average across episodes
    that had at least one compaction. This is a *last-compaction*
    estimator, not the per-event mean SWE-Gym reports.
  * WebArena: no per-event ratio persisted. We derive
    `max_tokens_seen / token_count` per episode where compaction_count
    > 0, dropping episodes where the ratio is < 1.05 (those are runs
    where the agent kept growing past the cap and `token_count` ended
    up tied to `max_tokens_seen`, giving a meaningless 1.0).
  * SWE-Gym: pulled directly from `memory_stats.avg_compression_ratio`
    in summary.json. This IS the mean of per-event ratios.

Run-mode: designed to execute on EC2 via `/run-script` so the
`${HOME}/results/` tree is local. `--results-root` defaults to
that path. Outputs a JSON dict at `--output`.

Usage (EC2):
    python -m memgym.training.scripts.scan_compression_ratios \
        --results-root ${HOME}/results \
        --output ${REPO_ROOT}/training_output/compression_ratios.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CohortStats:
    cohort: str
    domain: str  # 'swegym' | 'tau2' | 'webarena'
    path: str
    n_episodes: int
    n_compaction_episodes: int
    total_compactions: int
    avg_compression_ratio: float | None
    median_compression_ratio: float | None
    method: str  # which derivation was used
    notes: list[str] = field(default_factory=list)


def _load_json(p: Path) -> dict[str, Any] | None:
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return None


def scan_swegym(results_root: Path) -> list[CohortStats]:
    """SWE-Gym: read `memory_stats.avg_compression_ratio` from summary.json."""
    out: list[CohortStats] = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = _load_json(run_dir / "summary.json")
        if not summary:
            continue
        ms = summary.get("memory_stats") or {}
        avg = ms.get("avg_compression_ratio")
        if avg is None or avg == 1.0:
            # Either absent or no compactions happened — skip.
            continue
        out.append(CohortStats(
            cohort=run_dir.name,
            domain="swegym",
            path=str(run_dir),
            n_episodes=summary.get("total_instances")
                or summary.get("config", {}).get("n_instances", 0),
            n_compaction_episodes=ms.get("compaction_count", 0)
                and summary.get("total_instances", 0),
            total_compactions=ms.get("compaction_count", 0),
            avg_compression_ratio=float(avg),
            median_compression_ratio=None,
            method="memory_stats.avg_compression_ratio",
            notes=["Per-event mean recorded by agent.py:367-368."],
        ))
    return out


def scan_tau2(results_root: Path) -> list[CohortStats]:
    """τ2-bench: avg per-episode `wrapper_last_compression_ratio` filtered to compaction>0.

    Schema: per-episode `result.json` has `agent_stats.{agent,user}` each
    carrying `wrapper_compaction_count` + `wrapper_last_compression_ratio`.
    Older runs put the same fields at the top level. We try both.
    """
    out: list[CohortStats] = []
    tau2_root = results_root / "tau2_bench"
    if not tau2_root.is_dir():
        return out

    def _harvest_ratios(rec: dict[str, Any]) -> tuple[int, float | None]:
        """Return (compaction_count, last_ratio) summed across roles.

        Schema (HP_train runs): memory_stats is keyed by {"agent","user"}
        and each value carries `wrapper_compaction_count` and
        `wrapper_last_compression_ratio`. Top-level `agent_compaction_count`
        / `user_compaction_count` mirror the per-role counts.
        """
        cc = int(rec.get("agent_compaction_count", 0) or 0) + \
             int(rec.get("user_compaction_count", 0) or 0)
        last_ratios: list[float] = []
        ms = rec.get("memory_stats") or {}
        if isinstance(ms, dict):
            for role_stats in ms.values():
                if not isinstance(role_stats, dict):
                    continue
                rcc = role_stats.get("wrapper_compaction_count", 0) or 0
                lr = role_stats.get("wrapper_last_compression_ratio")
                if rcc > 0 and lr and lr > 1.0:
                    last_ratios.append(float(lr))
                cc = max(cc, int(rcc))
        return cc, (max(last_ratios) if last_ratios else None)

    for run_dir in sorted(tau2_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ratios: list[float] = []
        n_episodes = 0
        n_with_comp = 0
        total_comp = 0
        # τ2 layout: {run}/{baseline,memory}/{domain}/{ep}/result.json
        for phase in ("memory", "baseline"):
            phase_dir = run_dir / phase
            if not phase_dir.is_dir():
                continue
            for dom_dir in phase_dir.iterdir():
                if not dom_dir.is_dir():
                    continue
                for ep_dir in dom_dir.iterdir():
                    if not ep_dir.is_dir():
                        continue
                    rec = _load_json(ep_dir / "result.json")
                    if not rec:
                        continue
                    n_episodes += 1
                    cc, lr = _harvest_ratios(rec)
                    if cc > 0 and lr is not None:
                        ratios.append(lr)
                        n_with_comp += 1
                        total_comp += cc

        if not ratios:
            continue
        out.append(CohortStats(
            cohort=run_dir.name,
            domain="tau2",
            path=str(run_dir),
            n_episodes=n_episodes,
            n_compaction_episodes=n_with_comp,
            total_compactions=total_comp,
            avg_compression_ratio=statistics.fmean(ratios),
            median_compression_ratio=statistics.median(ratios),
            method="mean(wrapper_last_compression_ratio | comp>0)",
            notes=[
                "Last-compaction estimator only; per-event list not persisted.",
                f"Filtered to ratio>1.0 (n={len(ratios)}/{n_with_comp})",
            ],
        ))
    return out


def scan_webarena(results_root: Path) -> list[CohortStats]:
    """WebArena: derive max_tokens_seen / token_count per compactioned episode."""
    out: list[CohortStats] = []
    wa_root = results_root / "webarena"
    if not wa_root.is_dir():
        return out

    for run_dir in sorted(wa_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = _load_json(run_dir / "summary.json")
        if not summary or "results" not in summary:
            continue
        ratios: list[float] = []
        n_with_comp = 0
        total_comp = 0
        for r in summary["results"]:
            ms = r.get("memory_stats") or {}
            cc = ms.get("compaction_count", 0)
            if cc <= 0:
                continue
            n_with_comp += 1
            total_comp += int(cc)
            mts = ms.get("max_tokens_seen")
            tc = ms.get("token_count")
            if not mts or not tc or tc <= 0:
                continue
            ratio = mts / tc
            if ratio < 1.05:
                # Drops episodes where the agent ended at peak — ratio is
                # mathematically valid but uninformative about compaction.
                continue
            ratios.append(ratio)

        if not ratios:
            continue
        out.append(CohortStats(
            cohort=run_dir.name,
            domain="webarena",
            path=str(run_dir),
            n_episodes=len(summary["results"]),
            n_compaction_episodes=n_with_comp,
            total_compactions=total_comp,
            avg_compression_ratio=statistics.fmean(ratios),
            median_compression_ratio=statistics.median(ratios),
            method="mean(max_tokens_seen / token_count | comp>0, ratio>1.05)",
            notes=[
                "Per-event ratio not persisted; derivation has selection bias.",
                f"Filtered episodes: {len(ratios)}/{n_with_comp} usable.",
            ],
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="${HOME}/results", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if not args.results_root.is_dir():
        print(f"results-root not found: {args.results_root}")
        return 2

    rows: list[CohortStats] = []
    rows += scan_swegym(args.results_root)
    rows += scan_tau2(args.results_root)
    rows += scan_webarena(args.results_root)

    payload = {
        "results_root": str(args.results_root),
        "n_cohorts": len(rows),
        "rows": [asdict(r) for r in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)

    by_dom: dict[str, list[CohortStats]] = {}
    for r in rows:
        by_dom.setdefault(r.domain, []).append(r)
    print(f"wrote {args.output} with {len(rows)} cohorts")
    for dom, items in sorted(by_dom.items()):
        print(f"  {dom}: {len(items)} cohorts")
        for r in items[:5]:
            cr = r.avg_compression_ratio
            print(f"    {r.cohort}: avg_ratio={cr:.3f}  n_eps={r.n_compaction_episodes}  "
                  f"comp={r.total_compactions}  method={r.method}")
        if len(items) > 5:
            print(f"    ... +{len(items)-5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
