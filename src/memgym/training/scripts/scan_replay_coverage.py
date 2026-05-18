"""Pre-flight coverage scan for the reward-model dataset rebuild.

Walks aug_sft_pairs.jsonl and checks, for each (source_dir, instance_id)
unique pair, whether the corresponding _training.json + _replay.json
files exist under the baseline trajectory tree.

Read-only. Prints a per-source_dir coverage table; exits 0 on success
regardless of coverage so the caller can decide policy.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)


# Mapping from aug_sft_pairs.jsonl `source_dir` to the trajectory dir
# under data/world_model/trajectories/ that contains the matching
# `<iid>_training.json` + `<iid>_forked.json` files. Verified at scan
# time: each source_dir's full iid set is uniquely covered (100%) by
# exactly one trajectory dir below; no other dir reaches 100%.
#
# Note: trajectory dirs use `_forked.json` (the fork-batch output)
# instead of `_replay.json`. The loader.load_trajectory() helper
# already falls back to `_forked.json` when `_replay.json` is missing
# (loader.py:300-302), so downstream code works transparently.
FORK_TO_TRAJDIR: Dict[str, str] = {
    "sonnet_fork_gap554_scaleA_v1":      "fork_sonnet45_llmsumm_gap554_v2",
    "gptoss_fork_gap554_scaleA_v1":      "fork_gptoss_llmsumm_gap554_v2",
    "haiku45_fork_diverse500_scaleA_v1": "fork_haiku45_llmsumm_diverse500_haikusum_v1",
}


def _unique_pairs(pairs_in: Path) -> Dict[str, Set[str]]:
    """Return {source_dir: {instance_id, ...}} from the input pairs file."""
    by_dir: Dict[str, Set[str]] = defaultdict(set)
    with pairs_in.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_dir[row["source_dir"]].add(row["trajectory_id"])
    return by_dir


def _check_files(traj_dir: Path, iid: str) -> Tuple[bool, bool]:
    training = (traj_dir / f"{iid}_training.json").is_file()
    replay = (
        (traj_dir / f"{iid}_replay.json").is_file()
        or (traj_dir / f"{iid}_forked.json").is_file()
    )
    return training, replay


def scan(pairs_in: Path, trajectories_root: Path) -> Dict[str, Dict[str, int]]:
    by_dir = _unique_pairs(pairs_in)
    report: Dict[str, Dict[str, int]] = {}

    for source_dir, iids in sorted(by_dir.items()):
        traj_dirname = FORK_TO_TRAJDIR.get(source_dir)
        if traj_dirname is None:
            report[source_dir] = {
                "instances": len(iids),
                "training_found": 0,
                "replay_found": 0,
                "both_found": 0,
                "baseline_dir_missing": 1,
            }
            continue

        traj_dir = trajectories_root / traj_dirname / "trajectories"
        if not traj_dir.is_dir():
            report[source_dir] = {
                "instances": len(iids),
                "training_found": 0,
                "replay_found": 0,
                "both_found": 0,
                "baseline_dir_missing": 1,
            }
            continue

        training_found = 0
        replay_found = 0
        both_found = 0
        for iid in iids:
            t, r = _check_files(traj_dir, iid)
            training_found += int(t)
            replay_found += int(r)
            both_found += int(t and r)

        report[source_dir] = {
            "instances": len(iids),
            "training_found": training_found,
            "replay_found": replay_found,
            "both_found": both_found,
            "baseline_dir_missing": 0,
        }

    return report


def _format_report(report: Dict[str, Dict[str, int]]) -> str:
    out = []
    out.append(
        f"{'source_dir':<42}  {'iids':>5}  "
        f"{'train':>5}  {'replay':>6}  {'both':>5}  {'cov%':>5}"
    )
    out.append("-" * 80)
    total_iids = total_both = 0
    for sd, s in report.items():
        cov = (s["both_found"] / s["instances"] * 100) if s["instances"] else 0.0
        flag = " <-- baseline dir missing" if s["baseline_dir_missing"] else ""
        out.append(
            f"{sd:<42}  {s['instances']:>5}  "
            f"{s['training_found']:>5}  {s['replay_found']:>6}  "
            f"{s['both_found']:>5}  {cov:>5.1f}{flag}"
        )
        total_iids += s["instances"]
        total_both += s["both_found"]
    out.append("-" * 80)
    cov = (total_both / total_iids * 100) if total_iids else 0.0
    out.append(
        f"{'TOTAL':<42}  {total_iids:>5}  {'':>5}  {'':>6}  "
        f"{total_both:>5}  {cov:>5.1f}"
    )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pairs-in",
        type=Path,
        default=Path("data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl"),
    )
    p.add_argument(
        "--trajectories-root",
        type=Path,
        default=Path("data/world_model/trajectories"),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Scanning %s", args.pairs_in)
    logger.info("Trajectories root: %s", args.trajectories_root)
    logger.info("")

    report = scan(args.pairs_in, args.trajectories_root)
    print(_format_report(report))


if __name__ == "__main__":
    main()
