"""CLI: iterate the 5-slice strategy-OOD slate and build all pairs files.

Default slate matches Plan v3 §A0:

    1. structured_ms200      vs sonnet45_llm_summarizing baseline
    2. sliding_window_w75    vs same baseline
    3. obs_masking_w100      vs same baseline
    4. baseline_none         vs same baseline (n=82 sample)
    5. struct_haiku_ms100    vs haiku45 baseline (if available)

After this runs, extend `aug_pairs.SOURCE_MODEL_BY_DIRNAME` with the
five new dir → source_model entries, then run `aug_pairs.py` against
each `<out-root>/heldout_<strat>` dir to produce SFT-pair JSONL.

Example:
    python -m memgym.training.scripts.build_all_strategy_ood \\
        --traj-root  ${REPO_ROOT}/trajectories \\
        --ec2-root   ${REPO_ROOT}/ec2_trajectories \\
        --out-root   ${REPO_ROOT}/data/world_model/aug_pairs \\
        --slate      strategy_ood_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from memgym.training.data.memory_gain_pairs import (
    BuildStats,
    build_pairs_for_strategy,
)


def _sonnet45_slate(traj_root: Path, ec2_root: Path) -> List[Dict[str, str]]:
    """The Plan v3 §A0 slate, after a local-data + reeval inventory pass.

    Local `/trajectories/` (Mar 18-21) dirs were never patch-reeval'd, so
    their `outcome` is a meaningless default and Δr collapses to zero. We
    pivot to `/ec2_trajectories/` runs that DO have reeval JSONs. Slate:

    - Anchor (in-distribution): `train50_sonnet45_llmsummarizing` + its
      reeval (run by us; result lands at `*-llmsumm-reeval-for-rmv3-*`).
    - OOD slice 1: `train50_sonnet45_structured_250steps` (reeval'd
      on-EC2 as `*reeval-struct250v2.json`).
    - OOD slice 2: `train50_struct_haiku_ms100_r075` (cross-model
      crossover; reeval'd as `*run5-reeval.json`).

    Slice 3 (sonnet45 + sliding_window or obs_masking) is dropped from
    this batch — neither strategy has both `_training.json` files AND a
    reeval on EC2. Adding them would require re-running the agent on
    those strategies, which is out of scope for this OOD eval pass.

    The 'baseline-none' (`train50_sonnet45_baseline`) anchor is also
    dropped: 0 `_training.json` files (none-strategy never compacts), so
    we can't render the "Recorded action" template field at compaction
    steps that aren't in its message stream. It remains an open candidate
    for a future "memory-on vs no-memory" comparison if we extend the
    builder to read replay-only baselines.
    """
    anchor_dir = ec2_root / "train50_sonnet45_llmsummarizing"
    return [
        {
            "baseline_dir": str(anchor_dir),
            "baseline_reeval": str(
                anchor_dir
                / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0."
                  "llmsumm-reeval-for-rmv3-strategy-ood.json"
            ),
            "ood_dir": str(ec2_root / "train50_sonnet45_structured_250steps"),
            "ood_reeval": str(
                ec2_root / "train50_sonnet45_structured_250steps"
                / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0."
                  "reeval-struct250v2.json"
            ),
            "strategy_name": "sonnet45_structured_250steps",
        },
        {
            "baseline_dir": str(anchor_dir),
            "baseline_reeval": str(
                anchor_dir
                / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0."
                  "llmsumm-reeval-for-rmv3-strategy-ood.json"
            ),
            "ood_dir": str(ec2_root / "train50_struct_haiku_ms100_r075"),
            "ood_reeval": str(
                ec2_root / "train50_struct_haiku_ms100_r075"
                / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0."
                  "run5-reeval.json"
            ),
            "strategy_name": "haiku45_structured_ms100",
        },
    ]


SLATES = {"strategy_ood_v1": _sonnet45_slate}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-root", type=Path, required=True,
                        help="Root for /trajectories style dirs (sonnet45 sweep)")
    parser.add_argument("--ec2-root", type=Path, required=True,
                        help="Root for /ec2_trajectories style dirs "
                             "(haiku45 + apr-3 onward)")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="Where each heldout_<strat>/pairs_partial.jsonl lands")
    parser.add_argument("--slate", choices=sorted(SLATES.keys()),
                        default="strategy_ood_v1")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip slate entries whose baseline_dir or ood_dir "
                             "does not exist (e.g. haiku45 baseline). Without "
                             "this, the script raises on the first missing dir.")
    args = parser.parse_args()

    slate = SLATES[args.slate](args.traj_root, args.ec2_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, dict] = {}
    for entry in slate:
        strat = entry["strategy_name"]
        baseline_dir = Path(entry["baseline_dir"])
        ood_dir = Path(entry["ood_dir"])
        baseline_reeval = (Path(entry["baseline_reeval"])
                           if entry.get("baseline_reeval") else None)
        ood_reeval = (Path(entry["ood_reeval"])
                      if entry.get("ood_reeval") else None)
        missing = []
        for label, p in [("baseline_dir", baseline_dir), ("ood_dir", ood_dir),
                         ("baseline_reeval", baseline_reeval),
                         ("ood_reeval", ood_reeval)]:
            if p is not None and not p.exists():
                missing.append(f"{label}={p}")
        if missing:
            msg = f"missing for slice {strat!r}: " + ", ".join(missing)
            if args.skip_missing:
                logging.warning("SKIP %s — %s", strat, msg)
                summary[strat] = {"skipped": True, "reason": msg}
                continue
            raise FileNotFoundError(msg)

        out_dir = args.out_root / f"heldout_{strat}"
        logging.info("=== %s ===", strat)
        stats = build_pairs_for_strategy(
            baseline_dir=baseline_dir,
            ood_dir=ood_dir,
            strategy_name=strat,
            out_dir=out_dir,
            baseline_reeval=baseline_reeval,
            ood_reeval=ood_reeval,
        )
        summary[strat] = asdict(stats)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
