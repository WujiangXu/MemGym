"""CLI: build one strategy-OOD pair file via memory-gain labeling.

Wraps `memgym.training.data.memory_gain_pairs.build_pairs_for_strategy`.
For batch (5-slice slate) builds, see `build_all_strategy_ood.py`.

Example:
    python -m memgym.training.scripts.build_memory_gain_pairs \\
        --baseline-dir ${REPO_ROOT}/trajectories/sonnet45_llm_summarizing_ms100_r075_kf3 \\
        --ood-dir      ${REPO_ROOT}/trajectories/sonnet45_sliding_window_w75_kf3 \\
        --strategy-name sliding_window_w75 \\
        --out-dir      ${REPO_ROOT}/data/world_model/aug_pairs/heldout_sliding_window_w75
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from memgym.training.data.memory_gain_pairs import build_pairs_for_strategy


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ood-dir", type=Path, required=True)
    parser.add_argument("--strategy-name", required=True,
                        help="Short slug for the held-out strategy "
                             "(e.g. 'sliding_window_w75')")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-reeval", type=Path, default=None,
                        help="Optional reeval JSON for the baseline dir; "
                             "needed when the trajectory files don't carry a "
                             "`resolved` flag (most trajectories runs).")
    parser.add_argument("--ood-reeval", type=Path, default=None,
                        help="Same, for the OOD dir.")
    args = parser.parse_args()

    stats = build_pairs_for_strategy(
        baseline_dir=args.baseline_dir,
        ood_dir=args.ood_dir,
        strategy_name=args.strategy_name,
        out_dir=args.out_dir,
        baseline_reeval=args.baseline_reeval,
        ood_reeval=args.ood_reeval,
    )
    print(json.dumps(asdict(stats), indent=2), file=sys.stderr)
    print(args.out_dir / "pairs_partial.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
