"""Materialize the v2 reward-model train/eval split into a committed JSON manifest.

The manifest holds the **canonical instance-ID assignment** for the v2 SFT
dataset, so anyone with the raw `reward_model_pairs_v2.jsonl` (or a derivative)
can reproduce the documented 15,630 / 3,007 train/eval split deterministically
without re-running `assign_splits` or trusting an embedded `split` field.

Output (committed to git):
    data/world_model/reward_model_v2_split.json

Usage:
    python3 src/memgym/training/scripts/build_split_manifest.py

Determinism contract — the SHA256 over sorted "<repo>\\t<split>\\n" lines must
match the frozen reference in `verify_split_determinism.py`. Any drift trips
that test.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

REPO = Path(__file__).resolve().parents[4]
PAIRS = REPO / "data" / "world_model" / "training_output" / "reward_model_v2" / "reward_model_pairs_v2.jsonl"
OUT = REPO / "data" / "world_model" / "reward_model_v2_split.json"

ALGORITHM_VERSION = "v2-tie-broken-by-repo-name"
EVAL_FRAC_TARGET = 0.10


def assign_splits(trajectory_ids: Iterable[str], eval_frac_target: float = EVAL_FRAC_TARGET) -> Dict[str, str]:
    """Mirror of the canonical `reward_model_pairs.assign_splits`.
    Sort key is `(count, repo_name)` for byte-stable determinism."""
    by_repo: Counter = Counter()
    for tid in trajectory_ids:
        by_repo[tid.split("__")[0]] += 1
    total = sum(by_repo.values())
    target = int(total * eval_frac_target)
    ordered = sorted(by_repo.items(), key=lambda kv: (kv[1], kv[0]))
    eval_repos: Set[str] = set()
    cum = 0
    for repo, count in ordered:
        if cum >= target:
            break
        eval_repos.add(repo)
        cum += count
    return {repo: ("eval" if repo in eval_repos else "train") for repo in by_repo}


def main() -> int:
    print(f"reading {PAIRS}", flush=True)
    trajectory_ids: List[str] = []
    tids_per_split: Dict[str, Set[str]] = defaultdict(set)
    rows_per_split: Counter = Counter()
    embedded_repo_split: Dict[str, str] = {}
    with PAIRS.open() as f:
        for line in f:
            row = json.loads(line)
            tid = row["trajectory_id"]
            trajectory_ids.append(tid)
            sp = row["split"]
            tids_per_split[sp].add(tid)
            rows_per_split[sp] += 1
            repo = tid.split("__")[0]
            if repo in embedded_repo_split:
                if embedded_repo_split[repo] != sp:
                    print(f"  CORRUPT: repo {repo} has both splits in jsonl")
                    return 2
            else:
                embedded_repo_split[repo] = sp

    print(f"  rows: {sum(rows_per_split.values())}  ({dict(rows_per_split)})")

    repo_assignments = assign_splits(trajectory_ids)
    if repo_assignments != embedded_repo_split:
        diff = [(r, embedded_repo_split[r], repo_assignments[r])
                for r in embedded_repo_split if embedded_repo_split[r] != repo_assignments[r]]
        print(f"  ❌ embedded split disagrees with patched algorithm on {len(diff)} repos:")
        for r, e, n in diff[:10]:
            print(f"    {r}: embedded={e}  algorithm={n}")
        return 1
    print(f"  ✅ embedded jsonl split == patched algorithm output ({len(repo_assignments)} repos)")

    h_split = hashlib.sha256()
    for repo in sorted(repo_assignments):
        h_split.update(f"{repo}\t{repo_assignments[repo]}\n".encode())
    split_sha = h_split.hexdigest()
    print(f"  split SHA256: {split_sha}")

    manifest = {
        "meta": {
            "config": "reward_model_v2",
            "source_jsonl_relpath": "data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl",
            "source_jsonl_rows": sum(rows_per_split.values()),
            "algorithm_version": ALGORITHM_VERSION,
            "eval_frac_target": EVAL_FRAC_TARGET,
            "split_sha256": split_sha,
            "row_counts":   {"train": rows_per_split["train"], "eval": rows_per_split["eval"]},
            "trajectory_counts": {"train": len(tids_per_split["train"]), "eval": len(tids_per_split["eval"])},
            "repo_counts":  {
                "train": sum(1 for v in repo_assignments.values() if v == "train"),
                "eval":  sum(1 for v in repo_assignments.values() if v == "eval"),
            },
        },
        "repo_assignments": dict(sorted(repo_assignments.items())),
        "splits": {
            "train": sorted(tids_per_split["train"]),
            "eval":  sorted(tids_per_split["eval"]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"\n  wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
