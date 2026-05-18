"""Verify that the patched `assign_splits` reproduces the published v2 split.

Reads `data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl`,
extracts the (trajectory_id, split) ground truth, recomputes splits via the
patched `assign_splits`, and asserts every repo lands in the same partition.

Also emits a SHA256 over the sorted (repo, split) tuples so future runs can
hash-match against this frozen reference.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Set


def assign_splits(trajectory_ids: Iterable[str], eval_frac_target: float = 0.10) -> dict[str, str]:
    """Mirror of `reward_model_pairs.assign_splits` — inlined so this verifier
    runs without triggering `memgym/__init__.py`'s heavy memory imports.
    Must be kept byte-identical to the canonical implementation.
    """
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

REPO = Path(__file__).resolve().parents[4]
PAIRS = REPO / "data" / "world_model" / "training_output" / "reward_model_v2" / "reward_model_pairs_v2.jsonl"

# Frozen reference: SHA256 over sorted "<repo>\t<split>\n" lines, computed
# from the published reward_model_pairs_v2.jsonl. Any change to the patched
# `assign_splits` algorithm OR to the underlying repo→split assignment will
# flip this hash. Update only when consciously re-cutting the split.
REFERENCE_SHA256 = "0c9e22b35a23b7360d091e8e07b1202a08033b64b08dbe562830b1e4284e0716"


def main() -> int:
    observed: dict[str, str] = {}
    trajectory_ids: list[str] = []
    label_by_split: Counter = Counter()
    with PAIRS.open() as f:
        for line in f:
            row = json.loads(line)
            tid = row["trajectory_id"]
            sp = row["split"]
            repo = tid.split("__")[0]
            trajectory_ids.append(tid)
            label_by_split[sp] += 1
            if repo in observed:
                if observed[repo] != sp:
                    print(f"  CORRUPT: repo {repo} has BOTH splits in jsonl")
                    return 2
            else:
                observed[repo] = sp

    print(f"  rows               : {sum(label_by_split.values())}")
    print(f"  rows by split      : {dict(label_by_split)}")
    print(f"  unique repos       : {len(observed)}")
    print(f"  repos by split     : eval={sum(1 for v in observed.values() if v=='eval')}  "
          f"train={sum(1 for v in observed.values() if v=='train')}")

    recomputed = assign_splits(trajectory_ids, eval_frac_target=0.10)

    mismatches = [(r, observed[r], recomputed[r]) for r in observed if observed[r] != recomputed[r]]
    if mismatches:
        print(f"\n  MISMATCH on {len(mismatches)} repos:")
        for r, o, n in mismatches[:20]:
            print(f"    {r}: published={o}  recomputed={n}")
        return 1

    h = hashlib.sha256()
    for repo in sorted(recomputed):
        h.update(f"{repo}\t{recomputed[repo]}\n".encode())
    digest = h.hexdigest()
    print(f"\n  ✅ patched assign_splits reproduces the published split")
    print(f"  computed SHA256    : {digest}")
    print(f"  expected SHA256    : {REFERENCE_SHA256}")
    if digest != REFERENCE_SHA256:
        print("  ❌ HASH MISMATCH — split has drifted from the frozen reference")
        return 1
    print("  ✅ hash matches frozen reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
