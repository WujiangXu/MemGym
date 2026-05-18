"""List every file under the MemGym/memgym_data HF dataset repo.

The plan assumed `world_model/reward_model_v2/` is the canonical RM
dataset, but the user has indicated v2 may be tied to the old (wrong)
seed and a larger, newer dataset now exists. This probe enumerates
the whole repo so we can pick the right prefix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None)
    ap.add_argument("--repo", default="MemGym/memgym_data")
    args = ap.parse_args()

    import os
    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.exists():
            token = cached.read_text().strip()

    from huggingface_hub import HfApi  # type: ignore[import-not-found]
    api = HfApi(token=token)

    print(f"[list] {args.repo} (dataset)", flush=True)
    info = api.dataset_info(args.repo)
    print(f"  last_modified={info.last_modified}", flush=True)
    print(f"  private={info.private}", flush=True)
    print(f"  tags={info.tags}", flush=True)

    print("\n=== Top-level prefixes ===", flush=True)
    files = api.list_repo_files(args.repo, repo_type="dataset")
    by_prefix: dict[str, list[str]] = {}
    for f in files:
        head = f.split("/", 1)[0] if "/" in f else "(root)"
        by_prefix.setdefault(head, []).append(f)
    for head, fs in sorted(by_prefix.items()):
        print(f"  {head:<40s}  {len(fs)} files")

    print("\n=== Files under reward_model_* / world_model_* / coding_* ===", flush=True)
    relevant = [
        f for f in files
        if any(seg in f for seg in [
            "reward_model", "world_model", "rm_v", "rm_pair",
            "coding_", "trajectory", "trajectories"
        ])
    ]
    relevant.sort()
    # Get file sizes for the relevant subset.
    info_map = {s.rfilename: s for s in info.siblings or []}
    for f in relevant:
        sib = info_map.get(f)
        size = sib.size if sib and sib.size else None
        size_s = f"{size/2**20:>9.1f} MB" if size else "       ? MB"
        print(f"  {size_s}  {f}")

    print(f"\n[total] {len(files)} files in repo, {len(relevant)} match RM/WM/coding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
