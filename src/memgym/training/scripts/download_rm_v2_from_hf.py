"""One-shot: pull the rm_v2 long-context jsonl (and parquet) from a HF repo.

Repo: ``MemGym/memgym_data`` (dataset). The EC2 box currently has the
old 20 MB stub at
``data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl``;
the long-context build (8.1 GB) lives in HF and matches the datacard
prompt distribution (p50 ~73K chars, max ~194K chars).

Pass the token via ``--token`` argv. It is not persisted anywhere by
this script, but it WILL appear in the launch.py job-history record
for this call — rotate the token after the download completes.

Examples:
    # Probe what's in the repo:
      "module": "memgym.training.scripts.download_rm_v2_from_hf",
      "args": ["--token", "<HF_TOKEN>", "--list-only"]
    }

    # Actual download:
      "module": "memgym.training.scripts.download_rm_v2_from_hf",
      "args": [
        "--token", "<HF_TOKEN>",
        "--filename", "reward_model_pairs_v2.jsonl",
        "--out-dir", "data/world_model/training_output/reward_model_v2"
      ]
    }
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True, help="HF read token")
    parser.add_argument("--repo-id", default="MemGym/memgym_data")
    parser.add_argument("--repo-type", default="dataset",
                        choices=["dataset", "model"])
    parser.add_argument("--filename", default=None,
                        help="Path within the repo to download. Required "
                             "unless --list-only.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Local dir for the file (created if missing). "
                             "Required unless --list-only or --target-path.")
    parser.add_argument("--target-path", type=Path, default=None,
                        help="Override final file path. The HF repo's "
                             "internal directory structure is collapsed and "
                             "the downloaded file is renamed to exactly this "
                             "path. Mutually exclusive with --out-dir.")
    parser.add_argument("--list-only", action="store_true",
                        help="Just list repo files; no download.")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(
        args.repo_id, repo_type=args.repo_type, token=args.token,
    )
    print(f"repo {args.repo_id} ({args.repo_type}) — {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    if args.list_only:
        return 0

    if args.filename is None:
        print("ERROR: --filename is required without --list-only",
              file=sys.stderr)
        return 2
    if (args.out_dir is None) == (args.target_path is None):
        print("ERROR: provide exactly one of --out-dir or --target-path",
              file=sys.stderr)
        return 2

    if args.target_path is not None:
        # Stage into target_path's parent (so we still benefit from local_dir
        # avoiding the global cache symlink), then rename to the flat target.
        staging_dir = args.target_path.parent / "_hf_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        print(f"\ndownloading {args.filename} → {args.target_path} (staging in {staging_dir}) ...")
        out_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=args.filename,
            token=args.token,
            local_dir=str(staging_dir),
        )
        out_path_p = Path(out_path)
        args.target_path.parent.mkdir(parents=True, exist_ok=True)
        out_path_p.rename(args.target_path)
        # Best-effort cleanup of the now-empty subtree under staging_dir.
        for d in sorted(staging_dir.rglob("*"), reverse=True):
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass
        try:
            staging_dir.rmdir()
        except OSError:
            pass
        final = args.target_path
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\ndownloading {args.filename} → {args.out_dir} ...")
        final_str = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=args.filename,
            token=args.token,
            local_dir=str(args.out_dir),
        )
        final = Path(final_str)
    sz = final.stat().st_size
    print(f"OK: {final} ({sz} bytes, {sz / 1024**3:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
