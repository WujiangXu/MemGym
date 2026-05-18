#!/usr/bin/env python
"""
Consolidate trajectories from multiple run directories into a single collection.

Scans result directories for _replay.json, _training.json, .traj.json files,
deduplicates by instance_id, and merges into a unified output directory.
Also builds a merged preds.json per strategy for --resume dedup.

Usage:
    # Consolidate all run8 + train50 results
    python -m pipelines.memory_training.consolidate \
        --source-dirs results/run8_baseline_100 results/train50_sonnet45_baseline \
        --strategy baseline \
        --output results/trajectories_all/baseline

    # Auto-detect strategy from directory name
    python -m pipelines.memory_training.consolidate \
        --source-dirs results/run8_baseline_100 results/run8_llm_summarizing_100 \
        --output results/trajectories_all \
        --auto-detect

    # Consolidate from EC2 (remote mode)
    python -m pipelines.memory_training.consolidate \
        --ec2-url http://<YOUR_EC2_HOST>:30000 \
        --source-dirs run8_baseline_100 run8_llm_summarizing_100 run8_structured_summary_100 \
        --output results/trajectories_all \
        --auto-detect
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Strategy name detection from directory names
STRATEGY_PATTERNS = {
    "baseline": ["baseline", "none", "_none_"],
    "llm_summarizing": ["llm_summarizing", "llmsummarizing", "llm_sum"],
    "structured_summary": ["structured", "struct"],
    "observation_masking": ["obs_masking", "observation_masking"],
    "sliding_window": ["sliding_window"],
    "adaptive_token_budget": ["adaptive_token"],
    "pipeline": ["pipeline_mask"],
}


def detect_strategy(dir_name: str) -> Optional[str]:
    """Detect memory strategy from directory name."""
    dir_lower = dir_name.lower()
    for strategy, patterns in STRATEGY_PATTERNS.items():
        for pattern in patterns:
            if pattern in dir_lower:
                return strategy
    return None


def extract_instance_id(filename: str) -> Optional[str]:
    """Extract instance_id from trajectory filename.

    Examples:
        getmoto__moto-4833_replay.json -> getmoto__moto-4833
        getmoto__moto-4833.traj.json -> getmoto__moto-4833
        getmoto__moto-4833.traj_concise.json -> getmoto__moto-4833
    """
    name = filename
    # Remove known suffixes
    for suffix in ["_replay.json", ".traj_concise.json", ".traj.json", "_training.json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def scan_local_dir(source_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Scan a local directory for trajectory files.

    Returns: {instance_id: {file_type: path}}
    """
    instances = defaultdict(dict)
    traj_dir = source_dir / "trajectories"

    if not traj_dir.exists():
        print(f"  Warning: {traj_dir} does not exist, skipping")
        return instances

    for f in traj_dir.iterdir():
        if not f.is_file():
            continue
        instance_id = extract_instance_id(f.name)
        if instance_id is None:
            continue

        if f.name.endswith("_replay.json"):
            instances[instance_id]["replay"] = f
        elif f.name.endswith("_training.json"):
            instances[instance_id]["training"] = f
        elif f.name.endswith(".traj.json"):
            instances[instance_id]["traj"] = f
        elif f.name.endswith(".traj_concise.json"):
            instances[instance_id]["traj_concise"] = f

    return instances


def scan_remote_dir(ec2_url: str, source_dir: str) -> Dict[str, Dict[str, str]]:
    """Scan a remote EC2 directory for trajectory files.

    Returns: {instance_id: {file_type: remote_path}}
    """
    import urllib.request

    instances = defaultdict(dict)
    url = f"{ec2_url}/results/{source_dir}/trajectories"

    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Warning: Failed to list {url}: {e}")
        return instances

    items = data.get("items", [])
    for item in items:
        if item["type"] != "file":
            continue
        name = item["name"]
        instance_id = extract_instance_id(name)
        if instance_id is None:
            continue

        remote_path = f"{source_dir}/trajectories/{name}"
        if name.endswith("_replay.json"):
            instances[instance_id]["replay"] = remote_path
        elif name.endswith("_training.json"):
            instances[instance_id]["training"] = remote_path
        elif name.endswith(".traj.json"):
            instances[instance_id]["traj"] = remote_path
        elif name.endswith(".traj_concise.json"):
            instances[instance_id]["traj_concise"] = remote_path

    return instances


def download_file(ec2_url: str, remote_path: str, local_path: Path) -> bool:
    """Download a file from EC2."""
    import urllib.request

    url = f"{ec2_url}/download/{remote_path}"
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(local_path))
        return True
    except Exception as e:
        print(f"  Error downloading {remote_path}: {e}")
        return False


def load_preds(source_dir: Path) -> Dict:
    """Load preds.json from a result directory."""
    preds_path = source_dir / "preds.json"
    if preds_path.exists():
        with open(preds_path) as f:
            return json.load(f)
    return {}


def load_remote_preds(ec2_url: str, source_dir: str) -> Dict:
    """Load preds.json from EC2."""
    import urllib.request

    url = f"{ec2_url}/download/{source_dir}/preds.json"
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def consolidate_local(
    source_dirs: List[Path],
    output_dir: Path,
    strategy: Optional[str] = None,
    auto_detect: bool = False,
    copy_mode: str = "symlink",
) -> Dict[str, int]:
    """Consolidate trajectories from local directories.

    Args:
        source_dirs: List of result directories to scan
        output_dir: Target directory for consolidated output
        strategy: Strategy name (if all source dirs are same strategy)
        auto_detect: Auto-detect strategy from directory name
        copy_mode: "symlink", "copy", or "move"

    Returns: Stats dict
    """
    stats = defaultdict(int)
    all_preds = {}

    for src_dir in source_dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"Skipping {src_dir}: does not exist")
            continue

        # Determine strategy
        strat = strategy
        if auto_detect:
            strat = detect_strategy(src_path.name)
            if strat is None:
                print(f"  Warning: Cannot detect strategy from '{src_path.name}', skipping")
                continue

        print(f"Scanning {src_dir} (strategy={strat})...")
        instances = scan_local_dir(src_path)
        preds = load_preds(src_path)

        # Output directory for this strategy
        strat_dir = output_dir / strat / "trajectories"
        strat_dir.mkdir(parents=True, exist_ok=True)

        new_count = 0
        skip_count = 0

        for instance_id, files in instances.items():
            # Check if already exists in output (dedup)
            existing_replay = strat_dir / f"{instance_id}_replay.json"
            if existing_replay.exists():
                skip_count += 1
                continue

            # Copy/symlink trajectory files
            for file_type, src_file in files.items():
                dst_name = {
                    "replay": f"{instance_id}_replay.json",
                    "training": f"{instance_id}_training.json",
                    "traj": f"{instance_id}.traj.json",
                    "traj_concise": f"{instance_id}.traj_concise.json",
                }[file_type]
                dst_path = strat_dir / dst_name

                if copy_mode == "symlink":
                    if dst_path.exists():
                        dst_path.unlink()
                    dst_path.symlink_to(src_file.resolve())
                elif copy_mode == "copy":
                    shutil.copy2(src_file, dst_path)
                elif copy_mode == "move":
                    shutil.move(str(src_file), str(dst_path))

            new_count += 1

            # Merge predictions
            if instance_id in preds:
                all_preds[instance_id] = preds[instance_id]

        stats[f"{strat}_new"] += new_count
        stats[f"{strat}_skipped"] += skip_count
        print(f"  {strat}: {new_count} new, {skip_count} skipped (already exist)")

    # Save merged preds.json per strategy
    # Group preds by strategy
    for strat in set(
        k.rsplit("_", 1)[0] for k in stats.keys() if k.endswith("_new")
    ):
        preds_dir = output_dir / strat
        preds_path = preds_dir / "preds.json"

        # Load existing preds if any
        existing = {}
        if preds_path.exists():
            with open(preds_path) as f:
                existing = json.load(f)

        # Merge
        existing.update(all_preds)
        with open(preds_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Saved {len(existing)} predictions to {preds_path}")

    return dict(stats)


def consolidate_remote(
    ec2_url: str,
    source_dirs: List[str],
    output_dir: Path,
    strategy: Optional[str] = None,
    auto_detect: bool = False,
) -> Dict[str, int]:
    """Consolidate trajectories from EC2 to local.

    Downloads _replay.json files (primary) and preds.json.
    """
    stats = defaultdict(int)

    for src_dir in source_dirs:
        strat = strategy
        if auto_detect:
            strat = detect_strategy(src_dir)
            if strat is None:
                print(f"  Warning: Cannot detect strategy from '{src_dir}', skipping")
                continue

        print(f"Scanning {src_dir} on EC2 (strategy={strat})...")
        instances = scan_remote_dir(ec2_url, src_dir)
        preds = load_remote_preds(ec2_url, src_dir)

        strat_dir = output_dir / strat / "trajectories"
        strat_dir.mkdir(parents=True, exist_ok=True)

        new_count = 0
        skip_count = 0

        for instance_id, files in instances.items():
            existing_replay = strat_dir / f"{instance_id}_replay.json"
            if existing_replay.exists():
                skip_count += 1
                continue

            # Download trajectory files
            for file_type, remote_path in files.items():
                dst_name = {
                    "replay": f"{instance_id}_replay.json",
                    "training": f"{instance_id}_training.json",
                    "traj": f"{instance_id}.traj.json",
                    "traj_concise": f"{instance_id}.traj_concise.json",
                }[file_type]
                dst_path = strat_dir / dst_name
                download_file(ec2_url, remote_path, dst_path)

            new_count += 1

        # Save preds
        if preds:
            preds_dir = output_dir / strat
            preds_path = preds_dir / "preds.json"
            existing = {}
            if preds_path.exists():
                with open(preds_path) as f:
                    existing = json.load(f)
            existing.update(preds)
            with open(preds_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  Saved {len(existing)} predictions to {preds_path}")

        stats[f"{strat}_new"] += new_count
        stats[f"{strat}_skipped"] += skip_count
        print(f"  {strat}: {new_count} new, {skip_count} skipped")

    return dict(stats)


def print_summary(output_dir: Path):
    """Print summary of consolidated trajectories."""
    print("\n" + "=" * 60)
    print("CONSOLIDATION SUMMARY")
    print("=" * 60)

    for strat_dir in sorted(output_dir.iterdir()):
        if not strat_dir.is_dir():
            continue
        strategy = strat_dir.name

        traj_dir = strat_dir / "trajectories"
        if not traj_dir.exists():
            continue

        replay_files = list(traj_dir.glob("*_replay.json"))
        instance_ids = set()
        repos = defaultdict(int)

        for f in replay_files:
            iid = extract_instance_id(f.name)
            if iid:
                instance_ids.add(iid)
                repo = iid.rsplit("-", 1)[0].replace("__", "/")
                repos[repo] += 1

        # Check preds
        preds_path = strat_dir / "preds.json"
        preds_count = 0
        if preds_path.exists():
            with open(preds_path) as f:
                preds_count = len(json.load(f))

        print(f"\n{strategy}:")
        print(f"  Replay files: {len(replay_files)}")
        print(f"  Unique instances: {len(instance_ids)}")
        print(f"  Predictions: {preds_count}")
        if repos:
            print(f"  Repos: {len(repos)}")
            for repo, count in sorted(repos.items(), key=lambda x: -x[1])[:5]:
                print(f"    {repo}: {count}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Consolidate trajectories from multiple runs")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="Source result directories to scan",
    )
    parser.add_argument(
        "--output",
        default="results/trajectories_all",
        help="Output directory for consolidated trajectories",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy name (if all sources are same strategy)",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect strategy from directory name",
    )
    parser.add_argument(
        "--ec2-url",
        default=None,
        help="EC2 server URL for remote consolidation",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "move"],
        default="copy",
        help="How to transfer files (local only)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing consolidated data",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.summary_only:
        print_summary(output_dir)
        return

    if not args.strategy and not args.auto_detect:
        parser.error("Must specify --strategy or --auto-detect")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ec2_url:
        stats = consolidate_remote(
            ec2_url=args.ec2_url,
            source_dirs=args.source_dirs,
            output_dir=output_dir,
            strategy=args.strategy,
            auto_detect=args.auto_detect,
        )
    else:
        stats = consolidate_local(
            source_dirs=[Path(d) for d in args.source_dirs],
            output_dir=output_dir,
            strategy=args.strategy,
            auto_detect=args.auto_detect,
            copy_mode=args.copy_mode,
        )

    print(f"\nStats: {json.dumps(stats, indent=2)}")
    print_summary(output_dir)


if __name__ == "__main__":
    main()
