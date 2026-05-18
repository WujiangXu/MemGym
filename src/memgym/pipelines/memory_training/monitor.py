#!/usr/bin/env python
"""
Monitor Docker image pulls, disk usage, and evaluation job progress.

Runs in a loop, printing status every N seconds. Tracks:
  - Docker image count and pull progress (new images appearing)
  - Disk usage and Docker storage
  - Running containers and their ages
  - Evaluation job progress (predictions written so far)

Usage:
    # Monitor continuously (default 30s interval)
    python -m pipelines.memory_training.monitor

    # One-shot status
    python -m pipelines.memory_training.monitor --once

    # Custom interval and watch specific output dir
    python -m pipelines.memory_training.monitor --interval 60 --output-dir results/trajectories_all/baseline

    # Watch Docker pulls only
    python -m pipelines.memory_training.monitor --docker-only
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_cmd(cmd: List[str], timeout: int = 30) -> str:
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def get_disk_usage() -> Dict:
    """Get disk usage for root filesystem."""
    output = run_cmd(["df", "-h", "/"])
    lines = output.split("\n")
    if len(lines) < 2:
        return {}
    parts = lines[1].split()
    if len(parts) >= 5:
        return {
            "total": parts[1],
            "used": parts[2],
            "available": parts[3],
            "use_pct": parts[4],
        }
    return {}


def get_docker_storage() -> Dict:
    """Get Docker system storage summary."""
    output = run_cmd(["docker", "system", "df", "--format", "{{json .}}"])
    if not output:
        return {}
    result = {}
    for line in output.split("\n"):
        try:
            item = json.loads(line)
            result[item.get("Type", "unknown")] = {
                "total": item.get("TotalCount", 0),
                "active": item.get("Active", 0),
                "size": item.get("Size", "0B"),
                "reclaimable": item.get("Reclaimable", "0B"),
            }
        except json.JSONDecodeError:
            pass
    return result


def get_docker_images() -> List[Dict]:
    """Get list of Docker images with details."""
    output = run_cmd([
        "docker", "images",
        "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}",
    ])
    if not output:
        return []
    images = []
    for line in output.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 4:
            images.append({
                "repository": parts[0],
                "tag": parts[1],
                "size": parts[2],
                "created": parts[3],
            })
    return images


def get_docker_containers() -> List[Dict]:
    """Get running Docker containers."""
    output = run_cmd([
        "docker", "ps",
        "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.RunningFor}}",
    ])
    if not output:
        return []
    containers = []
    for line in output.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 4:
            containers.append({
                "name": parts[0],
                "image": parts[1],
                "status": parts[2],
                "running_for": parts[3],
            })
    return containers


def get_pulling_status() -> List[str]:
    """Check if Docker is currently pulling images by looking at docker events or ps."""
    # Check for containers in "Created" state (often during pull)
    output = run_cmd([
        "docker", "ps", "-a",
        "--filter", "status=created",
        "--format", "{{.Image}}",
    ])
    return output.split("\n") if output else []


def get_predictions_progress(output_dir: str) -> Dict:
    """Check evaluation progress by reading preds.json."""
    preds_path = Path(output_dir) / "preds.json"
    if not preds_path.exists():
        return {"total": 0, "submitted": 0, "errors": 0, "limits_exceeded": 0}

    try:
        with open(preds_path) as f:
            preds = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"total": 0, "submitted": 0, "errors": 0, "limits_exceeded": 0}

    total = len(preds)
    submitted = 0
    errors = 0
    limits_exceeded = 0
    repos = defaultdict(int)

    for iid, pred in preds.items():
        status = pred.get("status", "")
        if status == "Submitted":
            submitted += 1
        elif status == "Error":
            errors += 1
        elif status == "LimitsExceeded":
            limits_exceeded += 1
        else:
            submitted += 1  # Legacy entries without status

        repo = iid.rsplit("-", 1)[0].replace("__", "/")
        repos[repo] += 1

    return {
        "total": total,
        "submitted": submitted,
        "errors": errors,
        "limits_exceeded": limits_exceeded,
        "repos": dict(repos),
        "num_repos": len(repos),
    }


def get_trajectory_progress(output_dir: str) -> Dict:
    """Check trajectory file progress."""
    traj_dir = Path(output_dir) / "trajectories"
    if not traj_dir.exists():
        return {"replay_files": 0, "traj_files": 0}

    replay_files = list(traj_dir.glob("*_replay.json"))
    traj_files = list(traj_dir.glob("*.traj.json"))

    return {
        "replay_files": len(replay_files),
        "traj_files": len(traj_files),
    }


def format_status(
    disk: Dict,
    docker_storage: Dict,
    images: List[Dict],
    containers: List[Dict],
    preds_progress: Optional[Dict],
    traj_progress: Optional[Dict],
    prev_image_count: int,
    start_time: float,
) -> str:
    """Format a status report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  MONITOR STATUS — {now}  (elapsed: {elapsed_str})")
    lines.append(f"{'='*70}")

    # Disk
    lines.append(f"\n  DISK: {disk.get('used', '?')} / {disk.get('total', '?')} ({disk.get('use_pct', '?')})")

    # Docker storage
    if docker_storage:
        img_info = docker_storage.get("Images", {})
        ctr_info = docker_storage.get("Containers", {})
        lines.append(
            f"  DOCKER: {img_info.get('total', '?')} images ({img_info.get('size', '?')}), "
            f"{ctr_info.get('active', '?')} active containers ({ctr_info.get('size', '?')})"
        )

    # Docker images breakdown
    image_count = len(images)
    new_images = image_count - prev_image_count if prev_image_count >= 0 else 0

    # Group by namespace
    namespaces = defaultdict(lambda: {"count": 0, "newest": ""})
    for img in images:
        repo = img["repository"]
        ns = repo.split("/")[0] if "/" in repo else repo
        namespaces[ns]["count"] += 1
        if img["created"] > namespaces[ns]["newest"]:
            namespaces[ns]["newest"] = img["created"]

    lines.append(f"\n  DOCKER IMAGES: {image_count} total", )
    if new_images > 0:
        lines.append(f"  *** {new_images} NEW images since last check ***")

    for ns, info in sorted(namespaces.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"    {ns}: {info['count']} images")

    # Recently created images (last 5)
    swe_images = [img for img in images if "sweb.eval" in img["repository"] or "swesmith" in img["repository"]]
    if swe_images:
        swe_images.sort(key=lambda x: x["created"], reverse=True)
        lines.append(f"\n  LATEST SWE-BENCH IMAGES (newest first):")
        for img in swe_images[:5]:
            lines.append(f"    {img['repository']}:{img['tag']}  {img['size']}  ({img['created'][:19]})")

    # Running containers
    if containers:
        lines.append(f"\n  RUNNING CONTAINERS: {len(containers)}")
        for ctr in containers:
            lines.append(f"    {ctr['name'][:40]}  {ctr['status']}  ({ctr['running_for']})")

    # Evaluation progress
    if preds_progress and preds_progress["total"] > 0:
        p = preds_progress
        lines.append(f"\n  EVALUATION PROGRESS:")
        lines.append(
            f"    Predictions: {p['total']} total "
            f"({p['submitted']} submitted, {p['errors']} errors, {p['limits_exceeded']} LimExc)"
        )
        if p.get("num_repos"):
            lines.append(f"    Repos covered: {p['num_repos']}")
            for repo, count in sorted(p["repos"].items(), key=lambda x: -x[1])[:5]:
                lines.append(f"      {repo}: {count}")

    # Trajectory progress
    if traj_progress and traj_progress.get("replay_files", 0) > 0:
        lines.append(f"\n  TRAJECTORY FILES:")
        lines.append(f"    Replay: {traj_progress['replay_files']}, Traj: {traj_progress['traj_files']}")

    lines.append(f"{'='*70}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor Docker pulls and job progress")
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Seconds between checks (default: 30)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Watch evaluation output directory for progress",
    )
    parser.add_argument(
        "--docker-only", action="store_true",
        help="Only show Docker-related info",
    )

    args = parser.parse_args()
    start_time = time.time()
    prev_image_count = -1

    iteration = 0
    while True:
        iteration += 1

        # Gather data
        disk = get_disk_usage()
        docker_storage = get_docker_storage()
        images = get_docker_images()
        containers = get_docker_containers()

        preds_progress = None
        traj_progress = None
        if args.output_dir and not args.docker_only:
            preds_progress = get_predictions_progress(args.output_dir)
            traj_progress = get_trajectory_progress(args.output_dir)

        # Format and print
        status = format_status(
            disk=disk,
            docker_storage=docker_storage,
            images=images,
            containers=containers,
            preds_progress=preds_progress,
            traj_progress=traj_progress,
            prev_image_count=prev_image_count,
            start_time=start_time,
        )
        print(status)
        sys.stdout.flush()

        prev_image_count = len(images)

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
