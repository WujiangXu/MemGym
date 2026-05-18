#!/usr/bin/env python3
"""Pull missing Docker images for failed SWE-bench instances.

Usage:
    python -m pipelines.memory_training.pull_images --preds results/trajectories_all/baseline/preds.json
    python -m pipelines.memory_training.pull_images --preds results/trajectories_all/baseline/preds.json --workers 4
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_image_name(instance_id: str) -> str:
    """Convert instance_id to Docker image name (SWE-Gym convention)."""
    docker_id = instance_id.replace("__", "_s_")
    return f"xingyaoww/sweb.eval.x86_64.{docker_id}:latest"


def pull_image(image: str, idx: int, total: int) -> tuple[str, bool, str]:
    """Pull a single Docker image. Returns (image, success, message)."""
    print(f"[{idx}/{total}] Pulling {image}...")
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"[{idx}/{total}] OK: {image}")
            return (image, True, "")
        else:
            msg = result.stderr.strip()[:200]
            print(f"[{idx}/{total}] FAILED: {image} — {msg}")
            return (image, False, msg)
    except subprocess.TimeoutExpired:
        print(f"[{idx}/{total}] TIMEOUT: {image}")
        return (image, False, "timeout")
    except Exception as e:
        print(f"[{idx}/{total}] ERROR: {image} — {e}")
        return (image, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Pull missing Docker images")
    parser.add_argument("--preds", required=True, help="Path to preds.json")
    parser.add_argument("--workers", type=int, default=4, help="Parallel pull workers")
    parser.add_argument("--status", default="Error", help="Which status to pull images for")
    args = parser.parse_args()

    preds = json.loads(Path(args.preds).read_text())
    error_ids = [iid for iid, pred in preds.items() if pred.get("status") == args.status]
    images = [get_image_name(iid) for iid in sorted(error_ids)]

    print(f"Found {len(images)} images to pull (status={args.status})")
    if not images:
        print("Nothing to pull.")
        return

    # Check which are already present locally
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True
        )
        local_images = set(result.stdout.strip().split("\n"))
    except Exception:
        local_images = set()

    to_pull = [img for img in images if img not in local_images]
    already = len(images) - len(to_pull)
    print(f"Already local: {already}, need to pull: {len(to_pull)}")

    if not to_pull:
        print("All images already present.")
        return

    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(pull_image, img, i + 1, len(to_pull)): img
            for i, img in enumerate(to_pull)
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - t0
    ok = sum(1 for _, s, _ in results if s)
    fail = sum(1 for _, s, _ in results if not s)
    print(f"\nDone in {elapsed:.0f}s: {ok} pulled, {fail} failed")

    if fail:
        print("\nFailed images:")
        for img, s, msg in results:
            if not s:
                print(f"  {img}: {msg}")


if __name__ == "__main__":
    main()
