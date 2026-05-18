"""Download the v2 reward-model dataset from HF onto EC2.

Pulls only the JSONL (skipping the 1 GB Parquet mirror) to conserve the
~79 GB EC2 scratch budget, then symlinks it into the canonical local
path so `train_sft.py` and `eval_aug_sft.py` work unchanged.

Run as a module:
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO_ID = "MemGym/memgym_data"
REPO_TYPE = "dataset"
HF_PREFIX = "world_model/reward_model_v2"
JSONL_NAME = "reward_model_pairs_v2.jsonl"

DOWNLOAD_ROOT = Path("data/hf_download")
CANON_DIR = Path("data/world_model/training_output/reward_model_v2")
EXPECTED_ROWS = 18637

# Candidate repo IDs to probe when --probe is set or the primary 404s.
PROBE_CANDIDATES = [
    "MemGym/memgym_data",
    "memgym/memgym_data",
    "WujiangXu/memgym_data",
]


def whoami(token: str) -> dict:
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def probe_repo(repo_id: str, token: str | None) -> int:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default=None,
                    help="HF read token. Falls back to $HF_TOKEN / cached file.")
    ap.add_argument("--repo", default=DEFAULT_REPO_ID,
                    help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--probe", action="store_true",
                    help="Probe candidate repos without downloading.")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.exists():
            token = cached.read_text().strip()

    if token:
        try:
            info = whoami(token)
            print(f"[whoami] user={info.get('name')} email={info.get('email','?')}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[whoami] FAILED: {e}", flush=True)

    if args.probe:
        print("[probe] candidates:", flush=True)
        for cand in PROBE_CANDIDATES:
            code = probe_repo(cand, token)
            print(f"  {cand:<40s} HTTP {code}", flush=True)
        return 0

    primary_code = probe_repo(args.repo, token)
    print(f"[probe] {args.repo} HTTP {primary_code}", flush=True)
    if primary_code != 200:
        print(f"[probe] {args.repo} not accessible — probing alternatives:", flush=True)
        for cand in PROBE_CANDIDATES:
            if cand == args.repo:
                continue
            code = probe_repo(cand, token)
            print(f"  {cand:<40s} HTTP {code}", flush=True)
        return 3

    repo_id = args.repo
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={repo_id} prefix={HF_PREFIX}/{JSONL_NAME}", flush=True)
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        token=token,
        allow_patterns=[f"{HF_PREFIX}/{JSONL_NAME}"],
        local_dir=str(DOWNLOAD_ROOT),
    )
    print(f"[download] local_root={local_root}", flush=True)

    src = Path(local_root) / HF_PREFIX / JSONL_NAME
    if not src.exists():
        print(f"ERROR: expected file missing after download: {src}", file=sys.stderr)
        return 1
    size_gb = src.stat().st_size / 2**30
    print(f"[download] {src} size={size_gb:.2f} GB", flush=True)

    rc = subprocess.run(["wc", "-l", str(src)], capture_output=True, text=True)
    rows = int(rc.stdout.split()[0]) if rc.returncode == 0 else -1
    print(f"[download] rows={rows} (expected {EXPECTED_ROWS})", flush=True)
    if rows != EXPECTED_ROWS:
        print(
            f"WARNING: row count {rows} != expected {EXPECTED_ROWS} — proceed with care",
            file=sys.stderr,
        )

    CANON_DIR.parent.mkdir(parents=True, exist_ok=True)
    canon_jsonl = CANON_DIR / JSONL_NAME
    if CANON_DIR.is_symlink() or CANON_DIR.exists():
        # Don't trample an existing real directory; symlink the file directly.
        if canon_jsonl.is_symlink() or canon_jsonl.exists():
            canon_jsonl.unlink()
        CANON_DIR.mkdir(parents=True, exist_ok=True)
        canon_jsonl.symlink_to(src.resolve())
    else:
        CANON_DIR.mkdir(parents=True, exist_ok=True)
        canon_jsonl.symlink_to(src.resolve())
    print(f"[symlink] {canon_jsonl} -> {src.resolve()}", flush=True)

    first = src.open("r").readline()
    keys = set(json.loads(first).keys())
    required = {"messages", "prompt", "completion", "label", "split"}
    missing = required - keys
    print(f"[schema] first-row keys count={len(keys)} required_missing={sorted(missing)}", flush=True)
    if missing:
        print(f"ERROR: required keys missing: {missing}", file=sys.stderr)
        return 2

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
