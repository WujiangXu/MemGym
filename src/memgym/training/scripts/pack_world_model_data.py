"""Pack the world-model pipeline artifacts on EC2 into one tarball.

Writes `data/world_model_dump.tar.gz` containing:
  aug_pairs/<source_dir>/...            <- from augmentation_output/
  trajectories/<run_name>/...           <- from results/
  training_output/aug_sft_full_yn/...   <- eval metrics + per-row preds

The dir list is hard-coded to the minimum-sufficient set for
reproducing training + the training phase eval (six augmentation-source forks plus
their upstream baselines and memory replays, plus the SFT eval output
the offline threshold sweep operates on). ~5.5 GB compressed guess.

Intended caller: from the MemGym dev box, run via /run-script then
GET /download/data/world_model_dump.tar.gz.
"""
from __future__ import annotations

import os
import json
import sys
import tarfile
from pathlib import Path


# Minimum-sufficient world-model pipeline artifacts. Comments record
# why each entry is needed so a future reader can re-scope without
# re-deriving the whole plan.
RESULTS_DIRS = [
    # haiku45 — 85% of eval rows come from this teacher
    "haiku45_baseline_diverse500_v1",          # raw agent baseline
    "haiku45_llmsumm_diverse500_v1",           # memory-replay baseline
    "fork_haiku45_llmsumm_diverse500_haikusum_v1",  # perturbation forks (augmentation source)
    "diverse500_haiku45_llmsumm_ms100",         # alt memory-replay run
    # sonnet45 — eval includes 3 sonnet fork variants
    "sonnet45_baseline_diverse500",
    "sonnet45_baseline_gap554_v1",
    "sonnet45_llmsumm_diverse500_v1",
    "diverse500_sonnet45_llmsumm_ms100",        # alt memory-replay run
    "fork_sonnet45_llmsumm_diverse500_v1",
    "fork_sonnet45_llmsumm_diverse500_sonnetsumm_v1",
    "fork_sonnet45_llmsumm_gap554_v1",
    "fork_sonnet45_llmsumm_gap554_v2",
    # gpt-oss-120b
    "gpt_oss_120b_baseline_diverse500",
    "gptoss_baseline_gap554_v1",
    "gpt_oss_120b_llmsumm_diverse500_v1",
    "fork_gpt_oss_120b_llmsumm_diverse500_v1",
    "fork_gptoss_llmsumm_gap554_v2",
    "fork_gptoss_llmsumm_gap554_gptosssum_v1",
    "fork_gptoss_llmsumm_diverse500_v2",
    "fork_gptoss_llmsumm_diverse500_gptosssum_v1",
    # consolidated trajectory index (best-effort, may not exist)
    "trajectories_all",
]

# All 11 augmentation-output dirs; they're only ~60 MB total so grab
# everything. The six aug_pairs.SOURCE_MODEL_BY_DIRNAME entries are the
# ones that actually fed training; the others are pilots/smoke.
AUG_OUTPUT_ALL = True

TRAINING_OUTPUT_DIRS = [
    "aug_sft_full_yn",  # aug_sft_pairs.jsonl + eval_results_v2.json + threshold_sweep.json
]

OUT_PATH = Path("data/world_model_dump.tar.gz")


def _add(tar: tarfile.TarFile, src: Path, arcbase: str) -> dict:
    """Add a directory recursively; return (files, bytes) stats.

    Skips obvious fluff (*.log, __pycache__) to keep the tarball lean.
    """
    files = 0
    nbytes = 0
    if not src.is_dir():
        return {"files": 0, "bytes": 0, "missing": True}
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn.endswith(".log") or fn.endswith(".pyc"):
                continue
            full = Path(dp) / fn
            try:
                size = full.stat().st_size
            except OSError:
                continue
            rel = full.relative_to(src)
            tar.add(full, arcname=f"{arcbase}/{rel}", recursive=False)
            files += 1
            nbytes += size
    return {"files": files, "bytes": nbytes, "missing": False}


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = {"tar": str(OUT_PATH), "included": {}, "missing": []}

    with tarfile.open(OUT_PATH, "w:gz", compresslevel=3) as tar:
        # augmentation_output — all or nothing (tiny)
        if AUG_OUTPUT_ALL and Path("augmentation_output").is_dir():
            for entry in sorted(Path("augmentation_output").iterdir()):
                if not entry.is_dir():
                    continue
                key = f"aug_pairs/{entry.name}"
                s = _add(tar, entry, key)
                stats["included"][key] = s
                if s.get("missing"):
                    stats["missing"].append(key)

        # results/<name>
        for name in RESULTS_DIRS:
            src = Path("results") / name
            key = f"trajectories/{name}"
            s = _add(tar, src, key)
            stats["included"][key] = s
            if s.get("missing"):
                stats["missing"].append(key)

        # training_output/<name>
        for name in TRAINING_OUTPUT_DIRS:
            src = Path("training_output") / name
            key = f"training_output/{name}"
            s = _add(tar, src, key)
            stats["included"][key] = s
            if s.get("missing"):
                stats["missing"].append(key)

        # manifest: write the stats into the tarball itself for audit
        manifest = tarfile.TarInfo("manifest.json")
        manifest_bytes = json.dumps(stats, indent=2).encode()
        manifest.size = len(manifest_bytes)
        import io
        tar.addfile(manifest, io.BytesIO(manifest_bytes))

    tar_size = OUT_PATH.stat().st_size
    print(json.dumps({
        "output": str(OUT_PATH),
        "tar_bytes": tar_size,
        "tar_gb": round(tar_size / 1e9, 2),
        "sources": stats["included"],
        "missing": stats["missing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
