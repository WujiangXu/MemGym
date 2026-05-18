"""Publish the strategy-OOD eval bundle to a Hugging Face Hub dataset repo.

The bundle is everything a collaborator needs to reproduce Plan v3
Phase A on a new RM checkpoint:

    rm_strategy_ood_eval/
      data/
        rm_strategy_ood_pairs.jsonl   # combined eval JSONL (Y/N labels,
                                      #  long-context prompts ~22-38K tokens
                                      #  matching v2 RM training distribution)
        per_slice/
          heldout_sonnet45_obs_masking_w100__pairs_partial.jsonl
          heldout_sonnet45_pipeline_mask__pairs_partial.jsonl
      reeval/
        run7a_n20__llmsumm-baseline-n20.json
        run6b_n20__obs-masking-w100.json
        run7e_n20__pipeline-mask.json
      manifest.json                   # provenance + counts + slate
      README.md                       # how to use
      LICENSE                         # MIT (matches MemGym)

NOTE: defaults below match the current (post-2026-05-06) v2 slate. Older
defaults pointed at `heldout_sonnet45_structured_250steps` and
`heldout_haiku45_structured_ms100` — those dirs were built with a
pre-reeval-fix labeler and have stale `delta_r=0, label=1` rows. Per
`the internal handoff doc` they are deliberately
excluded from the v2 bundle.

Token handling: the HF write token is read from ``$HF_TOKEN`` (or
``$HUGGING_FACE_HUB_TOKEN``) at runtime. **Never hardcode the token in
this file or any other file in the repo.** Set it in your shell before
running:

    export HF_TOKEN=hf_...
    python -m memgym.training.scripts.publish_strategy_ood_dataset \\
        --repo-id MemGym/memgym_data --subdir rm_strategy_ood_eval

The script supports a ``--dry-run`` mode that stages the bundle in a
local directory and prints what would be uploaded.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_PAIRS_JSONL = REPO_ROOT / "training_output/strategy_ood_v1/strategy_ood_pairs.jsonl"
DEFAULT_PARTIAL_DIRS = [
    REPO_ROOT / "data/world_model/aug_pairs/heldout_sonnet45_obs_masking_w100",
    REPO_ROOT / "data/world_model/aug_pairs/heldout_sonnet45_pipeline_mask",
]
DEFAULT_REEVAL_FILES = [
    # Baseline (llm_summarizing) reeval used by all v2 OOD slices for
    # Δr labeling. Sonnet45 + run7a_n20 cohort.
    REPO_ROOT / "trajectories/sonnet45_llm_summarizing_ms100_r075_kf3/run7a_n20"
    / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-baseline-llmsumm-n20.json",
    # OOD strategy reevals (per-slice patch-eval reports).
    REPO_ROOT / "trajectories/sonnet45_obs_masking_w100/run6b_n20"
    / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-ood-obs-masking-w100.json",
    REPO_ROOT / "trajectories/sonnet45_pipeline_mask_struct_ms100_r075_kf3/run7e_n20"
    / "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1:0.rmv3-ood-pipeline-mask.json",
]


def _resolve_token() -> str:
    """Read the HF write token from env. Fail loudly if absent.

    We deliberately do NOT accept a token via CLI flag — that would
    encourage users to paste the secret into shell history. Env-var-only
    keeps the secret out of `ps`, history, and accidental commits.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(name)
        if tok:
            return tok
    raise SystemExit(
        "no HF token found. export HF_TOKEN=hf_... before running.\n"
        "(do NOT pass the token as a CLI flag — env-var only.)"
    )


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _stage_bundle(
    pairs_jsonl: Path,
    partial_dirs: List[Path],
    reeval_files: List[Path],
    stage_dir: Path,
    license_path: Optional[Path],
) -> Dict:
    """Lay out the upload tree. Returns manifest dict (caller writes it)."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "data" / "per_slice").mkdir(parents=True)
    (stage_dir / "reeval").mkdir(parents=True)

    if not pairs_jsonl.exists():
        raise FileNotFoundError(f"pairs jsonl missing: {pairs_jsonl}")
    shutil.copy2(pairs_jsonl, stage_dir / "data" / "rm_strategy_ood_pairs.jsonl")

    slices = []
    for d in partial_dirs:
        src = d / "pairs_partial.jsonl"
        if not src.exists():
            logger.warning("skip slice — missing %s", src)
            continue
        dst_name = f"{d.name}__pairs_partial.jsonl"
        shutil.copy2(src, stage_dir / "data" / "per_slice" / dst_name)
        slices.append({"name": d.name, "n_pairs": _count_jsonl(src)})

    reevals = []
    for r in reeval_files:
        if not r.exists():
            logger.warning("skip reeval — missing %s", r)
            continue
        dst_name = f"{r.parent.name}__{r.name}"
        shutil.copy2(r, stage_dir / "reeval" / dst_name)
        with r.open() as fh:
            d = json.load(fh)
        resolved_ids = set(d.get("resolved_ids", []))
        completed_ids = set(d.get("completed_ids", [])) | resolved_ids \
            | set(d.get("unresolved_ids", []))
        reevals.append({
            "src_dir": r.parent.name,
            "src_name": r.name,
            "dst_name": dst_name,
            "resolved": len(resolved_ids),
            "completed": len(completed_ids),
        })

    if license_path and license_path.exists():
        shutil.copy2(license_path, stage_dir / "LICENSE")

    return {
        "schema_version": 1,
        "bundle": "rm_strategy_ood_eval",
        "n_pairs_combined": _count_jsonl(pairs_jsonl),
        "slices": slices,
        "reevals": reevals,
    }


def _write_readme(stage_dir: Path, manifest: Dict, repo_id: str, subdir: str) -> None:
    n_pairs = manifest["n_pairs_combined"]
    by_slice = "\n".join(
        f"- `{s['name']}` — {s['n_pairs']} pairs" for s in manifest["slices"]
    ) or "- (no slices)"
    by_reeval = "\n".join(
        f"- `{r['dst_name']}` — {r['resolved']} resolved / {r['completed']} completed"
        for r in manifest["reevals"]
    ) or "- (no reevals)"
    readme = f"""# `{repo_id}` — strategy-OOD reward-model eval bundle (`{subdir}`)

This subfolder ships the strategy-OOD eval bundle for the MemGym v2
reward model (RM). It is the data half of Plan v3 Phase A
(memory-strategy out-of-distribution evaluation, no forking pipeline).

**Bundle revision (2026-05-06):** prompts are now rendered in the
**long-context training-distribution shape** (~22-38K tokens, full
conversation history + active compaction summary + judgment turn) via
`reward_model_pairs._build_row` against each pair's OOD trajectory
training/replay JSONs. The earlier short-template prompts (~600 tokens)
are deprecated — the prompt-length distribution shift was confounding
the strategy-shift signal.

## What's in here

```
{subdir}/
├── data/
│   ├── rm_strategy_ood_pairs.jsonl   # combined eval JSONL ({n_pairs} pairs, split=eval)
│   └── per_slice/                    # raw pairs_partial.jsonl per OOD slice
├── reeval/                           # SWE-Bench harness reports (resolved/unresolved IDs)
├── manifest.json                     # provenance + counts
├── README.md                         # this file
└── LICENSE                           # MIT
```

### Slices ({len(manifest["slices"])})

{by_slice}

### Reeval reports ({len(manifest["reevals"])})

{by_reeval}

## Quick start — score your own RM checkpoint

```bash
pip install datasets huggingface_hub
huggingface-cli download {repo_id} --repo-type dataset --local-dir ./memgym_data
python -m memgym.training.scripts.run_strategy_ood_eval \\
    --checkpoint /path/to/your/rm_checkpoint \\
    --base-model Qwen/Qwen3-1.7B-Base \\
    --dataset ./memgym_data/{subdir} \\
    --output-dir ./my_strategy_ood_results
```

The driver chains `eval_aug_sft.py` + `threshold_sweep.py` +
`per_source_threshold_sweep.py` and writes a `report.md` summarizing
AUROC / ECE / SAFE-F1 / HARMFUL-F1 per slice.

**Use `--max-length 32768`** when scoring this bundle; the prompts are
sized to match v2 training context.

## How labels were derived

For each shared instance between (baseline, OOD) trajectory dirs:

```
Δr  = int(ood.resolved) − int(baseline.resolved)
Δr ≥ 0  → SAFE     (label 1)   memory did NOT regress task resolution
Δr <  0 → HARMFUL  (label 0)   memory hurt
```

One pair is emitted per **memory-trigger step** of the OOD trajectory
(rising edge of `was_compacted`, falling back to `_replay.json`
tool_calls when `_training.json` left `action` empty).

The flat prompt for each pair is rendered using the **OOD trajectory's
own view** at step k (its `step.memory.filtered_msgs` and any
compaction summary it stored), then appended with the
`recorded_action` / `predicted_action` / "Is the proposed action safe?"
judgment turn. This matches how v2 RM training data was rendered, so
the eval prompt-length distribution overlaps the train distribution.

See `the internal handoff doc` and
`the internal handoff doc` in the source repo for
end-to-end provenance and per-slice row-count rationale.

## Scope notes

- Sonnet 4.5 trajectories only. Haiku model-axis crossover is not in
  this bundle (the older haiku heldout dir was built with a stale-label
  pipeline; documented in `strategy_ood_dataset_scope.md`).
- 1.7B target — the Plan v2 lightweight variant is the published RM.
  The 8B baseline (`rm_v2_8b_yn_cw3`) is intentionally excluded from
  this bundle's recommended A/B comparison; pair-level numbers do not
  benefit from the size axis when n is small.

## License

MIT (mirrors the source MemGym repo).
"""
    (stage_dir / "README.md").write_text(readme)


def _push(stage_dir: Path, repo_id: str, subdir: str, token: str) -> str:
    """Upload `stage_dir/*` → `<repo_id>/<subdir>/*` via huggingface_hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    info = api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=subdir,
        commit_message=f"chore(rm-ood): publish {subdir} bundle",
    )
    return info.commit_url if hasattr(info, "commit_url") else str(info)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="MemGym/memgym_data",
                        help="HF Hub dataset repo id")
    parser.add_argument("--subdir", default="rm_strategy_ood_eval",
                        help="Path inside the repo to land the bundle")
    parser.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS_JSONL,
                        help="Combined eval JSONL (one row per pair, split=eval)")
    parser.add_argument("--partial-dir", type=Path, action="append", dest="partial_dirs",
                        default=None, help="Repeatable: per-slice pairs_partial dir")
    parser.add_argument("--reeval-file", type=Path, action="append", dest="reeval_files",
                        default=None, help="Repeatable: SWE-Bench reeval JSON to publish")
    parser.add_argument("--license-file", type=Path,
                        default=REPO_ROOT / "LICENSE",
                        help="License file to ship in the bundle. Falls back to "
                             "an MIT note in the README if missing.")
    parser.add_argument("--stage-dir", type=Path,
                        default=REPO_ROOT / "training_output/strategy_ood_v1/_hf_stage",
                        help="Local staging dir (overwritten each run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage but do not upload; print manifest")
    args = parser.parse_args()

    partial_dirs = args.partial_dirs or DEFAULT_PARTIAL_DIRS
    reeval_files = args.reeval_files or DEFAULT_REEVAL_FILES
    license_path = args.license_file if args.license_file and args.license_file.exists() else None

    manifest = _stage_bundle(
        pairs_jsonl=args.pairs_jsonl,
        partial_dirs=partial_dirs,
        reeval_files=reeval_files,
        stage_dir=args.stage_dir,
        license_path=license_path,
    )
    (args.stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_readme(args.stage_dir, manifest, args.repo_id, args.subdir)

    print(json.dumps(manifest, indent=2), file=sys.stderr)

    if args.dry_run:
        print(f"DRY RUN — staged at {args.stage_dir}")
        return 0

    token = _resolve_token()
    url = _push(args.stage_dir, args.repo_id, args.subdir, token)
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
