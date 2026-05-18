"""Stage the rescued handoff artifacts into the HF dataset layout.

Usage:
    python -m scripts.stage_hf_dataset \
        --source data/handoff \
        --staging data/hf_staging \
        [--upload]   # also run `hf upload MemGym/memgym_data . --repo-type=dataset`

Layout produced (mirrors docs/ir/RESUME.md Section 4):

    <staging>/
      README.md
      verified/
        3hop_verified.jsonl
        4hop_paper_run.jsonl
        56hop_clean.jsonl
      benchmark/
        b2_n200_6methods.json
        b2_n200_summary.md
      meta/
        hop56_topics_used.txt
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

SRC_TO_DST = [
    ("hop3_verified.jsonl",       "verified/3hop_verified.jsonl"),
    ("hop4_paper_run.jsonl",      "verified/4hop_paper_run.jsonl"),
    ("hop56_clean.jsonl",         "verified/56hop_clean.jsonl"),
    ("benchmark_n200_6methods.json", "benchmark/b2_n200_6methods.json"),
    ("benchmark_n200_summary.md",    "benchmark/b2_n200_summary.md"),
    ("topics_used.txt",           "meta/hop56_topics_used.txt"),
]


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def build_readme(staging: Path, counts: dict[str, int]) -> str:
    n3 = counts["verified/3hop_verified.jsonl"]
    n4 = counts["verified/4hop_paper_run.jsonl"]
    n56 = counts["verified/56hop_clean.jsonl"]
    return f"""# MemGym-IR — multi-hop QA benchmark data

Private mirror of the verified instance corpus and the the training phase N=200 benchmark
produced in MemGym-IR. See the main repo for the pipeline and eval harness.

## Manifest

| Path | Lines | Purpose |
|---|---|---|
| `verified/3hop_verified.jsonl` | {n3} | 3-hop instances, verified by `harden_and_verify` (the training phase B1). |
| `verified/4hop_paper_run.jsonl` | {n4} | 4-hop verified instances from the paper run (subset that passed verification — the raw pool had 1005 lines; 916 survived). |
| `verified/56hop_clean.jsonl` | {n56} | 5/6-hop instances after merge + garble/MIT postfix (the training phase + the training phase B4). |
| `benchmark/b2_n200_6methods.json` | — | Full per-instance benchmark results. **See "Known defects" below.** |
| `benchmark/b2_n200_summary.md` | — | Rendered markdown summary table for B2. |
| `meta/hop56_topics_used.txt` | — | Topics already consumed by the 5/6-hop pipeline — avoid duplicate regen. |

## Schema

Each JSONL row is a `MemGymIRInstance` with (roughly):

```
{{
  "instance_id": str,
  "question":    str,
  "answer":      str,
  "hops":        [ {{...}} ],          # grounding facts per hop
  "documents":   [ {{"text": str, ...}} ],
  "verification": {{
      "score_no_memory":  float,
      "score_all_memory": float,
      "memory_gap":       float
  }} | null
}}
```

See `src/memgym/pipelines/memgym_ir/models.py` in the main repo for the full schema.

## Benchmark JSON shape

`benchmark/b2_n200_6methods.json` nests as:

```
{{
  "config": {{...}},
  "per_stratum": {{
    "3hop"|"4hop"|"56hop": {{
      "strategies": {{
        "ir_passthrough"|...: {{
          "avg_score":  float,
          "avg_recall": float,
          "per_instance": [ {{"instance_id": str, "score": float, ...}} ]
        }}
      }}
    }}
  }}
}}
```

## Known defects

1. **`ir_structured` row in B2 is DEGRADED.** The summarizer defaulted to
   `gpt-4o-mini` (no OpenAI key on the EC2 box), LiteLLM silently retried, a
   broad `except` swallowed the 401, and the code fell through to a
   `forgotten_text[:2000]` concatenation fallback. The B2 ir_structured scores
   (0.678 / 0.369 / 0.268) are from that fallback path — not real structured
   summarization. Fix lives in the main repo at commit `7798a3e`
   (`src/memgym/memory/ir/ir_structured_summary.py:124` → Bedrock Haiku default).
   Re-run `ir_structured` at N=200 against Bedrock before citing.

2. **the training phase (4-hop scale-up to N=916) never completed.** EC2 credentials
   expired mid-run; the collaborator should rerun on their own Bedrock box.

3. **LightMem packaging is fragile on Python 3.13.** The adapter at
   `src/memgym/memory/ir/ir_lightmem.py` soft-fails via ImportError catch; the
   benchmark skips `ir_lightmem` with a warning if setup is not complete.
   See `scripts/setup_lightmem_venv.py` for the uv-venv recipe that works.

## How this was produced

- 3-hop: `scripts/truncate_to_3hop.py` on the 4-hop pool, then
  `scripts/verify_3hop.py` with Sonnet 4.5 as verifier (250 → {n3} survived).
- 4-hop: original paper run (`scripts.run_deep_research` × 285 topics,
  Haiku 4.5 worker + Sonnet 4.5 verifier).
- 5/6-hop: `scripts/batch_hop56.py` across 45+ topics, merged + postfixed
  (garble-token reverse-map, MIT license strip).
- Benchmark: `scripts/run_ir_benchmark.py --limit 200 --workers 8` with six
  memory strategies.

## License / use

Internal research artifact. Not for redistribution.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/handoff"))
    ap.add_argument("--staging", type=Path, default=Path("data/hf_staging"))
    ap.add_argument("--upload", action="store_true",
                    help="After staging, run `hf upload MemGym/memgym_data . --repo-type=dataset`.")
    ap.add_argument("--repo", type=str, default="MemGym/memgym_data")
    args = ap.parse_args()

    args.staging.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for src_name, dst_rel in SRC_TO_DST:
        src = args.source / src_name
        dst = args.staging / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")
        print(f"stage {src} -> {dst}")
        shutil.copy2(src, dst)
        if dst.suffix == ".jsonl":
            counts[dst_rel] = count_lines(dst)

    readme = args.staging / "README.md"
    readme.write_text(build_readme(args.staging, counts))
    print(f"wrote {readme}")

    print(f"\nstaging tree at {args.staging}:")
    for p in sorted(args.staging.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(args.staging)}  ({p.stat().st_size} B)")

    cmd = ["hf", "upload", args.repo, ".", "--repo-type=dataset"]
    print(f"\nto upload, from {args.staging}:\n  {' '.join(cmd)}")
    if args.upload:
        print("\nuploading now...")
        subprocess.check_call(cmd, cwd=args.staging)


if __name__ == "__main__":
    main()
