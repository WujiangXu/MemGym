"""Three unexplored EC2 dirs + a deep peek at compaction-positive rows.

Earlier two probes proved: trajectory.json never persists summary text,
no sibling files in the task dir. But ${REPO_ROOT}/ has 14G
`output_ir/`, 2G `logs/`, 1.8G `job_logs/`, 225M `_artifact_pull/`,
56M `augmentation_output/` — none of which have been opened. AND the
131MB `reward_model_pairs_webarena_v2.jsonl` already has the right
long-context schema; we want to know whether compaction-positive rows
have summary BODY embedded in their `messages[]` array.

Sections:
  1 — `output_ir/` (14G) — top-level + depth-3 walk
  2 — `logs/` (2G) — top-level + recent files (mtime > Apr 12)
  3 — `_artifact_pull/` (225M) — full file listing
  4 — `augmentation_output/` (56M) — full file listing
  5 — webarena_v2.jsonl: filter to ood_was_compacted_here=True rows;
      dump full messages[] of the first 2 hits to see if a [Summary]
      block lives inside.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("${REPO_ROOT}")


def _sh(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<err: {e}>"


def _hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def section_1_output_ir() -> None:
    _hdr("SECTION 1 — ${REPO_ROOT}/output_ir/  (14G, never explored)")
    print("\n[ls -la]")
    print(_sh(f"ls -la {ROOT}/output_ir/ 2>&1 | head -20"))
    print("\n[depth-3 dirs]")
    print(_sh(f"find {ROOT}/output_ir -maxdepth 3 -type d 2>/dev/null | sort | head -60"))
    print("\n[depth-3 files, name pattern]")
    print(_sh(
        f"find {ROOT}/output_ir -maxdepth 3 -type f -printf '%p (%s B)\\n' "
        f"2>/dev/null | sort | head -40"
    ))
    print("\n[case-insensitive name search: *summary* / *compact* / *condens*]")
    print(_sh(
        f"find {ROOT}/output_ir -maxdepth 5 -type f \\( -iname '*summary*' "
        f"-o -iname '*compact*' -o -iname '*condens*' "
        f"-o -iname '*memory*' \\) 2>/dev/null | head -40"
    ))


def section_2_logs() -> None:
    _hdr("SECTION 2 — ${REPO_ROOT}/logs/  (2.0G, never explored)")
    print("\n[ls -la]")
    print(_sh(f"ls -la {ROOT}/logs/ 2>&1 | head -30"))
    print("\n[depth-3 dirs]")
    print(_sh(f"find {ROOT}/logs -maxdepth 3 -type d 2>/dev/null | sort | head -40"))
    print("\n[recent files (mtime within last 35 days, capped 30)]")
    print(_sh(
        f"find {ROOT}/logs -maxdepth 3 -type f -mtime -35 -printf "
        f"'%TY-%Tm-%Td %TH:%TM %s %p\\n' 2>/dev/null | sort -r | head -30"
    ))


def section_3_artifact_pull() -> None:
    _hdr("SECTION 3 — ${REPO_ROOT}/_artifact_pull/  (225M)")
    print("\n[ls -la]")
    print(_sh(f"ls -la {ROOT}/_artifact_pull/ 2>&1"))
    print("\n[depth-3 dir listing]")
    print(_sh(f"find {ROOT}/_artifact_pull -maxdepth 3 -type d 2>/dev/null | sort"))
    print("\n[all files, depth ≤ 5, capped 60]")
    print(_sh(
        f"find {ROOT}/_artifact_pull -maxdepth 5 -type f -printf '%p (%s B)\\n' "
        f"2>/dev/null | sort | head -60"
    ))


def section_4_aug_output() -> None:
    _hdr("SECTION 4 — ${REPO_ROOT}/augmentation_output/  (56M)")
    print(_sh(f"ls -la {ROOT}/augmentation_output/ 2>&1 | head -30"))
    print("\n[all files, depth ≤ 4]")
    print(_sh(
        f"find {ROOT}/augmentation_output -maxdepth 4 -type f -printf '%p (%s B)\\n' "
        f"2>/dev/null | sort | head -40"
    ))


def section_5_compacted_webarena_rows() -> None:
    _hdr("SECTION 5 — webarena_v2.jsonl: dump messages[] of compaction-positive rows")
    p = (
        ROOT / "data" / "world_model" / "training_output" /
        "reward_model_v2_scenario_ood" / "reward_model_pairs_webarena_v2.jsonl"
    )
    if not p.is_file():
        print("(missing)")
        return
    print(f"file: {p} ({p.stat().st_size:,} B)")
    n_total = 0
    n_compacted = 0
    n_compacted_with_marker = 0
    sample_rows = []
    try:
        with p.open() as fh:
            for line in fh:
                n_total += 1
                if n_total > 5000:  # cap counting
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("ood_was_compacted_here") is True:
                    n_compacted += 1
                    blob = json.dumps(obj.get("messages") or [])
                    if "[Summary]" in blob or "[SUMMARY:" in blob \
                            or "[summary]" in blob.lower():
                        n_compacted_with_marker += 1
                    if len(sample_rows) < 2:
                        sample_rows.append(obj)
    except Exception as e:
        print(f"<err: {e}>")
        return
    print(f"\nrows scanned (cap 5000): {n_total}")
    print(f"rows with ood_was_compacted_here=True: {n_compacted}")
    print(f"  ... of which carry a [Summary] marker in messages[]: {n_compacted_with_marker}")

    for i, row in enumerate(sample_rows):
        print(f"\n--- COMPACTED ROW {i} ---")
        print(f"  trajectory_id: {row.get('trajectory_id')}")
        print(f"  step:          {row.get('step')}")
        print(f"  webarena_cohort: {row.get('webarena_cohort')}")
        print(f"  n_compactions_active: {row.get('n_compactions_active')}")
        print(f"  n_messages_in_view:   {row.get('n_messages_in_view')}")
        print(f"  ood_compression_ratio: {row.get('ood_compression_ratio')}")
        print(f"  active_summary_chars:  {row.get('active_summary_chars')}")
        print(f"  delta_r:               {row.get('delta_r')}")
        print(f"  label:                 {row.get('label')}")
        msgs = row.get("messages") or []
        print(f"  messages count:        {len(msgs)}")
        for j, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            content = m.get("content") or ""
            role = m.get("role")
            print(f"\n  [msg {j}] role={role}  content_len={len(content)}")
            # First and last 400 chars to see if there's a [Summary] block
            if len(content) > 800:
                print(f"    head[:400]: {content[:400]!r}")
                print(f"    tail[-400:]: {content[-400:]!r}")
            else:
                print(f"    full: {content!r}")


def main() -> int:
    print(f"# probe_unexplored_dirs hostname={_sh('hostname')}")
    section_1_output_ir()
    section_2_logs()
    section_3_artifact_pull()
    section_4_aug_output()
    section_5_compacted_webarena_rows()
    print("\n# done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
