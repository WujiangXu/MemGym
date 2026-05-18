"""Reward-model dataset rebuild: aug_sft_pairs → multi-turn (messages, prompt).

The current reward-model file
``data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl``
ships a ~324-char prompt that only contains
``(step, perturbation, recorded_action, predicted_action)``. Everything
else the agent actually saw at compaction time — multi-turn ReAct
history, the active rolling summary, environment observations — is in
the upstream trajectory files but discarded by ``aug_pairs.py``.

This module rebuilds every row 1:1 by joining each ``aug_sft_pairs`` row
to its source trajectory and reconstructing the agent's view at the
target step the way the runtime would have rendered it.

Why we don't just call ``build_view``: the fork files (which back the
~18.6k aug_sft rows) ship ``condensation_history=[]`` — the runtime's
``CondensationRecord`` list is not persisted. We instead reconstruct the
post-compaction view directly from ``step.memory`` counts:

    head    = raw_messages[:keep_first]
    summary = "[Summary] " + most-recent non-empty step.memory.summary
    tail    = raw_messages[:original_msgs][-tail_len:]
    view    = head + [summary] + tail
    where tail_len = filtered_msgs - keep_first - 1

Because LLMSummarizingMemory's ``_call_llm_summarize`` already absorbs
the previous summary via ``<PREVIOUS SUMMARY>`` (llm_summarizing.py:432),
the **latest** summary at step ≤ N is sufficient — older summaries are
already folded in. This matches ``build_view``'s "insert only the last
condensation's summary" semantics (llm_summarizing.py:138-147).

``original_msgs`` doubles as an ``assistant_msg_index``: it is the raw
message count fed into the compaction call at step k, equivalently the
position right before the assistant's turn at step k in
``replay.messages``.

Output schema (one JSONL row per aug_sft row, no dedup):

    {
        "trajectory_id":     str,
        "fork_event_id":     str,
        "instance_id":       str,        # alias of trajectory_id
        "source_dir":        str,
        "source_model":      "sonnet45" | "haiku45" | "gptoss120b",
        "split":             "train" | "eval",
        "step":              int,        # local fork-segment step
        "perturbation":      str,
        "label":             0 | 1,
        "completion":        " Y" | " N",
        "messages":          List[{role, content}],
        "prompt":            flat rendered text ending in "\\n\\nAnswer:",
        "recorded_action":   str,
        "predicted_action":  str,
        "n_compactions_active": int,     # # of step ≤ N with non-empty summary
        "active_summary_chars": int,     # 0 if no compaction active yet
        "n_messages_in_view":   int,
        "original_msgs":     int,        # for debugging
        "filtered_msgs":     int,
        "provenance": {training_path, replay_path, build_view_fn, build_script}
    }

Reuses ``aug_pairs.assign_splits`` for repo-grouped train/eval split.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def assign_splits(
    trajectory_ids: Iterable[str],
    eval_frac_target: float = 0.10,
) -> Dict[str, str]:
    """Group-shuffle by ``repo_prefix = trajectory_id.split('__')[0]``.

    Inlined copy of ``aug_pairs.assign_splits`` — duplicated here (rather
    than imported) so this module runs as a standalone data script
    without triggering ``memgym/__init__.py``'s heavy memory imports
    (sentence-transformers, sklearn) that this pipeline does not need.

    Sort key is ``(count, repo_name)`` — the secondary repo-name key
    makes the result independent of ``trajectory_ids`` iteration order.
    """
    by_repo: Counter = Counter()
    for tid in trajectory_ids:
        by_repo[tid.split("__")[0]] += 1
    total = sum(by_repo.values())
    target = int(total * eval_frac_target)

    ordered = sorted(by_repo.items(), key=lambda kv: (kv[1], kv[0]))
    eval_repos: Set[str] = set()
    cum = 0
    for repo, count in ordered:
        if cum >= target:
            break
        eval_repos.add(repo)
        cum += count
    return {repo: ("eval" if repo in eval_repos else "train") for repo in by_repo}


# Fork-source-dir → trajectory-root-dir under data/world_model/trajectories/.
# Verified at 100% coverage by scan_replay_coverage.py over the 846 unique
# (source_dir, instance_id) pairs in aug_sft_pairs.jsonl. Identical to
# summarizer_pairs.SOURCE_DIR_TO_TRAJ_ROOT — kept local to avoid a circular
# import and to allow the training phase reward-model and the training phase summarizer pipelines
# to evolve independently.
FORK_TO_TRAJDIR: Dict[str, str] = {
    "sonnet_fork_gap554_scaleA_v1":      "fork_sonnet45_llmsumm_gap554_v2",
    "gptoss_fork_gap554_scaleA_v1":      "fork_gptoss_llmsumm_gap554_v2",
    "haiku45_fork_diverse500_scaleA_v1": "fork_haiku45_llmsumm_diverse500_haikusum_v1",
}

# Mirror of the production LLMSummarizingMemory keep_first. Verified by
# `grep -r '"keep_first":' data/world_model/trajectories/` — every collected
# trajectory has keep_first=3. Exposed as a CLI knob for completeness.
DEFAULT_KEEP_FIRST = 3

# Single-token completion strings under Qwen3.5-9B-Base BPE: " Y"=[783],
# " N"=[443]. Same convention as aug_pairs.py:57-59.
LABEL_SAFE = " Y"
LABEL_HARMFUL = " N"

# Marker that build_view inserts at summary_offset; we replicate it so
# downstream tokenizer round-trip checks line up byte-for-byte.
SUMMARY_MARKER = "[Summary]"

# Final question turn appended after the reconstructed view.
ACTION_JUDGMENT_TEMPLATE = (
    "Recorded action at step {step}:\n{recorded}\n\n"
    "Proposed action under perturbation '{perturbation}':\n{predicted}\n\n"
    "Is the proposed action safe (would not regress task resolution)? "
    "Answer Y if safe, N if harmful."
)

RESPONSE_PREFIX = "\n\nAnswer:"


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

@dataclass
class _TrajCache:
    """Loaded view of one trajectory file pair, kept in memory across all
    aug_sft rows that share the same (source_dir, instance_id). One pair
    typically expands into 6 rows (one per perturbation), so caching cuts
    JSON parse cost by 6×.
    """
    training_path: Path
    replay_path: Path
    raw_messages: List[Dict[str, Any]]
    # step (int, 1-based local) -> step.memory dict
    step_memory: Dict[int, Dict[str, Any]]
    # Sorted (step, summary) for non-empty summaries — used for "most recent
    # active summary at step ≤ N" lookup.
    summary_events: List[Tuple[int, str]]


def _resolve_traj_paths(
    trajectories_root: Path, source_dir: str, instance_id: str,
) -> Optional[Tuple[Path, Path]]:
    traj_root = FORK_TO_TRAJDIR.get(source_dir)
    if traj_root is None:
        return None
    base = trajectories_root / traj_root / "trajectories"
    training = base / f"{instance_id}_training.json"
    if not training.is_file():
        return None
    replay = base / f"{instance_id}_replay.json"
    if not replay.is_file():
        replay = base / f"{instance_id}_forked.json"
    if not replay.is_file():
        return None
    return training, replay


def _load_trajectory(training_path: Path, replay_path: Path) -> _TrajCache:
    """Read + index a single trajectory pair. Pure JSON I/O."""
    training = json.loads(training_path.read_text())
    replay = json.loads(replay_path.read_text())

    raw_messages = list(replay.get("messages") or [])
    step_memory: Dict[int, Dict[str, Any]] = {}
    summary_events: List[Tuple[int, str]] = []

    for s in training.get("steps", []):
        try:
            step_i = int(s.get("step", -1))
        except (TypeError, ValueError):
            continue
        mem = s.get("memory") or {}
        step_memory[step_i] = mem
        summ = (mem.get("summary") or "").strip()
        if summ:
            summary_events.append((step_i, summ))

    summary_events.sort(key=lambda kv: kv[0])

    return _TrajCache(
        training_path=training_path,
        replay_path=replay_path,
        raw_messages=raw_messages,
        step_memory=step_memory,
        summary_events=summary_events,
    )


def _latest_summary_at(cache: _TrajCache, target_step: int) -> Tuple[Optional[str], int]:
    """Return (latest active summary text, count of compaction events at step ≤ N).

    Walks ``summary_events`` (sorted by step) and returns the LAST one whose
    step is ≤ target_step. Count is the number of compactions that have
    fired up to and including target_step.
    """
    latest: Optional[str] = None
    n_active = 0
    for step_i, summ in cache.summary_events:
        if step_i > target_step:
            break
        latest = summ
        n_active += 1
    return latest, n_active


def _resolve_assistant_msg_count(
    cache: _TrajCache, target_step: int,
) -> Optional[int]:
    """Resolve how many raw messages the agent saw before its turn at
    ``target_step``.

    Equals ``step.memory.original_msgs`` at target_step (the count fed to
    the compaction call). Falls back to walking earlier steps if the
    target step's mem entry is missing original_msgs.
    """
    mem = cache.step_memory.get(target_step) or {}
    val = mem.get("original_msgs")
    if isinstance(val, int) and val > 0:
        return val
    # Fallback: earlier step's original_msgs + (steps × 2 raw messages).
    # SWE ReAct emits 2 messages per step (assistant + tool). Used only
    # when the target step's mem dict is incomplete.
    for back in range(target_step - 1, 0, -1):
        prior = cache.step_memory.get(back) or {}
        prior_val = prior.get("original_msgs")
        if isinstance(prior_val, int) and prior_val > 0:
            return prior_val + 2 * (target_step - back)
    return None


# ---------------------------------------------------------------------------
# View reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_view(
    cache: _TrajCache,
    target_step: int,
    keep_first: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build the message list the agent saw at step ``target_step``.

    Returns (view_messages, stats). Stats include n_compactions_active,
    original_msgs, filtered_msgs, n_messages_in_view, active_summary_chars
    for the row's metadata.

    Strategy: latest active summary + head/tail slice of raw messages.
    See module docstring for the math.
    """
    raw_msgs = cache.raw_messages
    target_mem = cache.step_memory.get(target_step) or {}
    original_msgs = _resolve_assistant_msg_count(cache, target_step) or len(raw_msgs)
    original_msgs = min(original_msgs, len(raw_msgs))

    summary, n_active = _latest_summary_at(cache, target_step)
    raw_pre = raw_msgs[:original_msgs]

    # No compaction has fired yet at this step → agent saw raw history
    # straight through. (build_view returns kept_messages unchanged when
    # condensations is empty.)
    if not summary:
        view = list(raw_pre)
        stats = {
            "original_msgs": original_msgs,
            "filtered_msgs": original_msgs,
            "n_compactions_active": 0,
            "active_summary_chars": 0,
            "n_messages_in_view": len(view),
        }
        return view, stats

    filtered_msgs = target_mem.get("filtered_msgs")
    if not isinstance(filtered_msgs, int) or filtered_msgs <= 0:
        # Sticky compacted step (mem.summary empty) — derive filtered_msgs
        # the way the runtime does for non-trigger steps: head + summary +
        # everything after the summary's forgotten-end. Without exact range
        # we approximate: keep the same compression target as the most
        # recent compaction.
        for back_step, _ in reversed(cache.summary_events):
            if back_step <= target_step:
                back_mem = cache.step_memory.get(back_step) or {}
                back_filtered = back_mem.get("filtered_msgs")
                if isinstance(back_filtered, int) and back_filtered > 0:
                    delta = original_msgs - (back_mem.get("original_msgs") or original_msgs)
                    filtered_msgs = back_filtered + max(0, delta)
                    break
        if not isinstance(filtered_msgs, int) or filtered_msgs <= 0:
            filtered_msgs = max(keep_first + 2, original_msgs)

    # Bound the slice by what we actually have in raw_pre.
    head_len = min(keep_first, len(raw_pre))
    tail_len = max(filtered_msgs - head_len - 1, 1)
    tail_len = min(tail_len, max(0, len(raw_pre) - head_len))

    head = raw_pre[:head_len]
    tail = raw_pre[len(raw_pre) - tail_len:] if tail_len > 0 else []
    summary_msg = {"role": "user", "content": f"{SUMMARY_MARKER} {summary}"}
    view = head + [summary_msg] + tail

    stats = {
        "original_msgs": original_msgs,
        "filtered_msgs": filtered_msgs,
        "n_compactions_active": n_active,
        "active_summary_chars": len(summary),
        "n_messages_in_view": len(view),
    }
    return view, stats


# ---------------------------------------------------------------------------
# Flat-prompt rendering
# ---------------------------------------------------------------------------

_ROLE_HEADERS = {
    "system":    "[System]",
    "user":      "[User]",
    "assistant": "[Assistant]",
    "tool":      "[Tool]",
    "exit":      "[Exit]",
}


def _stringify_message(msg: Dict[str, Any]) -> str:
    """Flatten one message dict (handling tool_calls / non-string content)
    into a printable role-tagged block. Mirrors
    ``replay._flatten_messages_for_chat`` so the flat prompt stays a
    superset of what the LLM saw.
    """
    role = msg.get("role", "user")
    header = _ROLE_HEADERS.get(role, f"[{role}]")
    content = msg.get("content")

    # Inline tool_calls (mini-swe-agent emits assistant turns where the bash
    # command lives in tool_calls, not in `content`).
    if role == "assistant":
        tool_calls = msg.get("tool_calls") or []
        parts: List[str] = []
        if isinstance(content, str) and content:
            parts.append(content)
        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "tool")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = args
            else:
                parsed = args
            if name == "bash" and isinstance(parsed, dict):
                cmd = parsed.get("command", "")
                parts.append(f"$ {cmd}" if cmd else f"{name}({parsed})")
            else:
                parts.append(f"{name}({parsed})")
        body = "\n".join(p for p in parts if p) or ""
    else:
        if isinstance(content, list):  # multimodal / structured
            chunks = []
            for c in content:
                if isinstance(c, dict):
                    chunks.append(c.get("text") or json.dumps(c)[:400])
                else:
                    chunks.append(str(c))
            body = "\n".join(chunks)
        else:
            body = str(content) if content is not None else ""

    return f"{header}\n{body}"


def _render_flat_prompt(messages: List[Dict[str, Any]]) -> str:
    """Render a list of role-tagged blocks separated by blank lines, ending
    in ``RESPONSE_PREFIX`` so the trainer's completion-only loss masks
    everything before the answer token.
    """
    blocks = [_stringify_message(m) for m in messages]
    body = "\n\n".join(b for b in blocks if b)
    return body + RESPONSE_PREFIX


# ---------------------------------------------------------------------------
# Per-row builder
# ---------------------------------------------------------------------------

@dataclass
class RowBuildResult:
    row: Optional[Dict[str, Any]]
    skip_reason: Optional[str] = None


def _source_model(source_dir: str) -> str:
    return {
        "sonnet_fork_gap554_scaleA_v1": "sonnet45",
        "gptoss_fork_gap554_scaleA_v1": "gptoss120b",
        "haiku45_fork_diverse500_scaleA_v1": "haiku45",
    }.get(source_dir, source_dir)


_RECORDED_HDR = "Recorded action:\n"
_PROPOSED_HDR = "\n\nProposed action:\n"
_TRAILER_HDR = "\n\nIs the proposed action safe"


def _parse_actions_from_input(input_text: str) -> Tuple[str, str]:
    """Pull (recorded_action, predicted_action) out of the rendered ``input``
    field that ``aug_pairs.py`` writes.

    The aug_sft_pairs schema does not expose the two strings as separate
    columns — they live only inside the rendered prompt body, between
    the ``Recorded action:`` / ``Proposed action:`` / ``Is the proposed``
    headers. Returns empty strings on parse failure rather than throwing,
    so a malformed row degrades to "no action context" instead of
    aborting the whole build.
    """
    if not input_text:
        return "", ""
    rec_hdr_idx = input_text.find(_RECORDED_HDR)
    if rec_hdr_idx < 0:
        return "", ""
    rec_start = rec_hdr_idx + len(_RECORDED_HDR)
    prop_hdr_idx = input_text.find(_PROPOSED_HDR, rec_start)
    if prop_hdr_idx < 0:
        return input_text[rec_start:].strip(), ""
    recorded = input_text[rec_start:prop_hdr_idx]
    prop_start = prop_hdr_idx + len(_PROPOSED_HDR)
    trailer_idx = input_text.find(_TRAILER_HDR, prop_start)
    if trailer_idx < 0:
        predicted = input_text[prop_start:]
    else:
        predicted = input_text[prop_start:trailer_idx]
    return recorded.strip(), predicted.strip()


def _build_row(
    src_row: Dict[str, Any],
    cache: _TrajCache,
    keep_first: int,
) -> RowBuildResult:
    target_step = int(src_row["step"])
    if target_step not in cache.step_memory:
        return RowBuildResult(None, "step_not_in_training_json")

    view_messages, stats = _reconstruct_view(cache, target_step, keep_first)
    if not view_messages:
        return RowBuildResult(None, "empty_view")

    recorded = src_row.get("recorded_action") or ""
    predicted = src_row.get("predicted_action") or ""
    if not recorded and not predicted:
        recorded, predicted = _parse_actions_from_input(src_row.get("input", "") or "")

    judgment = ACTION_JUDGMENT_TEMPLATE.format(
        step=target_step,
        recorded=recorded,
        predicted=predicted,
        perturbation=src_row.get("perturbation", "") or "",
    )
    judgment_msg = {"role": "user", "content": judgment}
    full_messages = view_messages + [judgment_msg]

    label = int(src_row["label"])
    completion = LABEL_SAFE if label == 1 else LABEL_HARMFUL
    flat_prompt = _render_flat_prompt(full_messages)

    iid = src_row["trajectory_id"]
    source_dir = src_row["source_dir"]
    out = {
        "trajectory_id": iid,
        "instance_id": iid,
        "fork_event_id": src_row.get(
            "fork_event_id", f"{iid}_{target_step}_{src_row.get('perturbation','')}"
        ),
        "source_dir": source_dir,
        "source_model": src_row.get("source_model") or _source_model(source_dir),
        "step": target_step,
        "perturbation": src_row.get("perturbation", ""),
        "label": label,
        "completion": completion,
        "target": completion,  # legacy column for GRPO/eval
        "messages": full_messages,
        "prompt": flat_prompt,
        "input": flat_prompt[: -len(RESPONSE_PREFIX)],  # legacy alias
        "recorded_action": recorded,
        "predicted_action": predicted,
        "diverged": bool(src_row.get("diverged", label == 0)),
        "n_compactions_active": stats["n_compactions_active"],
        "active_summary_chars": stats["active_summary_chars"],
        "n_messages_in_view": stats["n_messages_in_view"],
        "original_msgs": stats["original_msgs"],
        "filtered_msgs": stats["filtered_msgs"],
        "provenance": {
            "training_path": str(cache.training_path),
            "replay_path": str(cache.replay_path),
            "build_view_fn": "memgym.memory.llm_summarizing.build_view",
            "build_script": "src/memgym/training/data/reward_model_pairs.py",
            "keep_first": keep_first,
        },
    }
    return RowBuildResult(out)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def _iter_aug_sft_rows(pairs_path: Path) -> Iterable[Dict[str, Any]]:
    with pairs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_dataset(
    pairs_in: Path,
    trajectories_root: Path,
    out_path: Path,
    eval_frac: float = 0.10,
    keep_first: int = DEFAULT_KEEP_FIRST,
    limit: int = 0,
) -> Dict[str, Any]:
    """Walk every aug_sft_pairs row, emit a multi-turn reward-model row.

    No row-level dedup — perturbations are real augmentation rows. Splits
    are computed once across the full materialized stream (repo-grouped
    via ``aug_pairs.assign_splits``) and assigned in a second pass.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    skip_counts: Counter = Counter()
    cache_by_iid: Dict[Tuple[str, str], _TrajCache] = {}

    n_in = 0
    for src in _iter_aug_sft_rows(pairs_in):
        n_in += 1
        if limit and len(rows) >= limit:
            break
        key = (src["source_dir"], src["trajectory_id"])
        cache = cache_by_iid.get(key)
        if cache is None:
            paths = _resolve_traj_paths(trajectories_root, key[0], key[1])
            if paths is None:
                skip_counts["unresolvable_traj_paths"] += 1
                continue
            try:
                cache = _load_trajectory(*paths)
            except (OSError, json.JSONDecodeError) as exc:
                skip_counts["load_error"] += 1
                logger.warning("load failed for %s: %s", key, exc)
                continue
            cache_by_iid[key] = cache

        result = _build_row(src, cache, keep_first)
        if result.row is None:
            skip_counts[result.skip_reason or "unknown"] += 1
            continue
        rows.append(result.row)

    if not rows:
        raise RuntimeError(
            f"emitted zero rows; checked {n_in} input rows; skip counts: {dict(skip_counts)}"
        )

    splits = assign_splits((r["trajectory_id"] for r in rows), eval_frac)
    with out_path.open("w") as out_fh:
        for r in rows:
            r["split"] = splits[r["trajectory_id"].split("__")[0]]
            out_fh.write(json.dumps(r) + "\n")

    by_label = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_source = Counter(r["source_model"] for r in rows)
    by_pert = Counter(r["perturbation"] for r in rows)
    n_with_summary = sum(1 for r in rows if r["n_compactions_active"] > 0)
    msg_counts = [r["n_messages_in_view"] for r in rows]
    msg_counts.sort()
    def _pct(p: float) -> int:
        if not msg_counts:
            return 0
        return msg_counts[min(int(len(msg_counts) * p), len(msg_counts) - 1)]

    return {
        "input_rows_read": n_in,
        "written": len(rows),
        "skipped": dict(skip_counts),
        "by_label": dict(by_label),
        "by_split": dict(by_split),
        "by_source_model": dict(by_source),
        "by_perturbation": dict(by_pert),
        "rows_with_active_summary": n_with_summary,
        "rows_no_compaction_yet": len(rows) - n_with_summary,
        "n_messages_in_view_p10_p50_p90": [_pct(0.1), _pct(0.5), _pct(0.9)],
        "out_path": str(out_path),
        "keep_first": keep_first,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pairs-in", type=Path,
        default=Path("data/world_model/training_output/aug_sft_full_yn/aug_sft_pairs.jsonl"),
    )
    p.add_argument(
        "--trajectories-root", type=Path,
        default=Path("data/world_model/trajectories"),
    )
    p.add_argument(
        "--out-jsonl", type=Path,
        default=Path("data/world_model/training_output/reward_model_v2/reward_model_pairs_v2.jsonl"),
    )
    p.add_argument("--eval-frac", type=float, default=0.10)
    p.add_argument("--keep-first", type=int, default=DEFAULT_KEEP_FIRST)
    p.add_argument("--limit", type=int, default=0,
                   help="Smoke-run gate — stop after writing N rows (0 = unlimited).")
    args = p.parse_args()

    stats = build_dataset(
        pairs_in=args.pairs_in,
        trajectories_root=args.trajectories_root,
        out_path=args.out_jsonl,
        eval_frac=args.eval_frac,
        keep_first=args.keep_first,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(args.out_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
