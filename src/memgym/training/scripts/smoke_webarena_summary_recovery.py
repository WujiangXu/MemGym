"""Local smoke test for the webarena_long_context_pairs summary-recovery patch.

Synthesizes a 4-step trajectory.json with was_compacted=True at step 2 and
a [Summary] body in step.context, runs the patched loader + build_pair, and
asserts that the rendered prompt contains the summary body.

This is the read-only test fixture we use BEFORE running the builder on
EC2 cohorts — it certifies that the local code path produces
full-context prompts (history + [Summary] + task).

Usage::

    python3 -m memgym.training.scripts.smoke_webarena_summary_recovery
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from memgym.training.data.reward_model_pairs import SUMMARY_MARKER
from memgym.training.data.webarena_long_context_pairs import (
    build_pair,
    load_webarena_task,
)


SUMMARY_BODY = (
    "User wants to find a public repo named memgym. Earlier the agent "
    "navigated to the GitLab dashboard, opened the search bar, and "
    "issued the query 'memgym public'. No results yet."
)
TASK_INSTRUCTION = "Find the public repo memgym on GitLab and report its URL."


def _make_trajectory(out_path: Path, *, with_summary: bool) -> None:
    """Synthesize a 4-step trajectory.json.

    Step 2 has was_compacted=True; if ``with_summary`` is True, step.context
    holds the post-compaction view including a SUMMARY_MARKER message —
    matching the real webarena runner's persisted shape.
    """
    ctx_with_summary = [
        {"role": "system", "content": "You are a web-browsing agent."},
        {"role": "user", "content": f"{SUMMARY_MARKER} {SUMMARY_BODY}"},
        {"role": "user", "content": "OBSERVATION (truncated tail)"},
    ]

    steps = []
    for k in range(4):
        is_compaction_step = (k == 2)
        steps.append({
            "observation": f"obs at step {k}",
            "reasoning_action": f"thinking at step {k}",
            "action": {"action_type": "click", "selector": f"#button-{k}"},
            "memory_action": {
                "was_compacted": is_compaction_step,
                "strategy": "summarizing_ms10" if is_compaction_step else None,
                "tokens": 4321 if is_compaction_step else None,
                "compression_ratio": 1.42 if is_compaction_step else None,
            },
            "context": ctx_with_summary if (is_compaction_step and with_summary) else None,
        })

    traj = {
        "metadata": {"app_name": "gitlab", "instruction": TASK_INSTRUCTION, "reward": 1.0},
        "episodes": [{
            "metadata": {"app_name": "gitlab", "instruction": TASK_INSTRUCTION},
            "total_reward": 1.0,
            "steps": steps,
        }],
    }
    out_path.write_text(json.dumps(traj))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ood_dir = td / "gitlab" / "h0"
        baseline_dir = td / "baseline" / "gitlab" / "h0"
        ood_dir.mkdir(parents=True)
        baseline_dir.mkdir(parents=True)
        _make_trajectory(ood_dir / "trajectory.json", with_summary=True)
        # Use a baseline trajectory with NO compaction — just history.
        _make_trajectory(baseline_dir / "trajectory.json", with_summary=False)

        ood = load_webarena_task(ood_dir)
        base = load_webarena_task(baseline_dir)
        if ood.cache is None or base.cache is None:
            print("FAIL: load_webarena_task returned None cache")
            return 1

        print(f"ood: n_messages={ood.n_messages} n_steps={ood.n_steps} "
              f"n_compactions={ood.n_compactions}")
        print(f"  step_memory[2].summary head: "
              f"{(ood.cache.step_memory[2]['summary'] or '')[:60]!r}")
        print(f"  summary_events: {[(k, v[:40]) for k, v in ood.cache.summary_events]}")

        # Score step 3 — first step AFTER the compaction at step 2. The
        # active summary at step ≤ 3 is the one from step 2; _reconstruct_view
        # should inject it.
        result = build_pair(
            baseline_cache=base.cache,
            ood_cache=ood.cache,
            step_k=3,
            label=1,
            delta_r=0.0,
            baseline_reward=1.0,
            memory_reward=1.0,
            domain="gitlab",
            task_id="h0",
            cohort_name="smoke",
            ood_strategy="summarizing_ms10",
        )
        if result.row is None:
            print(f"FAIL: build_pair returned None ({result.skip_reason})")
            return 1

        prompt = result.row["prompt"]
        has_marker = SUMMARY_MARKER in prompt
        has_body = SUMMARY_BODY[:40] in prompt
        has_task = TASK_INSTRUCTION[:30] in prompt
        has_recorded = "Recorded action" in prompt
        print(f"\nprompt length: {len(prompt)}")
        print(f"contains [Summary] marker: {has_marker}")
        print(f"contains summary body:     {has_body}")
        print(f"contains task instruction: {has_task}")
        print(f"contains action template:  {has_recorded}")

        if not (has_marker and has_body and has_task and has_recorded):
            print("\nFAIL: rendered prompt is missing one of the required blocks.")
            print("--- prompt head 600 ---")
            print(prompt[:600])
            print("--- prompt tail 600 ---")
            print(prompt[-600:])
            return 1

        print("\nOK: rendered prompt contains history + [Summary] body + task + action template.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
