"""Intrinsic eval for a summarizer LoRA checkpoint ((see paper)).

Purpose — give the collaborator a no-Qwen3.6, no-vLLM signal they can run
on their training node to track progress during SFT or offline RL. Three
metrics, all computed from the held-out split of `summarizer_pairs.jsonl`:

1. **perplexity** on the teacher's completion — SFT sanity gate. Target
   < 1.5 from the H.2 plan.
2. **compression_ratio p50** (greedy-decoded summary length / input
   length, char-based to avoid tokenizer coupling). Reward-hack
   guardrail: must stay in [0.2, 0.7].
3. **summary_chars p50** of greedy-decoded outputs. Flag < 200 chars as
   a trivial-summary collapse.

An optional `--compute-action-match` flag computes the tier-1 metric from
H.3' (first-token agreement of proxy agent on post-compaction history
with student vs teacher summary). It is **stubbed to raise** today —
the collaborator implements it during H.3' when they build the GRPO
reward, and both the reward function and this script import the same
`score_action_match_row` primitive to avoid drift.

Usage:

    python -m memgym.training.scripts.eval_summarizer_offline \\
        --checkpoint checkpoints/summarizer_sft_8b_v1 \\
        --holdout-pairs data/world_model/training_output/summarizer_sft/summarizer_pairs.jsonl \\
        --proxy-agent Qwen/Qwen3-8B-Base \\
        --max-rows 50

Emits a single JSON line on stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memgym.training.data.summarizer_dataset import load_eval_rows

logger = logging.getLogger(__name__)


def _load_policy(checkpoint: Path, dtype: torch.dtype) -> tuple:
    """Load a LoRA-adapted causal LM. The checkpoint dir carries both the
    adapter config (pointing at the base model) and the tokenizer."""
    try:
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError("peft is required; `pip install peft`") from e

    adapter_cfg = json.loads((checkpoint / "adapter_config.json").read_text())
    base_name = adapter_cfg.get("base_model_name_or_path") or "Qwen/Qwen3-8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _row_ppl(model, tokenizer, prompt: str, completion: str, max_length: int) -> Optional[float]:
    """Per-token PPL of `completion` conditioned on `prompt`. Returns
    `None` when tokenization yields zero completion tokens (which can
    happen on a trivially-empty summary — excluded from the mean)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(
        prompt + completion, add_special_tokens=False, return_tensors="pt",
        truncation=True, max_length=max_length,
    ).input_ids
    full_ids = full_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    n_completion = full_ids.shape[1] - n_prompt
    if n_completion <= 0:
        return None

    out = model(full_ids, labels=full_ids)
    # HF `labels` shift includes the prompt's NLL — recompute on the
    # completion slice only for a faithful completion-only PPL.
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    # Mask out the prompt portion of the shifted target sequence. After
    # the shift, position i predicts token i+1, so the first
    # `n_prompt - 1` shifted positions are prompt→prompt or prompt→first
    # completion token. We mask everything up to (and including) the
    # position that predicts the prompt's last token.
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, n_prompt - 1:] = True
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    nll = -picked[mask].mean().item()
    return math.exp(nll)


@torch.no_grad()
def _greedy_summary(model, tokenizer, prompt: str, max_new: int = 512) -> str:
    """Greedy decode from the checkpoint on the given prompt. Temperature
    is not relevant here — this is a deterministic sample for telemetry,
    not a training rollout."""
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192,
    ).input_ids.to(model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Strip the prompt tokens — we only want the completion.
    generated = out[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def score_action_match_row(
    row: Dict[str, Any],
    student_summary: str,
    proxy_agent_name: str,
) -> float:
    """First-token-agreement proxy reward used by both the tier-1 eval
    and the H.3' GRPO reward function. Placeholder today — the
    collaborator implements this during H.3' reward-function design.

    Contract:
    - Load `proxy_agent_name` once (cache across calls).
    - Build `post_compaction_history_with(summary=student_summary)` from
      `row` — needs `aug_sft_pairs.jsonl` fields that are NOT currently
      carried over into `summarizer_pairs.jsonl` (`recorded_action`,
      `proposed_action`, `post_compaction_messages`). Joining back via
      `fork_event_id` is the likely implementation path.
    - Return 1.0 if proxy agent's first-token argmax on post-compaction-with-student
      == first token of `row["recorded_action"]`; else 0.0.
    """
    raise NotImplementedError(
        "action-match scoring lands in H.3' alongside the GRPO reward function. "
        "Pass --compute-action-match=false until then."
    )


def evaluate(
    checkpoint: Path,
    pairs_path: Path,
    proxy_agent: Optional[str],
    max_rows: int,
    max_new: int,
    max_length: int,
    compute_action_match: bool,
) -> Dict[str, Any]:
    model, tokenizer = _load_policy(
        checkpoint,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    rows = load_eval_rows(pairs_path)
    if max_rows:
        rows = rows[:max_rows]

    ppls: List[float] = []
    comp_ratios: List[float] = []
    summary_chars: List[int] = []
    action_matches: List[float] = []

    for r in rows:
        ppl = _row_ppl(model, tokenizer, r["prompt"], r["completion"], max_length)
        if ppl is not None:
            ppls.append(ppl)
        summary = _greedy_summary(model, tokenizer, r["prompt"], max_new=max_new)
        input_chars = r.get("num_input_chars") or max(1, len(r["prompt"]))
        comp_ratios.append(len(summary) / max(1, input_chars))
        summary_chars.append(len(summary))
        if compute_action_match:
            action_matches.append(score_action_match_row(r, summary, proxy_agent or ""))

    def _p50(xs): return statistics.median(xs) if xs else None
    def _mean(xs): return sum(xs) / len(xs) if xs else None

    result = {
        "checkpoint": str(checkpoint),
        "n_eval": len(rows),
        "ppl_mean": _mean(ppls),
        "ppl_p50": _p50(ppls),
        "compression_ratio_p50": _p50(comp_ratios),
        "summary_chars_p50": _p50(summary_chars),
    }
    if compute_action_match:
        result["action_match_rate"] = _mean(action_matches)
    else:
        result["action_match_rate"] = None  # stubbed until H.3'
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="LoRA checkpoint dir (must contain adapter_config.json + tokenizer).")
    parser.add_argument("--holdout-pairs", type=Path, required=True,
                        help="summarizer_pairs.jsonl — eval split is taken by split=='eval'.")
    parser.add_argument("--proxy-agent", default="Qwen/Qwen3-8B-Base",
                        help="Used only when --compute-action-match is set.")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Cap eval rows (0 = all held-out). Use e.g. 50 for a fast smoke.")
    parser.add_argument("--max-new", type=int, default=512,
                        help="Greedy-decode cap for summary telemetry.")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Tokenizer max_length for ppl computation.")
    parser.add_argument("--compute-action-match", action="store_true",
                        help="Compute tier-1 action-match rate (requires H.3' implementation).")
    args = parser.parse_args()

    result = evaluate(
        checkpoint=args.checkpoint,
        pairs_path=args.holdout_pairs,
        proxy_agent=args.proxy_agent,
        max_rows=args.max_rows,
        max_new=args.max_new,
        max_length=args.max_length,
        compute_action_match=args.compute_action_match,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
