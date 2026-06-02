# MemRM — Memory Reward Model

MemRM scores whether a memory-compaction event in an agent trajectory is **SAFE**
(the compacted context still supports the recorded next action) or **HARMFUL**.
It is the model behind paper Tab. `tab:memrm`.

- **Checkpoint:** [`MemGym/memgym-rm-1p7b`](https://huggingface.co/MemGym/memgym-rm-1p7b) (Apache-2.0)
- **Training data:** the paired-trajectory corpus — see [data.md](data.md#1-the-paired-trajectory-corpus-memrm-training-data)
- **CLI:** `memgym-eval-rm` (installed with the `[eval]` extra)

## Model configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` (Apache-2.0) |
| Adapter | LoRA / QLoRA (4-bit NF4 at train time) |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | **0.05** (the `adapter_config.json` value is authoritative; paper text says 0 — discrepancy **M3**) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | 32,768 |
| Paper checkpoint | `checkpoint-500` (stored as a subfolder, not a git revision) |
| Adapter size | **24.5 MB** (paper text says ~25.7 MB — discrepancy **M4**) |
| Decision threshold | `t* = 0.88` (the `memgym-eval-rm` default) |

## Reported metrics (checkpoint-500)

| Split | AUROC | ECE | Notes |
|---|---|---|---|
| IID held-out (n=3,007) | **0.985** | 0.009 | accuracy 97.3%, Safe-F1 0.849 |
| Scenario-OOD WebArena (n=426) | 0.748 | — | on n=87 covered subset |
| Strategy-OOD (n=166) | 0.714 | 0.850 | covered 26.5%; ECE flagged as **M2** ([data.md](data.md#known-data-discrepancies-tracked-not-hidden)) |

## Reproduce the table

```bash
uv pip install -e ".[eval]"        # bootstrap uv once: pip install uv

# Headline IID number (n=3,007); add --limit 8 for a quick GPU smoke
memgym-eval-rm --dataset iid-heldout --checkpoint MemGym/memgym-rm-1p7b

# Sweep every registered split and write a JSON table
memgym-eval-rm --dataset all --checkpoint MemGym/memgym-rm-1p7b \
    --output results/memrm_table.json
```

> Reproduced 2026-06-01 on 1×H100 80GB (4-bit NF4), ~34 min wallclock:
> AUROC = **0.985** (n=3007), ECE = 0.010, Cov@t = 95.9%, Acc@t = 0.985 —
> matches the IID row above. `resolve_checkpoint` descends into
> `checkpoint-500/` automatically, so the documented `--checkpoint
> MemGym/memgym-rm-1p7b` form works without a `subfolder=` workaround.

Registered `--dataset` short names: `iid-heldout`, `train-sanity`,
`scenario-ood-tau2`, `scenario-ood-wa-long`, `scenario-ood-webarena`,
`strategy-ood` (see [data.md](data.md#2-memrm-out-of-distribution-eval-splits)).
Pass a local `*.jsonl` path to score custom data. 4-bit inference needs CUDA +
`bitsandbytes`; use `--no-4bit` for fp16 or `--device cpu` (slow).

## Load it directly

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B-Base", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(
    base, "MemGym/memgym-rm-1p7b", subfolder="checkpoint-500")
tok = AutoTokenizer.from_pretrained("MemGym/memgym-rm-1p7b")

# P(HARMFUL) = softmax over the " Y"/" N" logits at the final position
```

A higher-level Python API (`MemoryWorldModel.from_checkpoint`,
`WorldModelEvaluator`) lives in `src/memgym/training/models/world_model.py`.

## Scoring your own memory strategy against MemRM

Use `memgym-eval-memory` to score a memory strategy on a raw-trajectory JSONL.
The signal is the **AUROC delta** between two strategies on the same file (the RM
was trained without a memory-output block, so absolute numbers drift from the
table above — the CLI prints this caveat). See the README section
"Evaluating your own memory against MemRM" and `testing.md` Tier 2.

## Discrepancies

The HF model card is the source of truth for **M2** (Strategy-OOD ECE),
**M3** (LoRA dropout) and **M4** (adapter size); all three are recorded in
[data.md](data.md#known-data-discrepancies-tracked-not-hidden). The repo also
contains an 8B development baseline (`reward_model_v2_run1/eval_results.json`) —
that is **not** this model, even though its AUROC also rounds to 0.985.
