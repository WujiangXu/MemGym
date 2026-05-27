# MemGym Data Catalog

Every dataset and model artifact MemGym **produces** (as opposed to the upstream
benchmarks it wraps — those are listed in [datasets-upstream.md](datasets-upstream.md)).
All artifacts are distributed on the Hugging Face Hub under the `MemGym/` org.

- Licensing for everything below is summarized in [licenses.md](licenses.md).
- The MemRM checkpoint has its own deep-dive page: [memrm.md](memrm.md).
- Each artifact's HF repo carries a `README.md` (dataset card) and a `schema.md`
  with the full per-field column inventory; this page is the index over them.

## Release status at a glance

| HF repo (`MemGym/…`) | What | Rows / size | License | Status |
|---|---|---|---|---|
| [`memgym-rm-1p7b`](https://huggingface.co/MemGym/memgym-rm-1p7b) | MemRM QLoRA checkpoint (+ eval JSONs) | 24.5 MB adapter | Apache-2.0 | **Public** |
| [`memgym-rm-iid-heldout`](https://huggingface.co/datasets/MemGym/memgym-rm-iid-heldout) | IID held-out eval split | 3,007 | MIT | **Public** |
| [`memgym-rm-train`](https://huggingface.co/datasets/MemGym/memgym-rm-train) | Paired-trajectory training split | 15,630 | MIT | **Public** |
| [`memgym-rm-scenario-ood-webarena`](https://huggingface.co/datasets/MemGym/memgym-rm-scenario-ood-webarena) | WebArena V2 scenario-OOD eval | 426 (+487 union) | MIT | **Public** |
| [`memgym-rm-scenario-ood-extras`](https://huggingface.co/datasets/MemGym/memgym-rm-scenario-ood-extras) | τ²-bench + WebArena-long scenario-OOD | 6,209 + 111 | MIT | **Public** |
| [`memgym-rm-strategy-ood`](https://huggingface.co/datasets/MemGym/memgym-rm-strategy-ood) | Strategy-OOD eval bundle (2 slices) | 22 | MIT | **Public** |
| [`memgym-dr-instances`](https://huggingface.co/datasets/MemGym/memgym-dr-instances) | MemGym-DR deep-research instances | 1,194 | MIT | **Public** |
| [`memgym-codeqa-instances`](https://huggingface.co/datasets/MemGym/memgym-codeqa-instances) | MemGym-CodeQA coding-QA instances | 4,289 | MIT | **Pending** |
| `memgym-rm-midtrain-sft` | Mid-train SFT data (not in paper) | 582 + 2,126 | — | **Not released** |

All released data and instances are **MIT**. The released MemGym-DR rows are all
`deep_research`-derived synthetic content (no Wikipedia passages ship), so no
CC-BY-SA-4.0 propagation applies; the only forward-looking caveat is that running
the DR pipeline against Wikipedia-based upstream corpora yourself would produce
CC-BY-SA-4.0 derivatives. See [licenses.md](licenses.md).

**Pending** = card published, data withheld until a collaborating institution's
review clears (MemGym-CodeQA). **Not released** = staged but intentionally kept
private (the mid-train SFT set is a generative SFT corpus with no classification
label and is not referenced anywhere in the paper).

---

## 1. The paired-trajectory corpus (MemRM training data)

The labeled corpus that trains and validates MemRM. Each row is one **memory
compaction event** drawn from an agent trajectory, labeled **SAFE** (the
compacted context still supports the recorded next action) or **HARMFUL** (it
does not).

- **Total: 18,637 events** → **15,630 train** ([`memgym-rm-train`](https://huggingface.co/datasets/MemGym/memgym-rm-train))
  + **3,007 eval** ([`memgym-rm-iid-heldout`](https://huggingface.co/datasets/MemGym/memgym-rm-iid-heldout)).
- **Label balance:** 16,357 HARMFUL (87.8%) / 2,280 SAFE (12.2%).
- **Split is deterministic** (SHA256 over `trajectory_id`), repo-grouped so the
  7 held-out repos in eval never appear in train.
- **Reasoners that generated the trajectories:** Claude Sonnet 4.5, Claude
  Haiku 4.5, GPT-OSS-120B.
- **Perturbation families** (how HARMFUL events are synthesized): `aggressive_0.5`,
  `summary_redaction`, `truncate_last_10`, `random_drop_0.2`, `attr_delete_paths`,
  `summary_noise`.

### Schema (23 fields, identical across train / eval)

The fields the MemRM eval/training code reads:

| Field | Type | Meaning |
|---|---|---|
| `label` | int | **0 = HARMFUL, 1 = SAFE** (the target) |
| `prompt` | string | Flattened `[System]/[User]/[Assistant]` prompt text |
| `messages` | list[dict] | The same prompt in chat-message form |
| `completion` / `target` | string | `" Y"` (SAFE) or `" N"` (HARMFUL) |
| `recorded_action` / `predicted_action` | string | The trajectory's actual vs. proposed next action |
| `split` | string | `"train"` or `"eval"` |

Plus provenance/diagnostic fields: `trajectory_id`, `instance_id`,
`fork_event_id`, `source_dir`, `source_model`, `step`, `perturbation`,
`diverged`, `n_compactions_active`, `active_summary_chars`, `n_messages_in_view`,
`original_msgs`, `filtered_msgs`, `provenance` (relative paths only),  `input`.
The full inventory with sample values lives in each repo's `schema.md`.

### Load it

```python
from datasets import load_dataset

train = load_dataset("MemGym/memgym-rm-train",       split="train")   # 15,630
eval_ = load_dataset("MemGym/memgym-rm-iid-heldout", split="train")   #  3,007
print(train[0]["label"], train[0]["completion"])
```

> The on-disk filenames are `reward_model_pairs_v2_train.jsonl` and
> `reward_model_pairs_v2_eval.jsonl`; `reward_model_v2_split.json` records the
> deterministic split assignment.

---

## 2. MemRM out-of-distribution eval splits

These four repos are the OOD rows behind `memgym-eval-rm --dataset all` (paper
Tab. `tab:memrm`). The short name in the table below is the `--dataset` value;
the registry lives in `src/memgym/training/eval/rm_eval.py`.

| `--dataset` | HF repo | File | n | Paper note |
|---|---|---|---|---|
| `scenario-ood-webarena` | `memgym-rm-scenario-ood-webarena` | `pairs_paper_eval.jsonl` | 426 | AUROC 0.748 on n=87 covered subset |
| `scenario-ood-tau2` | `memgym-rm-scenario-ood-extras` | `reward_model_pairs_tau2.jsonl` | 6,209 | τ²-bench compaction events |
| `scenario-ood-wa-long` | `memgym-rm-scenario-ood-extras` | `reward_model_pairs_webarena_longctx.jsonl` | 111 | WebArena long-context (small n) |
| `strategy-ood` | `memgym-rm-strategy-ood` | `data/rm_strategy_ood_pairs.jsonl` | 22 | covered subset — see note |

These rows share the corpus schema (`label`, `prompt`, `messages`, …) plus a few
split-specific extras (e.g. `delta_r`, `ood_strategy` for strategy-OOD).

> **`strategy-ood` is a 22-row covered subset**, not the paper's headline. Paper
> Tab. `tab:memrm` reports AUROC 0.714 on an n=166 broader sweep; the public
> 22-pair bundle (2 slices × 11, rendered in the long-context training shape) is
> the data-integrity-filtered subset for validating *your own* RM checkpoint. The
> `memgym-eval-rm` CLI prints a warning to this effect. Class balance is
> deliberately skewed (21 SAFE / 1 HARMFUL): the strategy swap barely moves
> SWE-bench resolution, so report per-slice deltas, not aggregate ROC.

> **`scenario-ood-webarena`** ships two files: `pairs_paper_eval.jsonl` (426, the
> paper eval) and `pairs_union.jsonl` (487, of which 61 rows are unscored from a
> failed inference shard — use the 426-row file for paper numbers).

---

## 3. MemGym-DR — deep-research instances

[`MemGym/memgym-dr-instances`](https://huggingface.co/datasets/MemGym/memgym-dr-instances)
— 1,194 verified multi-hop deep-research instances (paper Fig. 2b). Built by the
fictionalization pipeline in `src/memgym/pipelines/memgym_ir/`.

| Hop stratum | File | Rows |
|---|---|---|
| 3-hop | `3hop_verified.jsonl` | 161 |
| 4-hop | `4hop_paper_run.jsonl` | 916 |
| 5–6-hop | `56hop_clean.jsonl` | 117 |

Each row carries `question`, `answer`, `num_hops`, `decomposition`, `turns`,
`grounding_facts`, `distractors`, `memory_files`, `gold_supporting_paragraphs`,
`memory_required_facts`, `eviction_policy`, `context_tokens`, `total_tokens`,
`verification`, `rubric` (23 fields total; see the repo `schema.md`).

```python
from datasets import load_dataset
dr = load_dataset("MemGym/memgym-dr-instances",
                  data_files="4hop_paper_run.jsonl", split="train")  # 916
```

> **License: MIT.** All 1,194 released rows are `deep_research`-derived synthetic
> instances — no Wikipedia passages ship, so there is no CC-BY-SA-4.0 propagation
> and no per-row `license_tag`. (The pipeline *can* produce Wikipedia-derived rows
> from MuSiQue / 2WikiMultihopQA / HotpotQA; those would inherit CC-BY-SA-4.0, but
> none are included in this release.)

To regenerate (needs the `[swe]` extra + an LLM key), see
[tracks.md](tracks.md#memgym-dr) and `TESTING.md` Tier 3.

---

## 4. MemGym-CodeQA — coding-QA instances  ⏳ pending

[`MemGym/memgym-codeqa-instances`](https://huggingface.co/datasets/MemGym/memgym-codeqa-instances)
— **card published, data pending a collaborating institution's review.** Paper
Fig. 2a.

- **This set:** 4,289 verified instances (kept from a 4,999-attempt generation
  batch; Claude Opus 4 as both worker and verifier; `medium` difficulty).
- **Not the same as the pilot in the paper's verification table:** that table
  cites a separate 1,000-instance pilot (670 verified). The pilot and this
  4,289-row production set are independent runs — the differing counts are *two
  datasets*, not an error.
- Schema: `instance_id`, `task_prompt`, `memory_files`, `repo_context`,
  `repo_files`, `qa_pairs`, `grounding_facts`, `distractors`, `patch`,
  `num_critical_facts`, `num_external_facts`, `difficulty_preset`, … (see card).

Generate it yourself today (no review gate on the pipeline) — see
[tracks.md](tracks.md#memgym-codeqa) and `TESTING.md` Tier 3.

---

## Regenerating / retraining

- **Eval** pulls the split repos above automatically (`memgym-eval-rm` resolves
  HF repo IDs via `snapshot_download`). No manual download needed.
- **Retraining MemRM** uses the public corpus: train on
  `MemGym/memgym-rm-train` and validate on `MemGym/memgym-rm-iid-heldout`. (The
  legacy `download_rm_v2.py` script points at an internal monorepo that is not
  part of the public release; prefer the two split repos.)

## Known data discrepancies (tracked, not hidden)

These are documented on the respective HF cards and surfaced here so reproductions
line up with the paper:

- **Strategy-OOD ECE (M2):** paper Tab. `tab:memrm` reports ECE 0.850; on-disk
  recomputation of the historical 166-row file gives 0.578. Pending a paper
  cross-check. The public 22-row artifact is a *different, smaller* bundle and
  will not reproduce either number directly.
- **MemRM adapter (M3/M4):** `adapter_config.json` has `lora_dropout=0.05` and a
  24.5 MB adapter; the paper text states 0 and ~25.7 MB. The files are
  authoritative. See [memrm.md](memrm.md).
- **CodeQA counts:** 670 (paper pilot) vs 4,289 (this production set) — different
  runs, see §4.
