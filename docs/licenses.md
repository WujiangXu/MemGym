# Licenses

MemGym mixes permissive licenses across three kinds of artifact: **code**,
**data/instances**, and **model weights**. This page mirrors the paper's
Asset-Licenses appendix and is the authoritative summary for the public release.

## MemGym's own artifacts

| Artifact | License | Why |
|---|---|---|
| Repository code (the `memgym` package, wrappers, pipelines, scripts) | **Apache-2.0** | see [`../LICENSE`](../LICENSE) + [`../NOTICE`](../NOTICE) |
| Paired-trajectory corpus (`memgym-rm-train`, `memgym-rm-iid-heldout`) | **MIT** | MemGym-generated labels over MIT-licensed upstream trajectories |
| MemRM OOD eval splits (`memgym-rm-scenario-ood-*`, `memgym-rm-strategy-ood`) | **MIT** | same |
| MemGym-CodeQA instances (`memgym-codeqa-instances`) | **MIT** | synthesized from SWE-smith (MIT) |
| MemGym-DR instances (`memgym-dr-instances`) | **MIT** | all released rows are `deep_research`-derived synthetic content; no Wikipedia passages ship (see note below) |
| MemRM weights (`memgym-rm-1p7b`) | **Apache-2.0** | inherited from the `Qwen3-1.7B-Base` base model; **not a free choice** |

This matches the paper text: *"The MemGym wrappers, the paired-trajectory
corpus, and the synthetic MemGym-CodeQA / MemGym-DR instances are released under
MIT; MemRM weights inherit the Apache-2.0 license of the Qwen3-1.7B base."*

## Upstream assets (used under their original terms)

| Asset | License |
|---|---|
| SWE-Gym, SWE-bench, τ²-bench, SWE-smith, OpenHands condenser interface | MIT |
| WebArena, Qwen3-1.7B-Base, GPT-OSS-120B | Apache-2.0 |
| Multi-hop QA seeds for the DR pipeline (2WikiMultihopQA, MuSiQue, HotpotQA — Wikipedia-derived) | CC-BY-SA-4.0 |
| Claude Sonnet 4.5 / Haiku 4.5 (trajectory-collection reasoners) | accessed via AWS Bedrock under Anthropic's commercial API terms |

## Practical notes

- **MIT vs Apache-2.0** are both permissive (commercial use allowed). Apache-2.0
  adds an explicit patent grant and a `NOTICE` requirement; that is why the
  Qwen3-derived weights stay Apache-2.0.
- **No Wikipedia content ships in this release.** Every released MemGym-DR row is
  `deep_research`-derived synthetic text, so the corpus is plain MIT. CC-BY-SA-4.0
  only enters if *you* run the DR pipeline against the Wikipedia-derived QA seeds
  above — derivatives of those seeds inherit share-alike.
- Each HF repo's dataset/model card restates its own license in the YAML header.
