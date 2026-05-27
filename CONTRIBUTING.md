# Contributing to MemGym

Thanks for your interest in MemGym! This guide covers the dev setup and the
conventions that keep the tree clean.

## Dev setup

MemGym uses a `src/` layout and PEP 621 packaging (`pyproject.toml`). Install in
editable mode with the dev extras:

```bash
pip install uv            # one-time; every `pip` below can then be `uv pip`
uv pip install -e ".[dev]"
```

Add track extras as needed: `.[swe]` (SWE-Gym + the synthetic pipelines),
`.[tau2]`, `.[eval]` (MemRM), `.[train]`.

## Running the checks

```bash
pytest tests/unit          # fast unit suite (what CI runs)
ruff check src tests       # lint (line-length 120, configured in pyproject.toml)
```

Some tests require heavy optional deps (`torch`, `datasets`,
`sentence_transformers`); they `skip`/`importorskip` cleanly when those aren't
installed, so a partial install still gives a green run for the code you touched.

## Where things go

MemGym borrows its layout from Gymnasium (a clean core package + registry) and
verl (a `library / examples / scripts` split):

| Location | Purpose | Ships in the wheel? |
|----------|---------|---------------------|
| `src/memgym/` | The importable library. New code goes here. | Yes |
| `examples/` | Runnable, copy-pasteable how-tos, grouped by track. Import `memgym` as an installed package — no `sys.path` hacks. | No |
| `scripts/` | Dev/ops tooling (HF dataset staging, GPU cleanup, env setup). Run by hand, never imported. | No |
| `tests/` | `unit/` mirrors the package; `integration/` for end-to-end; `fixtures/` for shared data. | No |
| `docs/` | All long-form documentation. The root `README.md` stays an orientation page. | No |

Rule of thumb: **library code → `src/`**, **"here's how to run track X" → `examples/`**,
**"here's how we operate the cluster" → `scripts/`**.

## Adding a memory strategy

Memory strategies are resolved by *name* through a registry, so you never edit a
CLI to add one — implement the interface and register it:

```python
from memgym.memory.base import BaseMemoryManager, FilteredContext, register_memory_model

class MyMemory(BaseMemoryManager):
    def manage_context(self, original_context, current_observation, metadata=None):
        ...
        return FilteredContext(content=..., metadata={"tokens": ..., "strategy": "mine"})
    def reset(self) -> None:
        ...

register_memory_model("my_memory", MyMemory)
```

Built-in strategies live under `src/memgym/memory/strategies/`; track-tailored
adapters under `adapters/`; vendored third-party memory cores under `external/`.
See [`docs/architecture.md`](docs/architecture.md) for the full extension API.

## Pull requests

- Keep PRs focused; one logical change per PR.
- Run `pytest tests/unit` and `ruff check` before pushing.
- Update the relevant page under `docs/` (and `CHANGELOG.md`) when behavior changes.
- Code in this repo is Apache-2.0; by contributing you agree your contribution is
  licensed under the same terms.
