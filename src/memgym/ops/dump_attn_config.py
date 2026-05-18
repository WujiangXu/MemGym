"""One-shot dump: how does verl 0.7.1 pick `attn_implementation`?

Outputs the relevant lines from `verl.workers.fsdp_workers._build_model_optimizer`
plus any flash-attn fall-through points so we know which knob to flip
without installing flash-attn.
"""
from __future__ import annotations

import inspect

from verl.workers import fsdp_workers


def main() -> int:
    src = inspect.getsource(fsdp_workers._build_model_optimizer if hasattr(fsdp_workers, "_build_model_optimizer") else fsdp_workers.ActorRolloutRefWorker._build_model_optimizer)
    print(src[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
