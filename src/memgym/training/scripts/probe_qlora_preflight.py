"""Pre-flight checks for the QLoRA + 32K rm_v2 retrain.

Reports presence of the QLoRA-relevant deps (bitsandbytes, flash-attn,
peft) plus boolean visibility of WANDB_API_KEY in the inherited
environment — never the key itself.

Run via: POST /run-script module=memgym.training.scripts.probe_qlora_preflight
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


def _ver(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return f"NOT_IMPORTED ({type(exc).__name__}: {exc})"


def main() -> int:
    out = {
        "python": sys.executable,
        "versions": {
            name: _ver(name)
            for name in (
                "torch", "transformers", "peft", "accelerate", "trl",
                "bitsandbytes", "flash_attn",
            )
        },
        "env_present": {
            "WANDB_API_KEY": bool(os.environ.get("WANDB_API_KEY")),
            "WANDB_MODE": os.environ.get("WANDB_MODE") or None,
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT") or None,
            "HF_HOME": os.environ.get("HF_HOME") or None,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES") or None,
        },
        "wandb_key_file": {
            # Reports whether the bootstrap file written by
            # `bootstrap_wandb_key` exists; never echoes the key itself.
            "path": str(Path.home() / ".config" / "memgym" / "wandb_api_key"),
            "exists": (Path.home() / ".config" / "memgym" / "wandb_api_key").exists(),
        },
    }
    try:
        import torch  # type: ignore[import-not-found]

        out["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["cuda"]["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:  # noqa: BLE001
        out["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
