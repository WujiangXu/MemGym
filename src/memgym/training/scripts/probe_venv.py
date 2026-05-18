"""Probe the active venv for known training-stack package versions.

The the class-weighted CE checkpoint reward-model ship cell converged with `trl 0.16` per
`progress.md:138`. Smoke runs against the v2 dataset hang silently
post-init across every config we've tried (seq 1024 → 32K, 1.7B → 9B,
batch 1 → 4). This points at a venv drift, not at hyperparameters.

Run via:
"""
from __future__ import annotations

import importlib
import subprocess
import sys

PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "tokenizers",
    "flash_attn",
    "unsloth",
    "xformers",
]


def _try_import_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        return f"NOT IMPORTED ({type(e).__name__}: {e})"
    return getattr(mod, "__version__", "?")


def main() -> int:
    print(f"[python] {sys.executable}")
    print(f"[python] {sys.version.splitlines()[0]}")
    print()
    print("=== Import-time versions ===")
    for name in PACKAGES:
        print(f"  {name:<14s} {_try_import_version(name)}")

    print("\n=== pip show (selected) ===")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "show",
         "torch", "transformers", "trl", "peft",
         "accelerate", "bitsandbytes", "datasets"],
        capture_output=True, text=True,
    )
    print(rc.stdout)
    if rc.stderr.strip():
        print("--- pip show stderr ---")
        print(rc.stderr)

    print("=== torch CUDA + NCCL ===")
    try:
        import torch  # type: ignore[import-not-found]
        print(f"  torch.__version__       = {torch.__version__}")
        print(f"  torch.version.cuda      = {torch.version.cuda}")
        print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
        print(f"  torch.cuda.device_count = {torch.cuda.device_count()}")
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_getNCCLVersion"):
            try:
                print(f"  NCCL version            = {torch._C._cuda_getNCCLVersion()}")
            except Exception as e:  # noqa: BLE001
                print(f"  NCCL version            = (probe failed: {e})")
        else:
            print("  NCCL version            = (no _cuda_getNCCLVersion)")
        sdpa = torch.nn.functional.scaled_dot_product_attention
        print(f"  has SDPA                = {sdpa is not None}")
    except Exception as e:  # noqa: BLE001
        print(f"  torch probe failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
