"""Check torch ↔ CUDA-driver compatibility for the active venv.

Way A's isolated venv (`memgym-way-a`) installed verl 0.5.0 which
pulled a fresh torch wheel. The import probe surfaced a UserWarning:

    CUDA initialization: The NVIDIA driver on your system is too old
    (found version 12080). Please update your GPU driver ...

12080 = 12.8.0 — the actual NVIDIA driver. The warning means torch was
compiled for a CUDA toolkit newer than what the driver can run.

This probe prints the torch build version, the runtime CUDA the wheel
expects, the live driver version, whether `torch.cuda.is_available()`
returns True, and (if available) device count + bf16 support. It also
prints the same fields for the **other** venv (`memgym-world-model`)
when run there, so we can pick a torch+CUDA combo that matches.

Exit code: 0 always — the value is in the printed table.
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"python: {sys.version.split()[0]}")
    print(f"torch.__version__:        {torch.__version__}")
    print(f"torch.version.cuda:       {torch.version.cuda}")
    print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")

    try:
        avail = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {avail}")
        if avail:
            print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  cuda:{i}  name={torch.cuda.get_device_name(i)}  cc={torch.cuda.get_device_capability(i)}")
            print(f"torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}")
    except Exception as e:
        print(f"torch.cuda probe FAILED: {type(e).__name__}: {e}")

    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            timeout=10,
        ).decode().strip()
        print("\nnvidia-smi:")
        for line in out.splitlines():
            print(f"  {line}")
    except Exception as e:
        print(f"nvidia-smi probe FAILED: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
