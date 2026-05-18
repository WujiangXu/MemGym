"""Set env vars, then exec another Python module as __main__.

Mirrors `scripts.run_with_env` but lives inside the installed `memgym`
package so it's importable from any cwd via `pip install -e .` — the
top-level `scripts/` directory is not installed and not always on
sys.path under the EC2 `/run-script` runner.

Use this when you need to inject env vars (e.g. CUDA_VISIBLE_DEVICES,
OPENAI_API_KEY) into a job submitted via `/run-script`, which exposes
module + args + venv + cwd but no `env` field.

Usage:
    python -m memgym.training.scripts.run_with_env \
        --set-env CUDA_VISIBLE_DEVICES=4,5,6,7 \
        --set-env OPENAI_API_KEY=xxx \
        -- torch.distributed.run --nproc-per-node 4 \
           -m memgym.training.scripts.train_rl_online ...

The literal `--` separates wrapper flags from the child module + its
own argv. Env values pass through process memory only — they are NEVER
persisted to git.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        raise SystemExit(
            "run_with_env: missing `--` separator. "
            "Usage: --set-env K=V [...] -- <module> [args...]"
        )
    sep = argv.index("--")
    return argv[:sep], argv[sep + 1:]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_argv, child_argv = _split_argv(argv)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--set-env", action="append", default=[],
        help="K=V env var. Repeatable.",
    )
    args = p.parse_args(wrapper_argv)

    for kv in args.set_env:
        if "=" not in kv:
            raise SystemExit(f"--set-env value must be K=V, got: {kv!r}")
        k, v = kv.split("=", 1)
        os.environ[k] = v

    if not child_argv:
        raise SystemExit("run_with_env: no child module after `--`")

    child_module = child_argv[0]
    # `runpy.run_module` consults sys.argv[0] for the module's __main__,
    # so reset it so the child sees its own argv.
    sys.argv = [child_module] + child_argv[1:]
    runpy.run_module(child_module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
