"""AST-walk verl.experimental.agent_loop submodules to enumerate all imports.

After bumping verl 0.4.1 → 0.5.0, the agent_loop module surfaces missing
transitive deps one-by-one (cachetools first, fastapi next, ...). Rather
than burn one /run-script + /pip-install round-trip per dep, this probe
scans every .py file under `verl/experimental/agent_loop/`, records the
top-level import names, then attempts to import each — printing the ones
that fail so we can batch-install them in a single /pip-install call.

Run via the EC2 /run-script endpoint with venv=memgym-way-a. Exits 0
unconditionally; the value is in the printed list.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path


def _find_verl_dir() -> Path:
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("verl not installed in this venv")
    return Path(spec.origin).resolve().parent


def _collect_imports(py_files: list[Path]) -> set[str]:
    names: set[str] = set()
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    names.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".", 1)[0]
                    names.add(top)
    return names


def main() -> int:
    verl_dir = _find_verl_dir()
    print(f"verl install dir: {verl_dir}")
    print(f"verl version: {getattr(importlib.import_module('verl'), '__version__', '?')}")

    target_dir = verl_dir / "experimental" / "agent_loop"
    if not target_dir.exists():
        print(f"FAIL: {target_dir} does not exist")
        return 1

    py_files = sorted(target_dir.rglob("*.py"))
    print(f"\nFiles scanned: {len(py_files)}")
    for f in py_files:
        print(f"  {f.relative_to(verl_dir)}")

    imports = _collect_imports(py_files)
    print(f"\nUnique top-level imports: {len(imports)}")

    builtin = set(sys.builtin_module_names) | {
        "os", "sys", "json", "asyncio", "typing", "pathlib", "logging",
        "collections", "dataclasses", "abc", "functools", "itertools",
        "re", "time", "uuid", "weakref", "copy", "enum", "warnings",
        "concurrent", "contextlib", "inspect", "math", "random",
        "threading", "io", "string", "traceback", "argparse", "ast",
        "importlib", "pickle", "tempfile", "shutil", "subprocess",
        "datetime", "hashlib", "base64", "struct",
    }

    missing: list[tuple[str, str]] = []
    available: list[str] = []
    skipped: list[str] = []
    for name in sorted(imports):
        if name in builtin or name == "verl":
            skipped.append(name)
            continue
        try:
            importlib.import_module(name)
            available.append(name)
        except Exception as e:
            missing.append((name, f"{type(e).__name__}: {e}"))

    print(f"\nSkipped (stdlib/self): {len(skipped)} — {skipped}")
    print(f"\nAvailable ({len(available)}): {available}")
    print(f"\nMISSING ({len(missing)}):")
    for name, err in missing:
        print(f"  {name:30s}  {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
