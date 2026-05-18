"""Static probe: verl version + AgentLoopBase API surface.

Run on the Way A venv to catch verl API drift before the floor smoke
spends GPU on a stale scaffold. Reports the import path actually
available + the `run` method signature so we can compare against what
`way_a_verl/agent_loop.py` calls.
"""
from __future__ import annotations

import inspect
import sys


def main() -> int:
    import verl
    print(f"verl_version={verl.__version__}")

    # Original v0.4.x location
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase
        print("import_ok=verl.experimental.agent_loop.agent_loop.AgentLoopBase")
    except ImportError as exc:
        print(f"import_failed_old_path: {exc}")
        # Hunt for any module that looks like the agent loop in this version
        import pkgutil
        candidates = []
        for m in pkgutil.walk_packages(verl.__path__, prefix="verl."):
            name = m.name.lower()
            if "agent" in name and "loop" in name:
                candidates.append(m.name)
        print(f"agent_loop_module_candidates={candidates}")
        return 0

    # Method surface we depend on in agent_loop.py
    methods = [n for n, _ in inspect.getmembers(AgentLoopBase, predicate=inspect.isfunction)]
    print(f"methods={methods}")

    # Specifically the `run` signature drives what kwargs we can pass
    if hasattr(AgentLoopBase, "run"):
        try:
            sig = inspect.signature(AgentLoopBase.run)
            print(f"run_signature={sig}")
        except (ValueError, TypeError) as exc:
            print(f"run_signature_introspect_failed: {exc}")
    else:
        print("run_NOT_present_on_class")

    # Tool registry attribute we depend on (`self.tools[tool_cls_name]`)
    print(f"has_tools_attr={hasattr(AgentLoopBase, 'tools')}")
    print(f"has_generate_sequences={hasattr(AgentLoopBase, 'generate_sequences')}")

    # Hunt for replacement APIs in 0.5.0
    print()
    print("=== AgentLoopBase __init__ signature ===")
    try:
        print(f"init_signature={inspect.signature(AgentLoopBase.__init__)}")
    except (ValueError, TypeError) as exc:
        print(f"init_introspect_failed: {exc}")

    print()
    print("=== AgentLoopOutput shape ===")
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
        print(f"AgentLoopOutput_type={type(AgentLoopOutput).__name__}")
        if hasattr(AgentLoopOutput, "__annotations__"):
            print(f"AgentLoopOutput_fields={dict(AgentLoopOutput.__annotations__)}")
        if hasattr(AgentLoopOutput, "__init__"):
            try:
                print(f"AgentLoopOutput_init={inspect.signature(AgentLoopOutput.__init__)}")
            except (ValueError, TypeError):
                pass
    except ImportError as exc:
        print(f"AgentLoopOutput_import_failed: {exc}")

    print()
    print("=== Module-level public names in agent_loop module ===")
    import verl.experimental.agent_loop.agent_loop as al_mod
    public = [n for n in dir(al_mod) if not n.startswith("_")]
    print(f"public={public}")

    print()
    print("=== AsyncLLMServerManager API ===")
    try:
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
        print(f"AsyncLLMServerManager_methods={[n for n,_ in inspect.getmembers(AsyncLLMServerManager, predicate=inspect.isfunction)]}")
        for cand in ("generate", "async_generate", "submit", "submit_request", "complete"):
            if hasattr(AsyncLLMServerManager, cand):
                try:
                    sig = inspect.signature(getattr(AsyncLLMServerManager, cand))
                    print(f"{cand}_signature={sig}")
                except (ValueError, TypeError):
                    pass
    except ImportError as exc:
        print(f"AsyncLLMServerManager_import_failed: {exc}")

    print()
    print("=== Tool-related modules in verl (filesystem walk, no import) ===")
    # Filesystem walk avoids the verl.third_party.vllm vllm-version blowup.
    import os
    tool_files = []
    for root, _, files in os.walk(verl.__path__[0]):
        for f in files:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), verl.__path__[0])
            if "tool" in rel.lower() and "third_party" not in rel:
                tool_files.append(rel)
    print(f"tool_files={tool_files[:30]}")

    print()
    print("=== Try import verl.tools / verl.workers.tools ===")
    for path in ("verl.tools", "verl.workers.tools", "verl.experimental.tools",
                 "verl.experimental.agent_loop.tools",
                 "verl.experimental.agent_loop.tool_parser"):
        try:
            mod = __import__(path, fromlist=["*"])
            public = [n for n in dir(mod) if not n.startswith("_")]
            print(f"{path}_OK: {public[:15]}")
        except Exception as exc:
            print(f"{path}_FAIL: {type(exc).__name__}: {str(exc)[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
