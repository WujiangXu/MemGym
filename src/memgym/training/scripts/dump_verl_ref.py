"""Dump verl reference implementations we plan to mirror in Way A's scaffold.

Way A scaffold rewrite (#132) needs to copy the `request_id` + metrics +
trace decorator patterns from verl's canonical tool-using AgentLoop verbatim
to avoid drift from a fresh-think implementation. EC2's `/run-script` API
only accepts module names, so this is a thin wrapper around `inspect.getsource`
that prints the files we need to read into the conversation.

Behavior:
  1. **Smoke import** the three canonical 0.7.1 symbols the plan v4 the training phase
     gate requires (AgentLoopBase, AsyncLLMServerManager, AgentLoopOutput).
     If any fails, exits 2 with a diagnosis line — the install is bad.
  2. **Dumps** modules in priority order. Each dump is wrapped in a try so
     a moved/renamed submodule doesn't kill the rest of the run; it just
     records `<import_failed>` and continues. The plan's lift targets are:
       - tool_agent_loop (or whichever module subclasses AgentLoopBase
         with the canonical OpenAI-tool-calling loop)
       - base_tool / BaseTool (Way A's SWEMemoryEnv tool subclasses this)
       - tool_registry (registration mechanism — affects how `tool.py`
         exposes itself to verl's actor)
       - tool_parser (action parsing helpers; secondary, since our
         scaffold uses mini-swe-agent's backticks fence by default)
       - the agent_loop.py module itself (contains AgentLoopBase +
         AgentLoopOutput + AsyncLLMServerManager — single biggest lift
         target for #132)

The output is verbose-by-design (~1500 lines on 0.7.1). Pipe to a file
or capture via the EC2 /run-script job log; commit the result to
`docs/swegym/verl_071_ref_dump.txt` per plan v4 the training phase step 4.
"""
from __future__ import annotations

import inspect
import sys
from importlib import import_module


def _dump(label: str, obj) -> None:
    print(f"\n========== {label} ==========")
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        print(f"<getsource_failed: {exc}>")
        return
    print(src)


def _try_dump_module(label: str, dotted: str) -> None:
    """Import-then-dump a module. Records the failure in-band so the run
    survives single-module rename drift."""
    try:
        mod = import_module(dotted)
    except Exception as exc:
        print(f"\n========== {label} ==========")
        print(f"<import_failed: {dotted}: {type(exc).__name__}: {exc}>")
        return
    _dump(label, mod)


def _smoke_imports() -> int:
    """Plan v4 the training phase gate: the three symbols Way A's rewrite needs.
    Fail fast with rc=2 if any is missing — that's the trigger to flip to
    'Way B fix first' sequencing per the plan's fallback path."""
    print("========== smoke_imports ==========")
    try:
        import verl
        print(f"verl.__version__ = {getattr(verl, '__version__', '<missing>')}")
    except Exception as exc:
        print(f"verl import FAILED: {type(exc).__name__}: {exc}")
        return 2

    canonical = "verl.experimental.agent_loop.agent_loop"
    try:
        agent_loop = import_module(canonical)
    except Exception as exc:
        print(f"{canonical} import FAILED: {type(exc).__name__}: {exc}")
        return 2

    missing = []
    for name in ("AgentLoopBase", "AsyncLLMServerManager", "AgentLoopOutput"):
        if not hasattr(agent_loop, name):
            missing.append(name)
    if missing:
        print(f"FAIL: missing symbols in {canonical}: {missing}")
        print(f"available top-level names: "
              f"{[n for n in dir(agent_loop) if not n.startswith('_')][:40]}")
        return 2

    base = agent_loop.AgentLoopBase
    print(f"OK: AgentLoopBase = {base!r}")
    print(f"OK: AgentLoopBase.run signature = {inspect.signature(base.run)}")
    if base.run.__doc__:
        print(f"OK: AgentLoopBase.run.__doc__ = {base.run.__doc__[:200]!r}")
    print(f"OK: AsyncLLMServerManager = {agent_loop.AsyncLLMServerManager!r}")
    print(f"OK: AgentLoopOutput = {agent_loop.AgentLoopOutput!r}")
    print(f"OK: AgentLoopOutput fields = "
          f"{list(getattr(agent_loop.AgentLoopOutput, 'model_fields', {}).keys())}")
    return 0


def main() -> int:
    rc = _smoke_imports()
    if rc != 0:
        return rc

    # The canonical agent_loop module — biggest single lift target. Contains
    # AgentLoopBase, AgentLoopOutput Pydantic, AsyncLLMServerManager,
    # request-id plumbing. Rewrite #132 mirrors this verbatim.
    _try_dump_module(
        "verl.experimental.agent_loop.agent_loop",
        "verl.experimental.agent_loop.agent_loop",
    )

    # Tool-using AgentLoop subclass — pattern source for our SWE-Gym
    # MemoryAwareSWEAgentLoop. May be `tool_agent_loop`, `tool_using_agent_loop`,
    # or rolled into agent_loop.py itself depending on minor version.
    for candidate in (
        "verl.experimental.agent_loop.tool_agent_loop",
        "verl.experimental.agent_loop.tool_using_agent_loop",
    ):
        _try_dump_module(f"candidate {candidate}", candidate)

    # Tool base class + registry — Way A's tool.py subclasses BaseTool
    # and registers via tool_registry.
    _try_dump_module("verl.tools.base_tool", "verl.tools.base_tool")
    _try_dump_module("verl.tools.utils.tool_registry",
                     "verl.tools.utils.tool_registry")
    # Some 0.7.x lines moved registry into verl.tools directly.
    _try_dump_module("verl.tools.tool_registry (alt)", "verl.tools.tool_registry")

    # ToolParser abstract + Hermes concrete — secondary; our scaffold uses
    # mini-swe-agent's backticks-fence parser by default.
    _try_dump_module("verl.experimental.agent_loop.tool_parser",
                     "verl.experimental.agent_loop.tool_parser")

    return 0


if __name__ == "__main__":
    sys.exit(main())
