"""Backticks tool parser for mini-swe-agent's bash action format.

Why a parser, not an ``AgentLoopBase`` subclass: verl 0.7.1's
``ToolAgentLoop`` already implements the state machine (PENDING →
GENERATING → PROCESSING_TOOLS → TERMINATED) and dispatches to whatever
``ToolParser`` is registered under ``rollout_config.multi_turn.format``.
We just need to teach it how to read the action format that
mini-swe-agent emits in ``swebench_backticks.yaml``::

    THOUGHT: <reasoning>
    ```bash
    <command>
    ```

Each fenced ``bash`` block becomes one ``FunctionCall(name="MemGymSWETool",
arguments=json.dumps({"action": <command>}))``. ``ToolAgentLoop``'s
``_call_tool`` then routes those calls into our cached
``MemGymSWETool.execute()``.

Importing this module is enough to register the parser via
``ToolParser.register("backticks")``. ``train.py`` imports this module
at launch so the registration happens before
``ToolParser.get_tool_parser("backticks", tokenizer)`` is called inside
``ToolAgentLoop.__init__``.

We also expose ``get_agent_loop_class()`` for legacy callers that still
import the v0.4.x scaffold; it now returns verl's stock ``ToolAgentLoop``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# mini-swe-agent's action format — bash inside a fenced ```bash block.
# DOTALL so multi-line commands (heredocs, multi-statement scripts) work.
# Audit M1: mini-swe-agent's published examples use ```bash, ```sh, ```shell,
# and language-tag-less ``` blocks interchangeably. The original ``r"```bash"``
# regex silently dropped the latter three, leading to zero tool calls when the
# model produced a stylistic variant. Widen to accept all four forms.
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
# Defense-in-depth: strip any ``<think>...</think>`` that escaped the
# vLLM reasoning parser so the bash regex sees clean text.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Tool name must match ``MemGymSWETool.name`` from
# ``OpenAIFunctionToolSchema.function.name`` set in the tool config YAML.
_TOOL_NAME = "MemGymSWETool"


def _register_parser() -> None:
    """Idempotent — register ``BackticksToolParser`` once per process."""
    from verl.experimental.agent_loop.tool_parser import (  # type: ignore
        FunctionCall,
        ToolParser,
    )
    from verl.utils.ray_utils import get_event_loop  # type: ignore
    from verl.utils.rollout_trace import rollout_trace_op  # type: ignore

    if "backticks" in ToolParser._registry:
        return

    @ToolParser.register("backticks")
    class BackticksToolParser(ToolParser):
        """Parse mini-swe-agent's ```bash``` blocks into ``FunctionCall``s.

        verl calls ``extract_tool_calls(responses_ids, tools)`` once per
        assistant turn. Decoding happens in a thread (cheap for 8B Qwen3
        responses, ≲32K tokens). We return the cleaned content (with
        bash blocks removed) plus one ``FunctionCall`` per block.
        """

        @rollout_trace_op
        async def extract_tool_calls(
            self,
            responses_ids: List[int],
            tools=None,
        ) -> Tuple[str, List["FunctionCall"]]:
            loop = get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=True),
            )
            cleaned = _THINK_RE.sub("", text or "")
            calls: List[FunctionCall] = []
            for match in _BASH_BLOCK_RE.finditer(cleaned):
                command = match.group(1).strip()
                if not command:
                    continue
                calls.append(
                    FunctionCall(
                        name=_TOOL_NAME,
                        arguments=json.dumps({"action": command}, ensure_ascii=False),
                    )
                )
            content = _BASH_BLOCK_RE.sub("", cleaned)
            return content, calls


def register_parser() -> None:
    """Public entrypoint — ``train.py`` calls this before launching verl."""
    _register_parser()


def get_agent_loop_class():
    """Legacy compat — old callers expected an ``AgentLoopBase`` subclass.

    With verl 0.7.1 we route through the stock ``ToolAgentLoop`` plus our
    parser, so this returns ``ToolAgentLoop`` directly. The class is
    already registered as ``tool_agent`` in verl, so the YAML must set
    ``actor_rollout_ref.rollout.agent.agent_name: tool_agent``.
    """
    _register_parser()
    from verl.experimental.agent_loop.tool_agent_loop import (  # type: ignore
        ToolAgentLoop,
    )
    return ToolAgentLoop


# Eagerly register on import so anyone who does
# ``import memgym.training.rl.way_a_verl.agent_loop`` gets the parser
# registered as a side effect — matches the convention used by verl's
# built-in parsers (``HermesToolParser`` etc. register at module import).
try:
    _register_parser()
except Exception as exc:  # noqa: BLE001 — keeps non-verl envs importable
    logger.debug("verl tool parser registration deferred: %s", exc)


__all__ = ["register_parser", "get_agent_loop_class"]
