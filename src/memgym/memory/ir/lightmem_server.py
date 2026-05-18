"""LightMem JSON-over-stdio bridge server.

Runs inside the Python 3.11 uv-venv. Reads newline-delimited JSON from
stdin, holds one LightMemory instance warm for the process lifetime,
replies on stdout.

Protocol:
    1. Handshake. Client writes
       {"op": "init", "config": {...}, "metadata_prompt": "..."}; server
       replies {"ok": true, "ready": true}. ``metadata_prompt`` is
       optional — if present, the server caches it and forwards it as
       ``METADATA_GENERATE_PROMPT=`` on every subsequent ``add_memory``
       call so the client can override LightMem upstream's chat-only
       extraction prompt without forking the submodule.
    2. Ingest. Client writes {"op": "add", "text": "...", "ts": "..."};
       server replies {"ok": true}.
    3. Query. Client writes {"op": "retrieve", "query": "...", "k": 10};
       server replies {"ok": true, "results": "..."}.
    4. Any parse error or handler exception yields {"ok": false, "error": "..."}.
    5. Server exits on stdin EOF.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any


# Reserve fd 1 (the pipe the parent reads JSON from) as a dedicated channel.
# Anything lightmem / litellm / huggingface writes to "stdout" gets redirected
# to stderr so it can't corrupt the protocol.
_JSON_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def _send(obj: dict) -> None:
    _JSON_OUT.write(json.dumps(obj) + "\n")
    _JSON_OUT.flush()


def _handle(state: dict, msg: dict) -> dict:
    op = msg.get("op")

    if op == "init":
        # Lazy-import so a missing dep surfaces as an error on init rather
        # than crashing before the handshake reply can be written.
        from lightmem.memory.lightmem import LightMemory

        config = msg["config"]
        state["system"] = LightMemory.from_config(config)
        # Optional override of upstream chat-only METADATA_GENERATE_PROMPT.
        # Empty string is treated as "not set" so we don't accidentally
        # force LightMem onto an empty prompt.
        prompt = msg.get("metadata_prompt") or ""
        state["metadata_prompt"] = prompt if prompt else None
        return {"ok": True, "ready": True}

    system = state.get("system")
    if system is None:
        return {"ok": False, "error": "not initialized; send op=init first"}

    if op == "add":
        message = {
            "role": "user",
            "content": msg.get("text", ""),
            "time_stamp": msg.get("ts", ""),
        }
        kwargs = {"force_segment": True, "force_extract": True}
        prompt = state.get("metadata_prompt")
        if prompt:
            kwargs["METADATA_GENERATE_PROMPT"] = prompt
        system.add_memory(messages=[message], **kwargs)
        return {"ok": True}

    if op == "retrieve":
        result = system.retrieve(query=msg.get("query", ""), limit=int(msg.get("k", 10)))
        if isinstance(result, list):
            result = "\n".join(str(r) for r in result)
        return {"ok": True, "results": result or ""}

    if op == "ping":
        return {"ok": True, "pong": True}

    return {"ok": False, "error": f"unknown op: {op!r}"}


def main() -> int:
    state: dict[str, Any] = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"ok": False, "error": f"bad json: {e}"})
            continue
        try:
            reply = _handle(state, msg)
        except Exception as e:
            reply = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4),
            }
        _send(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
