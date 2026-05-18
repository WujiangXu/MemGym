"""LightMem (ICLR 2026) adapter for IR evaluation, via a uv-venv bridge.

LightMem pins `python >=3.10,<3.12`; our main benchmark venv is 3.13.
Rather than force-install and cascade downgrades, we run LightMem in an
isolated uv-managed Python 3.11 venv and talk to it over a JSON-over-stdio
subprocess (see `lightmem_server.py` and `scripts/setup_lightmem_venv.py`).

The adapter:
  - spawns one long-lived subprocess per manager instance,
  - handshakes via {"op": "init", "config": {...}},
  - forwards `manage_context` -> {"op": "add"} and
    `retrieve_for_question` -> {"op": "retrieve"},
  - tears down the subprocess in `reset()` and relaunches.

The venv check is lazy: importing this module does NOT require LightMem to
be installed. The first time a `LightMemMemoryManager` is constructed, the
adapter checks for the venv and raises if it is missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from .._lightmem_prompts import NEUTRAL_METADATA_PROMPT


_DEFAULT_VENV_PATH = Path.home() / ".cache" / "memgym" / "lightmem-venv"
_SERVER_PATH = Path(__file__).resolve().parent / "lightmem_server.py"


def _venv_dir() -> Path:
    return Path(os.environ.get("LIGHTMEM_VENV", str(_DEFAULT_VENV_PATH)))


def _venv_python() -> Path:
    return _venv_dir() / "bin" / "python"


def _require_venv() -> Path:
    py = _venv_python()
    if not py.exists():
        raise RuntimeError(
            f"LightMem venv not found at {_venv_dir()}. "
            f"Run `python scripts/setup_lightmem_venv.py` first, or "
            f"set $LIGHTMEM_VENV to an existing venv."
        )
    return py


class _LightMemClient:
    """Thin wrapper around a long-lived LightMem server subprocess."""

    def __init__(
        self,
        config: Dict[str, Any],
        repo_root: Path,
        metadata_prompt: Optional[str] = None,
    ):
        py = _require_venv()
        # Run the server by file path so the lightmem venv never tries to
        # import the broader `memgym` package (which pulls in gymnasium etc.
        # that aren't installed in the isolated 3.11 venv).
        self._proc = subprocess.Popen(
            [str(py), "-u", str(_SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(repo_root),
        )
        init_msg: Dict[str, Any] = {"op": "init", "config": config}
        if metadata_prompt:
            init_msg["metadata_prompt"] = metadata_prompt
        self._call(init_msg)

    def _call(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._proc.poll() is not None:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"lightmem server died (rc={self._proc.returncode}): {err}")
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"lightmem server closed stdout; stderr: {err}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"lightmem server error: {reply.get('error')}")
        return reply

    def add(self, text: str, ts: str) -> None:
        self._call({"op": "add", "text": text, "ts": ts})

    def retrieve(self, query: str, k: int) -> str:
        reply = self._call({"op": "retrieve", "query": query, "k": k})
        return reply.get("results", "") or ""

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _repo_root() -> Path:
    # src/memgym/memory/ir/ir_lightmem.py -> repo root is 4 levels up
    return Path(__file__).resolve().parents[4]


class IRLightMemMemory(BaseMemoryManager):
    """LightMem 3-stage memory for IR evaluation, proxied over subprocess."""

    def __init__(
        self,
        max_tokens: int = 100000,
        llm_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dims: int = 384,
        retrieve_k: int = 10,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_metadata: bool = True,
        enable_summary: bool = True,
        qdrant_dir: Optional[str] = None,
        question: str = "",
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._embedding_dims = embedding_dims
        self._retrieve_k = retrieve_k
        self._api_base = api_base
        self._api_key = api_key
        self._enable_metadata = enable_metadata
        self._enable_summary = enable_summary
        self._qdrant_dir = qdrant_dir or tempfile.mkdtemp(prefix="lightmem_ir_qdrant_")
        self._doc_counter = 0
        self._base_time = datetime(2024, 1, 1, 9, 0)
        self._all_observations: List[str] = []
        self._client: Optional[_LightMemClient] = None
        self._client = self._spawn_client()

    def _build_config(self) -> Dict[str, Any]:
        collection = f"memgym_ir_{uuid.uuid4().hex[:8]}"
        qdrant_path = os.path.join(self._qdrant_dir, collection)
        self._collection = collection
        return {
            "pre_compress": False,
            "topic_segment": False,
            "messages_use": "user_only",
            "metadata_generate": self._enable_metadata,
            "text_summary": self._enable_summary,
            "memory_manager": {
                "model_name": "bedrock",
                "configs": {
                    "model": self._llm_model,
                },
            },
            "extract_threshold": 0.0,
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": self._embedding_model,
                    "embedding_dims": self._embedding_dims,
                    "model_kwargs": {},
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": self._collection,
                    "embedding_model_dims": self._embedding_dims,
                    "path": qdrant_path,
                },
            },
            "update": "offline",
        }

    def _spawn_client(self) -> _LightMemClient:
        if self._api_key:
            os.environ.setdefault("OPENAI_API_KEY", self._api_key)
        if self._api_base:
            os.environ.setdefault("OPENAI_API_BASE", self._api_base)
        # Override LightMem upstream's chat-only extractor prompt
        # (LOCOMO Personal-Information-Extractor template) with the
        # shared domain-neutral prompt — see docs/coding/RESUME_RUNBOOK.md
        # §11.6 for the root-cause analysis. The bridge server passes
        # this on every `add_memory()` call as METADATA_GENERATE_PROMPT.
        return _LightMemClient(
            self._build_config(),
            _repo_root(),
            metadata_prompt=NEUTRAL_METADATA_PROMPT,
        )

    def _make_timestamp(self) -> str:
        ts = self._base_time + timedelta(hours=self._doc_counter)
        weekday = ts.strftime("%a")
        return ts.strftime(f"%Y/%m/%d ({weekday}) %H:%M")

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            assert self._client is not None
            self._client.add(text[:15000], self._make_timestamp())
            self._doc_counter += 1

        tokens = self.count_tokens(self._all_observations)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_lightmem",
                "num_docs": self._doc_counter,
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        k = top_k or self._retrieve_k
        assert self._client is not None
        result = self._client.retrieve(question, k)
        return result if result else "(no relevant memories found)"

    def get_notes_text(self) -> str:
        return ""

    def reset(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._doc_counter = 0
        self._all_observations.clear()
        self._qdrant_dir = tempfile.mkdtemp(prefix="lightmem_ir_qdrant_")
        self._client = self._spawn_client()
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def __del__(self):
        try:
            if getattr(self, "_client", None) is not None:
                self._client.close()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "num_docs": self._doc_counter,
            "retrieve_k": self._retrieve_k,
            "embedding_model": self._embedding_model,
            "via": "uv-venv-subprocess",
        })
        return stats


register_memory_model("ir_lightmem", IRLightMemMemory)
