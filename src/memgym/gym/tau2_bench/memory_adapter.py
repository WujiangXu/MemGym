"""
Memory adapter for tau2-bench: symmetric agent + user wrapping.

Plumbing layer between MemGym's modern ``BaseMemoryManager`` interface
(``manage_context`` → ``FilteredContext``) and tau2-bench's
``LLMAgent`` / ``LLMSoloAgent`` / ``UserSimulator`` classes — both of
which expose the same ``generate_next_message(message, state)`` shape.

Why this exists
===============
Tau2's ``Orchestrator`` calls ``agent.generate_next_message(...)`` and
``user.generate_next_message(...)`` directly. We can't slot the memory
layer in from the runner the way ``WebArenaRunner`` does, so we wrap
both sides — the agent and the user simulator — in subclasses that
intercept ``generate_next_message``, snapshot the conversation history,
hand it to a ``BaseMemoryManager.manage_context(...)`` call, and only
then delegate to the original implementation with the filtered state.

Symmetric wrapping is the central paper claim for the τ²-bench track:
both the agent's view and the user simulator's view get compressed by
their own memory managers. The "whose memory matters more" ablation
flips between dropping the agent-side and the user-side instance.

Round-trip risk (the budget item to watch)
==========================================
``BaseMemoryManager.manage_context(...)`` returns ``FilteredContext.
content`` as a list of message *dicts* (``{"role": ..., "content": ...,
"tool_calls": [...]}``), but tau2's state stores typed
``AssistantMessage`` / ``UserMessage`` / ``ToolMessage`` / ``SystemMessage``
objects. We have to convert in BOTH directions on every step:

  state.messages (tau2 Message objects)
      → list[dict]                       # _messages_to_dicts
      → memory.manage_context(...)
      → FilteredContext.content (dicts)
      → repair_tool_call_pairs (defense-in-depth)
      → list[tau2 Message]               # _coerce_to_tau2_messages
      → write back into state.messages

If any single item fails to coerce, we fall back to the anchor-only
view ``messages[:keep_first] + messages[-keep_last:]`` (no compression
applied this turn) and bump ``stats["coercion_failures"]`` so the issue
is visible in the trajectory output.

All ``tau2.*`` imports are deferred — this module loads cleanly even
when ``third_party/tau2-bench`` hasn't been cloned yet.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

from memgym.memory.base import BaseMemoryManager, FilteredContext, repair_tool_call_pairs

logger = logging.getLogger("memgym.tau2_bench.memory_adapter")


# =============================================================================
# Lazy class building
# =============================================================================
#
# The wrapper classes have to inherit from tau2's LLMSoloAgent /
# UserSimulator for tau2.Orchestrator's internal isinstance() checks.
# But importing those classes at module load time would break the
# "import-without-tau2-installed" property of memgym.gym.tau2_bench.
#
# So we cache one class per role and build it lazily inside the factory
# functions. The first call to wrap_agent_with_memory(...) triggers the
# import; subsequent calls reuse the cached class.

_CACHED_AGENT_CLASS: Optional[Type[Any]] = None
_CACHED_USER_CLASS: Optional[Type[Any]] = None


def _get_agent_wrapper_class() -> Type[Any]:
    """Build (or return cached) MemoryManagerAdapter subclass of tau2.LLMSoloAgent."""
    global _CACHED_AGENT_CLASS
    if _CACHED_AGENT_CLASS is not None:
        return _CACHED_AGENT_CLASS

    from .env import _ensure_tau2_on_path
    _ensure_tau2_on_path()
    from tau2.agent.llm_agent import LLMSoloAgent  # type: ignore

    class MemoryManagerAdapter(LLMSoloAgent):  # type: ignore[misc, valid-type]
        """Agent-side memory wrapper (tau2 LLMSoloAgent)."""

        # Set by _make_wrapper_instance() — never call __init__.
        _IS_TAU2_MEMORY_ADAPTER = True

        def generate_next_message(self, message: Any, state: Any) -> Tuple[Any, Any]:
            return _intercept_generate_next_message(self, message, state)

        # Tau2's BaseAgent has a `is_stop` classmethod that the orchestrator
        # calls on the wrapper class. Forward to the wrapped class so the
        # base agent's stop semantics (custom STOP_TOKEN, STOP_FUNCTION_NAME)
        # are preserved.
        @classmethod
        def is_stop(cls, message: Any) -> bool:
            from tau2.agent.llm_agent import LLMSoloAgent as _LLMSoloAgent  # type: ignore
            return _LLMSoloAgent.is_stop(message)

        def __getattr__(self, name: str) -> Any:
            # __getattr__ only fires when the normal lookup fails — so
            # _base_agent / _memory_manager / _stats / _role / etc. (set
            # via object.__setattr__) are reached without recursing here.
            base = object.__getattribute__(self, "__dict__").get("_base_agent")
            if base is None:
                raise AttributeError(name)
            return getattr(base, name)

    _CACHED_AGENT_CLASS = MemoryManagerAdapter
    return MemoryManagerAdapter


def _get_user_wrapper_class() -> Type[Any]:
    """Build (or return cached) UserMemoryManagerAdapter subclass of tau2.UserSimulator."""
    global _CACHED_USER_CLASS
    if _CACHED_USER_CLASS is not None:
        return _CACHED_USER_CLASS

    from .env import _ensure_tau2_on_path
    _ensure_tau2_on_path()
    from tau2.user.user_simulator import UserSimulator  # type: ignore

    class UserMemoryManagerAdapter(UserSimulator):  # type: ignore[misc, valid-type]
        """User-side memory wrapper (tau2 UserSimulator)."""

        _IS_TAU2_MEMORY_ADAPTER = True

        def generate_next_message(self, message: Any, state: Any) -> Tuple[Any, Any]:
            return _intercept_generate_next_message(self, message, state)

        # UserSimulator has its own is_stop. Forward to the underlying class.
        @classmethod
        def is_stop(cls, message: Any) -> bool:
            from tau2.user.user_simulator import UserSimulator as _UserSimulator  # type: ignore
            return _UserSimulator.is_stop(message)

        def __getattr__(self, name: str) -> Any:
            base = object.__getattribute__(self, "__dict__").get("_base_user")
            if base is None:
                raise AttributeError(name)
            return getattr(base, name)

    _CACHED_USER_CLASS = UserMemoryManagerAdapter
    return UserMemoryManagerAdapter


# =============================================================================
# Wrapper instance construction (init bypass)
# =============================================================================

def _make_wrapper_instance(
    cls: Type[Any],
    base: Any,
    memory_manager: BaseMemoryManager,
    role: str,
    base_attr_name: str,
) -> Any:
    """Allocate a wrapper instance WITHOUT calling tau2's __init__.

    tau2's ``LLMSoloAgent.__init__`` requires (tools, domain_policy, task,
    llm, llm_args) and ``UserSimulator.__init__`` requires (tools,
    instructions, llm, llm_args) — we don't want to pass any of that
    twice, so we bypass init entirely and proxy attribute access to the
    already-constructed base instance via __getattr__.
    """
    instance = cls.__new__(cls)  # no __init__ call
    object.__setattr__(instance, base_attr_name, base)
    object.__setattr__(instance, "_memory_manager", memory_manager)
    object.__setattr__(instance, "_role", role)
    object.__setattr__(instance, "_stats", {
        "total_calls": 0,
        "compaction_count": 0,
        "coercion_failures": 0,
        "last_compression_ratio": 1.0,
        # Per-call memory record log (consumed by the runner to write
        # {task_id}_training.json). Each entry describes one wrapper call:
        # turn_idx, was_compacted, msg/token counts, and on compaction the
        # summary + forgotten count. See `_intercept_generate_next_message`.
        "step_log": [],
    })
    return instance


def wrap_agent_with_memory(
    base_agent: Any,
    memory_manager: BaseMemoryManager,
    role: str = "agent",
) -> Any:
    """Return a tau2 LLMSoloAgent that intercepts generate_next_message.

    Args:
        base_agent: An already-constructed tau2 ``LLMAgent`` or
            ``LLMSoloAgent`` instance (built by ``Tau2BenchMemoryEnv.
            _build_base_agent``).
        memory_manager: A ``BaseMemoryManager`` instance constructed by
            the runner's memory factory.
        role: ``"agent"`` (default). Used by the memory manager to tag
            its filtering decisions and stats.
    """
    cls = _get_agent_wrapper_class()
    return _make_wrapper_instance(cls, base_agent, memory_manager, role, "_base_agent")


def wrap_user_with_memory(
    base_user: Any,
    memory_manager: BaseMemoryManager,
    role: str = "user",
) -> Any:
    """Return a tau2 UserSimulator that intercepts generate_next_message."""
    cls = _get_user_wrapper_class()
    return _make_wrapper_instance(cls, base_user, memory_manager, role, "_base_user")


# =============================================================================
# The intercept implementation (shared by agent + user wrappers)
# =============================================================================

def _fill_step_record_from_metadata(
    rec: Dict[str, Any], md: Dict[str, Any]
) -> None:
    """Copy the memory manager's FilteredContext.metadata into a step record.

    Kept out-of-band from ``_intercept_generate_next_message`` so both the
    non-compacted fast path and the compacted path populate the same shape.
    The training pipeline (``src/memgym/training/data/compaction.py``) reads
    these per-step fields to construct ``Tau2CompactionEvent`` instances.
    """
    # `was_compacted` reflects the adapter's own meaning: "the agent's view
    # currently contains a prior summary." It's sticky-True after the first
    # compaction event, so it's NOT the right signal for "did this step
    # perform a new summarization."
    rec["was_compacted"] = bool(md.get("was_compacted"))
    # `new_compaction` is the per-step training signal — True only on the
    # exact step where a new summarization ran (detected via presence of the
    # ``condensation_event`` metadata block, which adapters only emit on
    # actual compaction).
    cond = md.get("condensation_event")
    rec["new_compaction"] = cond is not None
    rec["strategy"] = md.get("strategy")
    rec["original_msgs"] = md.get("raw_size") or md.get("messages_before")
    rec["filtered_msgs"] = md.get("view_size") or md.get("messages_after")
    rec["original_tokens"] = md.get("original_tokens")
    rec["filtered_tokens"] = md.get("tokens")
    rec["compression_ratio"] = md.get("compression_ratio", 1.0)
    if cond:
        rec["summary"] = cond.get("summary_text")
        rec["forgotten"] = cond.get("forgotten_count")
        rec["summarizer_prompt_tokens"] = cond.get("summarizer_prompt_tokens", 0)
        rec["summarizer_completion_tokens"] = cond.get("summarizer_completion_tokens", 0)
        rec["summarization_time_seconds"] = cond.get("summarization_time_seconds", 0.0)


def _intercept_generate_next_message(
    wrapper: Any,
    message: Any,
    state: Any,
) -> Tuple[Any, Any]:
    """Filter ``state.messages`` through the memory manager, then delegate.

    The wrapper's ``__dict__`` carries (set by ``_make_wrapper_instance``):
        _base_agent / _base_user — the unwrapped tau2 instance to delegate to
        _memory_manager          — BaseMemoryManager
        _role                    — "agent" | "user"
        _stats                   — counters exposed via get_stats()

    The base instance is whichever of ``_base_agent`` / ``_base_user`` is set.
    """
    wd = object.__getattribute__(wrapper, "__dict__")
    base = wd.get("_base_agent") or wd.get("_base_user")
    memory: BaseMemoryManager = wd["_memory_manager"]
    role: str = wd["_role"]
    stats: Dict[str, Any] = wd["_stats"]

    stats["total_calls"] += 1
    original_messages = list(getattr(state, "messages", []) or [])

    # 1) Snapshot history as dicts and pass to the memory manager.
    history_dicts = _messages_to_dicts(original_messages)
    current_dict = _message_to_dict(message)

    # The turn index used in _replay.json — this is the number of messages
    # in state BEFORE tau2 appends the current observation + new response.
    turn_idx = len(history_dicts)

    # Step record is assembled as control flow progresses and appended in
    # the `finally` clause below so every return path is covered uniformly.
    step_record: Dict[str, Any] = {
        "step": stats["total_calls"] - 1,
        "turn_idx": turn_idx,
        "was_compacted": False,
        "new_compaction": False,
        "memory_failed": False,
        "coercion_failed": False,
    }

    try:
        try:
            filtered: FilteredContext = memory.manage_context(
                original_context=history_dicts,
                current_observation=current_dict,
                metadata={
                    "role": role,
                    "turn": turn_idx,
                    "task_id": getattr(memory, "_episode_state", {}).get("task_id"),
                    "domain": getattr(memory, "_episode_state", {}).get("domain"),
                },
            )
        except Exception as exc:
            # Memory failure must not break the episode — fall back to
            # passing state through untouched. Logged so the issue surfaces.
            logger.exception("memory.manage_context failed for role=%s: %s", role, exc)
            step_record["memory_failed"] = True
            return base.generate_next_message(message, state)

        _fill_step_record_from_metadata(step_record, filtered.metadata)

        # 2) If the manager didn't compact, skip the conversion roundtrip
        #    entirely — pass the original state through.
        if not filtered.metadata.get("was_compacted"):
            return base.generate_next_message(message, state)

        # 3) Pull out the filtered content. Memory adapters may return either
        #    a list (the new view) or a single string (legacy). The current
        #    observation is typically the LAST item in the returned list when
        #    the adapter wants to keep it visible to the agent.
        filtered_content = filtered.content
        if not isinstance(filtered_content, list):
            # Legacy / non-list return → cannot round-trip; fall back.
            logger.warning(
                "memory adapter returned non-list content for role=%s (type=%s); "
                "falling back to original state",
                role, type(filtered_content).__name__,
            )
            stats["coercion_failures"] += 1
            step_record["coercion_failed"] = True
            return base.generate_next_message(message, state)

        # 3b) Strip the current observation from the filtered content.
        #     manage_context() appends current_observation to its message
        #     list, so the returned view already includes it (always as the
        #     last element). But tau2's generate_next_message() will ALSO
        #     append the current `message` to state.messages before the API
        #     call. Without stripping, the current message appears twice —
        #     and when it's a ToolMessage, Bedrock sees duplicate toolResult
        #     blocks exceeding the toolUse count → BadRequestError.
        #
        #     We also extract tool_call_ids from the stripped observation so
        #     that the repair function preserves the matching tool_calls in
        #     the preceding assistant message (tau2 will append the result).
        pending_tool_ids: set = set()
        if filtered_content:
            stripped = filtered_content[-1]
            filtered_content = filtered_content[:-1]
            # Extract tool_call_ids that will arrive when tau2 appends message.
            # Check all three representations — stripped view item, converted
            # dict, and raw tau2 object — because MultiToolMessage's
            # tool_messages may be lost during model_dump() conversion.
            pending_tool_ids = _extract_tool_ids_from_message(stripped)
            pending_tool_ids |= _extract_tool_ids_from_message(current_dict)
            pending_tool_ids |= _extract_tool_ids_from_message(message)

        # 4) Defense-in-depth: drop orphaned tool_call/tool_result pairs that
        #    would otherwise crash Bedrock / OpenAI tool-call APIs.
        #    Tau2 ToolMessage uses "id" (not "tool_call_id"), so we normalize
        #    before repair and denormalize after.
        repaired_dicts = _repair_tau2_tool_pairs(filtered_content, pending_tool_ids)

        # 5) Coerce dicts back to tau2 Message objects. The original messages
        #    are passed in for type-shape inference (anchor messages should
        #    keep their original class even if they round-tripped through dict).
        coerced, ok = _coerce_to_tau2_messages(repaired_dicts, original_messages)
        if not ok:
            stats["coercion_failures"] += 1
            step_record["coercion_failed"] = True
            logger.warning(
                "tau2 message coercion failed for role=%s; passing original state through",
                role,
            )
            return base.generate_next_message(message, state)

        # 6) Build a filtered state for the API call, but return the ORIGINAL
        #    state to the orchestrator.
        #
        #    Critical: the orchestrator saves the returned state as
        #    ``self.agent_state`` for the next turn. If we return the
        #    compressed state, the next call's ``state.messages`` would be
        #    the compressed list — but condensation records reference indices
        #    in the FULL history → index mismatch → corrupted view.
        #
        #    So: use the filtered state for this API call only, then update
        #    the ORIGINAL state with current message + agent response.
        filtered_state = deepcopy(state)
        filtered_state.messages = coerced

        stats["compaction_count"] += 1
        stats["last_compression_ratio"] = float(
            filtered.metadata.get("compression_ratio", 1.0)
        )

        # Delegate to the base agent with the filtered (compressed) state.
        try:
            assistant_message, _discarded_state = base.generate_next_message(message, filtered_state)
        except Exception as exc:
            if "toolResult" in str(exc) or "toolUse" in str(exc) or "BedrockException" in str(exc):
                # Dump message structure for diagnosis
                _log_bedrock_message_structure(
                    coerced, message, getattr(filtered_state, "system_messages", []),
                    role, stats, exc,
                )
            raise

        # Update the ORIGINAL state the same way tau2's generate_next_message
        # does internally — append current message + agent response — so the
        # orchestrator carries the full, uncompressed history forward.
        try:
            from .env import _ensure_tau2_on_path
            _ensure_tau2_on_path()
            from tau2.data_model.message import MultiToolMessage as _MTM  # type: ignore
            if isinstance(message, _MTM):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
        except ImportError:
            state.messages.append(message)
        state.messages.append(assistant_message)

        return assistant_message, state
    finally:
        stats["step_log"].append(step_record)


# =============================================================================
# Round-trip helpers (the main risk surface)
# =============================================================================

def _extract_tool_ids_from_message(msg: Any) -> set:
    """Extract tool_call_ids from a message (ToolMessage or MultiToolMessage).

    Must check for MultiToolMessage (``tool_messages`` list) BEFORE the
    single-ToolMessage path, because MultiToolMessage also has
    ``role="tool"`` — an early return on role would yield only the
    top-level id and miss the N individual sub-message IDs.
    """
    if not isinstance(msg, dict):
        # MultiToolMessage object — check FIRST (before single ToolMessage)
        if hasattr(msg, "tool_messages"):
            tms = getattr(msg, "tool_messages", [])
            if tms:
                ids = set()
                for tm in tms:
                    tid = getattr(tm, "id", None)
                    if tid:
                        ids.add(tid)
                return ids
        # Single ToolMessage object
        if hasattr(msg, "id") and hasattr(msg, "role"):
            role = getattr(msg, "role", "")
            if role == "tool":
                tid = getattr(msg, "id", None)
                return {tid} if tid else set()
        return set()

    # Dict path — check tool_messages FIRST (MultiToolMessage dict)
    tool_messages = msg.get("tool_messages", [])
    if tool_messages:
        ids = set()
        for tm in tool_messages:
            if isinstance(tm, dict):
                tid = tm.get("tool_call_id") or tm.get("id")
            else:
                tid = getattr(tm, "id", None)
            if tid:
                ids.add(tid)
        return ids

    # Single ToolMessage dict
    role = msg.get("role", "")
    if role == "tool":
        tid = msg.get("tool_call_id") or msg.get("id")
        return {tid} if tid else set()
    return set()


def _repair_tau2_tool_pairs(
    messages: List[Dict[str, Any]],
    pending_tool_ids: set | None = None,
) -> List[Dict[str, Any]]:
    """Tau2-aware wrapper around ``repair_tool_call_pairs``.

    Tau2's ``ToolMessage.model_dump()`` uses ``"id"`` for the tool-call
    reference, while the base ``repair_tool_call_pairs`` expects the
    litellm/OpenAI convention of ``"tool_call_id"``. We normalize before
    calling the shared repair function, then strip the synthetic key so
    downstream ``_coerce_to_tau2_messages`` still sees the tau2 shape.

    ``pending_tool_ids`` — tool call IDs that will be appended by the
    caller after repair (the current observation). Passed through to
    ``repair_tool_call_pairs`` so it preserves matching tool_calls.
    """
    # Normalize: ensure every tool dict has tool_call_id
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and not msg.get("tool_call_id"):
            # Tau2 ToolMessage stores the ref as "id"
            tid = msg.get("id")
            if tid:
                msg["tool_call_id"] = tid

    repaired = repair_tool_call_pairs(messages, pending_tool_ids=pending_tool_ids)

    # Denormalize: remove the synthetic tool_call_id we added so the
    # tau2 Message constructor doesn't choke on an unexpected field.
    for msg in repaired:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and "tool_call_id" in msg and "id" in msg:
            if msg["tool_call_id"] == msg["id"]:
                del msg["tool_call_id"]

    return repaired


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Best-effort conversion of a tau2 Message-like object to a dict."""
    if isinstance(msg, dict):
        return msg
    # Pydantic v2 (tau2 uses pydantic models for messages)
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    # Pydantic v1 fallback
    dict_method = getattr(msg, "dict", None)
    if callable(dict_method):
        try:
            return dict_method()
        except Exception:
            pass
    # Last resort: stringify
    return {"role": getattr(msg, "role", "user"), "content": str(msg)}


def _messages_to_dicts(messages: List[Any]) -> List[Dict[str, Any]]:
    return [_message_to_dict(m) for m in messages]


def _coerce_to_tau2_messages(
    dicts: List[Dict[str, Any]],
    original_messages: List[Any],
) -> Tuple[List[Any], bool]:
    """Convert filtered dicts back to tau2 Message objects.

    Returns ``(messages, ok)``. If ``ok`` is False the caller should fall
    back to the anchor-only view rather than using the partial result.

    Strategy:
    - tau2 ``Message`` subclasses are pydantic models with a ``role``
      discriminator. We pick the matching subclass per item and call its
      constructor with the dict's fields.
    - Any item that fails validation aborts the whole conversion (hard
      fail), because partial conversion would silently re-introduce the
      tool-call mismatch we're trying to avoid.
    """
    if not dicts:
        return [], True

    from .env import _ensure_tau2_on_path
    _ensure_tau2_on_path()
    try:
        from tau2.data_model.message import (  # type: ignore
            AssistantMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )
    except ImportError as exc:
        logger.warning("tau2.data_model.message import failed: %s", exc)
        return [], False

    role_to_cls: Dict[str, Type[Any]] = {
        "assistant": AssistantMessage,
        "user": UserMessage,
        "tool": ToolMessage,
        "system": SystemMessage,
    }

    out: List[Any] = []
    for item in dicts:
        # If memory passed an original tau2 Message through unchanged,
        # detect it via attribute presence (not isinstance — that would
        # require importing all four classes at the top).
        if not isinstance(item, dict):
            if hasattr(item, "role") and hasattr(item, "content"):
                out.append(item)
                continue
            return [], False

        role = item.get("role")
        cls = role_to_cls.get(role)
        if cls is None:
            return [], False

        try:
            out.append(cls(**item))
        except Exception as exc:
            logger.debug("coerce failed for role=%s: %s — item=%r", role, exc, item)
            return [], False

    return out, True


# =============================================================================
# Diagnostic logging for Bedrock tool pair errors
# =============================================================================

def _log_bedrock_message_structure(
    coerced: List[Any],
    current_message: Any,
    system_messages: List[Any],
    role: str,
    stats: Dict[str, Any],
    exc: Exception,
) -> None:
    """Log the message structure when a Bedrock toolResult/toolUse error occurs."""
    lines = [
        f"BEDROCK_DIAG role={role} call={stats.get('total_calls')} "
        f"compactions={stats.get('compaction_count')} error={exc}",
        f"  system_messages: {len(system_messages)}",
        f"  coerced: {len(coerced)} messages",
    ]
    for i, msg in enumerate(coerced):
        d = _message_to_dict(msg) if not isinstance(msg, dict) else msg
        r = d.get("role", "?")
        tc = d.get("tool_calls")
        tc_ids = [t.get("id", "?")[:12] for t in tc] if tc else []
        tid = d.get("tool_call_id") or d.get("id", "")
        content_len = len(str(d.get("content", "")))
        if r == "tool":
            lines.append(f"  [{i}] role={r} id={tid[:12] if tid else '?'} content_len={content_len}")
        elif tc_ids:
            lines.append(f"  [{i}] role={r} tool_calls={tc_ids} content_len={content_len}")
        else:
            lines.append(f"  [{i}] role={r} content_len={content_len}")

    # Current message
    cd = _message_to_dict(current_message) if not isinstance(current_message, dict) else current_message
    cr = cd.get("role", "?")
    ctid = cd.get("tool_call_id") or cd.get("id", "")
    # Check for MultiToolMessage
    tool_msgs = cd.get("tool_messages", [])
    if tool_msgs:
        sub_ids = []
        for tm in tool_msgs:
            if isinstance(tm, dict):
                sub_ids.append((tm.get("tool_call_id") or tm.get("id", "?"))[:12])
            else:
                sub_ids.append(str(getattr(tm, "id", "?"))[:12])
        lines.append(f"  [current] role={cr} MultiToolMessage ids={sub_ids}")
    elif hasattr(current_message, "tool_messages"):
        sub_ids = [str(getattr(tm, "id", "?"))[:12] for tm in current_message.tool_messages]
        lines.append(f"  [current] role={cr} MultiToolMessage(obj) ids={sub_ids}")
    else:
        lines.append(f"  [current] role={cr} id={ctid[:12] if ctid else ''} content_len={len(str(cd.get('content','')))}")

    logger.error("\n".join(lines))


# =============================================================================
# Convenience: pull stats out of a wrapper
# =============================================================================

def get_adapter_stats(wrapper: Any) -> Dict[str, Any]:
    """Read the per-wrapper memory stats dict.

    Returns ``{}`` if ``wrapper`` is not a memory adapter (e.g. base
    agents that were never wrapped).
    """
    if not getattr(type(wrapper), "_IS_TAU2_MEMORY_ADAPTER", False):
        return {}
    return dict(object.__getattribute__(wrapper, "__dict__").get("_stats") or {})
