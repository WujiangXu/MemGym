"""
WebArenaAgent — LiteLLM-backed agent for webarena-infinity trajectories.

Supports two observation modes (matching WebArenaMemoryEnv):

- **text**: The agent sees a numbered accessibility-tree snapshot + URL and
  responds with a text-shaped action like `click [12]`, `type [3] [hello]`,
  `goto [/search]`, `stop`. This is the default and the only mode used for
  the headline memory-strategy comparison.
- **vision**: The agent sees a base64 PNG screenshot + URL and responds
  with the same text action vocabulary. Kept strictly for the multimodal
  ablation; memory strategies in this mode require a multimodal adapter.

The agent returns `AgentAction(action=<text>, metadata=...)` where `<text>`
is the raw action string. The env's `_parse_text_action()` handles the
actual parsing at `step()` time so the agent and env agree on one vocabulary.

LLM calls go through `litellm.completion` exactly like the SWE-bench agent.
The agent name matches the webarena-infinity agent contract at the highest
level (`system prompt` + `numbered elements` + `action-string response`)
but does NOT import anything from the webarena-infinity repo — we want the
MemGym agent code to be self-contained so the memory manager has a clean
contract to operate over.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from memgym.agents.base import BaseAgent, AgentAction, register_agent
from memgym.memory.base import FilteredContext

logger = logging.getLogger("memgym.webarena.agent")


# =============================================================================
# System prompt
# =============================================================================

WEBARENA_SYSTEM_PROMPT_TEXT = """You are a web GUI agent controlling a browser to complete a task.

You are given:
- The TASK (a natural-language instruction)
- The CURRENT URL the browser is on
- An ACCESSIBILITY TREE listing every interactable element on the page, numbered in reading order

You must respond with EXACTLY ONE action on a single line, chosen from this vocabulary:

  click [N]                  — click the Nth element in the accessibility tree
  type [N] [text]            — type `text` into the Nth element (form field)
  select [N] [value]         — select `value` from the Nth element (a <select>)
  goto [url]                 — navigate to an absolute or relative URL
  back                       — navigate browser history back
  forward                    — navigate browser history forward
  key [combo]                — press a keyboard combo (e.g. Enter, Control+A)
  scroll [down] / [up]       — scroll the viewport
  stop                       — end the task as complete
  fail                       — end the task as infeasible

Rules:
1. Output ONLY the action line. No explanations, no code fences, no extra text.
2. Element ids (N) refer to the numbered accessibility tree of the CURRENT page,
   not an earlier page. If you navigated, the numbers reset.
3. Prefer `click` and `type` over `goto` for user-initiated UI flows — tests
   check that your actions match a real user's click path, not URL bashing.
4. When the task is complete, respond with `stop`. If you are certain the task
   cannot be done from current state, respond with `fail`.
"""

WEBARENA_SYSTEM_PROMPT_VISION = WEBARENA_SYSTEM_PROMPT_TEXT.replace(
    "An ACCESSIBILITY TREE listing every interactable element on the page, numbered in reading order",
    "A SCREENSHOT of the current page (element ids come from an implicit DOM ordering)",
)


# =============================================================================
# WebArenaAgent
# =============================================================================

class WebArenaAgent(BaseAgent):
    """LiteLLM-backed web GUI agent.

    Args:
        policy_model: litellm model id (e.g. `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`).
        observation_mode: "text" (default) or "vision" — must match the env.
        max_tokens: Max tokens for each agent response.
        temperature: Sampling temperature.
        system_prompt: Override the default system prompt.
        config: Arbitrary config dict.
    """

    def __init__(
        self,
        policy_model: str,
        observation_mode: str = "text",
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config)

        if observation_mode not in ("text", "vision"):
            raise ValueError(f"observation_mode must be text or vision, got {observation_mode!r}")

        self.policy_model = policy_model
        self.observation_mode = observation_mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            WEBARENA_SYSTEM_PROMPT_TEXT if observation_mode == "text"
            else WEBARENA_SYSTEM_PROMPT_VISION
        )

    # ---- BaseAgent API ----

    def act(self, context: Any) -> AgentAction:
        """Decide the next action from a FilteredContext (or raw list/obs).

        The runner passes a `FilteredContext` whose `.content` is the message
        history that memory has already filtered. The agent prepends the
        system prompt, appends the current observation as a user message, and
        calls the LLM.
        """
        messages = self._build_messages(context)
        try:
            response = self._call_llm(messages)
            raw_text = self._extract_text(response)
            action_text = self._clean_action_line(raw_text)
        except Exception as e:
            logger.exception("agent LLM call failed: %s", e)
            action_text = "fail"

        agent_action = AgentAction(
            action=action_text,
            metadata={
                "policy_model": self.policy_model,
                "observation_mode": self.observation_mode,
                "messages_sent": len(messages),
            },
        )
        self._record_action(agent_action)
        return agent_action

    def reset(self) -> None:
        self._action_history = []
        self._last_action = None

    def set_episode_state(self, **state: Any) -> None:
        """Capture the task instruction (and any other per-episode info).

        Webarena-infinity's upstream agent (`evaluation/agents.py:189-205`)
        treats the task as agent-level state, not per-step observation. We
        follow the same contract: the runner calls this once after env.reset()
        and the instruction stays in the prompt for every subsequent step.
        """
        super().set_episode_state(**state)
        self._task_instruction = str(state.get("instruction") or "").strip()
        self._app_name = str(state.get("app_name") or "").strip()

    # ---- message assembly ----

    def _build_messages(self, context: Any) -> List[Dict[str, Any]]:
        """Turn a FilteredContext (or raw list) into a litellm message list."""
        history: List[Any] = []
        current_obs: Any = None

        if isinstance(context, FilteredContext):
            content = context.content
            if isinstance(content, list):
                # memory returned a message list — the last element is typically
                # the current observation the runner passed in
                if content and self._looks_like_observation(content[-1]):
                    history = content[:-1]
                    current_obs = content[-1]
                else:
                    history = content
            else:
                current_obs = content
        elif isinstance(context, list):
            history = context
        else:
            current_obs = context

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        # Inject the episode-scoped task instruction right after the system
        # prompt. Upstream webarena-infinity (evaluation/agents.py:193) prepends
        # the server context to the task before handing it to the agent — we
        # do the same so the model never has to infer "what app am I on?" from
        # the URL bar alone.
        task_instruction = getattr(self, "_task_instruction", "")
        if task_instruction:
            app_name = getattr(self, "_app_name", "")
            app_hint = f" (app: {app_name})" if app_name else ""
            messages.append({
                "role": "user",
                "content": f"TASK{app_hint}: {task_instruction}",
            })
        messages.extend(self._normalize_history(history))
        if current_obs is not None:
            messages.append(self._format_observation(current_obs))
        return messages

    def _normalize_history(self, history: List[Any]) -> List[Dict[str, Any]]:
        """Ensure every history item is a {role, content} dict suitable for litellm."""
        out: List[Dict[str, Any]] = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                out.append(item)
            elif isinstance(item, dict):
                out.append({"role": "user", "content": json.dumps(item)})
            else:
                out.append({"role": "user", "content": str(item)})
        return out

    def _format_observation(self, obs: Any) -> Dict[str, Any]:
        """Turn an observation dict from WebArenaMemoryEnv into a user message."""
        if not isinstance(obs, dict):
            return {"role": "user", "content": str(obs)}

        if obs.get("mode") == "vision" or "screenshot_b64" in obs:
            url = obs.get("url", "")
            title = obs.get("title", "")
            screenshot = obs.get("screenshot_b64", "")
            parts: List[Dict[str, Any]] = [
                {"type": "text", "text": f"URL: {url}\nTitle: {title}\n\nChoose the next action."}
            ]
            if screenshot:
                parts.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                })
            return {"role": "user", "content": parts}

        # text mode
        url = obs.get("url", "")
        title = obs.get("title", "")
        tree = obs.get("accessibility_tree", "")
        body = (
            f"URL: {url}\n"
            f"Title: {title}\n\n"
            f"ACCESSIBILITY TREE:\n{tree}\n\n"
            f"Choose the next action."
        )
        return {"role": "user", "content": body}

    @staticmethod
    def _looks_like_observation(msg: Any) -> bool:
        """Heuristic: does this item look like an env observation (not a chat msg)?"""
        return isinstance(msg, dict) and (
            "accessibility_tree" in msg or "screenshot_b64" in msg or "mode" in msg
        )

    # ---- LLM call ----

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Any:
        from litellm import completion  # lazy import
        return completion(
            model=self.policy_model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    # Action vocabulary patterns — match the system prompt at the top of this
    # file. Used by `_clean_action_line` to extract a valid action from a
    # possibly-verbose LLM response. Bracketed forms allow nested brackets in
    # the inner text/url/value field by using a non-greedy match anchored to
    # the line end (or to the next backtick fence).
    # Each verb has both a strict bracketed form (the original spec) and a
    # permissive bare form (what LLMs naturally emit, especially for prose
    # values like URLs and free text). Both are unambiguous because the only
    # bracketed token in any action is the optional element id `[N]`.
    _ACTION_PATTERNS = [
        r"click\s*\[\d+\]",
        r"type\s*\[\d+\]\s*\[[^\n]*\]",
        r"type\s*\[\d+\]\s+[^\n]+",
        r"select\s*\[\d+\]\s*\[[^\n]*\]",
        r"select\s*\[\d+\]\s+[^\n]+",
        r"goto\s*\[[^\n]*\]",
        r"goto\s+\S[^\n]*",
        r"key\s*\[[^\n]*\]",
        r"key\s+\S[^\n]*",
        r"scroll\s*\[(?:up|down)\](?:\s*\[\d+\])?",
        r"scroll\s+(?:up|down)(?:\s+\d+)?",
        r"\bback\b",
        r"\bforward\b",
        r"\bstop\b",
        r"\bdone\b",
        r"\bfail\b",
    ]
    _ACTION_RE = re.compile(
        "(" + "|".join(_ACTION_PATTERNS) + ")",
        re.IGNORECASE,
    )

    @classmethod
    def _clean_action_line(cls, text: str) -> str:
        """Extract a valid action from a possibly-verbose LLM response.

        The system prompt asks for ONE action line, but in practice the model
        often returns "Thought: ... Action: click [12]" or even pure prose on
        the first step before it learns the format. We do three passes:

        1. Strip code fences and ``Action:`` prefixes.
        2. If the cleaned text is exactly one line that matches the action
           grammar, return it (fast path).
        3. Otherwise, search the WHOLE response for the LAST occurrence of a
           valid action pattern. The last match is preferred because models
           tend to think first then act, so the action sits at the end.

        Returns the unmodified first line as a final fallback so the env's
        ``_parse_text_action`` can still log it as ``unknown`` rather than
        the agent's parser eating the response silently.
        """
        text = (text or "").strip()
        # Strip ``` fences
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        # Strip a leading "Action:" prefix on the first non-empty line, if
        # present, before pattern-searching — gives the regex a clean shot.
        lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
        if not lines:
            return text
        first = lines[0]
        if first.lower().startswith("action:"):
            first = first.split(":", 1)[1].strip()
            lines[0] = first

        # Fast path: a single line that already looks like a valid action.
        if len(lines) == 1 and cls._ACTION_RE.fullmatch(lines[0]):
            return lines[0]

        # Last-match search across the whole response.
        matches = list(cls._ACTION_RE.finditer(text))
        if matches:
            return matches[-1].group(1).strip()

        # Final fallback — return the first non-empty line so downstream
        # logging can flag it as an unparseable action rather than crashing.
        return first


# Register in agent registry
register_agent("webarena", WebArenaAgent)
