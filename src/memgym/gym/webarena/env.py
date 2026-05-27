"""
WebArenaMemoryEnv — Playwright-backed browsing environment for MemGym.

Each instance owns (1) a running `apps/<name>/server.py` subprocess provided
by a `WebArenaServerPool`, (2) a Playwright browser session pointed at that
server, and (3) the task's programmatic verifier. This mirrors the structural
role of `SWEMemoryEnv` on the coding track: the env is pure (reset/step/close)
and the runner owns the episode loop.

Default observation modality is **text** — a pruned accessibility tree plus
the current URL. Text observations are the primary modality for memory
strategy comparison because they feed directly into the proven text-based
memory adapters (`webarena_summarizing`, `webarena_structured`) without any
multimodal adapter layer. Setting `observation_mode="vision"` switches to
base64 screenshots for the multimodal-pressure ablation.

All heavy imports (`playwright.sync_api`) are performed lazily inside methods
so importing this module does not fail when Playwright is absent. The gym
`__init__.py` defers `register_env` for the same reason.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgym.envs.base import BaseMemoryEnvironment, BaseMemoryState, register_env

logger = logging.getLogger("memgym.webarena.env")


# =============================================================================
# Task loading
# =============================================================================

@dataclass
class WebArenaTask:
    """A single webarena-infinity task loaded from disk.

    Fields:
        app_name: Which app (gmail/gitlab/etc.) this task belongs to.
        task_id: Task identifier (integer index or string key).
        instruction: Natural-language task description for the agent.
        start_url: Path (relative to the server root) where the browser starts.
        verifier_config: Opaque dict passed to the task verifier. The exact
            shape depends on webarena-infinity's task schema and is consumed
            by `_run_verifier()` via a lazy import.
        raw: The full raw task dict (kept for debug / trajectory logging).
    """
    app_name: str
    task_id: str
    instruction: str
    start_url: str = "/"
    verifier_config: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_task(webarena_dir: Path, app_name: str, task_id: str) -> WebArenaTask:
    """Load a single task from `apps/<app>/tasks/<task_id>.json` (or similar).

    WebArena-Infinity ships tasks under each app directory. The precise layout
    (`tasks/*.json`, `benchmark/*.json`, `evaluation/tasks.jsonl`, etc.) varies
    by version. We try the common candidates in order and raise a descriptive
    error if none match so the user can point us at the right path explicitly.
    """
    app_dir = webarena_dir / "apps" / app_name
    if not app_dir.exists():
        raise FileNotFoundError(f"app not found: {app_dir}")

    candidates = [
        app_dir / "tasks" / f"{task_id}.json",
        app_dir / "benchmark" / f"{task_id}.json",
        app_dir / "eval" / f"{task_id}.json",
        app_dir / "tasks.jsonl",
        app_dir / "benchmark.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return _parse_task_file(path, app_name, task_id)

    # WebArena-Infinity layout: each app ships one or more task-manifest arrays.
    # Real tasks live in `real-tasks.json`, procedural function tasks in
    # `function-tasks.json`. Each is a JSON list of {id, instruction, verify, ...}.
    manifest_arrays = [
        app_dir / "real-tasks.json",
        app_dir / "function-tasks.json",
    ]
    for manifest in manifest_arrays:
        if not manifest.exists():
            continue
        with manifest.open() as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if str(row.get("id") or row.get("task_id")) == str(task_id):
                return _task_from_dict(row, app_name)

    raise FileNotFoundError(
        f"Could not find task {task_id} for app {app_name}. "
        f"Tried: {[str(c) for c in candidates + manifest_arrays]}"
    )


def _parse_task_file(path: Path, app_name: str, task_id: str) -> WebArenaTask:
    """Parse a task JSON or JSONL file and return a WebArenaTask."""
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if str(row.get("id") or row.get("task_id")) == str(task_id):
                    return _task_from_dict(row, app_name)
        raise KeyError(f"task_id {task_id} not found in {path}")

    with path.open() as f:
        data = json.load(f)
    return _task_from_dict(data, app_name)


def _task_from_dict(data: Dict[str, Any], app_name: str) -> WebArenaTask:
    # WebArena-Infinity stores the verifier as a path-to-script in a `verify`
    # field (e.g. "real-tasks/task_e1.py"). Older layouts used `verifier`/`eval`
    # as inline dicts. Preserve both forms so the runtime can dispatch.
    verifier_config = data.get("verifier") or data.get("eval") or {}
    if not verifier_config and data.get("verify"):
        verifier_config = {"script": str(data["verify"])}
    return WebArenaTask(
        app_name=app_name,
        task_id=str(data.get("id") or data.get("task_id") or "unknown"),
        instruction=str(data.get("instruction") or data.get("intent") or ""),
        start_url=str(data.get("start_url") or data.get("url") or "/"),
        verifier_config=verifier_config,
        raw=data,
    )


# =============================================================================
# DOM walker — tags interactive elements with data-memgym-id
# =============================================================================
#
# Returns a list of {id, role, name, value, tag, kind} dicts in DOM order. The
# JS clears any prior data-memgym-id attributes first so a re-run after page
# navigation produces a fresh consistent numbering.
#
# Filtering: only "interactive" elements (anything a user can click, type into,
# select, toggle, or focus). Hidden / 0-area / display:none / visibility:hidden
# elements are skipped because the agent cannot meaningfully act on them.

_TAG_INTERACTIVE_JS = r"""
() => {
  const SELECTOR = [
    'a[href]', 'button', 'input', 'textarea', 'select', 'summary',
    '[role="button"]', '[role="link"]', '[role="checkbox"]',
    '[role="radio"]', '[role="tab"]', '[role="menuitem"]',
    '[role="combobox"]', '[role="textbox"]', '[role="searchbox"]',
    '[role="switch"]', '[role="option"]', '[role="treeitem"]',
    '[contenteditable="true"]', '[tabindex]:not([tabindex="-1"])',
  ].join(', ');

  // Clear any previous tags so numbering is fresh.
  document.querySelectorAll('[data-memgym-id]').forEach(el => {
    el.removeAttribute('data-memgym-id');
  });

  const isVisible = (el) => {
    if (!el.isConnected) return false;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    const cs = window.getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') return false;
    if (parseFloat(cs.opacity || '1') === 0) return false;
    return true;
  };

  const accessibleName = (el) => {
    // Prefer aria-label; fall back to associated <label>; then text; then
    // placeholder/value/title for inputs.
    let n = (el.getAttribute('aria-label') || '').trim();
    if (n) return n;
    if (el.id) {
      const lbl = document.querySelector(`label[for="${el.id}"]`);
      if (lbl) {
        const t = (lbl.innerText || lbl.textContent || '').trim();
        if (t) return t;
      }
    }
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      const ph = el.getAttribute('placeholder') || '';
      if (ph) return ph;
    }
    const txt = (el.innerText || el.textContent || '').trim();
    if (txt) return txt.split('\n')[0];
    const title = el.getAttribute('title') || '';
    if (title) return title;
    return '';
  };

  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'summary') return 'button';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
      return 'textbox';
    }
    return tag;
  };

  // ----- Phase 1: standard interactive selectors --------------------------
  const nodes = Array.from(document.querySelectorAll(SELECTOR)).filter(isVisible);

  const out = [];
  nodes.forEach((el, i) => {
    const id = i + 1;
    el.setAttribute('data-memgym-id', String(id));

    let value = '';
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      try { value = (el.value || '').toString(); } catch (e) { value = ''; }
    }
    const checked = (tag === 'input' && (el.type === 'checkbox' || el.type === 'radio'))
      ? !!el.checked : null;

    out.push({
      id,
      tag,
      role: roleOf(el),
      name: accessibleName(el).slice(0, 160),
      value: value.slice(0, 160),
      checked,
    });
  });

  // ----- Phase 2: clickable container divs --------------------------------
  // Many modern web apps (gitlab-plan-and-track, list views, card grids)
  // attach click handlers to plain <div> / <li> rows with no <a>, <button>,
  // or role attribute. The first phase misses these entirely, so the agent
  // sees the issue title in the page outline but has no element id to click.
  //
  // Heuristic: walk container-ish tags, keep elements that (a) have text,
  // (b) look clickable (cursor:pointer or onclick attribute), (c) contain
  // no already-tagged descendant (so we don't double-tag a row that has an
  // inner button), and (d) have no already-tagged ancestor (so we don't
  // tag deeply-nested children of an outer card). This catches the
  // outermost click target, which is what a real user would aim at.
  const CONTAINER_TAGS = 'div, li, article, section, tr, td, span';
  const containers = document.querySelectorAll(CONTAINER_TAGS);
  const vw = window.innerWidth || 1280;
  const vh = window.innerHeight || 800;
  const VIEWPORT_AREA = vw * vh;

  for (const el of containers) {
    if (el.hasAttribute('data-memgym-id')) continue;
    // Skip if any already-tagged descendant covers it.
    if (el.querySelector('[data-memgym-id]')) continue;
    // Skip if any ancestor is already tagged.
    let hasTaggedAncestor = false;
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
      if (p.hasAttribute('data-memgym-id')) { hasTaggedAncestor = true; break; }
    }
    if (hasTaggedAncestor) continue;

    const txt = (el.innerText || '').trim();
    if (!txt) continue;

    if (!isVisible(el)) continue;

    // Only the most expensive checks last.
    const hasOnclick = el.onclick !== null || el.hasAttribute('onclick');
    let cursor = '';
    if (!hasOnclick) {
      cursor = window.getComputedStyle(el).cursor;
      if (cursor !== 'pointer') continue;
    }

    // Skip page-sized wrappers (some themes set body { cursor: pointer }
    // or wrap the whole main column in a clickable container).
    const r = el.getBoundingClientRect();
    if (r.width * r.height > VIEWPORT_AREA * 0.6) continue;

    const id = out.length + 1;
    el.setAttribute('data-memgym-id', String(id));
    out.push({
      id,
      tag: el.tagName.toLowerCase(),
      role: 'clickable',
      name: txt.replace(/\s+/g, ' ').slice(0, 200),
      value: '',
      checked: null,
    });
  }

  return out;
}
""".strip()


def _format_interactive_element(el: Dict[str, Any]) -> str:
    """Render one interactive element as a single ``[N] role "name"`` line.

    Compact on purpose — the numbered list is the agent's primary actionable
    surface and we want to keep token cost predictable. Optional fields
    (current value for inputs, checked state for checkboxes) only appear when
    they carry information.
    """
    idx = el.get("id")
    role = el.get("role") or "element"
    name = (el.get("name") or "").replace("\n", " ").strip()
    if len(name) > 100:
        name = name[:100] + "…"

    parts = [f'[{idx}] {role}']
    if name:
        parts.append(f'"{name}"')

    value = (el.get("value") or "").strip()
    if value:
        if len(value) > 60:
            value = value[:60] + "…"
        parts.append(f'(value="{value}")')

    checked = el.get("checked")
    if checked is True:
        parts.append("(checked)")
    elif checked is False:
        parts.append("(unchecked)")

    return " ".join(parts)


# =============================================================================
# WebArenaMemoryEnv
# =============================================================================

class WebArenaMemoryEnv(BaseMemoryEnvironment):
    """
    Playwright-backed webarena environment.

    Args:
        webarena_dir: Path to the webarena-infinity repo (`src/memgym/envs/
            webarena_infinity/` by default). Required so we can load task
            files and apply per-app verifiers.
        app_name: Which app this env runs tasks from.
        task_id: The specific task identifier.
        server_pool: A pre-initialized `WebArenaServerPool`. Required: the
            env does NOT spawn its own server because a single pool is
            shared by every worker process in a batch.
        observation_mode: "text" (default) returns an accessibility-tree
            snapshot + current URL. "vision" returns a base64 PNG screenshot.
        headless: Run Chromium headless. Always True on headless servers (no display).
        max_a11y_chars: Truncate the accessibility tree to this many chars
            to avoid unbounded per-step observations on giant pages.
        memory_state: Optional BaseMemoryState for the env's own bookkeeping
            (legacy compat — actual memory is managed by the MemoryManager).
        config: Arbitrary config dict.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        webarena_dir: Path | str,
        app_name: str,
        task_id: str,
        server_pool: Any,
        observation_mode: str = "text",
        headless: bool = True,
        max_a11y_chars: int = 20_000,
        memory_state: Optional[BaseMemoryState] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(memory_state=memory_state, config=config)

        if observation_mode not in ("text", "vision"):
            raise ValueError(
                f"observation_mode must be 'text' or 'vision', got {observation_mode!r}"
            )

        self.webarena_dir = Path(webarena_dir).resolve()
        self.app_name = app_name
        self.task_id = task_id
        self.server_pool = server_pool
        self.observation_mode = observation_mode
        self.headless = headless
        self.max_a11y_chars = max_a11y_chars

        # Populated by reset()
        self._task: Optional[WebArenaTask] = None
        self._server = None
        self._playwright = None  # Playwright context manager handle
        self._browser = None
        self._context = None
        self._page = None
        self._terminated_by_agent: bool = False
        # Episode-static fields populated by reset() — cloned into every
        # step() info dict so per-step keys (error, clicked, ...) don't
        # accumulate across steps. Carrying state in `_last_info` previously
        # caused stale `error: "unknown action type: 'unknown'"` to leak from
        # step 1 into every successful step thereafter.
        self._episode_info: Dict[str, Any] = {}

    # ---- gym API ----

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset browser + acquire a fresh server + load task."""
        # Gym base bookkeeping (this assigns self.np_random etc.)
        try:
            super().reset(seed=seed)
        except TypeError:
            # base class may not accept seed in some versions
            pass
        self._episode_step = 0
        self._terminated_by_agent = False

        # Close any leftover state from a previous episode
        self._teardown_browser()
        if self._server is not None:
            self.server_pool.release(self._server)
            self._server = None

        # Load task
        self._task = load_task(self.webarena_dir, self.app_name, self.task_id)

        # Acquire server
        self._server = self.server_pool.acquire(self.app_name)

        # Launch Playwright + open page at start_url
        self._launch_browser()
        start = self._absolute_url(self._task.start_url)
        self._page.goto(start)
        self._page.wait_for_load_state("domcontentloaded", timeout=30_000)

        obs = self._capture_observation()
        self._episode_info = {
            "task_id": self._task.task_id,
            "app_name": self._task.app_name,
            "instruction": self._task.instruction,
            "start_url": start,
            "observation_mode": self.observation_mode,
        }
        return obs, dict(self._episode_info)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one action against the live browser."""
        self._episode_step += 1
        self._total_steps += 1

        if self._page is None:
            raise RuntimeError("step() called before reset()")

        terminated = False
        truncated = False
        action_info: Dict[str, Any] = {"raw_action": action}

        try:
            action_info.update(self._execute_action(action))
        except Exception as e:
            logger.exception("action execution failed: %r", action)
            action_info["error"] = str(e)

        # Some actions finalize the episode directly
        if action_info.get("done"):
            self._terminated_by_agent = True
            terminated = True

        obs = self._capture_observation()
        reward = self._calculate_reward(obs, action, action_info)

        # Task verifier decides final termination on "done"-like actions or
        # whenever the verifier signals success. We run it every step when
        # cheap; if it proves expensive, gate this on terminated.
        if terminated or self._should_probe_verifier(action_info):
            verdict = self._run_verifier()
            action_info["verifier"] = verdict
            if verdict.get("success"):
                reward = max(reward, 1.0)
                terminated = True

        # Rebuild info from immutable episode-static fields each step so
        # per-step keys (error, clicked, typed, verifier, ...) cannot leak
        # across steps. _episode_info is set once in reset() and never
        # mutated; the per-step keys come exclusively from action_info.
        info = {
            **self._episode_info,
            "step": self._episode_step,
            **action_info,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Release browser + server, if held."""
        self._teardown_browser()
        if self._server is not None:
            try:
                self.server_pool.release(self._server)
            finally:
                self._server = None

    # ---- observation ----

    def _capture_observation(self) -> Dict[str, Any]:
        if self._page is None:
            return {"url": "", "content": ""}

        url = self._page.url
        title = ""
        try:
            title = self._page.title()
        except Exception:
            pass

        if self.observation_mode == "text":
            a11y = self._capture_accessibility_tree()
            return {
                "mode": "text",
                "url": url,
                "title": title,
                "accessibility_tree": a11y,
            }

        # vision
        try:
            png_bytes = self._page.screenshot(full_page=False, type="png")
            screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")
        except Exception as e:
            logger.warning("screenshot failed: %s", e)
            screenshot_b64 = ""
        return {
            "mode": "vision",
            "url": url,
            "title": title,
            "screenshot_b64": screenshot_b64,
        }

    def _capture_accessibility_tree(self) -> str:
        """Capture an accessibility snapshot of the current page as text.

        Strategy: tag every visible interactive element on the live page with a
        sequential ``data-memgym-id="N"`` attribute via JavaScript and return a
        numbered list keyed on those Ns. The system prompt promises the agent
        a ``click [N]`` / ``type [N] [text]`` vocabulary, so the observation
        MUST surface those Ns — otherwise the LLM hallucinates indices that
        don't correspond to anything (which is exactly what happened in the
        first W-0 run: aria_snapshot returned a YAML tree with zero numeric
        IDs and the agent emitted ``click [2]`` for ten straight steps).

        ``_resolve_locator(N)`` later uses ``data-memgym-id`` to locate the
        element exactly — no CSS nth-of-type guessing, no per-tag heuristics.

        We also append a compact ARIA outline so the agent retains spatial
        context (groupings, headings, list rows) that the flat numbered list
        loses. The outline is informational only — only the numbered block
        carries actionable IDs.

        Falls back to ``page.inner_text('body')`` if the JS evaluator throws
        (e.g. about: pages, network errors mid-load) so the agent still gets
        *something* and can call ``fail`` rather than crashing.
        """
        # Step 1: tag interactive elements + collect their metadata in one
        # round-trip. Doing this in JS rather than Python avoids N+1 trips
        # for getBoundingClientRect / aria-label / value lookups.
        try:
            elements = self._page.evaluate(_TAG_INTERACTIVE_JS)
        except Exception as e:
            logger.debug("interactive tagging failed: %s", e)
            elements = None

        if elements:
            lines = ["INTERACTIVE ELEMENTS (use these IDs with click/type/select):"]
            for el in elements:
                lines.append(_format_interactive_element(el))
            interactive_block = "\n".join(lines)
        else:
            interactive_block = "INTERACTIVE ELEMENTS: (none found)"

        # Step 2: spatial outline via aria_snapshot (best-effort, optional).
        outline = ""
        try:
            outline = self._page.locator("body").aria_snapshot() or ""
        except Exception as e:
            logger.debug("aria_snapshot for outline failed: %s", e)
            try:
                outline = self._page.inner_text("body") or ""
            except Exception:
                outline = ""

        # Budget: leave most of max_a11y_chars for the numbered list (it's the
        # only part the agent can actually act on); cap the outline to whatever
        # remains.
        out_parts = [interactive_block]
        remaining = self.max_a11y_chars - len(interactive_block)
        if outline and remaining > 200:
            if len(outline) > remaining - 50:
                outline = outline[: remaining - 50] + "\n[... outline truncated ...]"
            out_parts.append("\nPAGE OUTLINE:\n" + outline)

        text = "\n".join(out_parts)
        if len(text) > self.max_a11y_chars:
            text = text[: self.max_a11y_chars] + "\n[... truncated ...]"
        return text

    # ---- action execution ----

    def _execute_action(self, action: Any) -> Dict[str, Any]:
        """Translate an agent action into Playwright calls.

        Supported action shapes:

        1. A dict with a `type` key:
           {"type": "click", "element_id": 12}
           {"type": "type", "element_id": 3, "text": "hello"}
           {"type": "goto", "url": "/search?q=foo"}
           {"type": "back"} / {"type": "forward"}
           {"type": "key", "keys": "Enter"}
           {"type": "scroll", "direction": "down", "amount": 500}
           {"type": "done"} / {"type": "fail"}

        2. A string like "click [12]" / "type [3] hello" — converted via
           `_parse_text_action()`. Supports the webarena-infinity textual
           action conventions.

        Element IDs refer to the numbered accessibility-tree snapshot from
        the MOST RECENT observation; we resolve them to Playwright locators
        using a "nth-interactive" heuristic (Playwright does not expose a
        direct handle from the accessibility snapshot).
        """
        if isinstance(action, str):
            action = _parse_text_action(action)

        if not isinstance(action, dict):
            return {"error": f"unsupported action shape: {type(action).__name__}"}

        atype = action.get("type", "").lower()

        if atype == "done":
            return {"done": True}
        if atype == "fail":
            return {"done": True, "failed": True}

        if atype == "goto":
            url = self._absolute_url(str(action.get("url", "/")))
            self._page.goto(url)
            self._page.wait_for_load_state("domcontentloaded", timeout=30_000)
            return {"navigated_to": url}

        if atype == "back":
            self._page.go_back()
            return {"navigated": "back"}
        if atype == "forward":
            self._page.go_forward()
            return {"navigated": "forward"}

        if atype == "key":
            keys = str(action.get("keys", ""))
            # Direct the keypress at whatever element currently has focus
            # (e.g. the search input we just typed into) by routing through
            # Playwright's locator.press(). The previous implementation used
            # page.keyboard.press() which dispatches a synthetic event to the
            # browser's global keyboard — many SPAs (gitlab-plan-and-track's
            # search box among them) listen for the keypress on the input
            # element directly, so the global event never reaches their
            # submit handler. Routing via locator.press() guarantees the full
            # focus -> keydown -> keyup -> input handler chain fires.
            #
            # If the focused element isn't one we tagged (or nothing is
            # focused at all), fall back to the global keyboard so non-input
            # combos like Tab still work.
            focused_id = None
            try:
                focused_id = self._page.evaluate(
                    "() => { const a = document.activeElement;"
                    " return a && a.getAttribute && a.getAttribute('data-memgym-id'); }"
                )
            except Exception as e:
                logger.debug("activeElement probe failed: %s", e)

            if focused_id:
                try:
                    target = self._page.locator(
                        f'[data-memgym-id="{focused_id}"]'
                    ).first
                    target.press(keys, timeout=3000)
                    return {"keys": keys, "targeted_id": int(focused_id)}
                except Exception as e:
                    logger.debug(
                        "targeted key press on id=%s failed: %s — falling back",
                        focused_id, e,
                    )

            self._page.keyboard.press(keys)
            return {"keys": keys}

        if atype == "scroll":
            direction = str(action.get("direction", "down"))
            amount = int(action.get("amount", 500))
            dy = amount if direction == "down" else -amount
            self._page.mouse.wheel(0, dy)
            return {"scrolled": dy}

        if atype in ("click", "type", "select"):
            locator = self._resolve_locator(action.get("element_id"))
            if locator is None:
                return {"error": f"could not resolve element_id {action.get('element_id')}"}
            if atype == "click":
                locator.click(timeout=5000)
                return {"clicked": action.get("element_id")}
            if atype == "type":
                text = str(action.get("text", ""))
                locator.fill(text, timeout=5000)
                # Re-assert focus on the field so a follow-up `key [Enter]`
                # from the agent reliably targets it. Playwright's fill()
                # already focuses-then-types under the hood, but some apps
                # re-render the input on `input` events and the focus
                # reference can drift. focus() is cheap and idempotent.
                try:
                    locator.focus(timeout=1000)
                except Exception as e:
                    logger.debug("focus retention after type failed: %s", e)
                return {"typed": text, "into": action.get("element_id")}
            if atype == "select":
                value = str(action.get("value", ""))
                locator.select_option(value, timeout=5000)
                return {"selected": value, "in": action.get("element_id")}

        return {"error": f"unknown action type: {atype!r}"}

    def _resolve_locator(self, element_id: Any):
        """Resolve an interactive element id back to a Playwright locator.

        Uses the ``data-memgym-id`` attribute that ``_capture_accessibility_tree``
        wrote on every visible interactive element. This is exact: the same
        ID the LLM saw in the numbered list maps 1:1 to the same DOM node we
        click on. Returns None if the id is missing or stale (e.g. the page
        navigated since the snapshot was taken).
        """
        if element_id is None:
            return None
        try:
            idx = int(element_id)
        except (TypeError, ValueError):
            return None
        if idx <= 0:
            return None
        locator = self._page.locator(f'[data-memgym-id="{idx}"]').first
        try:
            locator.wait_for(state="attached", timeout=2000)
            return locator
        except Exception:
            return None

    # ---- verifier ----

    def _should_probe_verifier(self, action_info: Dict[str, Any]) -> bool:
        """Heuristic: run the verifier only on terminal-ish actions to save time."""
        return bool(action_info.get("done"))

    def _run_verifier(self) -> Dict[str, Any]:
        """Execute the per-task verify script that ships with each task.

        WebArena-Infinity tasks store ``verify: "<subdir>/<script>.py"`` (e.g.
        ``"function-tasks/task_1.py"``) in their manifest. The script lives at
        ``apps/<app_name>/<verify_path>`` and exports a ``verify(server_url)
        -> tuple[bool, str]`` function that hits the running app's REST API
        directly (e.g. ``GET /api/state``) to inspect post-action state.

        Notably the verifier does NOT need a Playwright handle — it talks to
        the app over HTTP, the same way a browser would. So we just import
        the script via ``importlib.util.spec_from_file_location`` and pass it
        ``self._server.url``.

        Returns a dict with ``success: bool`` plus either ``reason`` (the
        verifier's human-readable explanation) or ``error`` (an exception
        message if the script itself crashed).
        """
        cfg = self._task.verifier_config if self._task else {}
        script_rel = cfg.get("script") if isinstance(cfg, dict) else None
        if not script_rel:
            return {"success": False, "reason": "no verify script declared on task"}
        if self._server is None:
            return {"success": False, "reason": "no server bound to env"}

        script_path = (
            self.webarena_dir / "apps" / self.app_name / script_rel
        ).resolve()
        if not script_path.exists():
            return {
                "success": False,
                "reason": f"verify script not found: {script_path}",
            }

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"webarena_verify_{self.app_name}_{self._task.task_id}",
                str(script_path),
            )
            if spec is None or spec.loader is None:
                return {
                    "success": False,
                    "reason": f"could not build import spec for {script_path}",
                }
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            verify_fn = getattr(module, "verify", None)
            if verify_fn is None:
                return {
                    "success": False,
                    "reason": f"verify script {script_path.name} has no verify() function",
                }

            result = verify_fn(self._server.url)
            # Upstream returns (bool, str); be permissive about bare bool too.
            if isinstance(result, tuple) and len(result) == 2:
                success, reason = bool(result[0]), str(result[1])
            else:
                success, reason = bool(result), ""
            return {"success": success, "reason": reason, "script": str(script_path)}
        except Exception as e:
            logger.warning("verifier %s crashed: %s", script_path.name, e)
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "script": str(script_path),
            }

    # ---- reward + memory eval ----

    def _calculate_reward(
        self, observation: Any, action: Any, info: Dict[str, Any]
    ) -> float:
        """Sparse reward: 1.0 on verifier success, 0 otherwise.

        Dense shaping (per-step) is explicitly out of scope — memory
        strategies should be compared on the terminal task-completion
        signal only.
        """
        if info.get("verifier", {}).get("success"):
            return 1.0
        if info.get("failed"):
            return 0.0
        return 0.0

    def _evaluate_memory(self) -> Dict[str, Any]:
        """Legacy hook from BaseMemoryEnvironment — memory eval is done by
        the MemoryManager itself via get_stats(); the env returns a minimal
        placeholder dict to satisfy the interface."""
        return {"env_step": self._episode_step, "app": self.app_name}

    # ---- browser lifecycle ----

    def _launch_browser(self) -> None:
        from playwright.sync_api import sync_playwright  # lazy import

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 960},
            user_agent="MemGym-WebArena/0.1",
        )
        self._page = self._context.new_page()

    def _teardown_browser(self) -> None:
        for attr in ("_page", "_context", "_browser"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def _absolute_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if not path_or_url.startswith("/"):
            path_or_url = "/" + path_or_url
        return f"{self._server.url}{path_or_url}"


# =============================================================================
# Text-action parsing
# =============================================================================

def _parse_text_action(text: str) -> Dict[str, Any]:
    """Parse webarena-style text actions into the dict form used by step().

    Examples:
        "click [12]"                -> {"type": "click", "element_id": 12}
        "type [3] [hello world]"    -> {"type": "type", "element_id": 3, "text": "hello world"}
        "goto [/search?q=foo]"      -> {"type": "goto", "url": "/search?q=foo"}
        "key [Enter]"               -> {"type": "key", "keys": "Enter"}
        "scroll [down]"             -> {"type": "scroll", "direction": "down"}
        "stop"                       -> {"type": "done"}
    """
    import re

    text = text.strip()
    lower = text.lower()

    if lower in ("stop", "done", "finish"):
        return {"type": "done"}
    if lower == "fail":
        return {"type": "fail"}
    if lower in ("back",):
        return {"type": "back"}
    if lower in ("forward",):
        return {"type": "forward"}

    # click [N]
    m = re.match(r"click\s*\[(\d+)\]\s*$", text, re.IGNORECASE)
    if m:
        return {"type": "click", "element_id": int(m.group(1))}

    # type [N] [text...] — original strict form (preferred when given).
    m = re.match(r"type\s*\[(\d+)\]\s*\[(.*)\]\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {"type": "type", "element_id": int(m.group(1)), "text": m.group(2)}

    # type [N] text  — permissive form. LLMs trained on browser-action examples
    # tend to drop the inner brackets when the value is plain prose, so we
    # accept `type [2] Implement user profile badges` as well as the bracketed
    # form. Strips surrounding quotes if present so `type [2] "hello"` works.
    m = re.match(r"type\s*\[(\d+)\]\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        value = m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        return {"type": "type", "element_id": int(m.group(1)), "text": value}

    # select [N] [value] — strict form first.
    m = re.match(r"select\s*\[(\d+)\]\s*\[(.*)\]\s*$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {"type": "select", "element_id": int(m.group(1)), "value": m.group(2)}

    # select [N] value — permissive form, mirrors the type fallback above.
    m = re.match(r"select\s*\[(\d+)\]\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        value = m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        return {"type": "select", "element_id": int(m.group(1)), "value": value}

    # goto [url]  — strict bracketed form
    m = re.match(r"goto\s*\[(.*)\]\s*$", text, re.IGNORECASE)
    if m:
        return {"type": "goto", "url": m.group(1).strip()}

    # goto url  — permissive bare form. The LLM consistently emits
    # `goto http://127.0.0.1:18001/#/issues/19` instead of the bracketed
    # form, especially when the URL contains slashes/hashes. Accept it.
    m = re.match(r"goto\s+(.+)$", text, re.IGNORECASE)
    if m:
        url = m.group(1).strip()
        if len(url) >= 2 and url[0] == url[-1] and url[0] in ('"', "'"):
            url = url[1:-1]
        return {"type": "goto", "url": url}

    # key [combo]  — strict bracketed form
    m = re.match(r"key\s*\[(.*)\]\s*$", text, re.IGNORECASE)
    if m:
        return {"type": "key", "keys": m.group(1).strip()}

    # key combo  — permissive bare form (e.g. `key Enter`, `key Control+A`).
    m = re.match(r"key\s+(.+)$", text, re.IGNORECASE)
    if m:
        return {"type": "key", "keys": m.group(1).strip()}

    # scroll [dir] (amount?)  — strict bracketed form
    m = re.match(r"scroll\s*\[(up|down)\](?:\s*\[(\d+)\])?\s*$", text, re.IGNORECASE)
    if m:
        out: Dict[str, Any] = {"type": "scroll", "direction": m.group(1).lower()}
        if m.group(2):
            out["amount"] = int(m.group(2))
        return out

    # scroll dir (amount?)  — permissive bare form (e.g. `scroll down`,
    # `scroll up 300`).
    m = re.match(r"scroll\s+(up|down)(?:\s+(\d+))?\s*$", text, re.IGNORECASE)
    if m:
        out = {"type": "scroll", "direction": m.group(1).lower()}
        if m.group(2):
            out["amount"] = int(m.group(2))
        return out

    return {"type": "unknown", "raw": text}


# =============================================================================
# Registry
# =============================================================================
#
# Register under both `webarena` and `web` so callers can use either name.
# This module does NOT import Playwright at load time (playwright is imported
# lazily inside `_launch_browser()`), so this registration is safe to run on
# any host including dev machines without Playwright installed.
register_env("webarena", WebArenaMemoryEnv)
register_env("web", WebArenaMemoryEnv)
