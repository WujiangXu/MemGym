"""Coding QA Instance Viewer — inspect rendered benchmark instances.

Modes:
  terminal (default) — Rich terminal output with ANSI colors
  agent-view         — Shows ONLY what an evaluation agent sees
  html               — Self-contained HTML with collapsible sections
  summary            — Table overview of all instances

Usage:
    python -m memgym.pipelines.coding_synthetic.viewer \
        data/coding_synthetic/coding_qa_v2_100.jsonl \
        --index 0

    python -m memgym.pipelines.coding_synthetic.viewer \
        data/coding_synthetic/coding_qa_v2_100.jsonl \
        --id coding_qa__swesmith__oauthlib_... --agent-view

    python -m memgym.pipelines.coding_synthetic.viewer \
        data/coding_synthetic/coding_qa_v2_100.jsonl \
        --summary

    python -m memgym.pipelines.coding_synthetic.viewer \
        data/coding_synthetic/coding_qa_v2_100.jsonl \
        --index 0 --html --output viewer.html
"""

import argparse
import html as html_lib
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from .types.schemas import CodingQAInstance
from .utils.token_counter import count_tokens


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

class _C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def _header(text: str) -> str:
    return f"\n{_C.BOLD}{_C.CYAN}{'═' * 70}\n  {text}\n{'═' * 70}{_C.RESET}\n"


def _subheader(text: str) -> str:
    return f"\n{_C.BOLD}{_C.YELLOW}── {text} ──{_C.RESET}\n"


def _dim(text: str) -> str:
    return f"{_C.DIM}{text}{_C.RESET}"


# ---------------------------------------------------------------------------
# Token breakdown
# ---------------------------------------------------------------------------

def _token_breakdown(inst: CodingQAInstance) -> dict:
    tok_task = count_tokens(inst.task_prompt)
    tok_memory = sum(count_tokens(c) for c in inst.memory_files.values())
    tok_repo_ctx = count_tokens(inst.repo_context or "")
    tok_repo_files = sum(count_tokens(c) for c in inst.repo_files.values())
    tok_facts = sum(count_tokens(f.content) for f in inst.grounding_facts)
    tok_distractors = sum(count_tokens(d.content) for d in inst.distractors)
    total = tok_task + tok_memory + tok_repo_ctx + tok_repo_files + tok_facts + tok_distractors
    return {
        "task_prompt": tok_task,
        "memory_files": tok_memory,
        "repo_context": tok_repo_ctx,
        "repo_files": tok_repo_files,
        "grounding_facts": tok_facts,
        "distractors": tok_distractors,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Terminal mode
# ---------------------------------------------------------------------------

def _view_terminal(inst: CodingQAInstance, full_repo: bool = False) -> str:
    lines = []
    tb = _token_breakdown(inst)

    # Header
    lines.append(_header("CODING QA INSTANCE"))
    lines.append(f"  {_C.BOLD}ID:{_C.RESET}        {inst.instance_id}")
    lines.append(f"  {_C.BOLD}Base:{_C.RESET}      {inst.base_instance}")
    lines.append(f"  {_C.BOLD}QA Pairs:{_C.RESET}  {len(inst.qa_pairs)}")
    lines.append(f"  {_C.BOLD}Facts:{_C.RESET}     {len(inst.grounding_facts)} ({inst.num_critical_facts} critical, {inst.num_external_facts} external)")
    lines.append(f"  {_C.BOLD}Distractors:{_C.RESET} {len(inst.distractors)}")
    lines.append(f"  {_C.BOLD}Tokens:{_C.RESET}    {tb['total']:,}")
    lines.append(f"  {_C.BOLD}Image:{_C.RESET}     {inst.image_name or '(none)'}")

    # Task prompt
    lines.append(_subheader("TASK PROMPT"))
    lines.append(inst.task_prompt)

    # Memory files
    lines.append(_subheader(f"MEMORY FILES ({len(inst.memory_files)} files, {tb['memory_files']:,} tokens)"))
    for fname, content in inst.memory_files.items():
        ftok = count_tokens(content)
        lines.append(f"\n{_C.BOLD}{_C.GREEN}=== {fname} === {_C.RESET}{_dim(f'({ftok:,} tokens)')}")
        lines.append(content[:3000])
        if len(content) > 3000:
            lines.append(_dim(f"\n  ... ({len(content) - 3000} chars truncated, use --full for complete)"))

    # Repo files
    n_repo = len(inst.repo_files)
    if n_repo > 0:
        lines.append(_subheader(f"REPO FILES ({n_repo} files, {tb['repo_files']:,} tokens)"))
        for fpath, content in sorted(inst.repo_files.items()):
            ftok = count_tokens(content)
            n_lines = content.count("\n") + 1
            lines.append(f"  {_C.BLUE}{fpath}{_C.RESET}  {_dim(f'{ftok:,} tok, {n_lines} lines')}")
            if full_repo:
                lines.append(f"```\n{content}\n```")
            else:
                preview = "\n".join(content.split("\n")[:30])
                lines.append(f"```\n{preview}\n```")
                if n_lines > 30:
                    lines.append(_dim(f"  ... ({n_lines - 30} more lines)"))
    else:
        lines.append(_subheader("REPO FILES"))
        lines.append(_dim("  (empty — run extract-repo to populate)"))

    # Repo context (skeleton)
    if inst.repo_context:
        lines.append(_subheader(f"REPO CONTEXT (skeleton, {tb['repo_context']:,} tokens)"))
        lines.append(inst.repo_context[:2000])
        if len(inst.repo_context) > 2000:
            lines.append(_dim(f"  ... truncated"))

    # QA Pairs
    lines.append(_subheader(f"QA PAIRS ({len(inst.qa_pairs)})"))
    fact_map = {f.id: f for f in inst.grounding_facts}
    for qp in inst.qa_pairs:
        diff_color = {"easy": _C.GREEN, "medium": _C.YELLOW, "hard": _C.RED}
        dc = diff_color.get(qp.difficulty, "")
        synth_tag = " [synthesis]" if qp.requires_synthesis else ""
        lines.append(
            f"\n  {_C.BOLD}{qp.qa_id}{_C.RESET} "
            f"{dc}[{qp.difficulty}]{_C.RESET}"
            f" {_dim(f'{qp.num_linked_facts} facts, {qp.signal_tokens} signal tok')}"
            f"{synth_tag}"
        )
        lines.append(f"  {_C.BOLD}Q:{_C.RESET} {qp.question}")
        lines.append(f"  {_C.BOLD}A:{_C.RESET} {qp.gold_answer[:300]}")
        if qp.gold_answer and len(qp.gold_answer) > 300:
            lines.append(_dim("    ... (truncated)"))
        linked = [fact_map[fid] for fid in qp.linked_fact_ids if fid in fact_map]
        for lf in linked:
            disc_tag = f"{_C.GREEN}discoverable{_C.RESET}" if lf.discoverable else f"{_C.RED}external{_C.RESET}"
            lines.append(f"    {_C.DIM}→ {lf.id} [{lf.category}] {disc_tag}: {lf.content[:120]}{_C.RESET}")

    # Grounding facts
    lines.append(_subheader(f"GROUNDING FACTS ({len(inst.grounding_facts)}, {tb['grounding_facts']:,} tokens)"))
    for f in inst.grounding_facts:
        disc = f"{_C.GREEN}disc{_C.RESET}" if f.discoverable else f"{_C.RED}ext{_C.RESET}"
        lines.append(
            f"  {_C.BOLD}{f.id}{_C.RESET} [{f.category}] {disc} "
            f"{_C.DIM}{f.relevance}{_C.RESET}: {f.content[:200]}"
        )

    # Distractors
    lines.append(_subheader(f"DISTRACTORS ({len(inst.distractors)}, {tb['distractors']:,} tokens)"))
    by_source: dict = {}
    for d in inst.distractors:
        by_source.setdefault(d.source, []).append(d)
    for source, ds in sorted(by_source.items()):
        lines.append(f"  {_C.MAGENTA}{source}{_C.RESET}: {len(ds)} distractors")
        for d in ds[:3]:
            lines.append(f"    {_C.DIM}{d.id}: {d.content[:150]}{_C.RESET}")
        if len(ds) > 3:
            lines.append(_dim(f"    ... ({len(ds) - 3} more)"))

    # Token breakdown
    lines.append(_subheader("TOKEN BREAKDOWN"))
    for k, v in tb.items():
        bar = "█" * min(50, v // max(1, tb["total"] // 50))
        lines.append(f"  {k:>18}: {v:>8,}  {_C.BLUE}{bar}{_C.RESET}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent-view mode
# ---------------------------------------------------------------------------

def _view_agent(inst: CodingQAInstance) -> str:
    """Show ONLY what an evaluation agent sees — no labels, no gold answers."""
    lines = []
    lines.append(_header("AGENT VIEW"))
    lines.append(f"Instance: {inst.instance_id}")
    tb = _token_breakdown(inst)
    lines.append(f"Total context: {tb['total']:,} tokens\n")

    # Task prompt
    lines.append(_subheader("BUG REPORT"))
    lines.append(inst.task_prompt)

    # Memory files (with embedded facts and distractors, as agent sees them)
    lines.append(_subheader(f"DEVELOPER NOTES ({len(inst.memory_files)} files)"))
    for fname, content in inst.memory_files.items():
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {fname}")
        lines.append(f"{'=' * 60}")
        lines.append(content)

    # Repo context
    if inst.repo_context:
        lines.append(_subheader("REPO SKELETON"))
        lines.append(inst.repo_context)

    # Repo files
    if inst.repo_files:
        lines.append(_subheader(f"SOURCE FILES ({len(inst.repo_files)} files)"))
        for fpath, content in sorted(inst.repo_files.items()):
            lines.append(f"\n### {fpath}")
            lines.append(f"```python\n{content}\n```")

    # Questions only (no answers)
    lines.append(_subheader(f"QUESTIONS ({len(inst.qa_pairs)})"))
    for qp in inst.qa_pairs:
        lines.append(f"  {qp.qa_id}. {qp.question}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------

def _view_summary(instances: List[CodingQAInstance]) -> str:
    lines = []
    lines.append(_header(f"CODING QA SUMMARY — {len(instances)} instances"))

    # Table header
    hdr = (
        f"{'#':>4} {'Instance ID':<50} {'Repo':<20} "
        f"{'QA':>4} {'Facts':>5} {'Ext':>4} "
        f"{'E':>3} {'M':>3} {'H':>3} {'Tokens':>8}"
    )
    lines.append(hdr)
    lines.append("─" * len(hdr))

    total_qa = 0
    total_tokens = 0
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, inst in enumerate(instances):
        tb = _token_breakdown(inst)
        repo = (inst.image_name or "").split(".")[-1][:20] if inst.image_name else "?"
        n_easy = sum(1 for q in inst.qa_pairs if q.difficulty == "easy")
        n_med = sum(1 for q in inst.qa_pairs if q.difficulty == "medium")
        n_hard = sum(1 for q in inst.qa_pairs if q.difficulty == "hard")
        total_qa += len(inst.qa_pairs)
        total_tokens += tb["total"]
        diff_counts["easy"] += n_easy
        diff_counts["medium"] += n_med
        diff_counts["hard"] += n_hard

        lines.append(
            f"{i:>4} {inst.instance_id:<50} {repo:<20} "
            f"{len(inst.qa_pairs):>4} {len(inst.grounding_facts):>5} {inst.num_external_facts:>4} "
            f"{n_easy:>3} {n_med:>3} {n_hard:>3} {tb['total']:>8,}"
        )

    lines.append("─" * len(hdr))
    lines.append(
        f"{'':>4} {'TOTAL':<50} {'':<20} "
        f"{total_qa:>4} {'':<5} {'':<4} "
        f"{diff_counts['easy']:>3} {diff_counts['medium']:>3} {diff_counts['hard']:>3} "
        f"{total_tokens:>8,}"
    )
    avg_tok = total_tokens / max(1, len(instances))
    avg_qa = total_qa / max(1, len(instances))
    lines.append(f"\nAvg: {avg_qa:.1f} QA/instance, {avg_tok:,.0f} tokens/instance")
    pct_e = diff_counts["easy"] / max(1, total_qa) * 100
    pct_m = diff_counts["medium"] / max(1, total_qa) * 100
    pct_h = diff_counts["hard"] / max(1, total_qa) * 100
    lines.append(f"Difficulty: {pct_e:.0f}% easy, {pct_m:.0f}% medium, {pct_h:.0f}% hard")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML mode
# ---------------------------------------------------------------------------

def _view_html(inst: CodingQAInstance) -> str:
    """Generate self-contained HTML with collapsible sections."""
    tb = _token_breakdown(inst)
    fact_map = {f.id: f for f in inst.grounding_facts}

    def esc(s: str) -> str:
        return html_lib.escape(s)

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Coding QA Viewer</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }
h2 { color: #ffa500; cursor: pointer; }
h2:before { content: "▶ "; font-size: 0.8em; }
h2.open:before { content: "▼ "; }
.section { display: none; padding: 10px 0; }
.section.open { display: block; }
.meta { color: #888; font-size: 0.9em; }
.fact-ext { color: #ff6b6b; }
.fact-disc { color: #51cf66; }
.diff-easy { color: #51cf66; font-weight: bold; }
.diff-medium { color: #ffd43b; font-weight: bold; }
.diff-hard { color: #ff6b6b; font-weight: bold; }
pre { background: #16213e; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
code { background: #16213e; padding: 2px 6px; border-radius: 3px; }
.qa-pair { border-left: 3px solid #00d4ff; padding-left: 12px; margin: 12px 0; }
.token-bar { display: inline-block; height: 14px; background: #00d4ff; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; }
td, th { padding: 4px 8px; text-align: left; border-bottom: 1px solid #333; }
input[type=text] { width: 100%; padding: 8px; margin: 10px 0; background: #16213e;
                   color: #e0e0e0; border: 1px solid #444; border-radius: 4px; font-size: 1em; }
.highlight { background: #ffd43b33; }
</style>
<script>
function toggleSection(id) {
  var s = document.getElementById(id);
  var h = s.previousElementSibling;
  s.classList.toggle('open');
  h.classList.toggle('open');
}
function searchContent() {
  var q = document.getElementById('search').value.toLowerCase();
  var els = document.querySelectorAll('.searchable');
  els.forEach(function(el) {
    el.innerHTML = el.textContent; // reset
    if (q && el.textContent.toLowerCase().includes(q)) {
      el.innerHTML = el.textContent.replace(new RegExp('(' + q.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi'), '<span class="highlight">$1</span>');
    }
  });
}
</script></head><body>
""")

    # Header
    parts.append(f"<h1>Coding QA Instance</h1>")
    parts.append(f"<table>")
    parts.append(f"<tr><td><b>ID</b></td><td>{esc(inst.instance_id)}</td></tr>")
    parts.append(f"<tr><td><b>Base</b></td><td>{esc(inst.base_instance)}</td></tr>")
    parts.append(f"<tr><td><b>QA Pairs</b></td><td>{len(inst.qa_pairs)}</td></tr>")
    parts.append(f"<tr><td><b>Facts</b></td><td>{len(inst.grounding_facts)} ({inst.num_critical_facts} critical, {inst.num_external_facts} external)</td></tr>")
    parts.append(f"<tr><td><b>Tokens</b></td><td>{tb['total']:,}</td></tr>")
    parts.append(f"<tr><td><b>Image</b></td><td>{esc(inst.image_name or '(none)')}</td></tr>")
    parts.append(f"</table>")

    parts.append(f'<input type="text" id="search" placeholder="Search content..." oninput="searchContent()">')

    # Task prompt
    parts.append(f'<h2 onclick="toggleSection(\'task\')">Task Prompt</h2>')
    parts.append(f'<div id="task" class="section open"><pre class="searchable">{esc(inst.task_prompt)}</pre></div>')

    # Memory files
    parts.append(f'<h2 class="open" onclick="toggleSection(\'memory\')">Memory Files ({len(inst.memory_files)} files, {tb["memory_files"]:,} tokens)</h2>')
    parts.append(f'<div id="memory" class="section open">')
    for fname, content in inst.memory_files.items():
        ftok = count_tokens(content)
        parts.append(f"<h3>{esc(fname)} <span class='meta'>({ftok:,} tokens)</span></h3>")
        parts.append(f'<pre class="searchable">{esc(content)}</pre>')
    parts.append(f"</div>")

    # Repo files
    parts.append(f'<h2 onclick="toggleSection(\'repo\')">Repo Files ({len(inst.repo_files)} files, {tb["repo_files"]:,} tokens)</h2>')
    parts.append(f'<div id="repo" class="section">')
    if inst.repo_files:
        for fpath, content in sorted(inst.repo_files.items()):
            ftok = count_tokens(content)
            n_lines = content.count("\n") + 1
            parts.append(f"<h3>{esc(fpath)} <span class='meta'>({ftok:,} tok, {n_lines} lines)</span></h3>")
            parts.append(f'<pre class="searchable">{esc(content)}</pre>')
    else:
        parts.append("<p class='meta'>(empty — run extract-repo to populate)</p>")
    parts.append(f"</div>")

    # QA Pairs
    parts.append(f'<h2 class="open" onclick="toggleSection(\'qa\')">QA Pairs ({len(inst.qa_pairs)})</h2>')
    parts.append(f'<div id="qa" class="section open">')
    for qp in inst.qa_pairs:
        dc = f"diff-{qp.difficulty}"
        synth = " [synthesis]" if qp.requires_synthesis else ""
        parts.append(f'<div class="qa-pair">')
        parts.append(
            f"<b>{esc(qp.qa_id)}</b> "
            f"<span class='{dc}'>[{qp.difficulty}]</span> "
            f"<span class='meta'>{qp.num_linked_facts} facts, {qp.signal_tokens} signal tok{synth}</span>"
        )
        parts.append(f"<p><b>Q:</b> <span class='searchable'>{esc(qp.question)}</span></p>")
        parts.append(f"<p><b>A:</b> <span class='searchable'>{esc(qp.gold_answer)}</span></p>")
        linked = [fact_map[fid] for fid in qp.linked_fact_ids if fid in fact_map]
        if linked:
            parts.append("<ul>")
            for lf in linked:
                cls = "fact-disc" if lf.discoverable else "fact-ext"
                parts.append(f"<li class='{cls}'>{esc(lf.id)} [{esc(lf.category)}]: {esc(lf.content[:200])}</li>")
            parts.append("</ul>")
        parts.append("</div>")
    parts.append("</div>")

    # Grounding facts
    parts.append(f'<h2 onclick="toggleSection(\'facts\')">Grounding Facts ({len(inst.grounding_facts)}, {tb["grounding_facts"]:,} tokens)</h2>')
    parts.append(f'<div id="facts" class="section">')
    for f in inst.grounding_facts:
        cls = "fact-disc" if f.discoverable else "fact-ext"
        parts.append(f"<p class='{cls}'><b>{esc(f.id)}</b> [{esc(f.category)}] {f.relevance}: <span class='searchable'>{esc(f.content)}</span></p>")
    parts.append("</div>")

    # Distractors
    by_source: dict = {}
    for d in inst.distractors:
        by_source.setdefault(d.source, []).append(d)
    parts.append(f'<h2 onclick="toggleSection(\'dist\')">Distractors ({len(inst.distractors)}, {tb["distractors"]:,} tokens)</h2>')
    parts.append(f'<div id="dist" class="section">')
    for source, ds in sorted(by_source.items()):
        parts.append(f"<h3>{esc(source)} ({len(ds)})</h3><ul>")
        for d in ds:
            parts.append(f"<li class='meta'>{esc(d.id)}: <span class='searchable'>{esc(d.content[:200])}</span></li>")
        parts.append("</ul>")
    parts.append("</div>")

    # Token breakdown
    parts.append(f'<h2 class="open" onclick="toggleSection(\'tokens\')">Token Breakdown</h2>')
    parts.append(f'<div id="tokens" class="section open"><table>')
    for k, v in tb.items():
        pct = v / max(1, tb["total"]) * 100
        bar_w = int(pct * 3)
        parts.append(
            f"<tr><td>{k}</td><td style='text-align:right'>{v:,}</td>"
            f"<td><span class='token-bar' style='width:{bar_w}px'></span> {pct:.1f}%</td></tr>"
        )
    parts.append("</table></div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Review export mode — human-readable markdown with repo code
# ---------------------------------------------------------------------------

def _find_relevant_repo_snippets(
    inst: CodingQAInstance,
    fact: "GroundingFact",
    max_snippet_lines: int = 40,
) -> List[Tuple[str, str]]:
    """Find repo file snippets relevant to a grounding fact.

    Heuristic: search for key identifiers from the fact content in repo files.
    Returns list of (file_path, snippet) tuples.
    """
    if not inst.repo_files:
        return []

    # Extract identifiers from fact: words that look like code (snake_case, CamelCase, dotted)
    import re
    code_words = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_.]{4,}\b', fact.content))
    # Filter to likely code identifiers (not plain English)
    code_ids = {w for w in code_words if '_' in w or '.' in w or (w[0].isupper() and any(c.islower() for c in w))}
    if not code_ids:
        return []

    snippets = []
    for fpath, content in sorted(inst.repo_files.items()):
        lines = content.split("\n")
        # Find lines matching any code identifier
        match_lines = []
        for i, line in enumerate(lines):
            for cid in code_ids:
                if cid in line:
                    match_lines.append(i)
                    break

        if not match_lines:
            continue

        # Build snippets around matches (context window)
        regions = []
        for ml in match_lines:
            start = max(0, ml - 5)
            end = min(len(lines), ml + max_snippet_lines - 5)
            regions.append((start, end))

        # Merge overlapping regions
        merged = [regions[0]]
        for s, e in regions[1:]:
            if s <= merged[-1][1] + 3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        for start, end in merged[:2]:  # max 2 snippets per file
            numbered = "\n".join(
                f"{i+1:4d} | {lines[i]}" for i in range(start, min(end, len(lines)))
            )
            snippets.append((fpath, numbered))

    return snippets[:3]  # max 3 total snippets per fact


def _extract_patch_context(patch: str, max_lines: int = 60) -> str:
    """Extract a readable summary of the gold patch."""
    if not patch:
        return "(no patch available)"
    lines = patch.split("\n")
    if len(lines) <= max_lines:
        return patch
    # Show first and last hunks
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def export_review_markdown(inst: CodingQAInstance) -> str:
    """Generate a human-readable markdown review of one instance."""
    from .types.schemas import GroundingFact

    tb = _token_breakdown(inst)
    fact_map = {f.id: f for f in inst.grounding_facts}
    repo_name = inst.instance_id.split("__")[1] if "__" in inst.instance_id else "unknown"

    parts = []

    # Header
    parts.append(f"# Instance: {inst.instance_id}\n")
    parts.append(f"| Field | Value |")
    parts.append(f"|-------|-------|")
    parts.append(f"| Repo | `{repo_name}` |")
    parts.append(f"| QA Pairs | {len(inst.qa_pairs)} |")
    parts.append(f"| Grounding Facts | {len(inst.grounding_facts)} ({inst.num_critical_facts} critical, {inst.num_external_facts} memory-required) |")
    parts.append(f"| Distractors | {len(inst.distractors)} |")
    parts.append(f"| Total Tokens | {tb['total']:,} |")
    parts.append(f"| Repo Files | {len(inst.repo_files)} ({tb['repo_files']:,} tokens) |")
    parts.append(f"| Image | `{inst.image_name or '(none)'}` |")
    parts.append("")

    # Task prompt (what the agent sees as the bug report)
    parts.append("## Bug Report (Task Prompt)\n")
    parts.append(f"> {inst.task_prompt}\n")

    # Gold patch (what actually needs to change)
    parts.append("## Gold Patch\n")
    parts.append("```diff")
    parts.append(_extract_patch_context(inst.patch, max_lines=80))
    parts.append("```\n")

    # Memory files (developer notes the agent has access to)
    parts.append(f"## Memory Files ({len(inst.memory_files)} files)\n")
    for fname, content in inst.memory_files.items():
        ftok = count_tokens(content)
        parts.append(f"### {fname} ({ftok:,} tokens)\n")
        # Show first 2000 chars
        preview = content[:2000]
        parts.append(f"```")
        parts.append(preview)
        if len(content) > 2000:
            parts.append(f"\n... ({len(content) - 2000} chars truncated)")
        parts.append("```\n")

    # QA Pairs with linked facts and repo code
    parts.append(f"## QA Pairs ({len(inst.qa_pairs)})\n")

    for qp in inst.qa_pairs:
        mem_tag = " (MEMORY-REQUIRED)" if qp.requires_memory else " (discoverable)"
        synth_tag = " [synthesis]" if qp.requires_synthesis else ""

        parts.append(f"### {qp.qa_id} [{qp.difficulty}]{mem_tag}{synth_tag}\n")
        parts.append(f"**Question:** {qp.question}\n")
        parts.append(f"**Gold Answer:** {qp.gold_answer}\n")

        # Linked grounding facts
        linked = [fact_map[fid] for fid in qp.linked_fact_ids if fid in fact_map]
        if linked:
            parts.append("**Linked Facts:**\n")
            for lf in linked:
                disc_tag = "discoverable" if lf.discoverable else "MEMORY-ONLY"
                parts.append(f"- **{lf.id}** [{lf.category}, {disc_tag}]: {lf.content}\n")

                # Show relevant repo code for this fact
                snippets = _find_relevant_repo_snippets(inst, lf)
                if snippets:
                    for fpath, snippet in snippets:
                        parts.append(f"  <details><summary>Relevant code: <code>{fpath}</code></summary>\n")
                        parts.append(f"  ```python")
                        parts.append(f"  {snippet}")
                        parts.append(f"  ```")
                        parts.append(f"  </details>\n")

        parts.append("---\n")

    # Key repo files (show the most referenced ones)
    if inst.repo_files:
        # Find which files are most relevant (appear in referenced_files or patch)
        patch_files = set()
        for line in inst.patch.split("\n"):
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                patch_files.add(line.split("/", 1)[-1] if "/" in line else "")

        parts.append(f"## Key Repo Files\n")
        shown = 0
        for fpath in sorted(inst.repo_files.keys()):
            # Prioritize patch-related files
            is_patch_file = any(fpath.endswith(pf) for pf in patch_files if pf)
            if not is_patch_file and shown >= 3:
                continue

            content = inst.repo_files[fpath]
            n_lines = content.count("\n") + 1
            ftok = count_tokens(content)

            tag = " **(modified by patch)**" if is_patch_file else ""
            parts.append(f"### `{fpath}`{tag} ({ftok:,} tokens, {n_lines} lines)\n")

            # Show first 100 lines for patch files, 40 for others
            max_lines = 100 if is_patch_file else 40
            preview_lines = content.split("\n")[:max_lines]
            parts.append("```python")
            parts.append("\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(preview_lines)))
            if n_lines > max_lines:
                parts.append(f"     | ... ({n_lines - max_lines} more lines)")
            parts.append("```\n")
            shown += 1

        if len(inst.repo_files) > shown:
            parts.append(f"*({len(inst.repo_files) - shown} more repo files not shown)*\n")

    # Distractor summary
    by_source: dict = {}
    for d in inst.distractors:
        by_source.setdefault(d.source, []).append(d)
    parts.append(f"## Distractors ({len(inst.distractors)} total)\n")
    parts.append("| Source | Count | Example |")
    parts.append("|--------|-------|---------|")
    for source, ds in sorted(by_source.items()):
        example = ds[0].content[:100].replace("|", "\\|").replace("\n", " ")
        parts.append(f"| {source} | {len(ds)} | {example}... |")
    parts.append("")

    return "\n".join(parts)


def export_review_batch(
    instances: List[CodingQAInstance],
    output_dir: str,
    per_repo: bool = True,
) -> None:
    """Export multiple instances as markdown files for human review.

    If per_repo=True, groups instances by repo into separate files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if per_repo:
        by_repo: dict = {}
        for inst in instances:
            repo = inst.instance_id.split("__")[1] if "__" in inst.instance_id else "unknown"
            by_repo.setdefault(repo, []).append(inst)

        for repo, insts in sorted(by_repo.items()):
            parts = [f"# {repo} — {len(insts)} instances\n"]
            for inst in insts[:5]:  # max 5 per repo file
                parts.append(export_review_markdown(inst))
                parts.append("\n" + "=" * 80 + "\n")
            if len(insts) > 5:
                parts.append(f"\n*({len(insts) - 5} more instances not shown)*\n")

            fname = f"review_{repo}.md"
            (out / fname).write_text("\n".join(parts))
            print(f"  {fname}: {len(insts)} instances")
    else:
        for i, inst in enumerate(instances):
            md = export_review_markdown(inst)
            fname = f"review_{i:04d}.md"
            (out / fname).write_text(md)

    print(f"\nExported to {output_dir}/")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_instances(path: str, limit: Optional[int] = None) -> List[CodingQAInstance]:
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(CodingQAInstance(**json.loads(line)))
            if limit and len(instances) >= limit:
                break
    return instances


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coding QA Instance Viewer — inspect benchmark instances"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to coding_qa_instances.jsonl",
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="View instance at this index (0-based)",
    )
    parser.add_argument(
        "--id", type=str, default=None,
        help="View instance with this ID",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show summary table of all instances",
    )
    parser.add_argument(
        "--agent-view", action="store_true",
        help="Show only what an evaluation agent sees (no labels, no gold answers)",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate self-contained HTML output",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write output to file (for --html mode)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full repo file contents (not just first 30 lines)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max instances to load (for --summary mode)",
    )
    parser.add_argument(
        "--export-review", action="store_true",
        help="Export human-readable markdown review files (one per repo)",
    )

    args = parser.parse_args()

    # Export review mode
    if args.export_review:
        instances = load_instances(args.input, limit=args.limit)
        output_dir = args.output or "review_export"
        export_review_batch(instances, output_dir, per_repo=True)
        return

    # Summary mode
    if args.summary:
        instances = load_instances(args.input, limit=args.limit)
        output = _view_summary(instances)
        if args.output:
            # Strip ANSI for file output
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', output)
            Path(args.output).write_text(clean)
            print(f"Summary written to {args.output}")
        else:
            print(output)
        return

    # Single instance mode
    instances = load_instances(args.input, limit=args.limit)

    inst = None
    if args.id:
        for i in instances:
            if i.instance_id == args.id:
                inst = i
                break
        if inst is None:
            print(f"Instance '{args.id}' not found in {len(instances)} instances")
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(instances):
            print(f"Index {args.index} out of range (0-{len(instances) - 1})")
            sys.exit(1)
        inst = instances[args.index]
    else:
        # Default to first instance
        if not instances:
            print("No instances found in file")
            sys.exit(1)
        inst = instances[0]

    # Generate output
    if args.html:
        output = _view_html(inst)
        if args.output:
            Path(args.output).write_text(output)
            print(f"HTML written to {args.output}")
        else:
            print(output)
    elif args.agent_view:
        print(_view_agent(inst))
    else:
        print(_view_terminal(inst, full_repo=args.full))


if __name__ == "__main__":
    main()
