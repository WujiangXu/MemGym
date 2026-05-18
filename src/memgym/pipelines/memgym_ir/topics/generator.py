"""Topic generation for scaling MemGym-IR from 20 → 1000+ instances.

Tier 1: Decompose 20 parent topics into ~4 sub-mechanisms each → ~80 topics.
Tier 2: Mine arxiv categories for mechanism-level topics → ~300 more.

Usage:
    # Generate Tier 1 topics (LLM-decomposed sub-mechanisms)
    python -m memgym.pipelines.memgym_ir.topics.generator \
        --tier 1 --model bedrock/us.anthropic.claude-opus-4-6-v1 \
        --output output_ir/topic_pool.jsonl

    # Generate Tier 2 topics (arxiv-category-mined)
    python -m memgym.pipelines.memgym_ir.topics.generator \
        --tier 2 --model bedrock/us.anthropic.claude-opus-4-6-v1 \
        --output output_ir/topic_pool.jsonl

    # List all topics in the pool
    python -m memgym.pipelines.memgym_ir.topics.generator --list \
        --output output_ir/topic_pool.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# The 20 parent topics from the dev set (Steps 1-5 verification runs)
# ─────────────────────────────────────────────────────────────────────
PARENT_TOPICS = [
    "exoplanet_atmospheres",
    "crispr",
    "dark_matter",
    "mrna_vaccines",
    "superconductors",
    "quantum_sensing",
    "fusion_plasma",
    "protein_folding",
    "computer_vision",
    "diffusion_models",
    "federated_learning",
    "nas",
    "self_supervised",
    "continual_learning",
    "coral_reefs",
    "lithium_batteries",
    "attention",
    "gnn",
    "nlp",
    "rl",
]

# ─────────────────────────────────────────────────────────────────────
# Tier 1: LLM-decomposed sub-mechanisms
# ─────────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
You are helping build a research-knowledge benchmark. Given a broad research \
area, decompose it into exactly {count} narrow sub-mechanisms that satisfy ALL \
of these constraints:

1. **Paper-dense**: Each sub-mechanism should have ≥50 published papers on \
arXiv or major venues. Think "technique X for task Y" not "obscure niche Z".
2. **Entity-disjoint**: The canonical researchers, models/systems, datasets, \
and institutions for each sub-mechanism should be DIFFERENT from siblings. \
(E.g., for "superconductors": cuprate researchers ≠ iron-pnictide researchers.)
3. **Searchable**: The sub-mechanism name should work as a search query in \
an academic paper database — use the vocabulary that paper titles actually use.
4. **Non-overlapping**: A paper that belongs to sub-mechanism A should rarely \
belong to sub-mechanism B.

Research area: {topic}

For each sub-mechanism, provide:
- `slug`: a lowercase_underscore identifier (3-5 words), suitable as a \
directory name and search query
- `display_name`: a human-readable label (5-10 words)
- `seed_queries`: 2-3 specific search queries that would retrieve papers \
about this sub-mechanism

Respond in JSON:
{{"sub_mechanisms": [
    {{"slug": "...", "display_name": "...", "seed_queries": ["...", "..."]}},
    ...
]}}
"""

DECOMPOSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_decomposition",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sub_mechanisms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string"},
                            "display_name": {"type": "string"},
                            "seed_queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["slug", "display_name", "seed_queries"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["sub_mechanisms"],
            "additionalProperties": False,
        },
    },
}


def decompose_topic(
    parent: str,
    client: LLMClient,
    count: int = 4,
) -> List[Dict]:
    """Decompose a parent topic into sub-mechanisms via LLM.

    Returns a list of dicts with keys: slug, display_name, seed_queries, parent.
    """
    # Convert slug to readable form for the prompt
    readable = parent.replace("_", " ")

    prompt = DECOMPOSE_PROMPT.format(topic=readable, count=count)
    data = client.get_json_completion(
        prompt=prompt,
        response_format=DECOMPOSE_SCHEMA,
        temperature=0.5,
    )

    subs = data.get("sub_mechanisms", [])
    # Tag each with parent and validate
    results = []
    for sub in subs:
        slug = sub.get("slug", "").strip().lower().replace(" ", "_").replace("-", "_")
        if not slug or len(slug) < 3:
            continue
        results.append({
            "slug": slug,
            "display_name": sub.get("display_name", slug),
            "seed_queries": sub.get("seed_queries", []),
            "parent": parent,
            "tier": 1,
        })

    logger.info(f"  {parent} → {len(results)} sub-mechanisms: "
                f"{[r['slug'] for r in results]}")
    return results


def generate_tier1(
    client: LLMClient,
    parents: Optional[List[str]] = None,
    count_per_parent: int = 4,
) -> List[Dict]:
    """Generate Tier 1 topics by decomposing parent topics.

    Returns flat list of sub-mechanism dicts.
    """
    parents = parents or PARENT_TOPICS
    all_subs = []
    for parent in parents:
        subs = decompose_topic(parent, client, count=count_per_parent)
        all_subs.extend(subs)
    return all_subs


# ─────────────────────────────────────────────────────────────────────
# Tier 2: arxiv-category-mined topics
# ─────────────────────────────────────────────────────────────────────

# Categories with large paper counts in the local DB, grouped by domain.
# ~60 categories → ~300 topics at 5/category.
ARXIV_CATEGORIES = [
    # CS — ML & AI
    "cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "cs.IR", "cs.RO",
    "cs.MA", "cs.SD", "cs.HC", "cs.CR", "cs.SE", "cs.DB", "cs.DC",
    "cs.SI", "cs.CY",
    # Statistics & Math
    "stat.ML", "stat.ME", "math.OC", "math.ST",
    # Physics
    "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.SR",
    "cond-mat.mtrl-sci", "cond-mat.supr-con", "cond-mat.str-el",
    "hep-ph", "hep-th", "quant-ph", "physics.optics",
    "physics.bio-ph", "physics.comp-ph",
    # Biology & Chemistry
    "q-bio.BM", "q-bio.NC", "q-bio.GN", "q-bio.QM",
    # Electrical engineering
    "eess.SP", "eess.IV", "eess.AS",
]

EXTRACT_MECHANISMS_PROMPT = """\
Below are the titles and abstracts of {n} recent papers from the arXiv \
category "{category}". For each paper, extract the ONE specific \
mechanism, method, or phenomenon it introduces or studies. Then group and \
deduplicate: output exactly {count} distinct research topics that together \
cover these papers without redundancy.

Papers:
{papers_text}

For each topic, provide:
- `slug`: a lowercase_underscore identifier (3-5 words), suitable as a \
search query in an academic paper database
- `display_name`: a human-readable label (5-10 words)
- `seed_queries`: 2-3 search queries that would retrieve papers about \
this specific topic

Respond in JSON:
{{"topics": [
    {{"slug": "...", "display_name": "...", "seed_queries": ["...", "..."]}},
    ...
]}}
"""

EXTRACT_MECHANISMS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "extract_mechanisms",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string"},
                            "display_name": {"type": "string"},
                            "seed_queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["slug", "display_name", "seed_queries"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["topics"],
            "additionalProperties": False,
        },
    },
}


def _sample_papers_from_category(
    category: str,
    n: int = 10,
    db_path: str = "",
) -> List[Dict]:
    """Sample recent papers from a specific arxiv category via local DB."""
    import sqlite3
    default_path = os.path.expanduser("~/.memgym/arxiv_papers.db")
    path = db_path or os.environ.get("LOCAL_ARXIV_DB", default_path)
    if not os.path.exists(path):
        logger.warning(f"Local arXiv DB not found at {path}")
        return []

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Sample recent papers from this category, preferring recent years
    rows = conn.execute(
        """SELECT title, abstract, year FROM papers
           WHERE categories LIKE ?
           ORDER BY year DESC, RANDOM()
           LIMIT ?""",
        (f"%{category}%", n),
    ).fetchall()
    conn.close()

    return [{"title": r["title"], "abstract": r["abstract"][:300], "year": r["year"]}
            for r in rows]


def extract_mechanisms_from_category(
    category: str,
    client: LLMClient,
    count: int = 5,
    sample_n: int = 10,
    db_path: str = "",
) -> List[Dict]:
    """Sample papers from an arxiv category and extract mechanism-level topics."""
    papers = _sample_papers_from_category(category, n=sample_n, db_path=db_path)
    if len(papers) < 3:
        logger.warning(f"  {category}: only {len(papers)} papers — skipping")
        return []

    papers_text = "\n\n".join(
        f"[{i+1}] ({p['year']}) {p['title']}\n{p['abstract']}"
        for i, p in enumerate(papers)
    )

    prompt = EXTRACT_MECHANISMS_PROMPT.format(
        n=len(papers), category=category, count=count, papers_text=papers_text,
    )
    data = client.get_json_completion(
        prompt=prompt,
        response_format=EXTRACT_MECHANISMS_SCHEMA,
        temperature=0.5,
    )

    topics = data.get("topics", [])
    results = []
    for t in topics:
        slug = t.get("slug", "").strip().lower().replace(" ", "_").replace("-", "_")
        if not slug or len(slug) < 3:
            continue
        results.append({
            "slug": slug,
            "display_name": t.get("display_name", slug),
            "seed_queries": t.get("seed_queries", []),
            "parent": category,
            "tier": 2,
        })

    logger.info(f"  {category} → {len(results)} topics: "
                f"{[r['slug'] for r in results]}")
    return results


def generate_tier2(
    client: LLMClient,
    categories: Optional[List[str]] = None,
    count_per_category: int = 5,
    sample_n: int = 10,
    db_path: str = "",
) -> List[Dict]:
    """Generate Tier 2 topics by mining arxiv categories."""
    cats = categories or ARXIV_CATEGORIES
    all_topics = []
    for cat in cats:
        topics = extract_mechanisms_from_category(
            cat, client, count=count_per_category,
            sample_n=sample_n, db_path=db_path,
        )
        all_topics.extend(topics)
    return all_topics


# ─────────────────────────────────────────────────────────────────────
# Topic pool I/O
# ─────────────────────────────────────────────────────────────────────

def save_topic_pool(topics: List[Dict], path: str, append: bool = True):
    """Save topics to a JSONL file. Deduplicates by slug."""
    existing = load_topic_pool(path) if append else []
    existing_slugs = {t["slug"] for t in existing}

    new_count = 0
    with open(path, "a" if append else "w") as f:
        for topic in topics:
            if topic["slug"] not in existing_slugs:
                f.write(json.dumps(topic) + "\n")
                existing_slugs.add(topic["slug"])
                new_count += 1

    logger.info(f"Saved {new_count} new topics to {path} "
                f"(total: {len(existing_slugs)})")
    return new_count


def load_topic_pool(path: str) -> List[Dict]:
    """Load topics from a JSONL file."""
    if not os.path.exists(path):
        return []
    topics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    return topics


def get_topic_slugs(path: str) -> List[str]:
    """Get just the slug list from a topic pool file."""
    return [t["slug"] for t in load_topic_pool(path)]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate topics for MemGym-IR scaling"
    )
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1,
                        help="Tier 1: decompose parents; Tier 2: arxiv-mined")
    parser.add_argument("--model", type=str,
                        default="bedrock/us.anthropic.claude-opus-4-6-v1",
                        help="LLM for topic decomposition")
    parser.add_argument("--output", type=str,
                        default="output_ir/topic_pool.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--count-per-parent", type=int, default=4,
                        help="Sub-mechanisms per parent topic (Tier 1)")
    parser.add_argument("--parents", nargs="+", default=None,
                        help="Override parent topics (default: all 20)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Override arxiv categories for Tier 2")
    parser.add_argument("--count-per-category", type=int, default=5,
                        help="Topics per arxiv category (Tier 2)")
    parser.add_argument("--sample-n", type=int, default=10,
                        help="Papers to sample per category (Tier 2)")
    parser.add_argument("--db-path", type=str, default="",
                        help="Path to local arxiv DB (Tier 2)")
    parser.add_argument("--list", action="store_true",
                        help="List existing topics in the pool")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pool (default: append)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list:
        topics = load_topic_pool(args.output)
        if not topics:
            print(f"No topics in {args.output}")
            return
        by_tier = {}
        for t in topics:
            tier = t.get("tier", "?")
            by_tier.setdefault(tier, []).append(t)
        for tier, ts in sorted(by_tier.items()):
            print(f"\nTier {tier}: {len(ts)} topics")
            by_parent = {}
            for t in ts:
                by_parent.setdefault(t.get("parent", "?"), []).append(t)
            for parent, subs in sorted(by_parent.items()):
                print(f"  {parent}:")
                for s in subs:
                    print(f"    {s['slug']:40s}  {s['display_name']}")
        print(f"\nTotal: {len(topics)} topics")
        return

    client = LLMClient(model=args.model)

    if args.tier == 1:
        print(f"Generating Tier 1 topics ({args.count_per_parent} per parent)...")
        topics = generate_tier1(
            client,
            parents=args.parents,
            count_per_parent=args.count_per_parent,
        )
    elif args.tier == 2:
        cats = args.categories or ARXIV_CATEGORIES
        print(f"Generating Tier 2 topics ({args.count_per_category} per category, "
              f"{len(cats)} categories)...")
        topics = generate_tier2(
            client,
            categories=cats,
            count_per_category=args.count_per_category,
            sample_n=args.sample_n,
            db_path=args.db_path,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    new = save_topic_pool(topics, args.output, append=not args.force)
    print(f"\nGenerated {len(topics)} topics, {new} new written to {args.output}")


if __name__ == "__main__":
    main()
