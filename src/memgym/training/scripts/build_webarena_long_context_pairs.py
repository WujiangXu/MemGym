"""CLI shim around ``memgym.training.data.webarena_long_context_pairs``.

Walks one or more (baseline_cohort, ood_cohort) pairs under a WebArena
phase root and emits a single long-context pairs JSONL with rows whose
prompt-length distribution overlaps the v2 RM training distribution
(~22-38K tokens of accessibility-tree+action history + judgment turn).

Basic (single-root) usage::

    python -m memgym.training.scripts.build_webarena_long_context_pairs \\
        --webarena-root ${REPO_ROOT}/results/webarena/phase_d_prime \\
        --baseline-cohort paypal_none \\
        --ood-cohorts paypal_struct_ms10 paypal_struct_ms20 paypal_summ_ms15 \\
        --out reward_model_pairs_webarena_longctx.jsonl

Split-root usage (baseline and OOD live in different directories)::

    python -m memgym.training.scripts.build_webarena_long_context_pairs \\
        --baseline-root ${REPO_ROOT}/results/webarena/phase_d_prime \\
        --ood-root      ${REPO_ROOT}/results/webarena/prompt_fix_v2 \\
        --baseline-cohort-template "{app}_none" \\
        --ood-cohorts gitlab_struct_ms10 gmail_summ_ms10 paypal_summ_ms15 \\
        --out reward_model_pairs_webarena_v2.jsonl

When ``--baseline-root`` / ``--ood-root`` are provided they override
``--webarena-root`` for that side.  When ``--baseline-cohort-template``
is provided the per-OOD-cohort baseline dir is derived by substituting
``{app}`` with the app prefix extracted from the OOD cohort name
(e.g. ``gitlab_struct_ms10`` → app = ``gitlab`` →
baseline cohort = ``gitlab_none``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from memgym.training.data.reward_model_pairs import DEFAULT_KEEP_FIRST
from memgym.training.data.webarena_long_context_pairs import build_pairs_for_cohort

logger = logging.getLogger("build_webarena_long_context_pairs")

# Known app prefixes used in cohort names (longest-match wins).
_KNOWN_APPS = ("paypal", "gitlab", "gmail", "linear", "superhuman")


def _extract_app_from_cohort(cohort_name: str) -> Optional[str]:
    """Return the app token in a cohort name.

    Handles both bare ``<app>_<strategy>`` (e.g. ``gitlab_struct_ms10``)
    and prefixed ``<runid>_<app>_<strategy>`` (e.g.
    ``grid_idfix_gmail_struct_ms10``) by scanning the underscore-tokens
    for the first known app. Apps are pairwise distinct so order
    doesn't matter.
    """
    tokens = cohort_name.split("_")
    for tok in tokens:
        if tok in _KNOWN_APPS:
            return tok
    # Fallback: use the part before the first underscore.
    parts = cohort_name.split("_", 1)
    return parts[0] if parts else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build long-context WebArena OOD pairs (v2-RM-distribution-shaped).",
    )
    # ------------------------------------------------------------------ roots
    parser.add_argument(
        "--webarena-root", type=Path, default=None,
        help="Parent directory containing one subdir per cohort. "
             "Used as both baseline and OOD root when the split-root "
             "flags are not provided (backwards-compatible).",
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=None,
        help="Root directory for baseline cohorts. "
             "Overrides --webarena-root for the baseline side.",
    )
    parser.add_argument(
        "--ood-root", type=Path, default=None,
        help="Root directory for OOD cohorts. "
             "Overrides --webarena-root for the OOD side.",
    )
    # -------------------------------------------------------- cohort selection
    parser.add_argument(
        "--baseline-cohort", default=None,
        help="Single cohort dirname used as the BASELINE for all OOD "
             "cohorts (e.g. 'paypal_none'). Mutually exclusive with "
             "--baseline-cohort-template.",
    )
    parser.add_argument(
        "--baseline-cohort-template", default=None,
        metavar="TEMPLATE",
        help="Template for deriving the per-OOD baseline cohort dir. "
             "Use '{app}' as a placeholder that is filled with the app "
             "prefix of each OOD cohort (e.g. '{app}_none' → "
             "'gitlab_none' for OOD cohort 'gitlab_struct_ms10'). "
             "Mutually exclusive with --baseline-cohort.",
    )
    parser.add_argument(
        "--ood-cohorts", nargs="+", required=True,
        help="One or more cohort dirnames used as OOD strategies.",
    )
    # --------------------------------------------------------------- output
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output JSONL path. Overwritten if it exists.",
    )
    parser.add_argument(
        "--keep-first", type=int, default=DEFAULT_KEEP_FIRST,
        help="Number of leading messages always retained in the "
             "reconstructed view (matches the SWE/τ² builders).",
    )
    parser.add_argument(
        "--no-gate-compacted", action="store_true", default=False,
        help="Disable the was_compacted gate and emit every aligned "
             "step (legacy behaviour; use only for testing).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---------------------------------------------------------------- validate
    if args.baseline_cohort and args.baseline_cohort_template:
        logger.error("--baseline-cohort and --baseline-cohort-template are "
                     "mutually exclusive.")
        return 2
    if not args.baseline_cohort and not args.baseline_cohort_template:
        logger.error("One of --baseline-cohort or --baseline-cohort-template "
                     "is required.")
        return 2

    baseline_root: Optional[Path] = args.baseline_root or args.webarena_root
    ood_root: Optional[Path] = args.ood_root or args.webarena_root
    if baseline_root is None:
        logger.error("Provide --baseline-root (or --webarena-root) for the "
                     "baseline side.")
        return 2
    if ood_root is None:
        logger.error("Provide --ood-root (or --webarena-root) for the OOD "
                     "side.")
        return 2
    if not baseline_root.is_dir():
        logger.error("baseline root not a directory: %s", baseline_root)
        return 2
    if not ood_root.is_dir():
        logger.error("ood root not a directory: %s", ood_root)
        return 2

    gate_on_compacted = not args.no_gate_compacted

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overall: Counter = Counter()
    n_cohorts_ok = 0
    with args.out.open("w") as fh:
        for cohort_name in args.ood_cohorts:
            ood_dir = ood_root / cohort_name
            if not ood_dir.is_dir():
                logger.warning("skipping missing ood cohort: %s", ood_dir)
                overall["cohorts_skipped_missing"] += 1
                continue

            # Resolve baseline cohort dir for this OOD cohort.
            if args.baseline_cohort_template:
                app = _extract_app_from_cohort(cohort_name)
                if app is None:
                    logger.warning(
                        "cannot extract app from OOD cohort '%s' — skipping",
                        cohort_name,
                    )
                    overall["cohorts_skipped_app_unknown"] += 1
                    continue
                baseline_cohort_name = args.baseline_cohort_template.replace(
                    "{app}", app
                )
            else:
                baseline_cohort_name = args.baseline_cohort

            baseline_dir = baseline_root / baseline_cohort_name
            if not baseline_dir.is_dir():
                logger.warning(
                    "skipping missing baseline cohort: %s "
                    "(derived from OOD cohort '%s')",
                    baseline_dir, cohort_name,
                )
                overall["cohorts_skipped_missing_baseline"] += 1
                continue

            logger.info(
                "building pairs: %s vs %s (gate_compacted=%s)",
                baseline_cohort_name, cohort_name, gate_on_compacted,
            )
            stats = build_pairs_for_cohort(
                baseline_cohort=baseline_dir,
                ood_cohort=ood_dir,
                out_handle=fh,
                keep_first=args.keep_first,
                gate_on_compacted=gate_on_compacted,
            )
            for k, v in stats.items():
                overall[k] += v
            overall["cohorts_processed"] += 1
            n_cohorts_ok += 1
            logger.info("  cohort %s stats: %s", cohort_name,
                        {k: v for k, v in stats.items() if v})

    summary = {
        "out_path": str(args.out),
        "baseline_root": str(baseline_root),
        "ood_root": str(ood_root),
        "baseline_cohort": args.baseline_cohort,
        "baseline_cohort_template": args.baseline_cohort_template,
        "ood_cohorts": list(args.ood_cohorts),
        "cohorts_processed": n_cohorts_ok,
        "gate_on_compacted": gate_on_compacted,
        "stats": dict(overall),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if n_cohorts_ok > 0 and overall.get("pairs_written", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
