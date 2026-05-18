"""Shared HippoRAG memory core (used by both coding-synthetic and IR pipelines).

Named ``hipporag_core`` (not ``hipporag``) to avoid shadowing the upstream
``hipporag`` PyPI package — `memgym/memory/` is on ``sys.path`` for legacy
AMem reasons (see ``amem/model.py``), so a sub-package called ``hipporag``
would resolve as top-level ``hipporag``.
"""
from .system import HippoRAGSystem

__all__ = ["HippoRAGSystem"]
