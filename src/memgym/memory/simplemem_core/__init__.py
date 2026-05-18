"""Shared SimpleMem memory core (used by both coding-synthetic and IR pipelines).

Named ``simplemem_core`` (not ``simplemem``) to avoid shadowing the upstream
``simplemem`` PyPI package — `memgym/memory/` is on ``sys.path`` for legacy
AMem reasons (see ``amem/model.py``), so a sub-package called ``simplemem``
would resolve as top-level ``simplemem`` and break ``from simplemem import
SimpleMemSystem``.
"""
from .system import SimpleMemBackend

__all__ = ["SimpleMemBackend"]
