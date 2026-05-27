"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: per-track smoke tests that need Docker, a "
        "WebArena server, or other external deps. Skipped by default; "
        "run explicitly with `pytest -m integration`.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.integration tests unless explicitly selected.

    Without this hook a plain ``pytest`` run executes the integration
    suite, which is what CI does — but those tests need Docker /
    WebArena / GPU and would noisily fail on a clean checkout. Honor
    the explicit ``-m integration`` opt-in instead.
    """
    if config.getoption("markexpr"):
        # User explicitly passed -m ...; let pytest decide.
        return
    skip_integration = pytest.mark.skip(
        reason="integration tests require external deps; pass `-m integration` to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
