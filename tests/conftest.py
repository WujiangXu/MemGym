"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

# Ensure src/ is on path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
