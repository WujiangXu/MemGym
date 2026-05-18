"""Backward compat shim — real code moved to utils/search_tools.py."""
from .utils.search_tools import *  # noqa: F401,F403
from .utils.search_tools import get_search_fn, search_ddg, search_wikipedia, combined_search  # noqa: F401
