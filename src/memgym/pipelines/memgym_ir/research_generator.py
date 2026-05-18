"""Backward compat shim — real code moved to generators/research_generator.py."""
from .generators.research_generator import *  # noqa: F401,F403
from .generators.research_generator import main  # noqa: F401

if __name__ == "__main__":
    main()
