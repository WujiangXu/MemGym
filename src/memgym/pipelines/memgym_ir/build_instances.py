"""Backward compat shim — real code moved to generators/build_instances.py."""
from .generators.build_instances import *  # noqa: F401,F403
from .generators.build_instances import main  # noqa: F401

if __name__ == "__main__":
    main()
