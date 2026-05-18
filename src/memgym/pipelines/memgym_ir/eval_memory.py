"""Backward compat shim — real code moved to eval/memory.py."""
from .eval.memory import *  # noqa: F401,F403

if __name__ == "__main__":
    from .eval.memory import main
    main()
