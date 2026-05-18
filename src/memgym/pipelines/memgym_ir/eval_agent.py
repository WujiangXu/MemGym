"""Backward compat shim — real code moved to eval/agent.py."""
from .eval.agent import *  # noqa: F401,F403

if __name__ == "__main__":
    from .eval.agent import main
    main()
