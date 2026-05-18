"""Backward compat shim — real code moved to eval/delayed_reveal.py."""
from .eval.delayed_reveal import *  # noqa: F401,F403

if __name__ == "__main__":
    from .eval.delayed_reveal import main
    main()
