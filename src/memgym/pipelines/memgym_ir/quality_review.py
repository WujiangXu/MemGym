"""Backward compat shim — real code moved to eval/quality_review.py."""
from .eval.quality_review import *  # noqa: F401,F403

if __name__ == "__main__":
    from .eval.quality_review import main
    main()
