"""Backward compat shim — real code moved to generators/question_grower.py."""
from .generators.question_grower import *  # noqa: F401,F403
from .generators.question_grower import main, grow_batch, grow_question  # noqa: F401

if __name__ == "__main__":
    main()
