#!/usr/bin/env python
"""Run the SWE-bench / SWE-Gym evaluation track.

Thin convenience wrapper around ``memgym.gym.swe_bench.evaluate:main`` — the
same entry point exposed as the ``memgym-evaluate`` console script and as
``python -m memgym.gym.swe_bench``. Requires ``pip install -e ".[swe]"`` first.
See ``docs/swe_bench.md`` for the full flag reference.
"""
from memgym.gym.swe_bench.evaluate import main

if __name__ == "__main__":
    main()
