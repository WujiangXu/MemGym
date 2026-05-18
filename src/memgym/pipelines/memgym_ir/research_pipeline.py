"""Backward compat shim — real code moved to orchestrators/research_pipeline.py."""
from .orchestrators.research_pipeline import run_research_pipeline, main  # noqa: F401

if __name__ == "__main__":
    main()
