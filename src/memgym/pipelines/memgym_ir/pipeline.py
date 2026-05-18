"""Backward compat shim — real code moved to orchestrators/dataset_pipeline.py and config.py."""
from .config import DifficultyConfig, PipelineConfig  # noqa: F401
from .orchestrators.dataset_pipeline import run_pipeline  # noqa: F401
