"""
Coding-Synthetic pipeline: Convert SWE-smith instances to multi-turn memory-annotated benchmarks.

Usage:
    python pipelines/run_coding_synthetic.py --output ./output --limit 10
"""

from .pipeline import run_pipeline, run_pipeline_from_args, PipelineConfig

__all__ = ["run_pipeline", "run_pipeline_from_args", "PipelineConfig"]
