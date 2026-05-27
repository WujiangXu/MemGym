# Minimal shim so `pip install -e .` works on older pip versions.
# All packaging metadata lives in pyproject.toml (PEP 621).
from setuptools import setup

setup()
