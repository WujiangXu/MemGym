"""
MemGym setup configuration.

Install with:
    pip install -e .
    # or
    uv pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="memgym",
    version="0.2.0",
    description="A modular framework for testing memory abilities of AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MemGym Team",
    author_email="",
    url="https://github.com/WujiangXu/MemGym",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "tau2": [
            "fs",
            "rich",
            "plotly>=6.0.0",
            "scikit-learn>=1.6.1",
            "tabulate>=0.9.0",
            "fastapi>=0.115.11",
            "uvicorn>=0.34.0",
            "pydantic-argparse>=0.10.0",
            "pandas>=2.2.3",
            "psutil>=7.0.0",
            "loguru>=0.7.3",
            "docstring-parser>=0.16",
            "litellm>=1.65.0",
            "tenacity>=9.0.0",
            "matplotlib>=3.10.1",
            "seaborn>=0.13.2",
            "redis>=5.2.1",
            "deepdiff>=8.4.2",
            "addict>=2.4.0",
            "PyYAML>=6.0.2",
            "toml>=0.10.2",
            "langfuse>=2.60.7",
            "gymnasium>=1.2.2",
        ],
        "swe": [
            "beautifulsoup4",
            "chardet",
            "datasets",
            "docker",
            "ghapi",
            "GitPython",
            "python-dotenv",
            "requests",
            "rich",
            "tenacity",
            "tqdm",
            "unidiff",
            "openai",
            "anthropic",
            "tiktoken",
            "transformers",
            "jedi",
            "litellm>=1.65.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        # Training stack — versions floor at torch 2.4+, transformers
        # 4.44+, trl 0.10+, peft 0.12+, accelerate 0.34+. Pinned loosely
        # so contributors can recreate the venv on their own hardware.
        "train": [
            "torch>=2.4.0",
            "transformers>=4.44.0",
            "trl>=0.10.0",
            "peft>=0.12.0",
            "accelerate>=0.34.0",
            "datasets",
            "bitsandbytes",
            "sentencepiece",
            "huggingface-hub>=0.25.0",
            "wandb>=0.16.0",
        ],
        # Agentic-RL via VeRL. Pinned at v0.7.1 (latest stable). vllm
        # >=0.7.0 is explicit because verl 0.7's floor was raised but
        # pip may resolve an older cached vllm otherwise. Install in an
        # isolated venv if you also use the base "train" extras —
        # numpy<2 (VeRL) conflicts with numpy>=2 elsewhere.
        "rl-way-a": [
            "verl==0.7.1",
            "vllm>=0.7.0",
            "ray[default]>=2.10.0",
            "pyarrow>=15.0.0",
            "omegaconf>=2.3.0",
            "hydra-core>=1.3.0",
        ],
        # Lightweight extras for `memgym-eval-rm`. Pulled separately so
        # users who only want to score the reward model do not need the
        # full training stack.
        "eval": [
            "huggingface-hub>=0.25.0",
            "datasets",
            "transformers>=4.44.0",
            "peft>=0.12.0",
            "bitsandbytes",
            "accelerate>=0.34.0",
            "torch>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memgym-evaluate=memgym.gym.swe_bench.evaluate:main",
            "memgym-eval-rm=memgym.training.eval.rm_cli:main",
            "memgym-eval-memory=memgym.training.eval.memory_eval_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="memory agents llm benchmark evaluation",
    project_urls={
        "Documentation": "https://github.com/WujiangXu/MemGym",
        "Source": "https://github.com/WujiangXu/MemGym",
        "Tracker": "https://github.com/WujiangXu/MemGym/issues",
    },
)
