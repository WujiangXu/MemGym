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
        # (see paper) training stack — SFT + offline-RL for the 8B summarizer.
        # Pinned deliberately loosely; the collaborator recreates the venv
        # on their own hardware. Versions proven in the training phase the class-weighted CE checkpoint SFT:
        # torch 2.4+, transformers 4.44+, trl 0.10+, peft 0.12+, accelerate 0.34+.
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
        # Way A only — agentic-RL via VeRL. Pinned at v0.7.1 (Mar 2026
        # stable, latest before the 0.8.0.dev branch). 0.5.0 was the
        # original pin but its scaffold draft hit verl 0.4.x-shaped APIs
        # that 0.5.0 already removed (Apr-2026 probe v3 confirmed the
        # drift: `AgentLoopBase.__init__` is DI-injected, `run` is sync,
        # `AsyncLLMServerManager.generate` takes token-id I/O). 0.7.1 is
        # the closest stable release to the local 0.8.0.dev reference at
        # ${HOME}/code/AutoMLE/verl/, so the rewrite can
        # cross-reference both. vllm>=0.7.0 added explicitly because
        # verl 0.7's requirements floor was raised but pip can still
        # resolve to an older cached vllm if Way B's stack pre-cached one.
        # numpy<2 conflict with Way B's numpy>=2 stack persists — install
        # Way A in an isolated venv (`memgym-way-a`) per the handoff doc.
        # Way B users do NOT need this group — keeps the base install lean.
        "rl-way-a": [
            "verl==0.7.1",
            "vllm>=0.7.0",
            "ray[default]>=2.10.0",
            "pyarrow>=15.0.0",
            "omegaconf>=2.3.0",
            "hydra-core>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memgym-evaluate=memgym.gym.swe_bench.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
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
