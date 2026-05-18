#!/bin/bash
# setup_swebench.sh — One-shot setup for SWE-bench/SWE-Gym experiments
#
# Creates a venv, installs MemGym + dependencies, downloads the dataset,
# and optionally pulls Docker images.
#
# Usage:
#   bash scripts/setup_swebench.sh                              # defaults: swe-gym, slice 0:20
#   bash scripts/setup_swebench.sh --dataset swe-gym --slice 0:100
#   bash scripts/setup_swebench.sh --singularity                # skip Docker pull
#   bash scripts/setup_swebench.sh --venv ~/env/custom_name
#   bash scripts/setup_swebench.sh --offline-dir /shared/data   # prepare for offline use

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET="swe-gym"
SLICE="0:20"
VENV="$HOME/env/memgym"
SINGULARITY=false
OFFLINE_DIR=""
WORKERS=4

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2"; shift 2 ;;
        --slice)        SLICE="$2"; shift 2 ;;
        --venv)         VENV="$2"; shift 2 ;;
        --singularity)  SINGULARITY=true; shift ;;
        --offline-dir)  OFFLINE_DIR="$2"; shift 2 ;;
        --workers)      WORKERS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/setup_swebench.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET     Dataset: lite, verified, full, swe-gym, swe-smith (default: swe-gym)"
            echo "  --slice START:END     Instance slice (default: 0:20)"
            echo "  --venv PATH           Venv directory (default: ~/env/memgym)"
            echo "  --singularity         Skip Docker image pull (for HPC with Singularity)"
            echo "  --offline-dir PATH    Also cache dataset for offline transfer"
            echo "  --workers N           Parallel Docker pull workers (default: 4)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo " MemGym SWE-bench Setup"
echo "============================================"
echo " Dataset:     $DATASET"
echo " Slice:       $SLICE"
echo " Venv:        $VENV"
echo " Singularity: $SINGULARITY"
echo " Project dir: $PROJECT_DIR"
echo "============================================"
echo ""

# ── Step 1: Create venv + install dependencies ───────────────────────────────
echo "[1/4] Setting up Python virtual environment..."

if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    echo "  Venv already exists at $VENV, reusing it."
else
    echo "  Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "  Activated: $(which python)"

echo "  Installing MemGym + SWE-bench dependencies..."
pip install --upgrade pip -q
pip install -e "$PROJECT_DIR[swe]" -q
pip install "mini-swe-agent>=2.2.0" -q
pip install boto3 -q

echo "  Verifying installation..."
python -c "from memgym.gym.swe_bench.env import SWEMemoryEnv; print('  memgym.gym.swe_bench: OK')"
python -c "from memgym.memory import get_memory_model; print('  memgym.memory: OK')"
echo ""

# ── Step 2: Download HuggingFace dataset ──────────────────────────────────────
echo "[2/4] Downloading dataset..."

CACHE_ARGS="--dataset $DATASET"
if [ -n "$OFFLINE_DIR" ]; then
    CACHE_ARGS="$CACHE_ARGS --output $OFFLINE_DIR --list-images"
fi

python "$PROJECT_DIR/scripts/cache_dataset.py" $CACHE_ARGS
echo ""

# ── Step 3: Pull Docker images (unless --singularity) ────────────────────────
if [ "$SINGULARITY" = true ]; then
    echo "[3/4] Skipping Docker image pull (--singularity mode)"
    echo "  Singularity will lazily pull images on first use."
else
    echo "[3/4] Pulling Docker images for $DATASET [$SLICE]..."

    if ! command -v docker &> /dev/null; then
        echo "  WARNING: Docker not found. Skipping image pull."
        echo "  Install Docker or re-run with --singularity."
    else
        python "$PROJECT_DIR/scripts/pull_images.py" \
            --dataset "$DATASET" --slice "$SLICE" --workers "$WORKERS"
    fi
fi
echo ""

# ── Step 4: Summary + next steps ─────────────────────────────────────────────
echo "[4/4] Setup complete!"
echo ""
echo "============================================"
echo " Next steps"
echo "============================================"
echo ""
echo "1. Activate the venv:"
echo "   source $VENV/bin/activate"
echo ""
echo "2. Set your API key:"
echo "   export ANTHROPIC_API_KEY='sk-ant-...'    # Anthropic"
echo "   # OR: aws configure                       # Bedrock (set region us-west-2)"
echo "   # OR: export OPENAI_API_KEY='sk-...'      # OpenAI"
echo ""

if [ "$SINGULARITY" = true ]; then
    echo "3. Run a smoke test (Singularity):"
    echo "   python -m memgym.gym.swe_bench \\"
    echo "       --dataset $DATASET --slice $SLICE \\"
    echo "       --model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \\"
    echo "       --memory structured_summary --max-size 100 --condensation-ratio 0.75 --keep-first 1 \\"
    echo "       --env-type singularity --workers 2 \\"
    echo "       -o results/smoke_test --verbose"
else
    echo "3. Run a smoke test (Docker):"
    echo "   python -m memgym.gym.swe_bench \\"
    echo "       --dataset $DATASET --slice $SLICE \\"
    echo "       --model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \\"
    echo "       --memory structured_summary --max-size 100 --condensation-ratio 0.75 --keep-first 1 \\"
    echo "       --workers 4 \\"
    echo "       -o results/smoke_test --verbose"
fi

echo ""
echo "See docs/swegym/instructions.md for the full guide."
