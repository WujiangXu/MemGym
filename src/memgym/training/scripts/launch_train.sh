#!/usr/bin/env bash
# Launch 8× A100 DDP training for the memory world model.
#
# Usage:
#   bash src/memgym/training/scripts/launch_train.sh \
#       --dataset /path/to/dataset.jsonl \
#       --output-dir /path/to/checkpoints
#
# Requires the memgym-world-model venv at ${VENV_DIR}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/accelerate_8gpu.yaml"
VENV="${VENV:-${VENV_DIR}}"

if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

exec accelerate launch \
    --config_file "${CONFIG}" \
    -m memgym.training.scripts.train \
    "$@"
