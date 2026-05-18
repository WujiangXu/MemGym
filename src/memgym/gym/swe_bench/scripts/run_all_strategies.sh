#!/bin/bash
# Run all memory strategies on the same SWE-bench instances for comparison.
#
# Usage:
#   ./scripts/run_all_strategies.sh [SLICE] [DATASET] [MODEL]
#
# Examples:
#   ./scripts/run_all_strategies.sh 0:5 lite openai/gpt-5-mini   # 5 instances, lite dataset
#   ./scripts/run_all_strategies.sh 0:1 lite openai/gpt-5-mini   # Quick smoke test
#   ./scripts/run_all_strategies.sh 0:10 verified openai/gpt-5    # 10 verified instances

set -e

SLICE="${1:-0:5}"
DATASET="${2:-lite}"
MODEL="${3:-openai/gpt-5-mini}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/comparison_${TIMESTAMP}"

STRATEGIES=(
    "none"
    "naive"
    "observation_masking"
    "llm_summarizing"
    "structured_summary"
    "pipeline_masking_summarizing"
    "pipeline_masking_structured"
    "sliding_window"
)

# Check API key based on model provider
case "$MODEL" in
    anthropic/*)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "ERROR: ANTHROPIC_API_KEY not set (required for anthropic/ models)"
            exit 1
        fi
        ;;
    *)
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "ERROR: OPENAI_API_KEY not set"
            exit 1
        fi
        ;;
esac

echo "============================================================"
echo "MemGym Strategy Comparison"
echo "============================================================"
echo "Dataset:    $DATASET"
echo "Slice:      $SLICE"
echo "Model:      $MODEL"
echo "Output:     $BASE_DIR"
echo "Strategies: ${#STRATEGIES[@]}"
echo "============================================================"
echo ""

mkdir -p "$BASE_DIR"

FAILED=0
for i in "${!STRATEGIES[@]}"; do
    strategy="${STRATEGIES[$i]}"
    idx=$((i + 1))
    total=${#STRATEGIES[@]}

    echo ""
    echo "============================================================"
    echo "[$idx/$total] Running strategy: $strategy"
    echo "============================================================"

    if python gym/swe_bench/evaluate.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --memory "$strategy" \
        --slice "$SLICE" \
        --output "$BASE_DIR/$strategy" \
        --verbose; then
        echo "Strategy $strategy completed successfully."
    else
        echo "WARNING: Strategy $strategy failed!"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "All strategies completed ($FAILED failures)"
echo "============================================================"
echo ""

# Print comparison table
python gym/swe_bench/scripts/compare.py "$BASE_DIR"
