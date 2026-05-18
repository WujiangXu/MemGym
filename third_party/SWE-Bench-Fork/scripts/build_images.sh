DATASET_NAME=$1 
if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME="princeton-nlp/SWE-bench_Lite"
    echo "Using default dataset name: $DATASET_NAME"
fi
SPLIT=$2
if [ -z "$SPLIT" ]; then
    SPLIT="test"
    echo "Using default split: $SPLIT"
fi

MAX_WORKERS=$3
if [ -z "$MAX_WORKERS" ]; then
    MAX_WORKERS=4
    echo "Using default max workers: $MAX_WORKERS"
fi

RUN_ID="build-images-${DATASET_NAME//\//__}-${SPLIT}"
echo "Using dataset name: $DATASET_NAME"
echo "Using split: $SPLIT"
echo "Using max workers: $MAX_WORKERS"
echo "Using run id: $RUN_ID"
echo "================================================"

python -m swebench.harness.run_evaluation \
    --dataset_name $DATASET_NAME \
    --split $SPLIT \
    --predictions_path gold \
    --max_workers $MAX_WORKERS \
    --run_id $RUN_ID \
    --cache_level instance
