REPO_NAME=$1 # e.g. getmoto__moto
VERSION_DATA=data/interim/versioned/${REPO_NAME}_versions.non-empty.jsonl

python swebench/harness/run_validation.py \
    --dataset_name $VERSION_DATA \
    --run_id test \
    --cache_level instance \
    --max_workers 8

# --force_rebuild true
