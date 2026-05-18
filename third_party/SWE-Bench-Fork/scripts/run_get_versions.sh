
REPO_NAME=$1  # e.g., getmoto__moto
INSTANCE_PATH=/SWE-Bench/data/raw/${REPO_NAME}.jsonl
OUTPUT_DIR=/SWE-Bench/data/interim/versioned
CONDA_PATH=/miniconda3/condabin/conda
TESTBED_PATH=/SWE-Bench/data/testbed

pushd swebench/versioning

python get_versions.py \
    --instances_path $INSTANCE_PATH \
    --retrieval_method github \
    --conda_env temp \
    --num_workers 4 \
    --path_conda $CONDA_PATH \
    --output_dir $OUTPUT_DIR \
    --testbed $TESTBED_PATH

popd

OUTPUT_PATH=$OUTPUT_DIR/${REPO_NAME}_versions.json
python3 scripts/filter_empty_version.py $OUTPUT_PATH
