#!/bin/bash
# =============================================================================
# Official SWE-bench evaluation using swebench harness
#
# Uses the same Docker-based evaluation as published SWE-bench benchmarks,
# ensuring identical conditions and accurate scoring.
#
# Usage:
#   ./examples/swe_bench/run_official_eval.sh <predictions_path> [dataset] [run_id]
#
# Examples:
#   ./examples/swe_bench/run_official_eval.sh results/test_baseline/preds.json
#   ./examples/swe_bench/run_official_eval.sh results/comparison_*/none/preds.json lite memgym-none
#   ./examples/swe_bench/run_official_eval.sh results/my_test/preds.json verified my-run
# =============================================================================

set -euo pipefail

PREDS_PATH="${1:?Usage: $0 <predictions_path> [dataset] [run_id]}"
DATASET="${2:-lite}"
RUN_ID="${3:-memgym-eval-$(date +%Y%m%d_%H%M%S)}"

# Map dataset name to full SWE-bench dataset name
case "$DATASET" in
    lite)    DATASET_NAME="princeton-nlp/SWE-bench_Lite" ;;
    verified) DATASET_NAME="princeton-nlp/SWE-bench_Verified" ;;
    full)    DATASET_NAME="princeton-nlp/SWE-bench" ;;
    *)       DATASET_NAME="$DATASET" ;;
esac

echo "======================================================================"
echo "Official SWE-bench Evaluation"
echo "======================================================================"
echo "Predictions: $PREDS_PATH"
echo "Dataset:     $DATASET_NAME"
echo "Run ID:      $RUN_ID"
echo "======================================================================"

# Verify predictions file exists
if [ ! -f "$PREDS_PATH" ]; then
    echo "ERROR: Predictions file not found: $PREDS_PATH"
    exit 1
fi

# Show predictions summary
echo ""
echo "Predictions summary:"
python3 -c "
import json
with open('$PREDS_PATH') as f:
    preds = json.load(f)
if isinstance(preds, dict):
    preds = list(preds.values())
print(f'  Total predictions: {len(preds)}')
for p in preds:
    patch = p.get('model_patch', '')
    print(f'  {p[\"instance_id\"]}: {len(patch)} chars')
    if not patch.strip():
        print(f'    WARNING: Empty patch!')
"
echo ""

# Run official SWE-bench evaluation
REPORT_DIR="$(dirname "$PREDS_PATH")/swebench_reports"
mkdir -p "$REPORT_DIR"

echo "Running evaluation (this may take a while)..."
echo "Reports will be saved to: $REPORT_DIR"
echo ""

python3 -c "
from swebench.harness.run_evaluation import main

main(
    dataset_name='$DATASET_NAME',
    split='test',
    instance_ids=None,
    predictions_path='$PREDS_PATH',
    max_workers=4,
    force_rebuild=False,
    cache_level='env',
    clean=False,
    open_file_limit=4096,
    run_id='$RUN_ID',
    timeout=1800,
    namespace=None,
    rewrite_reports=False,
    modal=False,
    report_dir='$REPORT_DIR',
)
"

echo ""
echo "======================================================================"
echo "Evaluation complete!"
echo "======================================================================"
echo ""

# Show results if report exists
REPORT_FILE="$REPORT_DIR/$RUN_ID.json"
if [ -f "$REPORT_FILE" ]; then
    echo "Results from $REPORT_FILE:"
    python3 -c "
import json
with open('$REPORT_FILE') as f:
    report = json.load(f)

resolved = report.get('resolved', [])
total = report.get('total', 0) or len(report.get('submitted', []))
print(f'  Resolved: {len(resolved)}/{total}')
if total > 0:
    print(f'  Resolution rate: {100*len(resolved)/total:.1f}%')
print()
if resolved:
    print('  Resolved instances:')
    for inst in resolved:
        print(f'    - {inst}')
"
else
    echo "Report file not found. Check $REPORT_DIR for available reports."
    ls -la "$REPORT_DIR/" 2>/dev/null
fi
