#!/bin/bash
# Run 3 x 50-instance evaluations sequentially on EC2
# Usage: bash scripts/run_50_sequential.sh

EC2="${EC2_HOST:-http://<YOUR_EC2_HOST>:30000}"
POLL=60

wait_for_job() {
    local JOB_ID=$1
    local NAME=$2
    echo "  Waiting for $NAME (job $JOB_ID)..."
    while true; do
        STATUS=$(curl -s "$EC2/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
        if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
            echo "  $NAME finished: $STATUS"
            return
        fi
        sleep "$POLL"
    done
}

submit_job() {
    local MEMORY=$1
    local OUTPUT=$2
    local NAME=$3
    echo ""
    echo "=== Submitting $NAME ==="
    JOB_ID=$(curl -s -X POST "$EC2/evaluate" -H "Content-Type: application/json" -d "{
      \"model\": \"bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0\",
      \"memory\": \"$MEMORY\",
      \"dataset\": \"swe-gym\",
      \"split\": \"train\",
      \"slice\": \"0:50\",
      \"output\": \"$OUTPUT\",
      \"workers\": 4,
      \"skip_eval\": true,
      \"resume\": true
    }" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null)
    echo "  Job ID: $JOB_ID"
    wait_for_job "$JOB_ID" "$NAME"
}

echo "=========================================="
echo "  50-Instance Sequential Evaluation"
echo "  EC2: $EC2"
echo "=========================================="

# Job 1: baseline is already running as 8e750abe
echo ""
echo "=== Baseline (already submitted as 8e750abe) ==="
wait_for_job "8e750abe" "baseline"

# Job 2: llm_summarizing
submit_job "llm_summarizing" "results/train50_sonnet45_llmsummarizing" "llm_summarizing"

# Job 3: structured_summary
submit_job "structured_summary" "results/train50_sonnet45_structured" "structured_summary"

echo ""
echo "=========================================="
echo "  All 3 jobs completed!"
echo "=========================================="
