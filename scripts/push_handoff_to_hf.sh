#!/usr/bin/env bash
# Pushes data/s3_handoff/ to the private HF dataset under a coding_s3/
# prefix so it coexists with the IR track's artifacts in the same repo.
#
# Prerequisites:
#   - `hf` CLI installed (https://huggingface.co/docs/huggingface_hub/guides/cli)
#   - `hf auth login` has been run with a token that has write access to
#     the target dataset repo.
#   - The target repo `MemGym/memgym_data` exists and is set to private.
#
# Usage:
#   bash scripts/push_handoff_to_hf.sh
#
# Final layout in the HF repo:
#   coding_s3/
#     MANIFEST.md
#     MANIFEST.sha256
#     coding_qa_v4/...
#     s2/...
#     s3/...
#     s3_diag/...

set -euo pipefail

DATASET="MemGym/memgym_data"
SRC="${REPO_ROOT}/data/s3_handoff"
REMOTE_PREFIX="coding_s3"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC does not exist. Run pull_s3_handoff_from_ec2.sh first." >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not on PATH. Install with 'pip install huggingface_hub'" \
       "or 'brew install hf', then run 'hf auth login'." >&2
  exit 1
fi

echo "About to upload:"
du -sh "$SRC"
find "$SRC" -maxdepth 3 -type f | wc -l | xargs -I{} echo "  local files: {}"
echo "  target: $DATASET  (private dataset)"
echo "  remote path prefix: $REMOTE_PREFIX/"
echo

# `hf upload REPO LOCAL_PATH [PATH_IN_REPO]` — the third positional is the
# path prefix inside the HF repo, so no need to cd.
hf upload "$DATASET" "$SRC" "$REMOTE_PREFIX" \
  --repo-type=dataset \
  --commit-message "S3 coding-handoff snapshot: $STAMP"

echo
echo "Upload complete. Verify by re-downloading on any machine:"
echo "  hf download $DATASET --repo-type=dataset \\"
echo "    --include '$REMOTE_PREFIX/**' --local-dir /tmp/hf_verify"
echo "  cd /tmp/hf_verify/$REMOTE_PREFIX && sha256sum -c MANIFEST.sha256"
