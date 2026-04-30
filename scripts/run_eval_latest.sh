#!/usr/bin/env bash

set -euo pipefail

LATEST="${1:-4}"
OUTPUT_PATH="${2:-data/eval/ragas_latest${LATEST}_context_precision.jsonl}"

SSL_CERT_FILE="$(
  .venv/bin/python -c 'import certifi; print(certifi.where())'
)"

SSL_CERT_FILE="$SSL_CERT_FILE" \
.venv/bin/python -m dotenv run -- \
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path data/eval/live_sessions.jsonl \
  --output-path "$OUTPUT_PATH" \
  --latest "$LATEST" \
  --metric context_precision

jq '{cp: .metrics.context_precision, sources: .source_ids, domains: .domains}' \
  "$OUTPUT_PATH"
