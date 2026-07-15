# Historical Recorded-Session Evaluation

The authoritative current-product process is documented in
[Evaluation Methodology](eval.md). This page covers the separate historical
fixture runner retained for schema and evaluator compatibility.

## Scope

`scripts/eval_live_sessions.py` reads explicitly selected JSONL fixtures using
the historical live-session schema. It does not run current retrieval, prompt
rendering, or product generation, so its output is not the current Dashboard
result.

Committed fixtures under `demo/evaluation/` are synthetic or redacted. Raw
records under `data/eval/`, local Chroma excerpts, and workspace sessions remain
private local data.

## Dry-run inspection

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path data/eval/historical-fixture-results.jsonl \
  --dry-run
```

Dry run validates selection and manifests without evaluator scoring. Provider
scoring requires the explicit `--allow-provider-calls` flag, configured
credentials, network access, and an approved public or redacted payload.

## Interpretation

Historical fixture results validate runner behavior, metric schemas, and
failure handling. They do not measure the current retrieval index or establish
artistic quality, usability, accessibility, security, or broad statistical
performance.
