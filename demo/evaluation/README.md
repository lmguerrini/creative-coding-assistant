# Public Evaluation Evidence

This directory separates current-product summaries from sanitized historical
fixtures. Detailed methodology and commands live in
[docs/eval.md](../../docs/eval.md).

## Current-product evidence

- `current_product_ragas_evidence.json` is the canonical public-safe projection
  of the retained seven-case run.
- `current_product_ragas_evidence.schema.json` defines its public contract.
- `canonical_retrieval_report.json` records aggregate retrieval lineage and
  fingerprints without retrieved excerpt text.

These files contain benchmark, pipeline, model, eligibility, failure, and run
provenance. The Dashboard remains authoritative for the current dynamic result.

## Historical fixture

The `sanitized_*` JSONL files exercise the historical recorded-session evaluator
and its manifest with synthetic public inputs. Their results do not execute the
complete current retrieval, prompt, and generation path and are not the current
product score. Redacted recorded-session rows and detailed evaluator output stay
local even when their content has been reviewed.

Provider-scored reuse requires explicit authorization and review of the exact
payload. Raw `data/eval/` records, private session text, local Chroma excerpts,
workspace paths, and credentials are outside this public boundary.
