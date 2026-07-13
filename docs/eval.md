# Recorded-Session RAGAS Evaluation

Prepare a local evaluation over recorded assistant sessions without making
provider calls:

```bash
.venv/bin/python scripts/eval_live_sessions.py --limit 1 --dry-run
```

The command reads `data/eval/live_sessions.jsonl` by default and writes result
metadata to `data/eval/ragas_results.jsonl`. A real RAGAS run requires the
separate `--allow-provider-calls` flag.

## Requirements

Install the version-bounded evaluation extra before running:

```bash
.venv/bin/python -m pip install -e ".[evaluation]"
```

The optional evaluation stack has two no-fix advisories in the dated local
dependency audit. Use it only with trusted local inputs and protected cache
directories; review the
[Installation Guide](INSTALLATION_GUIDE.md#optional-evaluation-dependencies)
before installing or scoring.

Running RAGAs may call evaluator LLM APIs and can incur provider cost. Keep
`OPENAI_API_KEY` or the evaluator provider configuration available when the
selected RAGAs metrics require it.

Recorded live-session rows under `data/eval/` are private local runtime
artifacts. Do not send them to an external evaluator without privacy
approval.

## Examples

After reviewing the selected local row, provider boundary, model, cost, and
output path, run a low-volume provider-scored evaluation over the first
eligible sample:

```bash
.venv/bin/python scripts/eval_live_sessions.py --limit 1 --allow-provider-calls
```

Run selected metrics explicitly:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --limit 2 \
  --metric context_precision \
  --metric faithfulness \
  --allow-provider-calls
```

Run the approved synthetic/public fixture without overwriting committed
evidence:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path /tmp/cca-ragas-context-precision-results.jsonl \
  --metric context_precision \
  --allow-provider-calls
```

That fixture is synthetic and public/reviewer-safe. The 2026-07-08 run scored
4 of 4 eligible rows with zero skips, zero metric failures, and average context
precision `0.999999999925`.

## Behavior

The runner evaluates only recorded live samples that include:

- question
- answer
- retrieved contexts

Samples without retrieved contexts are skipped. The runner does not generate
synthetic samples or automatic ground truth; any sanitized fixture must be
created and reviewed separately before use.

By default, the CLI runs only the safer smoke metric:

- context precision without reference

Additional supported opt-in metrics are:

- faithfulness
- answer relevancy
- context relevancy

`faithfulness` can be slow and brittle for long code-heavy answers because
RAGAs asks an evaluator model to extract many statements. `answer_relevancy`
uses evaluator embeddings and may increase cost. `context_relevancy` measures
whether the retrieved excerpts contain information useful for the query.
Context recall is not supported for the approved fixtures because they do not
contain independently justified reference answers.

Failed metric scores are written as `null` with `metric_errors` metadata in the
local result rows. Detailed evaluator exceptions come from RAGAs logs.

## Configuration

Optional environment variables:

- `CCA_EVAL_DATA_PATH`
- `CCA_EVAL_RAGAS_RESULTS_PATH`
- `CCA_EVAL_RAGAS_MODEL`
- `CCA_EVAL_RAGAS_TIMEOUT_SECONDS`
- `CCA_EVAL_RAGAS_MAX_RETRIES`
- `CCA_EVAL_RAGAS_MAX_WORKERS`

Runtime result files under `data/eval/` remain local and ignored by git.
Public sanitized result files under `demo/evaluation/` may be tracked when they
contain no private session text, workspace paths, secrets, or local Chroma
content.

## Canonical retrieval engineering loop

Use the fixed seven-query, read-only retrieval report:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5
```

It sends only
the committed public query strings to the configured embedding provider; local
retrieved excerpts remain local. Each selected result includes non-text lineage
(record id, document title, chunk index, character count, distance, and score)
so source-diversity choices can be audited without publishing excerpt text. The
2026-07-13 loop improved expected-source overlap from 9/23 to a final 16/23 and
requested-domain coverage from 7/19 to 18/19. All seven queries returned five
results. These coverage ratios measure retrieval selection, not RAGAS quality
or overall product quality.

The complete verified progression was 9/23 + 7/19, 12/23 + 11/19 after intent
and result diversity, 15/23 + 17/19 after balanced per-domain candidate search,
15/23 + 18/19 after bounded requested-domain fallback, 17/23 + 18/19 after a
two-chunk source cap, and 19/23 + 18/19 after chunk-level Three.js manual
filtering plus unseen-source priority. Lineage inspection proved that the
19/23 peak included three title-only `p5_reference` chunks and a
`tone_js_docs` API-name index. Structural heading removal reduced the report to
16/23 + 18/19; excluding the verified index-only Tone.js source reduced it to
15/23 + 18/19; bounded post-filter candidate headroom then recovered substantive
Three.js manual evidence for the final 16/23 + 18/19 result.

The final result deliberately prefers lower truthful coverage to inflated raw
anchor overlap. It keeps substantive `three_manual` chunks, prefers unseen
sources before repeated chunks, and returns no evidence instead of
reintroducing filtered navigation/index material. No query, expected source,
fixture, score rule, top-k value, or source weight was changed to raise the
report. The remaining requested-domain gap is Shadertoy: the approved source has
no local chunks and returned HTTP 403 during focused sync, so that gap is
`BLOCKED_BY_EXECUTION_ENVIRONMENT` rather than scored as product failure.

`demo/evaluation/canonical_retrieval_report.json` records every selected
non-text lineage row for the final run. Its full-index metadata fingerprint
binds the report to the exact 1,445-chunk KB snapshot, while a separate selection
fingerprint binds the fixed benchmark, model, top-k, summary, and ranked result
lineage. Re-running against a changed index therefore produces visibly different
evidence rather than silently replacing history.

Do not run current local knowledge-base excerpts through an external generation
or evaluator provider. Current-product end-to-end RAGAS remains
`BLOCKED_BY_EXECUTION_ENVIRONMENT` until a reviewed committed sanitized/redacted
dataset represents that path. The latest defensible provider score remains the
committed transcribed product summary of the approved-fixture result; the raw
scored JSONL and run manifest are not tracked artifacts. `MISSING_EVIDENCE` and
blocked dimensions are never converted to zero.
