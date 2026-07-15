# Evaluation Methodology

Creative Coding Assistant separates retrieval selection, current-product RAGAS,
local product snapshots, and historical fixtures. These lanes answer different
questions and are never combined into a universal project or creativity score.

## Current-product benchmark

The primary evaluation runs the current application path rather than scoring a
detached answer fixture:

1. select the immutable seven-case public benchmark;
2. retrieve from the active official-document Chroma collection;
3. assemble context with the current ranking and filtering rules;
4. render the current Jinja prompt;
5. generate through the configured OpenAI adapter; and
6. evaluate the resulting answer and contexts with RAGAS.

The benchmark is versioned and fingerprinted. Case selection, authored
references, source expectations, and top-k settings must not be changed after
observing a run simply to improve a metric.

Full scope uses the same seven RAG cases and also records current local
Creative, Workflow, and Reliability snapshots. Those three lanes make no model
or evaluator calls, are not additional generated answers, and do not contribute
to Retrieval Quality.

## Metrics

The current-product macro is the equal-weight mean of five component metrics:

| Metric | What it measures | Uses the authored reference |
|---|---|---:|
| Context precision | Whether useful contexts are ranked ahead of less useful contexts | Yes |
| Faithfulness | Whether answer claims are supported by retrieved context | No |
| Answer relevancy | Whether the answer addresses the question | No |
| Context relevancy | Whether the retrieved material is useful for the question | No |
| Context recall | Whether retrieval covers the authored reference answer | Yes |

An ineligible case, skipped case, provider failure, or metric failure remains
explicit. Missing evidence is not converted to zero, and a partial metric set is
not promoted as the current five-metric score.

## Run the current-product path

Install the optional evaluator only with trusted local inputs:

```bash
.venv/bin/python -m pip install -e ".[evaluation]"
```

Prepare and fingerprint the seven-case selection without constructing or
calling the provider-bound stack:

```bash
.venv/bin/python -m creative_coding_assistant.eval.current_product_cli \
  --scope rag \
  --dry-run
```

A live diagnostic requires a populated compatible knowledge base, provider
credentials, network access, reviewed cost and privacy boundaries, and explicit
authorization:

```bash
.venv/bin/python -m creative_coding_assistant.eval.current_product_cli \
  --scope rag \
  --allow-provider-calls \
  --diagnostic-output data/eval/current-product-safe.json
```

This diagnostic path does not replace committed canonical evidence. Canonical
publication is a separate maintainer operation with stricter completeness,
schema, privacy, and provenance gates.

Provider-backed evaluation can call generation, query-embedding, evaluator-LLM,
and evaluator-embedding services and may incur cost. The CLI disables secondary
observability for an authorized live run so evaluation data is not duplicated
to an independently configured tracing service.

## Retrieval selection report

The separate retrieval report measures selection coverage without generating
answers or invoking RAGAS:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5
```

The command sends the committed public query strings to the configured embedding
provider, searches the local index, and prints a read-only report. Retrieved
excerpt text remains local. The report retains non-text lineage such as source
ID, document title, chunk index, distance, rank, and selection reason.

The committed `demo/evaluation/canonical_retrieval_report.json` is the
authoritative snapshot for exact counts and fingerprints. Retrieval coverage is
not answer quality, RAGAS quality, runtime correctness, or artistic quality.

## Evidence, provenance, and history

Every retained current-product result records:

- benchmark version and case IDs;
- dataset, retrieval-pipeline, prompt-pipeline, and result fingerprints;
- generation, evaluator, and embedding configuration;
- timestamps, run ID, eligibility, skips, and metric failures;
- component means and macro arithmetic; and
- source IDs and retrieval lineage without public question, answer, reference,
  or excerpt text.

The Dashboard owns the current dynamic score and rolling session history. The
committed `demo/evaluation/current_product_ragas_evidence.json` is a validated,
public-safe canonical projection loaded at build or reload. A Dashboard run does
not overwrite that file.

Historical synthetic and redacted fixtures remain useful for schema, evaluator,
and migration checks. They score recorded rows, often use fewer metrics, and do
not execute the current retrieval, prompt, or generation stack. For those
reasons, a historical fixture is never substituted for the current-product
score.

## Public and private data boundaries

The seven benchmark cases are committed public material approved for the
documented evaluator path. That decision does not authorize external evaluation
of arbitrary local content.

Public-safe evidence may include case IDs, status, aggregate metrics, model and
embedding identifiers, source IDs, timestamps, and fingerprints. It excludes:

- raw local questions and answers;
- authored references;
- retrieved Chroma excerpts;
- workspace/session records;
- image data, prompts, and generated private artifacts;
- API keys, environment contents, and personal paths; and
- exact private evaluator diagnostics.

Runtime result files under `data/eval/` remain local and ignored by git. Raw
live-session rows and arbitrary local Chroma excerpts must not be sent to an
external evaluator without a new, explicit content and privacy review.

## Historical recorded-session runner

The recorded-session CLI is retained for approved synthetic or independently
reviewed local fixtures. A dry run makes no evaluator calls:

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path data/eval/historical-ragas-results.jsonl \
  --dry-run
```

Its historical fixture does not contain an independently justified reference
answer, so context recall is unavailable there. Do not compare a four-metric
historical macro directly with the five-metric current-product macro.

## Limitations

- Seven cases provide reproducibility, not broad statistical coverage.
- RAGAS metrics inherit evaluator-model variance, version behavior, and prompt
  sensitivity.
- Code-heavy answers can make statement extraction and faithfulness scoring
  brittle or slow.
- A single retained run does not provide confidence intervals; repeated runs
  are needed to estimate evaluator variance.
- Retrieval metrics do not measure browser execution, usability, accessibility,
  safety, performance, originality, or artistic quality.
- The public-safe projection deliberately omits the text needed for deep error
  analysis; exact diagnostics must remain local and separately protected.

See the [Evaluation Workflow](../architecture/evaluation_workflow.md) for the
Dashboard, local snapshot, canonical publication, and historical lanes, and the
[Ethics and Privacy Assessment](ETHICS_PRIVACY_ASSESSMENT.md) for external-data
boundaries.
