# Live Session RAGAs Evaluation

Run local evaluation over real recorded Streamlit chat sessions:

```bash
.venv/bin/python scripts/eval_live_sessions.py --limit 1
```

The command reads `data/eval/live_sessions.jsonl` by default and writes result
rows to `data/eval/ragas_results.jsonl`.

## Requirements

Install RAGAs in the active virtual environment before running:

```bash
.venv/bin/python -m pip install ragas
```

Running RAGAs may call evaluator LLM APIs and can incur provider cost. Keep
`OPENAI_API_KEY` or the evaluator provider configuration available when the
selected RAGAs metrics require it.

Recorded live-session rows under `data/eval/` are private local runtime
artifacts. Do not send them to an external evaluator without HITL/privacy
approval.

## Examples

Run a low-cost smoke evaluation over the first eligible sample:

```bash
.venv/bin/python scripts/eval_live_sessions.py --limit 1
```

Run selected metrics explicitly:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --limit 2 \
  --metric context_precision \
  --metric faithfulness
```

Run the V8 release-candidate sanitized fixture:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl \
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

`faithfulness` can be slow and brittle for long code-heavy answers because
RAGAs asks an evaluator model to extract many statements. `answer_relevancy`
uses evaluator embeddings and may increase cost.

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
