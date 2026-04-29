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

## Behavior

The runner evaluates only recorded live samples that include:

- question
- answer
- retrieved contexts

Samples without retrieved contexts are skipped. No synthetic samples or
automatic ground truth are generated.

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
