# Live Session RAGAs Evaluation

Run local evaluation over real recorded Streamlit chat sessions:

```bash
.venv/bin/python scripts/eval_live_sessions.py
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

## Behavior

The runner evaluates only recorded live samples that include:

- question
- answer
- retrieved contexts

Samples without retrieved contexts are skipped. No synthetic samples or
automatic ground truth are generated.

The initial metric set is:

- faithfulness
- answer relevancy
- context precision without reference

## Configuration

Optional environment variables:

- `CCA_EVAL_DATA_PATH`
- `CCA_EVAL_RAGAS_RESULTS_PATH`

Runtime result files under `data/eval/` remain local and ignored by git.
