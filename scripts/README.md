# Scripts

Project automation and one-off operational entry points live here.

Evaluation helpers are local/manual by design:

- `eval_live_sessions.py` converts recorded live sessions into RAGAs-compatible
  evaluation runs.
- `run_eval_latest.sh` runs the latest eligible live-session samples and
  explicitly opts into provider calls.

Use `eval_live_sessions.py --dry-run` to prepare the dataset manifest without
calling evaluator LLM or embedding providers.
