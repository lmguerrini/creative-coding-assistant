# Scripts

Project automation and one-off operational entry points live here.

Evaluation helpers are local/manual by design:

- `eval_live_sessions.py` converts recorded live sessions into RAGAs-compatible
  evaluation runs.
- `run_eval_latest.sh` runs the latest eligible live-session samples and
  explicitly opts into provider calls.

Use `eval_live_sessions.py --dry-run` to prepare the dataset manifest without
calling evaluator LLM or embedding providers.

## V7.4 quality gates

`v7_quality_gates.py` provides deterministic local and CI checks for the V7.4
E2E Quality & CI Hardening capability.

- `python scripts/v7_quality_gates.py docs-mermaid` validates standalone
  Mermaid files and Markdown Mermaid fences.
- `python scripts/v7_quality_gates.py dashboard` validates the V7.4 roadmap,
  performance budgets, release checklist, and coverage dashboard.
- `python scripts/v7_quality_gates.py runtime-hygiene` validates Runtime Pack
  structure, final freeze artifacts, stale release wording, duplicate
  artifacts, and local Git hygiene.
- `python scripts/v7_quality_gates.py backend-log <log-path>` fails on
  backend log errors, tracebacks, and critical exceptions while allowing the
  controlled shutdown tracebacks produced by timeout-based smoke tests.

Frontend E2E gates live in `clients/nextjs`:

- `npm run test:e2e:smoke` runs the localhost and core creative journey smoke
  coverage.
- `npm run test:e2e` runs the full Playwright smoke and resilience suite.
