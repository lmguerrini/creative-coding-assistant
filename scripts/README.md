# Scripts

Project automation and one-off operational entry points live here.

Evaluation helpers are local/manual by design:

- `eval_live_sessions.py` converts recorded live sessions into RAGAs-compatible
  evaluation runs.
- `run_eval_latest.sh` runs the latest eligible live-session samples and
  explicitly opts into provider calls.
- `report_canonical_retrieval.py` reads the configured local KB, runs the seven
  canonical Capstone retrieval queries through the configured query-embedding
  provider, and prints per-case source/domain coverage plus non-text chunk
  lineage (record, title, index, length, and distance) as JSON without changing
  the index or evaluation datasets. It also fingerprints the complete non-text
  KB metadata snapshot and selected report lineage. Retrieved excerpt text
  remains local.

Use `eval_live_sessions.py --dry-run` to prepare the dataset manifest without
calling evaluator LLM or embedding providers.

## Product quality gates

`v7_quality_gates.py` provides deterministic local and CI checks for public
product quality surfaces.

- `python scripts/v7_quality_gates.py docs-mermaid` validates standalone
  Mermaid files and Markdown Mermaid fences.
- `python scripts/v7_quality_gates.py dashboard` validates product quality
  gates, performance budgets, release checklist, and coverage dashboard.
- `python scripts/v7_quality_gates.py backend-log <log-path>` fails on
  backend log errors, tracebacks, and critical exceptions while allowing the
  controlled shutdown tracebacks produced by timeout-based smoke tests.

Frontend E2E gates live in `clients/nextjs`:

- `npm run test:e2e:smoke` runs the localhost and core creative journey smoke
  coverage.
- `npm run test:e2e` runs the full Playwright smoke and resilience suite.
