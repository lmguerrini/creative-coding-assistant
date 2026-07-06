# V7.7 Production Deployment Foundation / Release Readiness Finalization - Task

## Task
Implement a production deployment foundation before V7 freeze.

## Contractual Roadmap Item
Items 1 through 15 in the V7.7 capability plan.

## Scope
- Add Dockerfile and optional docker-compose deployment artifacts.
- Add a production WSGI entrypoint and documented Gunicorn command.
- Make CORS environment-aware and guarded in production.
- Extend production configuration, Chroma posture, and readiness checklist
  reporting.
- Add CI coverage reporting and dependency security scan.
- Add deployment documentation, health-check guidance, and runtime checklist.
- Update Runtime Pack ledgers.
- Validate and commit before Codex Engineering Audit / HITL.

## Non-Goals
- Do not continue the V7 Grand Engineering Audit.
- Do not change creative/user-facing behavior.
- Do not change provider routing.
- Do not introduce auth or rate limiting except as documented future/edge
  recommendations.
- Do not alter LangGraph workflow order, prompt rendering, generated output,
  workspace persistence semantics, frontend UI behavior, merge, push, tag,
  freeze, or V8 start state.

## Required Files
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.github/workflows/ci.yml`
- `src/creative_coding_assistant/api/`
- `src/creative_coding_assistant/core/config.py`
- `docs/PRODUCTION_DEPLOYMENT.md`
- `tests/test_v7_7_production_deployment_foundation.py`
- `codex_starter_pack/docs/ROADMAP_DEFINITIVE_V7.md`
- `codex_starter_pack/docs/runtime/*.md`
- `codex_starter_pack/docs/capabilities/v7_7_production_deployment_foundation/*.md`

## Validation
- `git diff --check`
- Full-project Ruff over `src`, `tests`, and `scripts`
- `python -m compileall src tests scripts`
- Focused deployment/config tests
- Full backend pytest
- Frontend typecheck / Vitest because deployment docs reference frontend
  execution boundaries
- Docker build if feasible locally
- Local app smoke if required by Runtime Pack after validation

## Stop Conditions
Stop for required HITL gates, validation failures, product bugs, Runtime
Evolution proposals, or the final human-controlled merge/push/tag gate.

## Progress Update Requirements
Update capability progress, central progress, roadmap coverage, validation
evidence, product bug status, Runtime Evolution status, and historical
consistency before Codex Engineering Audit / HITL.
