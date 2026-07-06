# V7.7 Production Deployment Foundation / Release Readiness Finalization - Capability Spec

## Purpose
Close the production-readiness gaps Junie found after the V7 Grand Engineering
Audit without changing creative/user-facing behavior.

## Roadmap Contract
- Junie production-readiness finding intake.
- Phase -1 / Phase 0 Runtime Pack validation and V7.6 release reconciliation.
- Dockerfile.
- Optional docker-compose.
- Production WSGI server command/config using Gunicorn.
- Environment-aware CORS policy.
- Production deployment documentation.
- Health-check deployment guidance.
- Basic production runtime checklist.
- CI coverage reporting.
- CI security/dependency scan.
- Chroma dependency posture verification.
- Production configuration validation.
- Release/deployment readiness checklist.
- Runtime Pack ledger synchronization and capability validation.

## Architecture Boundaries
V7.7 owns deployment artifacts and production configuration posture for the
existing browser-facing WSGI API bridge. The recommended production host is
Gunicorn against `creative_coding_assistant.api.wsgi:application`; Uvicorn is
not recommended because the backend bridge is WSGI, not ASGI.

V7.7.1 records the release-readiness stabilization for CI security/dependency
posture and runtime verification evidence on top of the V7.7.0 deployment
foundation.

## Product Boundaries
V7.7 must not change creative generation behavior, provider/model routing,
LangGraph workflow order, prompt rendering, generated output semantics,
workspace persistence semantics, frontend UI behavior, authentication, rate
limiting, merge, push, tag, freeze, or V8 start state.

## Validation Contract
Required validation:

- `git diff --check`
- Full-project Ruff over `src`, `tests`, and `scripts`
- `python -m compileall src tests scripts`
- Focused deployment/config pytest
- Full backend pytest
- Frontend typecheck / Vitest because deployment documentation references the
  local Next.js production boundary
- Docker build if feasible locally
- Local app smoke if required after validation

## Runtime Evolution Contract
No product Runtime Evolution is expected. If deployment foundation work requires
changing runtime rules, creative behavior, provider routing, or production
auth/rate-limit enforcement, stop for Runtime Evolution HITL.
