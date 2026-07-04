# Production Deployment

V7.7 adds a deployment foundation for the browser-facing backend bridge without
changing creative generation, provider routing, persistence semantics, or local
development behavior.

## Local Development

Use the existing WSGI development bridge for local Next.js smoke work:

```bash
python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

The development bridge keeps permissive local CORS defaults and refuses to run
when `CCA_ENVIRONMENT=production` unless an operator passes the explicit
`--allow-production-dev-server` override.

## Production Backend

The production API surface is WSGI. Run it with Gunicorn or a compatible WSGI
host:

```bash
python -m pip install ".[server]"
CCA_ENVIRONMENT=production \
CCA_LOG_FORMAT=json \
CCA_CORS_ALLOWED_ORIGINS=https://your-frontend.example \
gunicorn creative_coding_assistant.api.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --threads 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

The WSGI entrypoint is `creative_coding_assistant.api.wsgi:application`.
Uvicorn is not the recommended production host for this backend because the API
apps are WSGI, not ASGI.

## Docker

Build and run the backend image:

```bash
docker build -t creative-coding-assistant-backend .
docker run --rm -p 8000:8000 \
  -e CCA_ENVIRONMENT=production \
  -e CCA_CORS_ALLOWED_ORIGINS=http://localhost:3000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/data:/app/data" \
  creative-coding-assistant-backend
```

Optional Compose workflow:

```bash
CCA_CORS_ALLOWED_ORIGINS=http://localhost:3000 docker compose up --build backend
```

The container persists Chroma, artifacts, and workspace sessions under
`/app/data`; mount a durable volume in real deployments.

## CORS

Local and test environments keep the existing wildcard behavior for browser
smoke compatibility. Production disables wildcard CORS and requires explicit
origins:

```text
CCA_ENVIRONMENT=production
CCA_CORS_ALLOWED_ORIGINS=https://app.example,https://admin.example
```

If production is started with `*` or no explicit origins, readiness reports a
guarded configuration and responses do not grant wildcard production CORS.

## Health Checks

Use these probes:

- Liveness: `GET /api/health/live`
- Readiness: `GET /api/health/ready`
- General contract probe: `GET /api/health`

Readiness is `503` when production configuration is guarded, including missing
credentials or unsafe CORS configuration. Liveness remains suitable for process
restart checks.

## Chroma Posture

Chroma is constrained to `chromadb>=0.6.3,<1.0.0`. The production dependency
report verifies the installed version without importing Chroma services:

```bash
python -c "from creative_coding_assistant.api.production import build_dependency_health_report; print(build_dependency_health_report().model_dump())"
```

## Runtime Checklist

- Set `CCA_ENVIRONMENT=production`.
- Set `CCA_CORS_ALLOWED_ORIGINS` to deployed frontend origins only.
- Provide `OPENAI_API_KEY` or `CCA_OPENAI_API_KEY`.
- Mount durable storage for `CCA_CHROMA_PERSIST_DIR`, `CCA_ARTIFACT_DIR`, and
  `CCA_WORKSPACE_SESSION_DB_PATH`.
- Use JSON logs with `CCA_LOG_FORMAT=json`.
- Wire `/api/health/live` and `/api/health/ready` to deployment probes.
- Confirm `build_deployment_readiness_checklist()` has no guarded items before
  release.
- Keep auth, rate limiting, WAF, TLS termination, and request-size enforcement
  at the platform edge until first-class middleware is added in a future scope.

## CI Deployment Gates

Push CI runs fast backend static gates, focused deployment/API pytest coverage,
runtime quality checks, frontend validation, and a dependency security audit.
The `pip-audit` job builds a frozen third-party requirements file that excludes
the unpublished local project package before auditing, so the gate still checks
installed dependencies without failing on the local package name.

Full backend pytest with coverage is a release verification gate for manual,
nightly, and version-tag workflows. This keeps normal push CI usable while
preserving full-suite validation before release decisions.
