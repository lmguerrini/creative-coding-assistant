# Production Deployment

Creative Coding Assistant includes a local two-process workstation and a
production WSGI packaging foundation. Current evidence supports the local
review/demo target; it does not establish a hosted public deployment,
authentication, rate limiting, managed storage, backup, or operational service
ownership.

## Local Development

Use the existing WSGI development bridge for local Next.js smoke work:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

The development bridge allows only `http://127.0.0.1:3000` and
`http://localhost:3000` by default. Other local ports or origins require an
explicit `CCA_CORS_ALLOWED_ORIGINS` value. The bridge refuses to run when
`CCA_ENVIRONMENT=production` unless an operator passes the explicit
`--allow-production-dev-server` override.

## Current Local Review Target

The reviewed Capstone target is a local workstation demo, not a public
deployment. The presenter runs the backend API on `127.0.0.1:8000` and the
Next.js workstation on the local Next.js development server. Showcase upload
and any release/deployment decision remain separate human actions.

Local demo startup path:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd clients/nextjs
npm run dev
```

Before the demo, verify:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/ready
```

Then open the primary in-app demo target:

```text
http://127.0.0.1:3000
```

Use **Demo Mode** inside Creative Coding Assistant to select one of the ten
current scenarios. Four are the canonical browser showcase sequence:
Polyrhythmic constellation, Recursive aurora garden, Kinetic orbit sculpture,
and Fractal solar bloom. The selected scenario loads its prompt into the normal
assistant composer and keeps input, runtime, expected artifact, validation, and
fallback boundaries visible.

Run the deterministic browser path from the repository when a local proof is
needed:

```bash
npm run test:e2e --prefix clients/nextjs -- e2e/demo-showcase-smoke.spec.js
```

That gate proves the asserted workstation, artifact, interaction, preview,
fullscreen, refinement, and persistence contracts with deterministic fixtures.
It is not a configured-provider generation or human-quality result.

Current fallback path:

- Frontend or provider failure: show a preflight-approved product artifact or
  the separately labelled deterministic browser fixture; do not imply a new
  provider response.
- Backend failure: use the [System Overview](SYSTEM_OVERVIEW.md) and current
  [Capstone Demo and Showcase Guide](CAPSTONE_DEMO_SHOWCASE.md); do not imply a
  live API response.
- Retrieval failure: use
  `demo/evaluation/canonical_retrieval_report.json` and state its date,
  fingerprint, and local-selection boundary; do not invent current citations.
- Preview failure: show source and the explicit code/export boundary rather
  than calling the renderer successful.

The retired V8 static launcher and eight-flow files remain historical evidence
only and are not a current fallback authority.

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

Local defaults allow the two loopback workstation origins on port 3000; they do
not grant wildcard access. Set an explicit comma-separated value for other
local origins. Production requires explicit deployed origins and rejects the
wildcard:

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
