# Installation Guide

This is the supported local installation for Creative Coding Assistant (CCA).
It runs the Python API and Next.js interface as separate local processes. It
does not describe a hosted production deployment.

## Prerequisites

- Git
- Python 3.11 or newer
- Node.js 22.13+ (22.x) or 24+ and npm
- A POSIX shell for the commands below
- Optional: an OpenAI API key for live generation, embeddings, knowledge-base
  refresh, or provider-scored evaluation

The committed lockfile is authoritative for frontend packages. A provider key
is not required to read the documentation, run most local tests, or browse the
UI's bounded local and fallback paths.

## 1. Create the Python environment

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

The base application and development validators are now installed.

### Optional evaluation dependencies

Add the evaluation extra only when you intend to run the RAGAS pipeline with
trusted local inputs:

```bash
.venv/bin/python -m pip install -e ".[dev,evaluation]"
```

Optional evaluator dependencies expand the dependency and external-service
boundary. Use only trusted local inputs, protect evaluation cache directories,
and do not pass attacker-controlled URL or file contexts to RAGAS. Check the
installed environment before use:

```bash
.venv/bin/python -m pip_audit
```

The optional `server` extra contains the production-process dependency; it is
not needed for the local development server used in this guide.

## 2. Install the frontend

```bash
cd clients/nextjs
npm ci
cd ../..
```

For browser tests, install the test browser once:

```bash
cd clients/nextjs
npx playwright install chromium
cd ../..
```

On a CI or Linux workstation that also needs system packages, the equivalent
Playwright command is `npx playwright install --with-deps chromium`; it may
require operating-system privileges.

## 3. Configure only what you need

The repository's `.env.example` is a catalogue, not a secret file. Create a
root `.env` containing only the settings you intend to use. At minimum, live
OpenAI generation normally uses:

```dotenv
OPENAI_API_KEY=replace-with-a-real-secret
CCA_OPENAI_MODEL=gpt-5-mini
CCA_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LANGSMITH_TRACING=false
```

`CCA_OPENAI_API_KEY` is a supported alias, but `OPENAI_API_KEY` takes
precedence when both are present. Do not commit `.env`, copy a real secret into
documentation, paste it into logs, or expose configuration objects that may
contain secret values. Optional tracing should remain off unless its external
data transfer is deliberately configured and authorized.

The application defaults to local data paths. Review every path and provider
setting before using real user data. See
[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

## 4. Start CCA

Terminal 1, from the repository root:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd clients/nextjs
npm run dev
```

Open `http://127.0.0.1:3000`.

The development server intentionally refuses a production environment unless
an explicit development-server override is supplied. Do not use that override
as a deployment strategy.

## 5. Verify the installation

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/live
curl --fail http://127.0.0.1:8000/api/health/ready
```

Then run the core deterministic checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/v7_quality_gates.py docs-mermaid
.venv/bin/python scripts/v7_quality_gates.py dashboard
npm run typecheck --prefix clients/nextjs
npm run test --prefix clients/nextjs
npm run test:e2e:smoke --prefix clients/nextjs
```

The canonical four-showcase browser path is:

```bash
npm --prefix clients/nextjs run test:e2e -- e2e/demo-showcase-smoke.spec.js
```

Tests that use a real provider, refresh embeddings, or score RAGAS evidence are
separate opt-in operations and may incur cost.

## 6. Optional knowledge-base refresh

The repository can be useful before a refresh. Refresh only when provider
credentials, embedding configuration, source licensing, network access, and
cost have been reviewed:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

To target one registered source or continue a batch after a source failure:

```bash
.venv/bin/python scripts/sync_official_kb.py --source-id SOURCE_ID
.venv/bin/python scripts/sync_official_kb.py --all --continue-on-error --summary-format json
```

The command can return exit code 0 (success), 1 (completed with source
failures), or 2 (configuration/usage failure). A registered source is not
necessarily indexed, retrieved, or cited. Full data guidance is in
[DATA_AND_KB.md](DATA_AND_KB.md).

## Next steps

- Product workflows: [USER_MANUAL.md](USER_MANUAL.md)
- Common failures: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Evaluation methodology: [eval.md](eval.md)
