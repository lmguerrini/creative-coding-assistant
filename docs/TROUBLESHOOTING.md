# Troubleshooting

Start with the visible error and the smallest relevant check. Preserve failure,
missing, and blocked states instead of forcing a green result. Do not paste API
keys, raw prompts, local retrieved excerpts, private session rows, or full
environment dumps into an issue or public diagnostic report.

## Backend does not start

From the repository root, confirm the supported interpreter and installation:

```bash
python3 --version
.venv/bin/python -m pip show creative-coding-assistant
.venv/bin/python -m creative_coding_assistant.api.dev_server --help
```

Python 3.11 or newer is required. If the package is missing, recreate/install
the local environment as described in [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Start the workstation API with:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

If port 8000 is already in use, stop the stale local process or deliberately
choose another port and update every frontend endpoint that targets the API.
The dev server refuses production posture by design; do not bypass that guard
as a deployment solution.

## Backend is live but not ready

Compare all three endpoints:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/live
curl --fail http://127.0.0.1:8000/api/health/ready
```

“Live” means the process can answer. “Ready” can still fail because a required
dependency, storage path, configuration, or provider-related prerequisite is
unavailable. Read the local server error, correct that specific dependency,
then retry once. Redact environment-specific details before sharing output.

## Frontend does not start or cannot find a package

Use Node.js 22.13+ (22.x) or 24+ and install from the committed lockfile:

```bash
cd clients/nextjs
node --version
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`. If the UI loads but requests fail, confirm the
backend and the `NEXT_PUBLIC_*` endpoint variables in
[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md). Restart Next.js after changing
public environment variables.

## Browser reports CORS or network errors

Confirm that the UI origin and API host/port match the configured environment.
For local development, avoid mixing `localhost` and another host unless both origins
are deliberately allowed. Production requires an explicit origin policy; do
not “fix” it with a credentialed wildcard.

Use the browser network panel to identify the exact failing route, such as
`/api/assistant/stream`, `/api/workspace/session`,
`/api/domain-experience`, `/api/evaluation/run`, or `/api/knowledge-base`.
Record response status and a sanitized error, not request secrets or image data.

## Provider generation is unavailable or times out

Check only whether the required variable is present; do not print its value.
`OPENAI_API_KEY` takes precedence over the `CCA_OPENAI_API_KEY` alias. Confirm
the configured model, timeout, provider account access, quota, and network
policy. The current defaults include model `gpt-5-mini`, maximum output tokens
`4000`, and timeout `90` seconds.

Repeated retries may incur cost. A missing/failed provider call must remain
labeled unavailable or failed; a local fallback is not evidence of provider
success.

## Image attachment is rejected

Current request bounds are:

- up to four images;
- PNG, JPEG, WebP, or GIF;
- no more than 1 MiB per image;
- non-empty content whose signature matches its declared media type.

Re-export or compress the image rather than changing only its filename or MIME
label. Remove an existing reference before adding a fifth. If the selection
clears after a successful submit, that is intentional request-scoped behavior.
Normal session restoration also omits image bytes.

If an image was selected but the output did not reflect it, verify that the
request was submitted with the attachment and that the configured provider
supports visual input. Current tests prove configured-provider payload
construction, not live receipt, use, or image influence. A thumbnail alone is
not proof of a multimodal provider call.

## Preview is blank or static

First identify the actual runtime in the artifact:

- p5.js, Three.js, and GLSL should use their in-app browser preview paths;
- Tone.js requires an explicit **Start** gesture and can remain silent under
  browser autoplay policy until that gesture;
- Hydra and React Three Fiber are code/export-only;
- TouchDesigner, Unreal, Blender Geometry Nodes, and Houdini require external
  handoff and cannot be previewed/validated as those tools inside CCA.

Inspect the artifact's preview error and browser console. Check for invalid
generated code, unavailable runtime assets, WebGL support, reduced-motion/audio
policy, and resource exhaustion. Do not interpret a nonblank canvas as proof of
correct artistic behavior. Use [DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md) for
the authoritative runtime matrix.

## Tone.js is silent

Select the visible **Start** control in the preview. Confirm that the tab is not
muted and that the browser has an active audio output. CCA does not upload,
capture, or analyze audio, so an audio-input troubleshooting path does not
apply.

## Session restoration looks stale

Confirm `/api/workspace/session` is reachable and that the configured local
SQLite path is writable by the current process. Refresh the browser once and
inspect the current session identity in the UI/Inspector.

Do not delete or edit the SQLite file while either process is using it. Back up
user data before any manual repair. A compact browser localStorage fallback can
restore state when the API is unavailable; do not clear it until you have
recorded the relevant session identity and accepted losing that fallback. Image
attachments intentionally do not return to the composer, even when other
workspace state restores.

## Knowledge Base is empty, partial, or stale

Check registered, indexed, retrieved, and cited states separately. A registered
source with zero active chunks needs source-specific investigation; a populated
index with irrelevant results is a retrieval issue, not an indexing failure.

Inspect the sync command without mutating data:

```bash
.venv/bin/python scripts/sync_official_kb.py --help
```

After approving network, source, provider, embedding, storage, and cost
boundaries, target one source:

```bash
.venv/bin/python scripts/sync_official_kb.py --source-id SOURCE_ID --summary-format json
```

Exit code 1 means at least one source failed; exit code 2 identifies a
configuration/usage failure. Preserve source-level failures instead of
injecting an expected source into results or deleting the active index to
conceal a gap. See [DATA_AND_KB.md](DATA_AND_KB.md).

## Evaluation will not run

Install the optional evaluation dependencies:

```bash
.venv/bin/python -m pip install -e ".[dev,evaluation]"
```

Before installing or using this optional stack, review its dependency-risk
guidance and local audit command in the
[Installation Guide](INSTALLATION_GUIDE.md#optional-evaluation-dependencies).

Verify the current-product evaluation path without provider calls:

```bash
.venv/bin/python -m creative_coding_assistant.eval.current_product_cli \
  --scope rag \
  --dry-run
```

Live scoring requires `--allow-provider-calls`, credentials, network access,
and an approved external-transfer decision. It can incur cost. The Dashboard
is authoritative for the current dynamic result; see [eval.md](eval.md) for
the current diagnostic command and evidence boundaries.

For the separate historical recorded-session fixture, inspect selection
without provider scoring:

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path data/eval/approved-ragas-results.jsonl \
  --dry-run
```

Do not point the historical runner at raw `data/eval/live_sessions.jsonl` or
arbitrary local Chroma excerpts.

The explicit tracing override is important: `--dry-run` blocks RAGAS provider
scoring, but an independently enabled telemetry integration is a separate
external boundary.

A `null` metric with `metric_errors` is a metric failure, not zero. Context
recall is missing for the historical fixture because it lacks independently
justified reference answers; it is present for the canonical seven-case
benchmark. See [eval.md](eval.md).

## Playwright cannot launch Chromium

Install the browser package:

```bash
cd clients/nextjs
npx playwright install chromium
```

On a CI/Linux workstation that needs browser system dependencies:

```bash
cd clients/nextjs
npx playwright install --with-deps chromium
```

Then run the targeted smoke:

```bash
npm run test:e2e:smoke
npm run test:e2e -- e2e/demo-showcase-smoke.spec.js
```

Current automated browser evidence is Chromium-based; a pass does not establish
Firefox/WebKit equivalence.

## A test fails

Run the smallest relevant command and preserve its first actionable failure:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python -m compileall src tests scripts
.venv/bin/python scripts/v7_quality_gates.py docs-mermaid
.venv/bin/python scripts/v7_quality_gates.py dashboard
cd clients/nextjs
npm run typecheck
npm run test
```

Do not delete snapshots, relax an assertion, alter a benchmark, or overwrite
committed evidence merely to make a check green. Separate a product regression,
environment prerequisite, provider block, and stale historical artifact.

## Prepare a safe diagnostic report

Include the failing command, exit code, sanitized error, relevant route, browser
and runtime, and whether the path was local, provider-backed, fixture-based, or
blocked. Exclude:

- `.env` contents and API keys;
- raw prompts, reference images, and generated private artifacts;
- `data/eval/live_sessions.jsonl` rows;
- local Chroma excerpts or a copy of the vector store;
- full session databases, absolute personal paths, and external tracing IDs
  unless specifically reviewed and redacted.
