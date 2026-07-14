# Reviewer Guide

This guide gives a reviewer a short, reproducible path through Creative Coding
Assistant (CCA). It separates what the current application demonstrates from
historical, fixture-based, blocked, or human evidence. For installation and
configuration details, use [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) and
[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

## Evidence rule

Use this order when resolving conflicting claims:

1. current behavior observed in the running application;
2. current automated validation;
3. current evaluation output;
4. current documentation;
5. historical evidence, clearly dated and labeled;
6. planned work, clearly labeled.

Missing or blocked evidence is not a zero, and it is not a pass. An automated
browser check establishes the interaction it asserts; it is not a substitute
for a human judgment of artistic quality or usability.

## 15-minute reviewer path

### 1. Start and verify the workstation (3 minutes)

Use two terminals from the repository root.

Before starting, review the uncommitted `.env`. For an ordinary local review,
keep `LANGSMITH_TRACING=false` so browsing or generation does not silently add
an external telemetry boundary.

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

```bash
cd clients/nextjs
npm run dev
```

Open `http://127.0.0.1:3000`. In a third terminal, confirm the backend:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/ready
```

The ready check can expose an unavailable dependency even when the process is
live. Do not paste health output into public material without first checking it
for environment-specific details.

### 2. Inspect the product surfaces (4 minutes)

- Open the Homepage and note the product boundary: a creative-coding assistant,
  not an external DCC controller or deployment service.
- Open Demo Mode. Select one of the four canonical live showcases, choose
  **Load prompt & run**, and inspect the generated artifact and preview:
  **Polyrhythmic constellation** (Tone.js), **Recursive aurora garden**
  (p5.js), **Kinetic orbit sculpture** (Three.js), or **Fractal solar bloom**
  (GLSL).
- Open the artifact refinement form, submit a refinement, and confirm that the
  interaction remains in the same creative session.
- Toggle Fullscreen Creative Session and return to the normal workspace.

The browser preview currently executes p5.js, Three.js, GLSL, and Tone.js.
Tone.js remains silent until the user explicitly starts audio. Hydra and React
Three Fiber are code/export-only. TouchDesigner, Unreal Engine, Blender
Geometry Nodes, and Houdini are external handoffs: CCA does not execute,
control, deploy to, or validate those tools. The definitive domain matrix is
[DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).

### 3. Check workflow and evidence fidelity (4 minutes)

- Run or inspect **Single Agent**, **Multi Agent**, and **Auto** modes. In Auto,
  review the resolved route shown by the UI. The graph and Inspector should
  describe the route that was actually published; do not infer concurrency or
  hidden provider work from a label alone.
- Review the Inspector panels for the current session, artifact, workflow,
  retrieval, evaluation, and telemetry information. Empty or unavailable
  evidence should remain explicit.
- Open Dashboard, then inspect Overview, Architecture, Workflow, Workspace,
  Runtime, Preview, Artifacts, Domains, Knowledge Base, AI & agents, Memory,
  Sessions, Telemetry, and Evaluation. User Guide and Settings are the two
  secondary destinations. Dashboard cards summarize evidence; the running
  surface or cited file remains authoritative.
- In Dashboard > Artifacts, open an artifact preview. In Dashboard > Evaluation,
  distinguish the approved-fixture score from current local retrieval evidence.

### 4. Exercise the bounded multimodal path (2 minutes)

Choose **Reference-guided palette study**, attach a supported image, then
submit it with the prompt. The selected file stays browser-local until that
submission. The backend validates its size, media type, and signature, then
includes accepted pixels in the configured-provider request payload for that
request. Current evidence does not prove live provider receipt, use, or image
influence. The composer clears the attachment after submission, and session
persistence does not restore it.

CCA accepts up to four PNG, JPEG, WebP, or GIF references, each no larger than
1 MiB. An export made *before* submission may include the queued image in the project
bundle. Review an export before sharing it. CCA does not provide audio upload or
audio analysis; Tone.js is browser playback only. See
[ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md).

### 5. Read the evaluation correctly (2 minutes)

The primary Retrieval Quality is now the canonical current-product run
`v9-current-product-final-retained`: seven of seven eligible RAG cases, no
skips, no metric failures, and an equal-weight five-metric macro of
`0.6803191571804` (68.03191571804%). Its component means are Context Precision
51.96428571169692%, Faithfulness 64.8989898989899%, Answer Relevancy
56.62963631284655%, Context Relevancy 85.71428571428571%, and Context Recall
80.95238095238094%. The dataset fingerprint is
`sha256:b5fbc0e7cc9a523658eee8b0fc5cd7c417aa10540f8919e10bc2c4e10a40705f`.

The old 61.44% display remains in History as a four-row synthetic approved
fixture with no context-recall result. It was obsolete as a primary score; the
root cause was `EVALUATION_PIPELINE_DEFECT`, because that fixture did not
execute the current retrieval, prompt, generation, and benchmark paths.

Current local retrieval on a fixed seven-query, top-five report improved from
9/23 to 16/23 substantive expected-source overlaps (69.57%) and from 7/19 to
18/19 requested-domain coverage (94.74%). All seven queries returned five
results. This retrieval-only report remains separate from the current-product
RAGAS macro even though both use seven public benchmark cases.

The Evaluation catalog contains 35 deduplicated prompt contracts. That number
is contract coverage, not execution. Full runs seven current-product RAG cases
and records current local Creative, Workflow, and Reliability snapshots; those
snapshots are not additional generated or RAGAS-scored cases.
No human evaluation is claimed. See
[EVALUATION_METRICS_SUMMARY.md](EVALUATION_METRICS_SUMMARY.md) and
[eval.md](eval.md).

## Focused 10-minute validation

Run the smallest relevant checks rather than treating one suite as proof of
every claim.

```bash
.venv/bin/python -m pytest -q \
  tests/test_workflow_documentation_alignment.py \
  tests/test_documentation_intelligence.py
.venv/bin/python scripts/v7_quality_gates.py docs-mermaid
.venv/bin/python scripts/v7_quality_gates.py dashboard
cd clients/nextjs
npm run typecheck
```

## Full local validation (allow 30+ minutes)

Run the complete suites when the review window allows. Duration depends on the
machine and browser cache; the backend suite alone has exceeded twenty minutes
in a dated local release run.

```bash
.venv/bin/python -m pytest -q
cd clients/nextjs
npm run test
npm run test:e2e:smoke
npm run test:e2e -- e2e/demo-showcase-smoke.spec.js
```

If Playwright reports that Chromium is missing, follow
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Provider-backed generation,
knowledge-base refresh, and provider-scored evaluation can incur cost and are
not implicit parts of these local checks.

## What a reviewer should be able to defend

| Claim | Current evidence | Important boundary |
|---|---|---|
| Four live creative runtimes | Current browser behavior and canonical Demo Mode smoke | p5.js, Three.js, GLSL, and Tone.js only |
| Three workflow choices | Current UI, published graph, Inspector, and tests | A mode name does not prove parallel execution |
| Image-guided request | Current request-scoped upload path and browser validation | Provider-payload inclusion occurs only on submit; live receipt/use/influence is unproven; no durable attachment restoration |
| Grounded retrieval | Local Chroma report bound to its chunk count and fingerprint | Registered, indexed, retrieved, and cited are different states |
| 68.03% Retrieval Quality | Seven-case current-product public RAGAS evidence | Five-metric RAG macro, not a project grade or artistic-quality score |
| 61.44% RAGAS macro | Historical four-row synthetic fixture | Obsolete as primary Retrieval Quality; context recall is missing |
| External-tool support | Generated code and export handoff packages | No external-tool execution, control, deployment, or validation |
| Usability or aesthetic quality | Human observation, if a reviewer performs it | No completed human study is claimed by the repository |

## Reviewer questions worth asking

- Which evidence is current, and which is fixture-based or historical?
- What data leaves the workstation for generation, embeddings, evaluation, or
  optional tracing?
- Does a cited knowledge source appear in the retrieved contexts for this
  request, or is it only registered/indexed?
- Is a preview executing inside CCA, or is it an external handoff?
- Which failure is represented as unavailable, missing, or blocked rather than
  silently converted into a successful score?
- Can the team reproduce the claimed test or report without exposing secrets or
  private session data?

Continue with [USER_MANUAL.md](USER_MANUAL.md),
[DATA_AND_KB.md](DATA_AND_KB.md), [FAQ.md](FAQ.md), and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).
