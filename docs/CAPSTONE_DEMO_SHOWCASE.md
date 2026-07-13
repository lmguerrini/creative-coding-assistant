# Capstone Demo and Showcase Guide

## Product in one sentence

Creative Coding Assistant is a local AI workstation that helps creative coders
turn audiovisual intent into source-grounded, inspectable browser artifacts
while keeping workflow, runtime, persistence, evaluation, and failure
boundaries visible.

## Reviewer path

1. Read the [portfolio case study](PORTFOLIO_CASE_STUDY.md).
2. Start the backend and frontend with the commands below.
3. Open Demo Mode and inspect the ten-scenario catalog.
4. Use the four canonical browser fixtures for a reliable visual walkthrough.
5. Open Dashboard → Evaluation for retrieval, RAGAS, creative, and reliability
   evidence.
6. Follow the [exact 10-minute talk](TEN_MINUTE_PRESENTATION.md) and
   [five-minute Q&A](FIVE_MINUTE_QA.md).

The prepared slide deck is
[creative-coding-assistant-capstone.pptx](../outputs/creative-coding-assistant-capstone.pptx).

## Start locally

Prerequisites are Python 3.11+, the project environment, Node.js 22, and
installed frontend dependencies. A configured generation service is optional
for the deterministic reviewer route.

Backend:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

Health checks:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/ready
```

Frontend in a second terminal:

```bash
cd clients/nextjs
npm run dev
```

Open `http://127.0.0.1:3000`.

## Problem, solution, and impact

Creative intent is usually underspecified for code: the user may describe
motion, mood, geometry, sound, and interaction without naming a runtime or
artifact contract. The product translates that intent through an observable
workflow, can retrieve registered technical knowledge, extracts bounded
artifacts, selects a controlled browser surface where supported, and preserves
the session for refinement and review.

The impact is a shorter and more reviewable path between concept and prototype.
The product does not claim to replace a creative coder, certify generated code,
or control external production tools.

## Architecture at a glance

| Layer | Current responsibility | Boundary |
|---|---|---|
| Next.js workstation | Prompting, sessions, artifacts, preview, Dashboard, Demo Mode | Local browser product surface |
| Python WSGI backend | Assistant stream, health, knowledge, sessions | Local service; public hosting is not claimed |
| Workflow runtime | Intake, routing, retrieval, generation, extraction, preview preparation, review | Bounded Single/Multi/Auto paths |
| Chroma and embeddings | Local indexed technical knowledge and ranked retrieval | Registered/indexed sources only |
| Provider adapter | Optional text and image-bearing generation requests | Availability and output must be verified per run |
| Preview sandbox | Controlled p5.js, Three.js, GLSL, and Tone.js surfaces used by the showcase | Unsupported source remains code/export-only |
| Evaluation Lab | Separate Retrieval, AI/RAGAS, Creative, and Reliability evidence | No global product score |

## Canonical ten-flow Demo Mode catalog

| # | Scenario | Mode | What to show | Evidence boundary |
|---:|---|---|---|---|
| 1 | Polyrhythmic constellation | Single-Agent | Silent-ready Tone.js spectrum and optional Start audio | Deterministic fixture; automated checks do not assess playback quality |
| 2 | Recursive aurora garden | Single-Agent | p5.js golden-angle field, pointer parallax, fullscreen, refinement | Deterministic fixture; aesthetic quality is not objectively scored |
| 3 | Kinetic orbit sculpture | Single-Agent | Bundled Three.js r176 scene, nested rigs, changing frame | Deterministic fixture; plain bounded JavaScript only |
| 4 | Fractal solar bloom | Single-Agent | WebGL 1 shader compile and animated field | Deterministic fixture; not a frame-rate benchmark |
| 5 | Source-grounded design brief | Auto | Current-run retrieval status and source boundary | Local retrieval can be shown; no citation is invented when unavailable |
| 6 | Multi-agent production plan | Multi-Agent | Requested/resolved route and five bounded roles | Not parallel agents or external-tool automation |
| 7 | Single-agent line study | Single-Agent | Direct generator route and p5.js artifact | Separate planning/review route is intentionally skipped |
| 8 | Export handoff package | Auto | Inspectable handoff and project bundle | External target does not run inside the app |
| 9 | Reference-guided palette study | Single-Agent | Image attachment, request-scoped status, self-contained p5.js output | Image transport is tested; configured-model visual influence is unverified |
| 10 | Failure-recovery rehearsal | Auto | Controlled fallback, Partial outcome, Retry/code actions | Use only the validation fixture; do not simulate failure in a normal session |

The canonical source is
[`clients/nextjs/src/lib/demo-mode.ts`](../clients/nextjs/src/lib/demo-mode.ts).
The deterministic fallback contract is
[`demo/v9_5_golden_demo_dataset.json`](../demo/v9_5_golden_demo_dataset.json).

## Four canonical browser showcases

These fixtures are the recommended visual sequence because each has current
automated browser evidence:

### 1. Polyrhythmic constellation

- Artifact: `polyrhythmic-constellation.tone.js`
- Runtime: controlled Tone.js surface
- Visible proof: spectrum mounts in a ready state
- Interaction: optional Start audio, Stop, and Mute
- Boundary: keep it silent unless room audio has passed preflight

### 2. Recursive aurora garden

- Artifact: `recursive-aurora-garden.p5.js`
- Runtime: controlled global-mode p5.js surface
- Visible proof: animated golden-angle field
- Interaction: pointer parallax, fullscreen, colder-palette refinement
- Boundary: the “recursive” look is an artistic construction

### 3. Kinetic orbit sculpture

- Artifact: `kinetic-orbit-capstone.three.js`
- Runtime: locally bundled Three.js r176
- Visible proof: nonblank frame energy above the regression threshold and a
  changing frame signature
- Interaction: fullscreen and slower-motion refinement
- Boundary: React, standalone HTML, and remote modules are not executed here

### 4. Fractal solar bloom

- Artifact: `fractal-solar-bloom.frag`
- Runtime: bounded WebGL 1 fragment surface
- Visible proof: compiled, linked, nonblank animated field
- Interaction: fullscreen and higher-contrast refinement
- Boundary: no textures, samplers, or performance-benchmark claim

The exact E2E streams are labeled local deterministic fixtures. They prove the
product and renderer path, not a configured-provider generation.

## Data and retrieval evidence

The dated canonical retrieval report records a 1,445-record local collection,
seven benchmark queries, and top-five selection:

| Measure | Current result |
|---|---:|
| Queries with results | 7/7 |
| Requested-domain coverage | 18/19 (94.74%) |
| Substantive expected-source overlap | 16/23 (69.57%) |

The report contains source IDs, domains, rankings, distances, scores, and
fingerprints without publishing retrieved excerpt text. It measures retrieval
selection only.

## Evaluation evidence

The approved RAGAS fixture is intentionally shown as a separate historical
lane:

| Dimension | Four-row approved fixture |
|---|---:|
| Context Precision | 100.00% |
| Faithfulness | 29.58% |
| Answer Relevancy | 47.43% |
| Context Relevancy | 68.75% |
| Equal-weight macro | 61.44% |
| Context Recall | Missing |

This fixture uses synthetic public-safe content. It is not a current-product
score and is not comparable to the local retrieval report. Current-product
external RAGAS execution is unavailable under the present privacy boundary.

## Dated engineering validation

The 2026-07-13 local release run recorded:

- 2,688 backend tests plus 423 subtests, zero failures;
- 77 frontend test files and 553 tests;
- TypeScript typecheck, full-repository Ruff, and production build;
- 11/11 combined showcase, Dashboard, responsive, and direct Three.js checks;
- 4/4 exact ordinary-click showcase refinements in 19.2 seconds;
- 28/28 complete Playwright checks;
- semantic browser inspection of all 16 Dashboard pages, artifact preview,
  Evaluation boundaries, fullscreen restoration, nine themes, and the ten-flow
  catalog.

The backend run also emitted 454 third-party deprecation warnings. The counts
are a dated local snapshot, not independent acceptance or a permanent guarantee
for a different checkout.

## Capstone case alignment

| Official case | Current alignment |
|---|---|
| Case 1 — RAG knowledge assistant | Strong: Chroma, embeddings, registered sources, ranking, lineage, and measured coverage |
| Case 2 — task automation | Strong but bounded: observable workflow roles and recovery, without unattended external actions |
| Case 5 — code generation/debugging | Strong: runtime-specific source, preview diagnostics, refinement, and failure states |
| Case 6 — advanced topics/tools | Strong: orchestration, evaluation separation, multimodal transport, and runtime hardening |
| Case 3 — document search | Supporting: registered/indexed technical sources only |
| Case 4 — multimodal | Partial: request transport and privacy proven; visual influence unverified |
| Case 7 — performance tuning | Not claimed complete: no controlled parameter experiment dataset |

## Honest fallback route

If configured generation is unavailable, keep the selected prompt visible and
open the matching deterministic artifact. Say:

> The configured generation service is unavailable, so I am switching to the
> deterministic local artifact. This proves the workstation and renderer path,
> not a new provider response.

If retrieval is unavailable, show the dated canonical report and do not claim
current citations. If a renderer is unavailable, show source and the explicit
code/export boundary without calling it a successful preview.

## Presentation and upload

- [Demo narrative](DEMO_NARRATIVE.md)
- [Exact 10-minute presentation](TEN_MINUTE_PRESENTATION.md)
- [Five-minute Q&A](FIVE_MINUTE_QA.md)
- [Manual demo checklist](../demo/manual_demo_checklist.md)
- [Showcase upload preparation](../demo/showcase_upload_preparation.md)
- [Public documentation boundary audit](PUBLIC_DOCUMENTATION_BOUNDARY_AUDIT.md)

No showcase upload, public deployment, merge, push, or release action is
performed by these documents.
