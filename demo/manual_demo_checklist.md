# Manual Demo Checklist

Run this checklist before the Capstone demo rehearsal and again before the
showcase session.

## Preflight

- Confirm branch: `version-review/v8` during Grand Review rehearsal.
- Confirm no merge, push, tag, or freeze action is planned without maintainer
  approval.
- Keep `README.md`, `docs/CAPSTONE_DEMO_SHOWCASE.md`, and
  `docs/CAPSTONE_EVALUATION_ETHICS.md` open.
- Keep `docs/V8_CAPSTONE_EVIDENCE_MATRIX.md` open for reviewer evidence
  questions.
- Keep `demo/golden_demo_dataset.json` and `demo/demo_prompt_library.md` open.
- Open Creative Coding Assistant locally and use integrated `Demo Mode` as the
  primary presenter surface.
- Keep `demo/final_demo_launcher.html` ready from the local static QA server as
  the fallback/reviewer evidence launcher.
- Keep `demo/final_demo_suite.json` open for exact demo starts, success
  criteria, fallback paths, and talking points.
- Keep `demo/evaluation/` open for sanitized and redacted latest-live RAGAs
  input/results.
- Keep `demo/golden_artifacts/` open for generated p5.js, Three.js, GLSL, and
  Hydra artifact QA evidence.
- Confirm `assets/preview_current.png`, `assets/preview_v1.png`, and
  `assets/preview_v2.png` exist.

## Reliability

- Start the local demo backend:
  `.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000`.
- Start the local workstation:
  `cd clients/nextjs` then `npm run dev`.
- Open `http://127.0.0.1:3000`, select `Demo Mode`, and verify selecting a
  scenario loads the prompt into the normal assistant composer.
- Rehearse the full 10-minute demo:
  1:15 problem/purpose, 1:30 architecture, 4:15 primary golden flow,
  1:30 evaluation/ethics, 1:30 challenges/next steps.
- Rehearse the 5-minute Q&A with the evidence matrix open:
  data/sources, RAGAs, live execution versus Typed Domain Intelligence Layers, provider outage,
  retrieval outage, preview limitations, deployment target, privacy, and
  Chroma warnings.
- Rehearse the primary flow in 7 minutes as a compressed fallback.
- Rehearse the case-alignment summary in 90 seconds.
- Rehearse the evaluation and ethics summary in 60 seconds.
- Rehearse the offline fallback in 30 seconds.
- Rehearse the top reviewer questions from the evidence matrix in 90 seconds.
- Practice stating provider, retrieval, preview, network, and timing failures
  clearly without implying live success.

## Provider Failure Recovery

If provider access fails:

1. Say the provider path is unavailable in the current environment.
2. Keep the selected Demo Mode prompt visible, or switch to the matching prompt
   in `demo/golden_demo_dataset.json`.
3. Show the prepared preview screenshot.
4. Continue with architecture, evaluation, ethics, and limitations.

## Retrieval Failure Recovery

If retrieval is unavailable:

1. Say retrieval is not being run live.
2. Show the retrieval demo pack and evaluation workflow references.
3. Show the redacted latest-live RAGAs result rows and sanitized RAGAs evidence.

## Preview Failure Recovery

If preview is unavailable:

1. Use `assets/preview_current.png`.
2. Explain the preview/runtime boundary without implying a live preview
   succeeded.
3. Show `demo/golden_artifacts/browser_full_runtime_qa_results.json` as browser QA
   evidence:
   p5, Three.js, and Hydra rendered nonblank through real temporary QA runtime
   packages, and GLSL rendered nonblank through WebGL.
4. Continue with code/artifact planning and critique/refinement explanation.

## Reviewer Answer Cards

- What is the deployment target?
  A local Capstone workstation demo: backend on `127.0.0.1:8000`, Next.js dev
  server locally, no public deployment without maintainer approval.
- What actually rendered in browser QA?
  p5.js rendered nonblank with `p5@2.3.0`, Three.js rendered nonblank with
  `three@0.185.1`, Hydra rendered nonblank with `hydra-synth@1.4.0`, and GLSL
  rendered nonblank through WebGL. These packages were temporary QA
  dependencies, not app dependency changes.
- Is this a full performance benchmark?
  No. It is render/failure-boundary QA with uncapped draw-loop frame timing, not
  display FPS, load, soak, or deployment validation.
- What should happen if a live dependency fails?
  Keep the integrated Demo Mode path as the presenter frame, then switch to the
  prepared offline dataset, prompt library, screenshots, sanitized RAGAs
  evidence, static launcher fallback, golden artifact QA, and architecture
  walkthrough without implying live success.

## Approval Gate

Stop for maintainer review before showcase upload, merge, push, tag, or
freeze.
