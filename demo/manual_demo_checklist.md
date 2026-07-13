# Manual Demo Checklist

Run this checklist on the presentation machine after the final repository
checkout is selected. Do not pre-fill any pass. Record the observed state and
use the matching fallback when a condition is unavailable.

## 1. Repository and privacy preflight

- [ ] `git status --short` shows only understood changes.
- [ ] No real `.env`, credential, raw session, local database, or private image
      is visible in screen-sharing folders, editor tabs, browser history, or
      terminal scrollback.
- [ ] `README.md`, `docs/CAPSTONE_DEMO_SHOWCASE.md`, the slide deck, and the
      spoken script all describe ten Demo Mode flows.
- [ ] The four showcase names are exactly Polyrhythmic constellation, Recursive
      aurora garden, Kinetic orbit sculpture, and Fractal solar bloom.
- [ ] Deterministic showcase artifacts are labeled non-provider evidence.
- [ ] The reference image, if used, is synthetic or explicitly safe for a
      public presentation.
- [ ] Notifications, password managers, and unrelated browser profiles are
      hidden.

## 2. Services

Start the backend:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

- [ ] The process starts without an unexpected traceback.
- [ ] `curl --fail http://127.0.0.1:8000/api/health` succeeds.
- [ ] `curl --fail http://127.0.0.1:8000/api/health/ready` returns the observed
      readiness state; guarded configuration is not relabeled as ready.

Start the frontend in a second terminal:

```bash
cd clients/nextjs
npm run dev
```

- [ ] `http://127.0.0.1:3000` opens.
- [ ] The workstation finishes restoring and the composer is usable.
- [ ] Browser console and terminal show no unexplained product error.
- [ ] Demo Mode opens and lists ten scenarios.
- [ ] Dashboard opens and lists all 16 pages.

## 3. Focused automated preflight

Run early enough to leave time for recovery:

```bash
.venv/bin/python -m pytest -q \
  tests/test_multimodal_provider_inputs.py \
  tests/test_retrieval_demo_pack.py \
  tests/test_langgraph_workflow_integration.py

cd clients/nextjs
npm run typecheck
npm run test -- src/lib/demo-mode.test.ts src/components/workstation-shell.test.tsx
npm run test:e2e -- \
  e2e/demo-showcase-smoke.spec.js \
  e2e/phase3-product-exploration.spec.js \
  e2e/preview-sandbox-three.spec.js
```

- [ ] Record pass/fail counts and date; do not copy an older count as a new run.
- [ ] If a check fails, determine whether the presentation route is affected.
- [ ] Do not weaken a test, force a click, or hide a failure for rehearsal.

## 4. Four canonical browser fixtures

The focused Playwright command is the deterministic fixture proof. For the
spoken presentation, use only an ordinary product artifact prepared during the
same-machine preflight and keep its provenance visible. If that artifact is
unavailable, mark the live product step unavailable and show the matching
browser evidence; do not seed or relabel a normal session during the talk.

### Polyrhythmic constellation

- [ ] Exact artifact name is visible.
- [ ] Tone.js runtime is ready and the spectrum is visible.
- [ ] The scene is silent before Start audio.
- [ ] Start audio is used only if room output and volume have passed preflight.
- [ ] Stop/Mute returns to a safe state.
- [ ] Fullscreen restore preserves the intended layout.
- [ ] The slower-tempo refinement uses an ordinary pointer click.
- [ ] Refined state survives reload.

### Recursive aurora garden

- [ ] The p5.js canvas is visible and animated.
- [ ] Pointer movement produces parallax.
- [ ] Fullscreen enter/exit restores sidebar, inspector, and preview state.
- [ ] The colder-palette refinement uses an ordinary pointer click.
- [ ] Refined state survives reload.

### Kinetic orbit sculpture

- [ ] The runtime identifies bundled Three.js revision 176.
- [ ] The canvas is nonblank and visibly changing.
- [ ] Sculpture, orbit rings, lighting, and camera motion are legible.
- [ ] Fullscreen works without horizontal overflow.
- [ ] The slower-motion refinement uses an ordinary pointer click.
- [ ] Refined state survives reload.

### Fractal solar bloom

- [ ] The shader compiles and links in the bounded WebGL 1 surface.
- [ ] The field is nonblank and animated.
- [ ] Fullscreen restores cleanly.
- [ ] The higher-contrast refinement uses an ordinary pointer click.
- [ ] Refined state survives reload.

For all four, state the provenance shown by the product. Call the automated
artifact a “deterministic browser fixture”; do not infer a configured-provider
run from visual quality or from an older session.

## 5. Remaining six Demo Mode flows

### Source-grounded design brief

- [ ] Selected mode is Auto.
- [ ] Retrieval status and source boundary are visible.
- [ ] Current citations are shown only if current retrieval actually ran.
- [ ] If unavailable, use the canonical report and state its date/boundary.

### Multi-agent production plan

- [ ] Requested mode is Multi-Agent.
- [ ] Runtime evidence shows the resolved mode and bounded roles.
- [ ] No claim of parallel agents or external-tool automation is made.

### Single-agent line study

- [ ] Requested/resolved mode is Single-Agent.
- [ ] The direct generator route is visible.
- [ ] The p5.js source and preview agree.

### Export handoff package

- [ ] The artifact is described as an inspectable handoff.
- [ ] Export contents can be inspected.
- [ ] The external target is not described as running inside the app.

### Reference-guided palette study

- [ ] Demo Mode disables the run until an image is attached.
- [ ] The attached image is public-safe.
- [ ] Image status is request-scoped and clears after submission.
- [ ] Reloaded session contains no image bytes.
- [ ] If configured-model visual influence was not captured, state that only
      transport/privacy is proven.

### Failure-recovery rehearsal

- [ ] Use the controlled validation fixture, never a fabricated failure in a
      normal session.
- [ ] The outcome is Partial or unavailable as appropriate, not Success.
- [ ] Retry and code/local-draft actions are visible.
- [ ] No preview is claimed after an unavailable-renderer fallback.

## 6. Dashboard and evidence

- [ ] All pages open: Overview, Architecture, Workflow, Workspace, Runtime,
      Preview, Artifacts, Domains, Knowledge Base, AI & agents, Memory,
      Sessions, Telemetry, Evaluation, User Guide, and Settings.
- [ ] Artifacts shows a live preview only for an eligible artifact.
- [ ] Evaluation shows four separate lanes.
- [ ] Retrieval evidence reads 7/7 queries, 18/19 domains, and 16/23 source
      anchors for the dated report.
- [ ] RAGAS reads 61.44% only as a four-row approved-fixture macro.
- [ ] Faithfulness 29.58%, Answer Relevancy 47.43%, and missing Context Recall
      remain visible.
- [ ] No current-product RAGAS score is implied.
- [ ] Typography and Workspace settings use the shared preference state.
- [ ] Theme switching works for the intended presentation theme.

## 7. Timed rehearsal

- [ ] Present [the exact talk](../docs/TEN_MINUTE_PRESENTATION.md) in 10:00 or
      less without skipping evaluation boundaries.
- [ ] Leave Demo Mode and Dashboard pages pre-positioned to reduce navigation.
- [ ] Practice the [five-minute Q&A](../docs/FIVE_MINUTE_QA.md) with the
      canonical report and Evaluation Lab open.
- [ ] Practice the configured-service, retrieval, preview, and audio fallback
      sentences verbatim.
- [ ] Rehearse once with the network unavailable.
- [ ] Rehearse once without starting audio.
- [ ] Confirm the slide deck opens from
      `outputs/creative-coding-assistant-capstone.pptx`.

## 8. Presentation-machine notes

Record immediately before presenting:

| Check | Observed state | Time |
|---|---|---|
| Backend health |  |  |
| Backend readiness |  |  |
| Frontend load |  |  |
| Configured generation |  |  |
| Local retrieval |  |  |
| Tone.js silent-ready |  |  |
| p5.js preview |  |  |
| Three.js r176 preview |  |  |
| WebGL 1 preview |  |  |
| Fullscreen restore |  |  |
| Refinement and reload |  |  |
| Evaluation Lab |  |  |
| Offline fallback |  |  |

## 9. After rehearsal or presentation

- [ ] Stop audio and local services.
- [ ] Remove public-demo image references and temporary exports.
- [ ] Review screenshots/video for credentials, personal paths, notifications,
      private prompts, and unrelated sessions.
- [ ] Keep observed failures in the rehearsal notes; do not convert them into a
      pass after the fact.
- [ ] Upload or publish only after the separate showcase checklist is complete.
