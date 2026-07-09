# Final Product Polish Roadmap

Branch: `version-review/v8`  
Registered: 2026-07-09  
Execution rule: run FP tasks sequentially, one scoped task at a time; commit
each accepted task before moving on. Stop for blockers and required HITL
checkpoints. Do not merge, push, tag, freeze, or start V9 from this roadmap.

## Current Task

| Field | Value |
|---|---|
| Current task | FP-12 Preview / Code / Saved Ecosystem |
| Status | ACCEPTED |
| Scope boundary | Panel role clarity, Preview/Code/Saved handoff, duplicated noise, tab labels, and focused panel screenshots only |
| Screenshot evidence | Captured at 1440, 1024, and 720 widths |
| Latest task commit | FP-11 accepted at `ddb3d1e0` |

## Status Legend

- NOT_STARTED: task has not been audited or changed.
- IN_PROGRESS: focused task audit or implementation is underway.
- HITL_REVIEW: definition of done appears satisfied and awaits human approval.
- ACCEPTED: HITL accepted the task.
- BLOCKED: task cannot proceed without external input or an accepted boundary.

## Roadmap Table

| Step | Capability | Task principal | Definition of Done | Status | Evidence / paths | Blockers / boundaries | Latest accepted commit |
|---|---|---|---|---|---|---|---|
| FP-01 | Artifact & Workspace Integrity | Audit artifacts; audit untracked files; clean repository; no orphan artifacts; workspace integrity | Git clean; artifact ledger coherent | ACCEPTED | `demo/golden_artifacts/qa_manifest.json`; `demo/final_demo_suite.json`; `tests/test_golden_artifacts.py`; FP-01 validation log in task final response | No visual screenshots needed; `.runtime_pack/` private ignored copy is not public release evidence | `22414b2b8e5f42bcc729f09e06736512ad71a6aa` |
| FP-02 | Preview UX Excellence | Preview unavailable redesign; preview available canvas-first; eliminate User Mode HUD; overlay controls; no huge black canvas; no User Mode debug boxes | Preview looks like a real artistic canvas | ACCEPTED | `/tmp/cca-v8-fp02-preview-ux/manifest.json`; preview-region contact sheets at 1440, 1024, and 720 widths; `clients/nextjs/src/components/workstation-shell.test.tsx`; `clients/nextjs/src/lib/preview-sandbox-runtime.test.ts` | Developer Mode may show diagnostics; FP-05 owns full Developer Mode polish | `edd9adbe7fca48dfe8b9331eef13ce61eaab8173` |
| FP-03 | Chat UX Excellence | No HTML/JS/GLSL dumps; summary only; code to Code panel; artifact to Artifacts; preview to Preview | Chat readable like ChatGPT | ACCEPTED | `/tmp/cca-v8-fp03-chat-ux/focused-manifest.json`; `/tmp/cca-v8-fp03-chat-ux/fp03-focused-contact-sheet.png`; `clients/nextjs/src/components/workstation-shell.test.tsx` | User Mode uses the product-facing `Saved` tab for artifact routing; broader panel visual polish stays with FP-04/FP-08/FP-12 | `17d976d9` |
| FP-04 | User Mode Excellence | Minimal User Mode; inspector closed; max 3 tabs; no technical internals; responsive layout | Looks like a consumer app | ACCEPTED | `/tmp/cca-v8-fp04-user-mode/manifest.json`; `/tmp/cca-v8-fp04-user-mode/fp04-contact-sheet.png`; `clients/nextjs/src/components/workstation-shell.tsx`; `clients/nextjs/src/app/globals.css` | Scope limited to User Mode; Developer Mode polish stays with FP-05 | `112ff182` |
| FP-05 | Developer Mode Excellence | Full technical information; no overlap; no truncated text; readable details | Looks like a professional IDE | ACCEPTED | `/tmp/cca-v8-fp05-developer-mode/manifest.json`; `/tmp/cca-v8-fp05-developer-mode/fp05-layout-contact-sheet.png`; `clients/nextjs/src/components/preview-runtime-stage.tsx` | Developer Mode remains intentionally dense; iframe HUD disabled to avoid duplicate diagnostic overlap | `89c67105` |
| FP-06 | Demo Mode UX | Minimal demo cards; metadata in Developer Mode; coherent categories; explicit capability; no internal terminology | Reviewer understands each demo in 5 seconds | ACCEPTED | `/tmp/cca-v8-fp06-demo-mode/manifest.json`; `/tmp/cca-v8-fp06-demo-mode/fp06-contact-sheet.png`; `clients/nextjs/src/lib/demo-mode.test.ts`; `clients/nextjs/src/components/workstation-shell.test.tsx` | Scope limited to Demo Mode UI; demo pack coverage stays with FP-07 | `1f92e6bb` |
| FP-07 | Demo Pack Coverage | Every demo maps to a capability; single-agent; hybrid; multi-domain; retrieval; preview; output; Capstone mapping | No capability without a demo | ACCEPTED | `clients/nextjs/src/lib/demo-mode.test.ts`; `demo/final_demo_suite.json`; `clients/nextjs/src/lib/demo-mode.ts` | No live multi-agent execution path is claimed; geometry/morphogenesis remains multi-domain, not multi-agent | `67aa6afd` |
| FP-08 | Artifacts & Saved UX | Human labels; Saved browser; Code browser; Preview browser; no technical filenames; responsive layout | Artifact management is clear | ACCEPTED | `/tmp/cca-v8-fp08-artifacts-saved/manifest.json`; `/tmp/cca-v8-fp08-artifacts-saved/fp08-contact-sheet.png`; `clients/nextjs/src/components/workstation-shell.test.tsx`; `clients/nextjs/e2e/workstation-smoke.spec.js` | Developer Mode may show raw filenames; User Mode uses human labels and hides raw artifact filenames | `4c6d037f` |
| FP-09 | Input Composer UX | Codex/ChatGPT-style composer; minimal plus; bottom send; auto-grow; no status clutter; no overlap | Composer matches Codex philosophy | ACCEPTED | `/tmp/cca-v8-fp09-composer/manifest.json`; `/tmp/cca-v8-fp09-composer/fp09-composer-contact-sheet.png`; `clients/nextjs/src/components/workstation-shell.test.tsx`; `clients/nextjs/src/app/globals.css` | Developer Mode may keep compact composer status text; User Mode omits it to avoid clutter | `e8651d4e` |
| FP-10 | Codex Design System | Codex philosophy across typography, whitespace, flat surfaces, hierarchy, interactions; theme changes only colors | App feels part of Codex ecosystem | ACCEPTED | `/tmp/cca-v8-fp10-design-system/manifest.json`; `/tmp/cca-v8-fp10-design-system/fp10-design-contact-sheet.png`; `clients/nextjs/src/app/globals.css` | Aqua and Matrix preserved; FP-10 changes are Codex-theme surface flattening only | `1f6799e5` |
| FP-11 | Typography & Layout QA | Fix glued words; overflow; line wrapping; subtitles; cards; padding; margins | No visible typography defects | ACCEPTED | `/tmp/cca-v8-fp11-typography-layout/manifest.json`; `/tmp/cca-v8-fp11-typography-layout/fp11-typography-layout-contact-sheet.png`; `clients/nextjs/src/app/globals.css` | Collapsed inspector rail text creates false positive overflow; no page-level overflow remains | `ddb3d1e0` |
| FP-12 | Preview / Code / Saved Ecosystem | Coordinate three panels; preview is preview; Code is code; Saved is artifacts; no duplicated noise | Every panel has a clear role | ACCEPTED | `/tmp/cca-v8-fp12-panel-ecosystem/manifest.json`; `/tmp/cca-v8-fp12-panel-ecosystem/fp12-panel-ecosystem-contact-sheet.png`; `clients/nextjs/src/components/workstation-shell.tsx`; `clients/nextjs/src/components/workstation-shell.test.tsx` | Preview remains visual/fallback, Code remains source, Saved remains artifact management; repeated same-runtime artifacts use numbered human labels | Pending FP-12 commit |
| FP-13 | KB & Retrieval UX | Global KB status; current retrieval state; Check KB; refresh if safe; correct wording; User vs Developer | Reviewer sees that RAG is real | NOT_STARTED | TBD | Do not fake refresh if API/command is unsafe | TBD |
| FP-14 | LangSmith Observability | Real trace; visible workflow; planner; retrieval; timing; demo instructions | ACTIVE TESTED | NOT_STARTED | TBD | Do not claim active tracing without credentials and evidence | TBD |
| FP-15 | Complete Smoke Coverage Matrix | Every important function has at least one full app smoke; Live/Mock/Artifact distinguished | Coverage 100% | NOT_STARTED | TBD | TBD | TBD |
| FP-16 | Extra Scenario Validation | Space colonization; DLA; differential growth; prompts outside Demo Pack | Robust beyond predefined demos | NOT_STARTED | TBD | Digital Morphogenesis references remain inspiration unless indexed | TBD |
| FP-17 | Failure Injection | Provider offline; retrieval fail; preview fail; artifact fail; timeout; backend restart; UI fallback | Demo resilient | NOT_STARTED | TBD | TBD | TBD |
| FP-18 | Performance & Demo Timing | Real times; real tokens; optimal demo order; presenter timing; optimized demo pack | 10-minute demo optimized | NOT_STARTED | TBD | Do not invent estimates | TBD |
| FP-19 | README & Docs | Professional README; Evaluator Start Here; future roadmap; public claims; Capstone sections | Reviewer understands project easily | NOT_STARTED | TBD | Public claims stay conservative | TBD |
| FP-20 | Internal Reviewer Simulation | STL; AI reviewer; senior reviewer; peer reviewer; Q&A; weakness fixing | No easy attack point | NOT_STARTED | TBD | Treat internal reviewers as advisory, not objective truth | TBD |
| FP-21 | Product Visual Review | Mandatory manual screenshots; human review over automated QA | Screenshots approved by HITL | NOT_STARTED | TBD | No pass without saved screenshot evidence | TBD |
| FP-22 | Codex Final Audit | Complete audit after all micro-passes; no fix unless blocker; final coverage | Codex PASS | NOT_STARTED | TBD | Must wait for FP-01 through FP-21 | TBD |
| FP-23 | Junie Independent Review | Independent review; last blocker fixes | Junie PASS | NOT_STARTED | TBD | Must wait for prior HITL approvals | TBD |
| FP-24 | Version Freeze | Merge; push; tag; release candidate; demo ready | V8 Frozen and ready for Turing Capstone Demo | NOT_STARTED | TBD | Requires explicit HITL freeze approval; not allowed in this task | TBD |

## FP-01 Acceptance Criteria

- Current branch is `version-review/v8`.
- Git status starts clean and ends clean after commit.
- `demo/golden_artifacts/qa_manifest.json` references only existing public
  artifact paths.
- `demo/final_demo_suite.json` contains 8 demo flows and does not depend on
  missing artifacts.
- No public artifact/demo file references `p5_sacred_geometry_sketch.js`.
- Public artifact/demo surfaces do not expose forbidden product/internal names.
- Generated/saved/preview artifact locations have no untracked public artifacts.
- Runtime Pack private artifact copy is ignored and not used as public evidence.

## FP-01 Artifact Audit Notes

- `demo/golden_artifacts/p5_sacred_geometry_sketch.js` is not tracked and is not
  present in `demo/golden_artifacts/`.
- The public p5 golden artifact is
  `demo/golden_artifacts/p5_generative_morphogenesis_sketch.js`.
- The prior internal copy is
  `.runtime_pack/active/runtime/private_artifacts/p5_geometry_sketch_internal.js`;
  it is ignored under `.runtime_pack/` and is not a public release artifact.
- `data/artifacts/` contains only `.gitkeep`; no saved/generated public artifact
  is orphaned there.
- Public golden artifact ledger currently contains 4 artifacts:
  `p5_generative_morphogenesis_sketch`,
  `three_audio_reactive_scene`,
  `glsl_kaleidoscope_field`, and `hydra_feedback_lattice`.

## FP-01 Validation Evidence

| Check | Result |
|---|---|
| `git status --short --branch` | Clean at audit start on `version-review/v8` |
| `tests/test_golden_artifacts.py` | Passed: 9 tests |
| Custom manifest/demo-suite path check | Passed: 4 manifest artifacts, 8 demo flows |
| Public artifact forbidden-name scan | Passed: no matches in `demo/golden_artifacts`, `demo/final_demo_suite.json`, `demo/final_demo_launcher.html`, or `demo/golden_demo_dataset.json` |
| Old public filename scan | Passed: no matches for `p5_sacred_geometry_sketch.js` in public demo/artifact surfaces |
| Runtime Pack private artifact status | Ignored `.runtime_pack/`; not public release evidence |

## FP-02 Acceptance Criteria

- Preview unavailable in User Mode is a compact fallback card with a short title,
  one-sentence explanation, and clear actions for Code and Saved.
- Preview available in User Mode is canvas-first, with the visual surface taking
  priority and only minimal overlay controls.
- User Mode hides runtime HUD text, renderer metrics, raw diagnostics, and debug
  boxes.
- Developer Mode still exposes preview runtime diagnostics.
- The preview unavailable path does not show a large empty black canvas.
- Screenshots are saved and manually inspected before status is set to
  HITL_REVIEW.

## FP-02 Preview Audit Notes

- Current implementation already routes unavailable User Mode preview through
  `.previewUserFallbackCard`.
- Current implementation renders available preview through `.previewShelf` with
  `data-user-mode="true"` in User Mode.
- Current implementation passes `showStatusOverlay: false` to the sandbox
  runtime in User Mode, hiding the iframe runtime status HUD.
- Developer Mode keeps the preview runtime overlay visible by design; full
  Developer Mode density and overlap review is reserved for FP-05.

## FP-02 Screenshot Evidence

| Width | Full-page evidence | Focused preview evidence |
|---|---|---|
| 1440 | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-contact-sheet-1440.png` | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-1440.png` |
| 1024 | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-contact-sheet-1024.png` | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-1024.png` |
| 720 | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-contact-sheet-720.png` | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-720.png` |

## FP-02 Validation Evidence

| Check | Result |
|---|---|
| Playwright screenshot capture | Passed: User unavailable, User available, and Developer diagnostics captured at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: focused preview sheets show compact fallback, canvas-first User Mode, no User Mode HUD, and Developer diagnostics retained |
| Automated preview metrics | Passed: `hasRuntimeHudTextInUserMode=false`, `overlayVisible=false` in User Mode, `overlayVisible=true` in Developer Mode across all captured widths |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/preview-sandbox-runtime.test.ts` (`100` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed: `npm run test:e2e:smoke` (`8` tests) |
| Hygiene | Passed: `git diff --check`; Runtime Pack hygiene OK |
| Accepted boundary | Developer Mode diagnostics can overlay the canvas; FP-05 owns full Developer Mode visual polish |

## FP-03 Acceptance Criteria

- User Mode chat does not display raw HTML, JavaScript, GLSL, TypeScript, or
  generated source-code blocks.
- Long or code-bearing assistant output is summarized in the conversation.
- Generated source remains inspectable in the Code panel.
- Generated artifacts remain inspectable through the User Mode Saved surface.
- Previewable output routes to the Preview surface.
- Regression coverage proves User Mode chat does not flood with generated code.

## FP-03 Chat UX Audit Notes

- `buildUserModeAssistantSummary` strips code fences, script/style tags, HTML
  tags, p5 setup/draw signatures, Three.js imports, and GLSL fragment shader
  signatures from User Mode assistant messages.
- `buildAssistantConversationSummary` summarizes code-bearing or long final
  answers before they are written into the conversation.
- User Mode artifact routing is intentionally labeled `Saved`, not `Artifacts`,
  while Developer Mode keeps the technical `Artifacts` inspector label.
- Full-page screenshots can show duplicated sticky headers as a Playwright
  artifact at narrow widths; focused region screenshots are the primary visual
  evidence for FP-03.

## FP-03 Screenshot Evidence

| Width | Chat evidence | Code evidence | Saved evidence | Preview evidence |
|---|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-chat-1440.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-code-1440.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-saved-1440.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-preview-1440.png` |
| 1024 | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-chat-1024.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-code-1024.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-saved-1024.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-preview-1024.png` |
| 720 | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-chat-720.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-code-720.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-saved-720.png` | `/tmp/cca-v8-fp03-chat-ux/fp03-focused-preview-720.png` |

Contact sheet: `/tmp/cca-v8-fp03-chat-ux/fp03-focused-contact-sheet.png`.

## FP-03 Validation Evidence

| Check | Result |
|---|---|
| Focused regression | Passed: `npx vitest run src/components/workstation-shell.test.tsx --testNamePattern "generated code\|mixed generated code"` (`2` tests) |
| Full workstation tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx` (`84` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright focused screenshot capture | Passed: chat, Code, Saved, and Preview captures at 1440, 1024, and 720 widths |
| Playwright smoke | Passed: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo/` excluding internal review roadmaps |
| Docs Mermaid check | Not applicable: no Mermaid blocks in touched docs |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | FP-03 proves chat/code routing; FP-04, FP-08, and FP-12 own broader User Mode, Saved, and panel ecosystem visual polish |

## FP-04 Acceptance Criteria

- User Mode opens with the right inspector collapsed.
- User Mode topbar uses human status copy and hides provider/token telemetry.
- Expanded User Mode inspector shows only Preview, Code, and Saved tabs.
- User Mode hides Overview, Runtime, Workflow, Telemetry, Retrieval, Artifacts,
  raw IDs, trace metadata, and runtime diagnostics.
- User Mode responsive layouts remain readable at 1440, 1024, and 720 widths.
- Human-visible screenshots are inspected before acceptance.

## FP-04 User Mode Audit Notes

- Browser screenshots found a real wide-screen blocker: the collapsed inspector
  rail still reserved the full inspector column, making the main workspace feel
  like a dense dashboard.
- The collapsed desktop inspector now renders as a compact overlay rail while
  the main workspace reclaims the layout width.
- The existing laptop/mobile stacked collapsed-inspector behavior is preserved.
- User Mode session status now shows `Ready / Start a prompt`,
  `Working / Generating response`, `Complete / Output ready`, or a compact
  attention state instead of workflow steps or provider/token telemetry.
- Empty-state copy now uses `Ways to work` and saved-output language rather than
  visible workflow-state terminology.
- Developer Mode retains diagnostic tabs and provider/usage metadata by design;
  FP-05 owns Developer Mode visual polish.

## FP-04 Screenshot Evidence

| Width | Default User Mode | Expanded User Inspector | Generated Output User Mode |
|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp04-user-mode/fp04-user-default-1440.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-expanded-1440.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-generated-1440.png` |
| 1024 | `/tmp/cca-v8-fp04-user-mode/fp04-user-default-1024.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-expanded-1024.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-generated-1024.png` |
| 720 | `/tmp/cca-v8-fp04-user-mode/fp04-user-default-720.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-expanded-720.png` | `/tmp/cca-v8-fp04-user-mode/fp04-user-generated-720.png` |

Contact sheet: `/tmp/cca-v8-fp04-user-mode/fp04-contact-sheet.png`.

## FP-04 Validation Evidence

| Check | Result |
|---|---|
| Focused User Mode tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx --testNamePattern "User Mode inspector\|compact User Mode fallback\|restores a persisted workspace session"` (`3` tests) |
| Full workstation tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx` (`84` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright screenshot capture | Passed: default, expanded-inspector, and generated-output User Mode captures at 1440, 1024, and 720 widths |
| Playwright smoke | Passed: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo/` excluding internal review roadmaps |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | User Mode is clean and responsive; Developer Mode density and diagnostics remain in scope for FP-05 |

## FP-05 Acceptance Criteria

- Developer Mode exposes the full inspector tab set: Overview, Preview, Runtime,
  Code, Workflow, Telemetry, Artifacts, and Retrieval.
- Developer Mode keeps provider, runtime, workflow, retrieval, trace, and
  telemetry details visible.
- Developer Mode remains readable at 1440, 1024, and 720 widths.
- No visible overlapping text, cropped labels, or glued words are accepted in the
  audited Developer Mode surfaces.
- Human-visible screenshots are inspected before acceptance.

## FP-05 Developer Mode Audit Notes

- Browser screenshots found a real narrow-width blocker: the iframe runtime HUD
  and React runtime overlay duplicated diagnostic text over the preview canvas.
- The iframe HUD is now disabled; Developer Mode keeps the React runtime overlay,
  Preview inspector, and Runtime console as the diagnostic surfaces.
- Workflow and Telemetry panels are intentionally long and dense, but individual
  cards and labels remain readable at 720 width.
- Developer Mode still exposes provider/usage pending states and workflow step
  labels by design; these are hidden only in User Mode.

## FP-05 Screenshot Evidence

| Width | Developer layout | Workflow | Runtime | Telemetry | Retrieval |
|---|---|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-layout-1440.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-workflow-1440.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-runtime-1440.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-telemetry-1440.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-retrieval-1440.png` |
| 1024 | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-layout-1024.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-workflow-1024.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-runtime-1024.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-telemetry-1024.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-retrieval-1024.png` |
| 720 | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-layout-720.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-workflow-720.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-runtime-720.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-telemetry-720.png` | `/tmp/cca-v8-fp05-developer-mode/fp05-developer-retrieval-720.png` |

Layout contact sheet:
`/tmp/cca-v8-fp05-developer-mode/fp05-layout-contact-sheet.png`.

## FP-05 Validation Evidence

| Check | Result |
|---|---|
| Focused preview/runtime tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/preview-sandbox-runtime.test.ts --testNamePattern "runtime\|Renderer health overlay\|preview runtime"` (`37` tests) |
| Full workstation and preview tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/preview-sandbox-runtime.test.ts` (`101` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright screenshot capture | Passed: Developer layout and Workflow, Runtime, Telemetry, Retrieval captures at 1440, 1024, and 720 widths |
| Playwright smoke | Passed: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo/` excluding internal review roadmaps |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | Developer Mode remains technical and denser than User Mode; the duplicate iframe HUD was removed to prevent canvas diagnostic overlap |

## FP-06 Acceptance Criteria

- Demo Mode opens inside Creative Coding Assistant.
- All 8 curated scenarios are visible.
- Scenario selection preloads the normal assistant composer.
- User Mode Demo cards show concise scenario title, capability, runtime,
  expected output, estimated time, short prompt preview, and load state.
- Developer Mode Demo cards expose full timing, token, provider, retrieval,
  preview, fallback, source-boundary, validation, and evidence metadata.
- Public category labels are coherent: Three.js, p5.js, GLSL, Hydra, Retrieval,
  Concept Translation, Visual Planning, and Installation Planning.
- Demo Mode contains no forbidden public/internal terminology.

## FP-06 Demo Mode Audit Notes

- No product code change was required for FP-06.
- Browser assertions verified all 8 scenario categories, normal composer preload,
  User Mode metadata hiding, Developer Mode metadata visibility, and forbidden
  public-term boundaries.
- User Mode intentionally keeps `Validates:` visible because the task requires
  every demo to state what capability it validates.
- Developer Mode remains denser because it is the appropriate surface for token,
  source-boundary, fallback, and evidence metadata.

## FP-06 Screenshot Evidence

| Width | User Mode Demo | Developer Mode Demo |
|---|---|---|
| 1440 | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-user-1440.png` | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-developer-1440.png` |
| 1024 | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-user-1024.png` | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-developer-1024.png` |
| 720 | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-user-720.png` | `/tmp/cca-v8-fp06-demo-mode/fp06-demo-developer-720.png` |

Contact sheet: `/tmp/cca-v8-fp06-demo-mode/fp06-contact-sheet.png`.

## FP-06 Validation Evidence

| Check | Result |
|---|---|
| Focused Demo Mode tests | Passed: `npx vitest run src/lib/demo-mode.test.ts src/components/workstation-shell.test.tsx --testNamePattern "Demo Mode\|demo mode\|final eight curated\|public Capstone"` (`7` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright screenshot capture | Passed: User and Developer Demo Mode captures at 1440, 1024, and 720 widths |
| Playwright smoke | Passed: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo` excluding internal review roadmaps |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | FP-06 covers Demo Mode UX only; FP-07 owns broader demo pack coverage mapping |

## FP-07 Acceptance Criteria

- All 8 Demo Mode scenarios map to a distinct public capability.
- Demo pack includes single-domain, hybrid retrieval, multi-domain, and planning
  workflow coverage.
- Retrieval, preview, fallback, expected output, Capstone presentation, evidence,
  source-boundary, and validation-path metadata are present for every scenario.
- Hydra remains bounded to the validated local `hydra-synth` artifact path.
- Geometry/morphogenesis remains multi-domain and is not presented as live
  multi-agent execution.
- No live multi-agent, Studio Mode, or critic-refinement execution claim appears
  in app-facing Demo Mode scenario metadata.

## FP-07 Demo Pack Coverage Audit Notes

- Capability coverage is complete across the final 8 scenarios:
  Three.js visual system, p5.js generative growth, GLSL shader validation,
  Hydra feedback-pattern runtime, retrieval-grounded answer, concept
  translation, multi-runtime morphogenesis, and installation planning.
- Workflow coverage is represented as single-domain generation, hybrid
  retrieval-grounded generation, bounded multi-domain routing/artifact QA, and
  planning with retrieval.
- Retrieval coverage is present in every scenario metadata row through either
  measured retrieved contexts or RAGAs/retrieval evidence.
- Preview coverage is explicit for previewable runtimes and explicitly bounded
  for answer/planning flows where preview is not required or depends on the
  generated artifact choice.
- Capstone mapping is present through `recommendedForDemo`, `presentationTime`,
  `talkingPoint`, `evidence`, `sourceBoundary`, and `validationPath`.
- Multi-agent coverage remains an accepted boundary: no live multi-agent path is
  validated, and the demo pack does not claim one.

## FP-07 Validation Evidence

| Check | Result |
|---|---|
| Focused Demo Pack tests | Passed: `npx vitest run src/lib/demo-mode.test.ts` (`7` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local server permission: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo` excluding internal review roadmaps |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | FP-07 verifies multi-domain and planning coverage, but does not claim live multi-agent execution |

## FP-08 Acceptance Criteria

- User Mode Saved, Code, and Preview surfaces use human-facing labels instead of
  raw artifact filenames.
- Saved outputs remain readable at 1440, 1024, and 720 widths without cropped or
  overlapping artifact labels.
- Code remains available through the Code panel while long generated source is
  kept out of the User Mode chat.
- Preview routes to the Preview panel and hides raw runtime source titles in
  User Mode.
- Developer Mode keeps raw filenames and full artifact metadata for debugging.
- First-run User Mode keeps the right inspector collapsed by default.

## FP-08 Artifacts & Saved Audit Notes

- User Mode now treats `Saved` as the artifact browser while Developer Mode
  keeps the technical `Artifacts` tab.
- Preview runtime status, diagnostics, and user-facing runtime errors replace
  raw source titles with the sanitized preview surface title when diagnostics are
  hidden.
- First-run workspace defaults now prefer User Mode minimalism: inspector
  collapsed and debug panels hidden unless Developer Mode is requested.
- E2E smoke expectations were updated to validate the accepted User Mode
  contract: collapsed inspector by default, human artifact labels in Saved,
  no raw filenames in User Preview/Saved, and retrieval internals available only
  after switching to Developer Mode.

## FP-08 Screenshot Evidence

| Width | Saved | Code | Preview | Developer artifacts |
|---|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-saved-1440.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-code-1440.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-preview-1440.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-developer-artifacts-1440.png` |
| 1024 | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-saved-1024.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-code-1024.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-preview-1024.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-developer-artifacts-1024.png` |
| 720 | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-saved-720.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-code-720.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-user-preview-720.png` | `/tmp/cca-v8-fp08-artifacts-saved/fp08-developer-artifacts-720.png` |

Contact sheet: `/tmp/cca-v8-fp08-artifacts-saved/fp08-contact-sheet.png`.

## FP-08 Validation Evidence

| Check | Result |
|---|---|
| Screenshot capture | Passed: User Saved, User Code, User Preview, and Developer Artifacts captured at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: User Mode uses human labels and avoids raw filenames/cropped artifact cards; Developer Mode remains dense but readable |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/workspace-persistence.test.ts src/lib/preview-sandbox-runtime.test.ts` (`112` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local server permission: `npm run test:e2e:smoke` (`8` tests) |
| Public docs/demo claim scan | Passed: no matches for forbidden public product terms in `README.md`, `docs/`, or `demo` excluding internal review roadmaps |
| Docs Mermaid check | Passed: `.venv/bin/python scripts/v7_quality_gates.py docs-mermaid` |
| Hygiene | Passed: `git diff --check`; Runtime Pack private directories remain ignored |
| Accepted boundary | Developer Mode intentionally keeps raw filenames and full artifact metadata; FP-12 owns broader panel ecosystem coordination |

## FP-09 Acceptance Criteria

- User Mode composer uses a calm single input frame with a minimal `+`
  attachment button.
- Send button is bottom-aligned and integrated into the composer edge.
- Textarea grows with multi-line prompts without overlapping the attachment menu
  or send control.
- User Mode does not show composer status clutter such as `Type a prompt to
  begin` or `Ready to generate`.
- Attachment menu opens without covering prompt text at 1440, 1024, or 720
  widths.
- Developer Mode may keep compact composer status text for operator feedback.

## FP-09 Input Composer Audit Notes

- User Mode no longer renders the visible composer status label; the top session
  status remains the user-facing working-state surface.
- Developer Mode retains the compact composer status text for diagnostic
  feedback.
- The send button is positioned inside the composer frame, matching the
  Codex/ChatGPT-style single composer control.
- The textarea now auto-sizes from content up to the existing maximum height and
  then scrolls internally.
- Browser screenshots found and fixed a real 720-width issue where the
  attachment menu overlapped a grown textarea. The menu now anchors to the full
  composer frame instead of the plus button.

## FP-09 Screenshot Evidence

| Width | Empty User composer | Filled User composer | Attachment menu | Developer composer |
|---|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp09-composer/fp09-user-composer-empty-1440.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-filled-1440.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-attachment-1440.png` | `/tmp/cca-v8-fp09-composer/fp09-developer-composer-filled-1440.png` |
| 1024 | `/tmp/cca-v8-fp09-composer/fp09-user-composer-empty-1024.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-filled-1024.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-attachment-1024.png` | `/tmp/cca-v8-fp09-composer/fp09-developer-composer-filled-1024.png` |
| 720 | `/tmp/cca-v8-fp09-composer/fp09-user-composer-empty-720.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-filled-720.png` | `/tmp/cca-v8-fp09-composer/fp09-user-composer-attachment-720.png` | `/tmp/cca-v8-fp09-composer/fp09-developer-composer-filled-720.png` |

Contact sheet: `/tmp/cca-v8-fp09-composer/fp09-composer-contact-sheet.png`.

## FP-09 Validation Evidence

| Check | Result |
|---|---|
| Screenshot capture | Passed: empty, filled, attachment-menu, and Developer composer captures at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: no User Mode status clutter, send aligned, textarea readable, attachment menu does not overlap prompt text |
| Focused composer tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx --testNamePattern "composer\|Add attachment\|Send prompt"` (`3` tests) |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/workspace-persistence.test.ts src/lib/preview-sandbox-runtime.test.ts` (`113` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local server permission: `npm run test:e2e:smoke` (`8` tests) |
| Accepted boundary | Developer Mode keeps compact composer status text; User Mode relies on top session state and conversation text |

## FP-10 Acceptance Criteria

- Codex theme reads as neutral graphite, restrained, flat, and professional.
- Primary User Mode workspace remains the dominant visual surface.
- Main panels, Demo Mode, Preview, and Developer Mode use consistent borders,
  shadows, and surface treatment.
- Existing layout, workflow behavior, Aqua theme, and Matrix theme are not
  changed by this pass.
- Screenshots are saved and inspected at 1440, 1024, and 720 widths before
  acceptance.

## FP-10 Codex Design System Audit Notes

- Browser screenshots showed the Codex theme was already close, but main panels
  still carried older shadow depth and stacked-card visual weight.
- The Codex theme now flattens topbar, session, demo, preview, and inspector
  surfaces with subtler borders and reduced shadows.
- Utility popovers keep a small shadow because they need layering over the
  workspace.
- Empty-state, demo-scenario, saved-output, preview, retrieval, artifact, and
  workflow cards share the same Codex-theme low-contrast surface treatment.
- This pass intentionally did not modify layout, component hierarchy, Aqua,
  Matrix, or demo scenario content.

## FP-10 Screenshot Evidence

| Width | User default | User Demo Mode | Developer Mode |
|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp10-design-system/fp10-user-default-1440.png` | `/tmp/cca-v8-fp10-design-system/fp10-user-demo-1440.png` | `/tmp/cca-v8-fp10-design-system/fp10-developer-default-1440.png` |
| 1024 | `/tmp/cca-v8-fp10-design-system/fp10-user-default-1024.png` | `/tmp/cca-v8-fp10-design-system/fp10-user-demo-1024.png` | `/tmp/cca-v8-fp10-design-system/fp10-developer-default-1024.png` |
| 720 | `/tmp/cca-v8-fp10-design-system/fp10-user-default-720.png` | `/tmp/cca-v8-fp10-design-system/fp10-user-demo-720.png` | `/tmp/cca-v8-fp10-design-system/fp10-developer-default-720.png` |

Contact sheet: `/tmp/cca-v8-fp10-design-system/fp10-design-contact-sheet.png`.

## FP-10 Validation Evidence

| Check | Result |
|---|---|
| Screenshot capture | Passed: User default, User Demo Mode, and Developer Mode captures at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: Codex surfaces are flatter and calmer; no layout regression or obvious overlap in audited screenshots |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx --testNamePattern "theme\|settings preferences\|User Mode\|Demo Mode"` (`11` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local server permission: `npm run test:e2e:smoke` (`8` tests) |
| Accepted boundary | FP-10 is a Codex-theme surface polish only; FP-11 owns deeper typography/overflow QA |

## FP-11 Acceptance Criteria

- No visible glued words, cropped labels, or awkward horizontal overflows in the
  audited User, Demo Mode, and Developer surfaces.
- Demo Mode scenario cards wrap cleanly at 720 width.
- Inspector tab labels guard against small-width overflow.
- No page-level horizontal overflow appears at 1440, 1024, or 720 widths.
- Screenshots are saved and manually inspected before acceptance.

## FP-11 Typography & Layout Audit Notes

- Browser-side overflow audit found no page-level horizontal overflow.
- The only real layout issue found was a 720-width Demo Mode scenario list
  overflow of about 20px. Scenario lists now use a safer responsive grid track
  and min-width guards.
- A tiny Developer `Overview` tab label overflow at 1024 width was fixed with a
  wrapping guard on inspector tab labels.
- Collapsed inspector rail text can appear as an overflow metric because it is a
  narrow rail control; it does not create page-level overflow or visible
  cropping in the screenshots.

## FP-11 Screenshot Evidence

| Width | User default | User Demo Mode | Developer Mode |
|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp11-typography-layout/fp11-user-default-1440.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-user-demo-1440.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-developer-1440.png` |
| 1024 | `/tmp/cca-v8-fp11-typography-layout/fp11-user-default-1024.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-user-demo-1024.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-developer-1024.png` |
| 720 | `/tmp/cca-v8-fp11-typography-layout/fp11-user-default-720.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-user-demo-720.png` | `/tmp/cca-v8-fp11-typography-layout/fp11-developer-720.png` |

Contact sheet:
`/tmp/cca-v8-fp11-typography-layout/fp11-typography-layout-contact-sheet.png`.

## FP-11 Validation Evidence

| Check | Result |
|---|---|
| Browser overflow audit | Passed: no page-level overflow and no relevant element overflow at 1440, 1024, or 720 widths after fixes |
| Screenshot capture | Passed: User default, User Demo Mode, and Developer Mode captures at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: no visible glued words, cropped labels, or unreadable card/button wrapping in audited screenshots |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx --testNamePattern "Demo Mode\|User Mode\|Developer Mode\|inspector\|theme"` (`25` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local server permission: `npm run test:e2e:smoke` (`8` tests) |
| Accepted boundary | FP-11 fixes visible typography/layout defects only; broader panel role coordination remains FP-12 |

## FP-12 Acceptance Criteria

- Preview panel stays visual-first and does not expose generated source code in
  User Mode.
- Code panel remains the source-code inspection surface and shows the generated
  code for the active artifact.
- Saved panel remains the artifact-management surface with human artifact labels.
- User Mode does not expose raw artifact filenames in Preview or Saved.
- Repeated same-runtime saved outputs stay distinguishable without falling back
  to technical filenames.
- Screenshots are saved and manually inspected at 1440, 1024, and 720 widths.

## FP-12 Panel Ecosystem Audit Notes

- User Mode Preview, Code, and Saved now have clearer boundaries: Preview shows
  the canvas/fallback, Code shows source, and Saved shows artifact management.
- Repeated same-runtime outputs are numbered with human labels such as
  `P5 Sketch 1` and `P5 Sketch 2`; raw filenames remain reserved for Developer
  Mode.
- A single saved output does not render an extra selectable Saved list, reducing
  repeated noise while keeping the active artifact summary and actions visible.
- Browser assertions verified that Preview text does not contain `createCanvas`,
  Code does contain the source, and Saved contains `P5 Sketch` without the raw
  `fp12-orbit-field.p5.js` filename.

## FP-12 Screenshot Evidence

| Width | Preview | Code | Saved |
|---|---|---|---|
| 1440 | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-preview-1440.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-code-1440.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-saved-1440.png` |
| 1024 | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-preview-1024.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-code-1024.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-saved-1024.png` |
| 720 | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-preview-720.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-code-720.png` | `/tmp/cca-v8-fp12-panel-ecosystem/fp12-user-saved-720.png` |

Contact sheet:
`/tmp/cca-v8-fp12-panel-ecosystem/fp12-panel-ecosystem-contact-sheet.png`.

## FP-12 Validation Evidence

| Check | Result |
|---|---|
| Screenshot capture | Passed: Preview, Code, and Saved captured at 1440, 1024, and 720 widths |
| Screenshot human inspection | Passed: Preview is visual, Code is source, Saved is artifact management with human labels and no raw filenames |
| Focused frontend tests | Passed: `npx vitest run src/components/workstation-shell.test.tsx src/lib/workspace-persistence.test.ts src/lib/preview-sandbox-runtime.test.ts` (`114` tests) |
| Typecheck | Passed: `npm run typecheck` |
| Playwright smoke | Passed after sandbox rerun with local browser permission: `npm run test:e2e:smoke` (`8` tests) |
| Accepted boundary | Developer Mode may keep raw artifact metadata; FP-12 only tightens User Mode panel handoff and labels |
