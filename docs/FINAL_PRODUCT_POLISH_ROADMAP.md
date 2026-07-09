# Final Product Polish Roadmap

Branch: `version-review/v8`  
Registered: 2026-07-09  
Execution rule: run exactly one FP task per HITL cycle. Do not merge, push,
tag, freeze, or start V9 from this roadmap.

## Current Task

| Field | Value |
|---|---|
| Current task | FP-07 Demo Pack Coverage |
| Status | ACCEPTED |
| Scope boundary | Demo scenario capability mapping, workflow coverage, Capstone mapping, unsupported multi-agent claim audit, and focused metadata tests only |
| Screenshot evidence | Not required unless UI coverage issues are found; FP-06 contains Demo Mode screenshots |
| Latest task commit | FP-07 accepted; pending commit creation |

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
| FP-07 | Demo Pack Coverage | Every demo maps to a capability; single-agent; hybrid; multi-domain; retrieval; preview; output; Capstone mapping | No capability without a demo | ACCEPTED | `clients/nextjs/src/lib/demo-mode.test.ts`; `demo/final_demo_suite.json`; `clients/nextjs/src/lib/demo-mode.ts` | No live multi-agent execution path is claimed; geometry/morphogenesis remains multi-domain, not multi-agent | Pending commit creation |
| FP-08 | Artifacts & Saved UX | Human labels; Saved browser; Code browser; Preview browser; no technical filenames; responsive layout | Artifact management is clear | NOT_STARTED | TBD | TBD | TBD |
| FP-09 | Input Composer UX | Codex/ChatGPT-style composer; minimal plus; bottom send; auto-grow; no status clutter; no overlap | Composer matches Codex philosophy | NOT_STARTED | TBD | TBD | TBD |
| FP-10 | Codex Design System | Codex philosophy across typography, whitespace, flat surfaces, hierarchy, interactions; theme changes only colors | App feels part of Codex ecosystem | NOT_STARTED | TBD | TBD | TBD |
| FP-11 | Typography & Layout QA | Fix glued words; overflow; line wrapping; subtitles; cards; padding; margins | No visible typography defects | NOT_STARTED | TBD | Human screenshots override automated checks | TBD |
| FP-12 | Preview / Code / Saved Ecosystem | Coordinate three panels; preview is preview; Code is code; Saved is artifacts; no duplicated noise | Every panel has a clear role | NOT_STARTED | TBD | TBD | TBD |
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
