# Final Product Polish Roadmap

Branch: `version-review/v8`  
Registered: 2026-07-09  
Execution rule: run exactly one FP task per HITL cycle. Do not merge, push,
tag, freeze, or start V9 from this roadmap.

## Current Task

| Field | Value |
|---|---|
| Current task | FP-02 Preview UX Excellence |
| Status | HITL_REVIEW |
| Scope boundary | Preview unavailable, preview available, User Mode HUD hiding, overlay controls, and Developer Mode diagnostic boundary only |
| Screenshot evidence | `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-1440.png`; `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-1024.png`; `/tmp/cca-v8-fp02-preview-ux/fp02-preview-region-contact-sheet-720.png` |
| Latest task commit | Pending FP-02 HITL acceptance; task implementation commit recorded in final response |

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
| FP-02 | Preview UX Excellence | Preview unavailable redesign; preview available canvas-first; eliminate User Mode HUD; overlay controls; no huge black canvas; no User Mode debug boxes | Preview looks like a real artistic canvas | HITL_REVIEW | `/tmp/cca-v8-fp02-preview-ux/manifest.json`; preview-region contact sheets at 1440, 1024, and 720 widths; `clients/nextjs/src/components/workstation-shell.test.tsx`; `clients/nextjs/src/lib/preview-sandbox-runtime.test.ts` | Developer Mode may show diagnostics; FP-05 owns full Developer Mode polish | Pending HITL |
| FP-03 | Chat UX Excellence | No HTML/JS/GLSL dumps; summary only; code to Code panel; artifact to Artifacts; preview to Preview | Chat readable like ChatGPT | NOT_STARTED | TBD | TBD | TBD |
| FP-04 | User Mode Excellence | Minimal User Mode; inspector closed; max 3 tabs; no technical internals; responsive layout | Looks like a consumer app | NOT_STARTED | TBD | TBD | TBD |
| FP-05 | Developer Mode Excellence | Full technical information; no overlap; no truncated text; readable details | Looks like a professional IDE | NOT_STARTED | TBD | Developer Mode may be denser than User Mode | TBD |
| FP-06 | Demo Mode UX | Minimal demo cards; metadata in Developer Mode; coherent categories; explicit capability; no internal terminology | Reviewer understands each demo in 5 seconds | NOT_STARTED | TBD | TBD | TBD |
| FP-07 | Demo Pack Coverage | Every demo maps to a capability; single-agent; hybrid; multi-domain; retrieval; preview; output; Capstone mapping | No capability without a demo | NOT_STARTED | TBD | Multi-agent must not be claimed unless live path is validated | TBD |
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
