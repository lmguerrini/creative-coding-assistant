# Public Documentation Boundary Audit

Date: 2026-07-08
Branch: `version-review/v8`

This audit classifies every tracked file under `docs/` and `demo/` after the
V8 release-candidate excellence pass. The goal is to keep reviewer/user
evidence and product documentation public while keeping private runtime evidence
out of Git.

No tracked `docs/` or `demo/` file was moved or removed in this pass. The
review found no file that should be reclassified as private engineering
evidence or obsolete/internal. Private live-session records, Runtime Pack state,
local Chroma data, environment files, and generated test output remain outside
the public docs/demo tree and are ignored.

## Docs Classification

| File | Classification | Action | Rationale |
|---|---|---|---|
| `docs/ARCHITECTURE_DECISIONS.md` | Public product documentation | Keep tracked | Explains architecture choices useful to reviewers and future maintainers. |
| `docs/CAPSTONE_DEMO_SHOWCASE.md` | Public reviewer/user evidence | Keep tracked | Defines purpose, problem, solution, 10-minute demo, 5-minute Q&A, fallback, SCR, and SMART framing. |
| `docs/CAPSTONE_EVALUATION_ETHICS.md` | Public reviewer/user evidence | Keep tracked | Documents evaluation evidence, ethics, privacy, and limitations. |
| `docs/IMPLEMENTATION_ROADMAP.md` | Public product documentation | Keep tracked | Gives high-level product/version context without private runtime data. |
| `docs/ORCHESTRATION_PACKAGE_BOUNDARIES.md` | Public product documentation | Keep tracked | Clarifies active/passive orchestration ownership and claim boundaries. |
| `docs/PRODUCTION_DEPLOYMENT.md` | Public product documentation | Keep tracked | Provides production deployment posture and setup guidance. |
| `docs/PROJECT_CONTEXT.md` | Public product documentation | Keep tracked | Summarizes project purpose and context. |
| `docs/PUBLIC_DOCUMENTATION_BOUNDARY_AUDIT.md` | Public reviewer/user evidence | Keep tracked | Records this public/private boundary review. |
| `docs/RUNTIME_VALIDATION.md` | Public product documentation | Keep tracked | Documents runtime validation posture and supported validation paths. |
| `docs/V8_CAPSTONE_EXCELLENCE_SCORECARD.md` | Public reviewer/user evidence | Keep tracked | Primary release-candidate scorecard and evidence map. |
| `docs/V8_GRAND_ENGINEERING_REVIEW.md` | Public reviewer/user evidence | Keep tracked | Final Grand Review release-candidate evidence and HITL boundaries. |
| `docs/eval.md` | Public product documentation | Keep tracked | Explains evaluation approach and reviewer-relevant commands. |
| `docs/eval_pipeline.md` | Public product documentation | Keep tracked | Details manual eval/RAGAs pipeline and privacy boundaries. |
| `docs/sync.md` | Public product documentation | Keep tracked | Documents source sync posture for the knowledge base. |

## Demo Classification

| File | Classification | Action | Rationale |
|---|---|---|---|
| `demo/README.md` | Public reviewer/user evidence | Keep tracked | Explains demo mode, golden flow, local QA evidence, and claim boundaries. |
| `demo/demo_prompt_library.md` | Public reviewer/user evidence | Keep tracked | Reviewer-facing prompt set for golden flows and fallback. |
| `demo/evaluation/README.md` | Public reviewer/user evidence | Keep tracked | Explains sanitized RAGAs fixture privacy posture. |
| `demo/evaluation/private_live_session_ragas_decision.json` | Public reviewer/user evidence | Keep tracked | Records the private live-session RAGAs HITL decision without exposing private row content. |
| `demo/evaluation/sanitized_ragas_live_sessions.jsonl` | Public reviewer/user evidence | Keep tracked | Synthetic, schema-valid, privacy-approved RAGAs input fixture. |
| `demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl` | Public reviewer/user evidence | Keep tracked | Exact sanitized RAGAs metric result rows. |
| `demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl.manifest.json` | Public reviewer/user evidence | Keep tracked | Exact sanitized RAGAs run manifest and metric summary. |
| `demo/evaluation/redacted_live_session_ragas_latest4.jsonl` | Public reviewer/user evidence | Keep tracked | Redacted latest-live RAGAs fixture derived from live-session structure without exposing private row text. |
| `demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl` | Public reviewer/user evidence | Keep tracked | Exact redacted latest-live RAGAs metric result rows. |
| `demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl.manifest.json` | Public reviewer/user evidence | Keep tracked | Exact redacted latest-live RAGAs run manifest and evaluator configuration. |
| `demo/final_demo_launcher.html` | Public reviewer/user evidence | Keep tracked | One-click local launcher for the eight-flow demo and evidence links. |
| `demo/final_demo_suite.json` | Public reviewer/user evidence | Keep tracked | Eight-flow final demo suite with prompt, expected behavior, fallback, success criteria, validation path, and talking point. |
| `demo/golden_artifacts/browser_full_runtime_qa.html` | Public reviewer/user evidence | Keep tracked | Full-runtime browser QA harness for temporary p5/Three/Hydra packages and GLSL WebGL checks. |
| `demo/golden_artifacts/browser_full_runtime_qa_results.json` | Public reviewer/user evidence | Keep tracked | Exact full-runtime browser QA result, classifications, frame timing, and accepted boundaries. |
| `demo/golden_artifacts/browser_render_qa.html` | Public reviewer/user evidence | Keep tracked | Offline browser QA harness for p5 shim, GLSL WebGL, and Three.js dependency-boundary checks. |
| `demo/golden_artifacts/browser_render_qa_results.json` | Public reviewer/user evidence | Keep tracked | Exact browser/render QA result and limitations. |
| `demo/golden_artifacts/README.md` | Public reviewer/user evidence | Keep tracked | Explains generated artifact scope and runtime boundaries. |
| `demo/golden_artifacts/p5_sacred_geometry_sketch.js` | Public reviewer/user evidence | Keep tracked | Generated p5.js artifact for output/creative quality inspection. |
| `demo/golden_artifacts/three_audio_reactive_scene.js` | Public reviewer/user evidence | Keep tracked | Generated Three.js artifact for output/creative quality inspection. |
| `demo/golden_artifacts/glsl_kaleidoscope_field.frag` | Public reviewer/user evidence | Keep tracked | Generated GLSL artifact for output/creative quality inspection. |
| `demo/golden_artifacts/hydra_feedback_lattice.js` | Public reviewer/user evidence | Keep tracked | Generated Hydra artifact for bounded hydra-synth browser QA inspection. |
| `demo/golden_artifacts/qa_manifest.json` | Public reviewer/user evidence | Keep tracked | Static and browser QA evidence plus runtime boundaries. |
| `demo/golden_demo_dataset.json` | Public reviewer/user evidence | Keep tracked | Rehearsal and offline fallback dataset. |
| `demo/manual_demo_checklist.md` | Public reviewer/user evidence | Keep tracked | Manual timed-demo reliability checklist. |
| `demo/showcase_upload_preparation.md` | Public reviewer/user evidence | Keep tracked | Showcase packaging and final public-claims review checklist. |

## Private/Obsolete Result

Private engineering evidence moved to `.runtime_pack/active/` or
`.chatgpt_context/`: none from `docs/` or `demo/`.

Obsolete/internal files removed or ignored: none from tracked `docs/` or
`demo/`.

Untracked sandbox-only RAGAs failure files created during the first network
attempt were removed before commit. The committed RAGAs result is the successful
privacy-approved external run over sanitized data.
