# V8 Capstone Evidence Matrix

Date: 2026-07-08
Branch: `version-review/v8`

This document is engineering evidence for the V8 Capstone release candidate.
It intentionally avoids public self-scoring. Reviewers should evaluate the
project from the evidence, tests, demo paths, architecture boundaries, and
remaining risks below.

## Evidence Base

- Backend validation: full pytest evidence is recorded in
  `docs/V8_GRAND_ENGINEERING_REVIEW.md`; focused demo/evaluation tests pass.
- Frontend validation: typecheck, Vitest, build, and Playwright smoke evidence
  cover the workstation and integrated Demo Mode.
- API validation: health, readiness, invalid assistant request, and workspace
  session smoke checks are documented.
- Retrieval validation: local Chroma retrieval returned results for all 7
  Capstone demo scenarios with expected-source overlap.
- RAGAs validation: sanitized public fixture and redacted latest-live fixture
  both ran with provider-backed evaluator calls.
- Golden artifact QA: p5.js, Three.js, GLSL, and Hydra artifacts have syntax
  and browser/runtime QA records.
- Demo validation: integrated in-app Demo Mode is primary; the static launcher
  remains fallback/reviewer evidence.
- Privacy boundary: raw private live-session rows remain local-only.
- Deployment boundary: Capstone target is local demo, not public cloud
  deployment.

## Category Evidence

| Category | Evidence | Remaining risk |
|---|---|---|
| Architecture | Python backend, Next.js workstation, local API, Chroma retrieval, Typed Domain Intelligence Layers, Demo Mode, and documented boundaries. | Some typed layers are planning/reasoning surfaces, not active runtime execution. |
| Code Quality | Typed models, focused tests, artifact QA tests, RAGAs foundation tests, and public-claim tests. | Chroma/Pydantic warnings remain third-party dependency noise. |
| Testing | Backend, frontend, smoke E2E, focused demo, focused RAGAs, and artifact checks. | Live provider/RAGAs tests remain opt-in because they use credentials and may incur cost. |
| Documentation | README, Capstone demo guide, ethics guide, deployment target, Grand Review, and demo docs. | Documentation is still broad; presenter should use the start-here path. |
| Demo Reliability | Integrated Demo Mode covers 8 scenarios; fallback launcher and manual checklist exist. | Final spoken rehearsal is still a human activity outside CI. |
| Capstone Alignment | Cases 1, 2, 3, 5, and 6 are mapped conservatively in demo docs. | Official reviewer preferences may differ from local evidence mapping. |
| Presentation Readiness | 10-minute demo, 5-minute Q&A, fallback triggers, and reviewer answers are documented. | Delivery quality depends on presenter timing and live environment. |
| Product Robustness | Local backend/frontend/API smoke, artifact QA, retrieval evidence, and fallback paths are present. | No broad chaos, load, soak, or display-FPS benchmark is claimed. |
| RAG/Retrieval Quality | Sanitized RAGAs, redacted latest-live RAGAs, retrieval smoke, and source-boundary docs. | Redacted latest-live fixture is p5.js-only and includes one weak faithfulness row. |
| Output Quality | Golden artifacts render nonblank in the strongest local QA path available for p5.js, Three.js, GLSL, and Hydra. | Runtime QA used temporary dependencies; product preview integration is separate. |
| Creative Quality | Demo prompts now emphasize audio-reactive systems, morphogenesis, feedback patterns, geometry, and installation planning. | Creative impact remains partly subjective and benefits from presenter curation. |
| Security/Privacy | Sanitized/redacted evaluator fixtures, ignored runtime data, local `.env`, and documented private-data boundaries. | Secret scan should rerun immediately before any public release action. |
| Production Readiness | Local demo target, fallback path, HITL boundaries, validation evidence, and dependency warning path are documented. | Final freeze, public release, tag, push, and Chroma upgrade require HITL or follow-up validation. |

## RAGAs And Retrieval Analysis

Sanitized public fixture:

- Input: `demo/evaluation/sanitized_ragas_live_sessions.jsonl`
- Result: `demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl`
- Metric: context precision
- Rows: 4 total, 4 eligible, 0 skipped, 0 metric failures
- Average context precision: `0.999999999925`

Redacted latest-live fixture:

- Input: `demo/evaluation/redacted_live_session_ragas_latest4.jsonl`
- Result: `demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl`
- Metrics: context precision, faithfulness, answer relevancy
- Rows: 4 total, 4 eligible, 0 skipped, 0 metric failures
- Averages after the public-safe wording refresh: context precision
  `0.7006944444230672`, faithfulness `0.6875`, answer relevancy
  `0.4419141765019863`

Weak-row analysis:

- Sample: `redacted_live_p5_demo_fallback_73f56121`
- Faithfulness: `0.0`
- Context precision: `0.32499999998375`
- Answer relevancy: `0.3121392372165798`
- Finding: retrieval returned some relevant fallback contexts, especially the
  browser-sketch failure fallback and V8 demo checklist rows, but the top
  p5.js setup/canvas contexts are only partially relevant to the question.
- Likely cause: a combination of retrieval ranking and evaluator sensitivity.
  The answer is broadly supported by the final two contexts, but it compresses
  multiple fallback actions into one sentence and does not explicitly anchor
  each claim to the retrieved context wording.
- Low-risk improvement: demo and README copy now explain the weak row directly;
  demo prompts were strengthened to ask for clearer fallback/source-boundary
  language. The existing metric is not hidden or overwritten.
- Not changed: raw private live-session scoring is still avoided; any new
  provider-backed RAGAs run should use sanitized or redacted fixture text only.

## Demo Quality Audit

| Scenario | Artistic review | Improvement applied |
|---|---|---|
| Three.js audio-reactive visual system | Strong wow factor from 3D motion, bloom, and audio mapping. | Kept as primary high-impact flow. |
| p5.js generative morphogenesis sketch | Clear educational value; needed richer growth vocabulary. | Added differential growth, diffusion-limited aggregation, branching, and clearer growth-story language. |
| GLSL shader / post-processing visual | Strong technical/visual credibility through WebGL QA. | Kept focused on glow, uniforms, failure risks, and fallback. |
| Hydra feedback-pattern demo | Strong live-code aesthetic when bounded honestly. | Added moire-like motion and kept support limited to local `hydra-synth` artifact QA. |
| Retrieval-grounded answer | Strong evaluator credibility. | Kept source-boundary and privacy language explicit. |
| Concept-to-visual translation | Good reviewer story for abstract-to-operational design. | Kept authority boundaries explicit. |
| Geometry / morphogenesis visual system | Stronger with rule-based growth vocabulary. | Added DLA/branching emphasis while avoiding source-indexing claims. |
| Installation / immersive scene planning | Strong capstone presentation path. | Kept planning/handoff boundaries instead of execution claims. |

## Digital Morphogenesis Boundary

Jason Webb's public Digital Morphogenesis resources are useful inspiration for
demo language around space-colonization-style branching, diffusion-limited
aggregation, differential growth, reaction diffusion, cellular automata,
L-systems, flow fields, particle systems, self-organization, and emergent
form.

The current KB/demo registry does not claim those URLs are indexed sources.
Demo Mode and demo docs use the ideas only as general public creative-coding
inspiration and continue to avoid a Jason Webb-specific source-coverage claim.

## Internal Reviewer Loop

Internal advisory surfaces used as engineering reviewers:

- production readiness review;
- creative readiness review;
- demo showcase plan;
- retrieval demo pack;
- workflow explainability;
- creative analytics;
- creative critic;
- self evaluation;
- creative director;
- creative reasoning signals.

These surfaces are treated as internal review evidence, not objective truth or
automatic grading. The practical improvements from this pass were README
clarity, public/private naming cleanup, prompt quality, RAGAs weak-row
analysis, and removal of public self-scores.

Latest advisory-review run:

- Production readiness review: status `guarded`, 6 records, 13 guarded
  findings, 0 blocking findings.
- Creative readiness review: status `guarded`, 6 records, 3 guarded findings,
  0 blocking findings.
- Demo showcase plan: 8 demo flows, 5 fallback plans, and 8 presentation
  segments.
- Retrieval demo pack: `capstone_kb_expansion_retrieval_demo_pack`, 7
  scenarios.

## Current Release-Candidate Judgment

V8 is ready for external reviewer evaluation with bounded claims and local demo
evidence. It is not autonomously frozen. Freeze, merge, push, tag, public
release, and public cloud deployment require explicit HITL approval.
