# Capstone Evaluation and Ethical AI Summary

This is the reviewer-facing capstone summary for Creative Coding Assistant
(CCA) as of 2026-07-14. It follows the repository evidence rule: current live
behavior first, then current automated validation and evaluation, then current
documentation; historical and planned evidence remain explicitly labeled.

## Capstone alignment

CCA is a working AI-assisted creative-coding workstation. Its primary AI
engineering case is retrieval-augmented generation over a registered/indexed
creative-coding knowledge base. The product also demonstrates workflow/agent
orchestration, code generation and refinement, a bounded request-scoped image
path, and retrieval/evaluation tuning.

The current reviewer experience includes:

- Single Agent, Multi Agent, and Auto routes with a published workflow graph;
- live browser previews for p5.js, Three.js, GLSL, and Tone.js;
- request-scoped PNG/JPEG/WebP/GIF image references included in the
  configured-provider payload only on explicit submission; current evidence
  does not establish live provider receipt, use, or image influence;
- local Chroma retrieval with source/chunk lineage;
- Dashboard and Inspector evidence surfaces;
- artifacts, local session persistence, refinement, fullscreen review, and
  user-mediated exports.

Hydra and React Three Fiber are code/export-only. TouchDesigner, Unreal Engine,
Blender Geometry Nodes, and Houdini are external handoffs. CCA does not claim
to execute, control, deploy to, or validate those tools.

## Current evaluation evidence

| Evidence | Result | Defensible interpretation |
|---|---|---|
| Canonical Demo Mode browser smoke | Four canonical live showcase paths covered | Current Chromium interaction/runtime evidence for the asserted paths, not artistic-quality judgment |
| Current-product RAGAS | 7/7 eligible/scored, 0 skipped, 0 metric failures; macro `0.6803191571804` (68.03191571804%) | Current seven-case public RAG evidence; not a project grade or artistic-quality judgment |
| Current RAGAS component means | precision `0.5196428571169692`; faithfulness `0.648989898989899`; answer relevancy `0.5662963631284655`; context relevancy `0.8571428571428571`; recall `0.8095238095238094` | Five reference-aware dimensions remain visible instead of hiding weak precision/relevancy behind the macro |
| Historical approved fixture | 4/4 eligible, 0 skipped/failures; four-metric macro 61.44%; no recall result | Preserved historical synthetic/public fixture; obsolete as the primary score |
| Current local retrieval | substantive expected-source overlap 16/23 (69.57%), from 9/23; requested-domain coverage 18/19 (94.74%), from 7/19; 7/7 queries returned five results | Fixed-query, top-five local coverage evidence; not RAGAS or answer-quality evidence |
| Evaluation contract catalog | 35 deduplicated product-authored prompt contracts | Contract inventory only; Full executes seven RAG cases and records three local snapshot lanes, not 35 model/evaluator calls |
| Human usability/aesthetic evaluation | no completed study | Human evidence remains missing; automation does not substitute for this judgment |

The final retrieval number is intentionally lower than an apparent 19/23 peak:
lineage review found title-only and index-only false positives, so they were
removed. The remaining requested-domain gap is Shadertoy, whose focused source
sync returned HTTP 403 and left no indexed chunks. Neither gap is concealed by
pinning expected sources into results.

Full definitions, provenance, fingerprints, and reproduction commands are in
[EVALUATION_METRICS_SUMMARY.md](EVALUATION_METRICS_SUMMARY.md) and
[eval.md](eval.md).

## Evaluation criteria mapping

| Capstone criterion | Evidence to present | Limitation to state |
|---|---|---|
| Outcome quality | Running workstation, canonical live showcases, artifacts/refinement, browser smoke | No completed human artistic-quality or usability study |
| Applied learning | RAG source/index/retrieval separation, workflow routes, code generation, image request path, retrieval tuning, tests | Provider/model labels do not prove a successful call |
| Ethics and responsibility | Explicit data boundaries, secret handling, approved fixture, blocked private-data rerun, provenance and handoff labels | Local storage is not automatic encryption/retention; model/source bias remains |
| Presentation | Reproducible reviewer path, current evidence labels, demo/recovery scenarios, concise limitations | Historical screenshots or fixture results cannot replace current live behavior |

## Ethical commitments

- **Source grounding:** distinguish registered, indexed, retrieved, and cited
  states. A citation or retrieved chunk does not guarantee correctness.
- **Privacy:** do not expose keys, private session JSONL, local Chroma excerpts,
  reference images, or unreviewed exports. Generation, embeddings, evaluation,
  tracing, and public sharing are separate consent/data-transfer boundaries.
- **Image consent:** accepted image pixels enter the configured-provider request
  payload only with explicit submission. Current evidence does not establish
  live provider receipt, use, or image influence. Attachments clear after
  submission and do not restore with the session; a pre-submit export can still
  include them.
- **Creative ownership:** users retain responsibility for originality,
  licensing, attribution, and rights to submitted images and published output.
- **Cultural framing:** geometric, symbolic, spiritual, and narrative language
  is creative inspiration, not religious, historical, medical, psychological,
  or metaphysical authority.
- **Provider transparency:** provider calls can fail and incur cost. A fallback
  or dry run must not be presented as provider evidence.
- **Agent transparency:** workflow labels describe bounded orchestration, not
  sentience, independent autonomy, or unobserved parallel work.
- **External handoffs:** export is not execution, deployment, compatibility, or
  validation in a target tool.

The detailed risk register and human review gate are in
[ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md).

## Demo statement

> CCA demonstrates a working retrieval-grounded creative-coding assistant with
> bounded agent workflows, four live browser runtimes, request-scoped image
> input, and reproducible evaluation evidence. The current seven-case RAGAS
> macro is 68.03%, with all five component means and fingerprints published.
> The old 61.44% synthetic-fixture macro remains historical, while human
> artistic-quality evaluation remains explicitly missing rather than overstated.

Start the review with [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). Operational
boundaries are in [USER_MANUAL.md](USER_MANUAL.md),
[DATA_AND_KB.md](DATA_AND_KB.md), and
[DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).
