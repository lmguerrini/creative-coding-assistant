# Creative Coding Assistant — Portfolio Case Study

## Outcome

Creative Coding Assistant is a local AI workstation that turns creative intent
into inspectable browser-native code, source-grounded guidance, and bounded
workflow evidence. Its strongest result is not a claim that every generated
idea works everywhere. It is a product that keeps prompt, route, source,
artifact, renderer, persistence, evaluation, and failure state visible enough
for a creative coder or reviewer to decide what is trustworthy.

## Problem

Creative technologists often begin with language such as “a recursive aurora
garden,” “an orbital sculpture,” or “a silent-first generative score.” Turning
that intent into working code requires several translations at once:

- artistic intent into geometry, motion, sound, interaction, and constraints;
- a concept into a runtime-specific source contract;
- API guidance into current, relevant documentation;
- generated source into a browser renderer with observable health;
- a promising result into a refinement that survives reload;
- an AI answer into evidence that can be reviewed without exposing private
  sessions.

A generic chat response can look plausible while using the wrong runtime,
inventing an API, hiding a failed preview, or losing the user’s creative goal.

## Solution

The product combines a Python WSGI backend, a bounded LangGraph-oriented
workflow, a local Chroma knowledge base, provider adapters, a Next.js
workstation, isolated browser preview surfaces, workspace persistence, and a
four-lane Evaluation Lab.

The user can choose a direct Single-Agent route, a bounded Multi-Agent route,
or Auto. The current runtime resolves that request into either:

- one `generator` role with no separate retrieval/review route; or
- `planner`, `researcher`, `generator`, `critic`, and `reviewer` roles with at
  most two refinement attempts, for up to three generation calls including the
  initial call.

Auto selects Single only for Explain or Debug with no attachments and no
resolved domains; every other Auto request resolves Multi. Successful Explain
generation branches directly to finalization without artifact extraction,
preview preparation, critique, or review. The public
`execution.max_refinement_loops` field still reports `1`, a documented drift
from the executable two-attempt limit.

The execution decision is emitted in the stream and rendered in the UI. It is
not inferred later from a decorative graph. Browser frame telemetry appears
after response hydration and does not feed the backend critic or reviewer.

## Data and knowledge

The retrieval layer uses registered technical sources, OpenAI embeddings, and
a local Chroma collection. The dated canonical retrieval snapshot contains
1,445 indexed records and seven benchmark queries, each returning five ranked
chunks.

The final retrieval report records:

| Measure | Result | Interpretation |
|---|---:|---|
| Benchmark queries with results | 7/7 | Execution coverage for the retrieval pack |
| Requested-domain coverage | 18/19 (94.74%) | Requested technical domains represented in the selected contexts |
| Expected-source-anchor overlap | 16/23 (69.57%) | Substantive expected anchors present in top-five selections |

Expected source IDs are diagnostic anchors, not items pinned into results.
The lower source-overlap figure is intentionally retained after heading-only
and index-only chunks were excluded. This report measures retrieval selection,
not final answer quality or a global product score.

## Demonstrated product flows

The in-app Demo Mode contains ten scenarios:

1. Polyrhythmic constellation
2. Recursive aurora garden
3. Kinetic orbit sculpture
4. Fractal solar bloom
5. Source-grounded design brief
6. Multi-agent production plan
7. Single-agent line study
8. Export handoff package
9. Reference-guided palette study
10. Failure-recovery rehearsal

Four browser-native fixtures are the canonical visual showcase:

| Fixture | Runtime evidence | Important boundary |
|---|---|---|
| Polyrhythmic constellation | Tone.js program parses; spectrum mounts silent-ready; explicit Start audio control | Automated validation does not start or assess audio playback |
| Recursive aurora garden | p5.js canvas runs with a visible signal and pointer parallax | Artistic quality is demonstrated, not objectively scored |
| Kinetic orbit sculpture | Bundled Three.js r176 renders a nonblank changing frame with nested rigs | The controlled runtime executes bounded plain JavaScript, not React or remote modules |
| Fractal solar bloom | WebGL 1 shader compiles, links, and renders a nonblank animated field | This is not a texture pipeline or frame-rate benchmark |

These four tests use deterministic, scenario-aware local streams. They prove
request identity, artifact hydration, renderer behavior, fullscreen restore,
refinement payloads, persistence, reload, and source-quality tokens. They are
not evidence of a fresh configured-provider generation.

## Evaluation

Evaluation is split so unlike evidence is not collapsed into one score.

| Lane | Current evidence | What it does not prove |
|---|---|---|
| Retrieval | 7/7 queries; 18/19 domain coverage; 16/23 source anchors | Generated-answer correctness |
| Current-product RAGAS | Seven public RAG cases; 68.03% equal-weight macro across five measured dimensions | Whole-product, private-session, or artistic quality |
| Historical RAGAS fixture | Four sanitized rows; 61.44% four-metric macro | Current-product quality or Context Recall |
| Browser/runtime | Four canonical fixtures, a direct Three.js r176 smoke, and full Playwright coverage | Configured-provider generation or aesthetic consensus |
| Engineering | 2,688 backend tests plus 423 subtests; 77 client files / 553 tests; typecheck, Ruff, build | Independent acceptance or public production readiness |

The current-product RAGAS components are Context Precision 51.96%, Faithfulness
64.90%, Answer Relevancy 56.63%, Context Relevancy 85.71%, and Context Recall
80.95%. The weak metrics remain visible instead of being hidden behind an
optimistic product claim.

## Multimodal boundary

The image-reference path has a tested request-construction contract: valid image
pixels are placed beside the text prompt as an `input_image` block in one
configured-provider payload. Current evidence does not prove live provider
receipt, use, or image influence. The composer clears the reference after
submission, persisted workspace snapshots
exclude image bytes, and diagnostics expose metadata rather than the data URL.

What is not yet demonstrated is equally important: the current evidence does
not establish that a configured model’s output was materially influenced by a
specific image. The product therefore proves multimodal input transport and
privacy boundaries, not complete text-image synergy.

## Hardest engineering decisions

### Retrieval quality without score gaming

Early retrieval favored semantically strong domains and allowed repeated or
non-substantive chunks to crowd the final context. The improvement loop added
bounded domain intent, per-domain candidates, source diversity, substantive
chunk filtering, and candidate headroom. The best-looking intermediate source
coverage was rejected when lineage showed that part of it came from titles and
an API-name index.

### Preview truth over optimistic UI

The product distinguishes supported browser runtimes, code/export-only
artifacts, unavailable renderers, provider fallback, and partial outcomes.
A renderer error cannot be relabeled as success just because source code
exists.

### Reliable refinement after fullscreen

A nested-scroll hit-test race could intercept an ordinary refinement click
after the Creative Session entered and exited fullscreen. The stacking and
hover geometry were corrected, the test retained a real pointer click, and the
four exact showcase refinements then passed 4/4 in 19.2 seconds. The complete
Playwright suite passed 28/28.

## Capstone alignment

| Case | Alignment | Evidence boundary |
|---|---|---|
| Case 1 — RAG knowledge assistant | Strong | Local Chroma retrieval, source lineage, and measured retrieval coverage |
| Case 2 — task automation | Strong but bounded | Observable Single/Multi/Auto routes; no unattended external-tool automation |
| Case 5 — code generation and debugging | Strong | Runtime-specific artifacts, preview diagnostics, refinement, and recovery |
| Case 6 — advanced topics/tools | Strong | Orchestration, evaluation separation, runtime contracts, and product hardening |
| Case 3 — document search | Supporting | Search is limited to registered/indexed creative-coding sources, not arbitrary uploads |
| Case 4 — multimodal | Partial | Image transport is proven; visual influence is not yet demonstrated |
| Case 7 — performance tuning | Not claimed as complete | Creativity controls exist, but no controlled parameter experiment dataset is present |

## Result and limits

The repository demonstrates a working, interactive, browser-focused AI
prototype with unusually explicit evidence boundaries. It does not demonstrate
public deployment, arbitrary code execution, external creative-tool control,
independent user acceptance, a configured-provider multimodal outcome, or a
statistically broad or private-session RAG quality beyond the retained
seven-case current-product benchmark.

The strongest portfolio lesson is that AI product quality is not one score.
It is the agreement between the user’s request, the chosen route, the supplied
evidence, the generated artifact, the observed runtime, and the limits stated
to the user.

## Evidence map

- [Architecture diagram guide](../architecture/README.md)
- [Capstone demo and showcase guide](CAPSTONE_DEMO_SHOWCASE.md)
- [Challenges and lessons](CHALLENGES_AND_LESSONS.md)
- [Future work](FUTURE_WORK.md)
- [Canonical retrieval report](../demo/evaluation/canonical_retrieval_report.json)
- [Evaluation fixture notes](../demo/evaluation/README.md)
- [Canonical Demo Mode source](../clients/nextjs/src/lib/demo-mode.ts)
- [Showcase browser smoke](../clients/nextjs/e2e/demo-showcase-smoke.spec.js)
- [Direct Three.js runtime smoke](../clients/nextjs/e2e/preview-sandbox-three.spec.js)
