# Multi-Lens Product Evaluation

Creative Coding Assistant is evaluated through separate lenses, each answering
a different question with a different method. Results are reported side by side
and never averaged into a single product score. The raw RAGAS result remains
unchanged; complementary evidence covers properties outside its scope, and
unmeasured dimensions remain explicit.

## Why no single metric is the product grade

The product combines retrieval, orchestration, generation, browser runtime
execution, a workstation UI, privacy boundaries, and documentation. RAGAS
evaluates a bounded subset of that surface: retrieval and grounded-answer
behavior on a frozen benchmark. Engineering gates, runtime QA, end-to-end UI
checks, and privacy controls measure other properties with other methods.
Because their scopes, scales, and error models differ, combining them would
remove information rather than add it. Each lens therefore retains its own
evidence, method, and limitations.

## Product quality matrix

The matrix separates measured results, qualitative observations, and open
questions. Confidence describes how directly the cited source supports a claim
within its stated coverage. It is neither a product score nor an uncertainty
interval and is not used in a composite.

| Lens | Result or status | Source | Coverage | Evidence confidence | Demonstrates | Not measured |
|---|---|---|---|---|---|---|
| Retrieval and grounding | **68.03% raw RAGAS macro** (`0.6803191571804`); single retained run | RAGAS 0.4.3 using `gpt-4o-mini` | Five metrics over seven frozen public cases; all eligible, no skips or metric failures | Direct retained-run evidence; no variance estimate | Benchmark-specific retrieval ranking and grounded-answer behavior | Generalization, runtime behavior, usability, safety, and creative quality |
| Repository engineering review | Qualitative review history; no published audit score | Internal and tool-assisted repository reviews | Selected engineering findings reflected in repository history | Limited; no committed scored review artifact | Review-informed hardening work | Independent external review and a reproducible quantified review result |
| Engineering, CI, and release readiness | Fast CI configured for pull requests and selected pushes; full backend verification on `version-review/**` and `version-freeze/**` pushes; no score | Automated repository workflows | Static checks, fast backend subset, dependency audits, frontend tests/build/E2E; full backend suite on release branches | Direct configuration evidence; current run outcomes are not claimed | The scope and separation of configured quality gates | Answer quality, production reliability, and a numeric engineering grade |
| Runtime and artifact validation | **4/4 committed golden artifacts validated** | Browser QA harness and retained result manifest | One fixed p5.js, Three.js, GLSL, and Hydra artifact, checked 2026-07-08 | Direct but narrow retained QA evidence | Non-blank browser execution with pixel checks for the fixed sample | Artistic quality, fresh-generation pass rate, and production performance |
| UX and workflow evidence | Automated behavior coverage; no usability score | Playwright suites and workflow contracts | Encoded smoke, resilience, sandbox, and design-system paths | Direct for encoded paths; no participant evidence | Documented flows and explicit runtime and failure states | User satisfaction, task usability, accessibility, and broad behavioral coverage |
| Privacy, safety, and provenance | Documented controls and boundaries; no privacy or security score | First-party documentation, code, configuration, and dependency-audit workflows | Declared provider boundaries, guardrails, evaluation gates, and source lineage | Evidence-backed but first-party; no external security review | Implemented and documented governance boundaries | Independent assurance, encryption at rest, and automatic deletion |
| Creator creative self-assessment | **creator self-assessment pending final scoring** | Creator/product owner; first-party | Frozen five-dimension rubric; fixed sample and ratings pending completion | Pending and conflict-limited; not independent validation | The assessment protocol and disclosure are prepared | Creator ratings and independent audience judgment |
| Independent human creative evaluation | Planned as a multi-rater study; not measured; no score | External human evaluators; study not yet completed | No participant results yet | No result evidence | The remaining evaluation need is explicitly defined | Artistic quality, creative usefulness, inter-rater variation, and external validation |

## Lens 1 — Retrieval and grounding (RAGAS, raw)

The published current-product RAGAS result is **68.03%** (macro
`0.6803191571804`), reported raw and unadjusted.

| Metric | Mean over seven cases |
|---|---:|
| Context precision | 0.5196 |
| Faithfulness | 0.6490 |
| Answer relevancy | 0.5663 |
| Context relevancy | 0.8571 |
| Context recall | 0.8095 |

The macro is the equal-weight arithmetic mean of the five metric means.

The [canonical evidence file](../demo/evaluation/current_product_ragas_evidence.json)
records the run identity below and is validated against its committed
[schema](../demo/evaluation/current_product_ragas_evidence.schema.json):

- benchmark: `current-product-retrieval.v1`, seven frozen public cases, all
  eligible, zero skips, zero metric failures;
- evaluator: RAGAS 0.4.3 with `gpt-4o-mini`;
- generation model: `gpt-5-mini-2025-08-07`; embeddings:
  `text-embedding-3-small`;
- evaluated at 2026-07-14T01:27:10Z;
- dataset, knowledge-base, retrieval, prompt, generation, and output
  fingerprints are recorded in the evidence file, so the exact evaluated system
  state is identifiable.

See [Evaluation Methodology](eval.md) for methodology, reproduction commands,
and data boundaries. The separate
[retrieval selection report](../demo/evaluation/canonical_retrieval_report.json)
records retrieval lineage for the same seven cases without generating answers.

### Observed methodological limitations

These are properties of the evaluation method observed in this product's
context. They are disclosed so the raw score can be read precisely; the score
itself is not adjusted for them.

- **Small, frozen sample.** Seven cases provide reproducibility, not
  statistical coverage, and the retained result is a single run without a
  variance estimate.
- **Stochastic evaluator.** RAGAS metrics are produced by an evaluator model;
  identical or near-identical inputs can receive materially different scores
  across runs and evaluator versions. Repeated runs would be required to
  estimate that spread.
- **Cross-domain context judging.** Several benchmark cases intentionally span
  multiple creative domains. When excerpts from complementary domains are
  judged independently, useful context can be scored as if it were noise. In
  the retained run, three of the seven cases — each spanning two or more
  domains — received a context precision of 0.0 while their context relevancy
  remained at 0.5–1.0.
- **Technical-creative answers.** Answers in this product mix documented facts
  with creative recommendations. Statement-level faithfulness scoring can
  penalize recommendation content that is appropriate but not literally present
  in retrieved documentation, and code-heavy answers make statement extraction
  brittle.
- **Artistic quality is out of scope.** No RAGAS metric measures aesthetic
  quality, creative usefulness, or runtime behavior of generated code.
- **Fingerprint comparability.** Every retained result carries a knowledge-base
  fingerprint. A run against a different fingerprint evaluates a different
  indexed state and is not directly comparable with the published result.

## Lens 2 — Repository engineering review

Repository history contains internal and tool-assisted engineering review
rounds (visible as `audit/*`, `review/*`, and `version-review/*` branches and
their merged findings). Their conclusions informed hardening work such as
preview-session integrity and reliability fixes on `main`. These rounds are not
presented as external independent audits. No scored review artifact is committed
as verifiable evidence, so no audit number is published; engineering review
would in any case measure code and release quality, not retrieval or grounding.
It is not a substitute for, and is never averaged with, the RAGAS result.

## Lens 3 — Engineering, CI, and release readiness

What it measures: static quality, contract and regression behavior, dependency
security, and build health — not answer quality.

Evidence:

- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) is configured for
  every pull request and selected pushes, including `main`, with backend
  lint/compile checks, a fast backend pytest subset with coverage, dependency
  security audits (`pip-audit`, `npm audit`), documentation/Mermaid gates, a
  backend startup log gate, and frontend lint, typecheck, unit tests, production
  build, and Playwright end-to-end suites.
- [`.github/workflows/backend-release-verification.yml`](../.github/workflows/backend-release-verification.yml)
  runs the full backend pytest suite with coverage on `version-review/**` and
  `version-freeze/**` branches.
- Backend tests live under [`tests/`](../tests); frontend unit tests and
  Playwright specs live under [`clients/nextjs`](../clients/nextjs).

Limitations: the per-merge CI lane runs a fast backend subset; the full backend
suite is a release-branch gate. Test volume is evidence of coverage intent, not
of product quality by itself.

## Lens 4 — Runtime and artifact validation

What it measures: whether committed reference artifacts actually execute and
render in real browser runtimes — execution correctness, not artistic quality.

Evidence comes from the golden-artifact QA harness and its retained
[`QA manifest`](../demo/golden_artifacts/qa_manifest.json) and
[`browser results`](../demo/golden_artifacts/browser_full_runtime_qa_results.json)
(checked 2026-07-08). All four committed golden artifacts — p5.js, Three.js,
GLSL, and Hydra — rendered non-blank in a Chromium browser with pixel-level
checks. The p5.js, Three.js, and Hydra checks used their runtime packages; GLSL
compiled and rendered in a generic WebGL harness. Separately, the product's
live preview path supports bounded p5.js, Three.js, GLSL, and Tone.js artifact
contracts, with sandbox isolation covered by Playwright specs under
[`clients/nextjs/e2e/`](../clients/nextjs/e2e).

Limitations: the QA set is a curated sample of one artifact per QA runtime; the
manifest itself records `creative_quality: not_measured`. A pass rate over
freshly generated artifacts has not yet been measured. Rendering non-blank
establishes execution, not aesthetic or creative value.

## Lens 5 — UX and workflow evidence

What it measures: whether the workstation behaves as documented and whether
workflow surfaces report what actually ran.

Evidence: Playwright smoke, resilience, preview-sandbox, and design-system
suites under [`clients/nextjs/e2e/`](../clients/nextjs/e2e) are configured in
CI; typed NDJSON workflow events distinguish route, retrieval, runtime, and
failure states, and failure states remain explicit rather than being converted
into fabricated success (see the [User Manual](USER_MANUAL.md) and
[Architecture Diagram Guide](../architecture/README.md)).

Limitations: a formal usability study and an accessibility audit have not yet
been completed. Automated end-to-end checks demonstrate behavior, not user
satisfaction.

## Lens 6 — Privacy, safety, and provenance

What it measures: data governance, external-boundary control, and source
lineage.

Evidence: the [Ethics and Privacy Assessment](ETHICS_PRIVACY_ASSESSMENT.md) and
[Evaluation and Responsible-AI Overview](CAPSTONE_EVALUATION_ETHICS.md) document
the declared external boundaries. Provider-scored evaluation requires an
explicit authorization flag, and input and generation guardrails live in the
[`security` package](../src/creative_coding_assistant/security/). Retrieval
distinguishes registered, indexed, retrieved, and cited sources. Committed
evaluation evidence is a public-safe projection that excludes raw prompts,
answers, references, and excerpts; CI audits backend and frontend dependencies.

Limitations: local storage is not encrypted or automatically deleted
(disclosed), and no independent external security review has been performed.

## Lens 7 — Creative quality

### Creator creative review (first-party evidence)

**Status: creator self-assessment pending final scoring.** The assessor is the
project creator and product owner. This is legitimate first-party evidence with
an inherent conflict of interest: the assessor defined and built the product
being judged. It is not independent human validation, is reported separately
from external evaluation, and cannot replace the planned multi-rater study.

The dimension names and assessment questions below were frozen under
`creator-creative-self-assessment.v1` before scoring. Each dimension must be
assessed against the same fixed artifact sample with a creator-entered score or
rating and an evidence note. Any rubric change requires a new version. No
weights or overall product-score calculation are defined.

| Assessment record | Entry |
|---|---|
| Assessor | Creator/product owner: [name to be entered] |
| Evidence class | First-party; not independent human validation |
| Fixed sample | [Artifact IDs, briefs, commit, and assessment date to be entered before scoring] |
| Rubric | `creator-creative-self-assessment.v1`; dimensions and questions frozen |
| Status | creator self-assessment pending final scoring |

| Frozen dimension | Assessment question | Creator score or rating | Evidence or rationale |
|---|---|---|---|
| Creative-intent fidelity | How faithfully does the artifact preserve the stated intent, mood, references, and constraints? | Pending | Pending |
| Aesthetic and conceptual coherence | Do visual, sonic, interactive, and technical choices form a coherent concept where applicable? | Pending | Pending |
| Creative usefulness | Does the result provide useful material, directions, or decisions for continued creative work? | Pending | Pending |
| Editability and continuation potential | Can the creator inspect, refine, extend, and continue the artifact without losing its intent? | Pending | Pending |
| Alignment with the product vision | Does the result help translate artistic vision into a technically grounded, inspectable creative system? | Pending | Pending |

Repository evidence may support bounded observations about product intent,
editability, or execution (see [Purpose](../README.md#purpose), the
[User Manual](USER_MANUAL.md), and the
[runtime QA manifest](../demo/golden_artifacts/qa_manifest.json)). It cannot
determine a creative score from architecture, tests, screenshots, or runtime
success alone.

### Independent human creative evaluation

Artistic quality and creative usefulness also require structured independent
human judgment using a fixed rubric, a fixed sample of briefs, and multiple
raters. The evaluation protocol has already been designed and its rubric frozen,
but execution remains pending. The study is **planned and not yet formally
measured**; no result exists yet, and no external score is assigned or implied.
A completed creator self-assessment remains a separate first-party record; it
does not count as independent validation or replace the multi-rater study.

## How to interpret the results

- **68.03% is the raw retrieval-and-grounding benchmark result** on a frozen
  seven-case benchmark. It is reported unchanged.
- **It is not the overall capstone grade.** It measures one bounded subset of
  the product.
- Product quality outside that subset is supported by separate automated and
  documented evidence: CI and test gates, browser runtime QA, end-to-end UI
  checks, and privacy and provenance controls.
- Creator self-assessment remains pending and first-party; independent human
  creative evaluation remains planned and unmeasured.
- Every limitation above remains visible rather than being manually corrected,
  and no lens result is averaged with, or substituted for, another.

## Future evaluation roadmap

Planned next steps, in likely order of value:

1. a repeated-run RAGAS series at a fixed knowledge-base fingerprint, published
   as a clearly labeled variance diagnostic beside — never in place of — the
   raw retained score;
2. a measured execution pass rate over freshly generated artifacts per live
   runtime, using the existing browser QA harness;
3. completion of the creator self-assessment against its frozen rubric and
   fixed sample, published only as first-party evidence;
4. an independent human creative evaluation with a fixed rubric, sample set,
   and multiple raters;
5. a usability walkthrough protocol and an accessibility baseline scan for the
   workstation.

No completion date is promised for these items.
