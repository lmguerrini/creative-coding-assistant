# SMART Presentation

This reviewer-facing **Specific–Measurable–Achievable–Relevant–Time-bound**
frame turns the product narrative into claims that can be checked during a
ten-minute demonstration and five-minute question period.

## One-slide version

| Dimension | Reviewable commitment | Evidence and boundary |
|---|---|---|
| **Specific** | Convert a text prompt or validated image reference into an inspectable creative-code response with the requested and resolved workflow, executed-node trace, sources, artifact, and preview metadata. | [Architecture Walkthrough](ARCHITECTURE_WALKTHROUGH.md); image data reaches the generation payload, but queued image bytes are request-scoped rather than restored from saved sessions. Browser frame telemetry remains local and does not feed backend critique. |
| **Measurable** | Demonstrate the four canonical live preview domains; publish route/node evidence; report fixed retrieval coverage and five current-product RAGAS component means without collapsing them into a whole-product claim. | [Capability Matrix](CAPABILITY_MATRIX.md), [retrieval report](../demo/evaluation/canonical_retrieval_report.json), and [Evaluation Metrics Summary](EVALUATION_METRICS_SUMMARY.md). |
| **Achievable** | Use a local Next.js 15 client, exact-path Python WSGI API, compiled LangGraph, local Chroma/SQLite stores, and one implemented OpenAI provider route with no more than two Multi refinements. | [System Overview](SYSTEM_OVERVIEW.md); optional provider/network dependencies are explicit, and the published one-refinement field is documented as drift from the executable two-attempt limit. |
| **Relevant** | Reduce the gap between a creative intention and a reviewable browser-oriented code artifact while teaching rather than hiding technical choices. | It addresses the official Outcome Quality, Learning Application, Ethical Considerations, and Presentation criteria; its primary implementation patterns align with Cases 1, 2, and 5. Case 7 is supporting and incomplete because no controlled parameter-experiment dataset exists. |
| **Time-bound** | Complete one end-to-end proof in the ten-minute capstone window, using evidence stamped by its own run time rather than implying continuous production monitoring. | The canonical retrieval artifact records `2026-07-13T05:05:33.306298+00:00`; it is a fixed report, not a live dashboard. |

## Ninety-second script

> The project objective is specific: given a creative prompt or bounded image
> reference, return more than an answer. The interface should show the chosen
> workflow, the route that actually ran, the executed nodes, any selected
> official context, an extractable code artifact, and preview metadata. It is
> measurable through four canonical live browser-preview domains and committed
> retrieval evidence: seven of seven fixed cases completed, sixteen of
> twenty-three expected source anchors were covered, and eighteen of nineteen
> requested domains were covered. Those are coverage measures, not artistic or
> grounded-answer quality. A seven-case current-product RAGAS run reports a
> 68.03 percent five-metric macro, while the old four-case synthetic fixture is
> labeled historical. The scope is achievable because the
> implementation is deliberately bounded: a Next.js client, a WSGI API, one
> compiled LangGraph, local Chroma and SQLite storage, one implemented generation
> provider, and at most two Multi refinements. It is relevant because creative
> coding needs fast iteration without concealing source, runtime, or privacy
> decisions. It is time-bound by the review itself: one full request and its
> evidence must be understandable within ten minutes, while every stored metric
> keeps its own timestamp and fixture boundary.

## Measurement card

| Measure | Current evidence | What it proves | What it does not prove |
|---|---:|---|---|
| Canonical live browser-preview domains | 4 | The public runtime contract supports p5.js, Three.js, GLSL, and Tone.js preview paths | Every listed/exported creative technology runs inside the product |
| Curated demo scenarios | 10 | The repository has diverse reviewer prompts across the supported experience | Ten fresh provider generations all pass aesthetic review |
| Fixed retrieval cases completed | 7/7 | The committed canonical run returned results for every fixed query | Every expected source or ideal grounded answer was found |
| Expected source-anchor coverage | 16/23 (69.57%) | How many expected anchors appeared in the fixed top-five results | Semantic correctness of every retrieved chunk |
| Requested-domain coverage | 18/19 (94.74%) | How broadly the results covered requested domains | Source-anchor precision or answer faithfulness |
| Current RAGAS context precision | 0.5196428571169692 | Retrieval ordering on seven public current-product cases | Universal product precision or a project grade |
| Current RAGAS faithfulness | 0.648989898989899 | Grounding on those seven answers | A universal hallucination rate |
| Current RAGAS answer relevancy | 0.5662963631284655 | Bounded answer alignment on the current benchmark | Human artistic relevance |
| Current RAGAS context relevancy | 0.8571428571428571 | Usefulness of selected contexts on the benchmark | Every source is ideal |
| Current RAGAS context recall | 0.8095238095238094 | Coverage against authored reference answers/context | Recall for unrelated product lanes |
| Current Retrieval Quality macro | 0.6803191571804 | Equal-weight summary of the five RAGAS means | Accuracy, project grade, or aesthetic judgment |
| Multi refinement budget | At most 2 after the initial generation | Review permits at most three generation calls total | That every retry improves artistic quality |

The primary **68.03%** macro is an arithmetic summary of the five current RAGAS
means. It is not a whole-product score or capstone grade. The old **61.44%**
four-metric fixture remains historical and has no context-recall denominator.

Runtime routing is equally narrow: Auto selects Single only for Explain or
Debug with no attachments and no resolved domains; every other Auto request
selects Multi. A successful Explain generation goes directly to finalization,
skipping artifact extraction, preview, critique, and review. The public
`execution.max_refinement_loops` field remains `1` even though the executable
review path permits two attempts.

## Live verification sequence

1. **Specific:** submit one supported-domain prompt and point out the requested
   mode, resolved route, node trace, source context, artifact, and preview.
2. **Measurable:** open the fixed retrieval report and keep its three coverage
   measures separate; show the current five RAGAS components and historical
   fixture boundary.
3. **Achievable:** show the system diagram and identify the single generation
   boundary, bounded review loop, and local stores.
4. **Relevant:** connect the artifact and explanation to the creative coder's
   iteration loop, then state the unsupported production claims.
5. **Time-bound:** finish by minute ten with the terminal response visible and
   reserve the next five minutes for route, metric, privacy, and runtime questions.

## Reviewer questions this frame should answer

- Can the stated outcome be demonstrated from a clean local start?
- Which measures come from current runtime behavior, fixed committed reports,
  synthetic fixtures, or human review?
- Why is the project scope credible for one implemented provider and four live
  preview domains?
- How do the goals support creative learning and responsible use rather than
  code generation for its own sake?
- Which timestamps and fixture definitions constrain the evidence?

## Honest completion test

The SMART claim is met only when the reviewer can trace a request from UI to
terminal event and inspect the artifact or truthful failure. It is not met by a
feature label, a static route diagram, an old score, or a preview screenshot
alone.

Use this frame after the [SCR Presentation](SCR_PRESENTATION.md), then follow
the [Reviewer Guide](REVIEWER_GUIDE.md), [Ten-Minute Presentation](TEN_MINUTE_PRESENTATION.md),
and [Five-Minute Q&A](FIVE_MINUTE_QA.md) for the complete review path. The
[Architecture Diagram Guide](../architecture/README.md) connects these claims
to the system, route, preview, recovery, and evaluation views.
