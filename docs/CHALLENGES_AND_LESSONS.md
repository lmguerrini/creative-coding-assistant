# Challenges and Lessons Learned

## 1. Creative language is not an executable contract

**Challenge.** A prompt such as “make an organic fractal sculpture” leaves the
runtime, artifact shape, dependencies, interaction, and validation ambiguous.
Plausible prose is easy; a bounded artifact that can actually run is harder.

**Response.** Curated prompts name one artifact, one runtime, allowed APIs,
forbidden dependencies, interaction expectations, and fallback behavior. The
preview layer then validates source shape before selecting a renderer.

**Evidence.** The four canonical fixtures use distinct Tone.js, p5.js,
Three.js, and GLSL contracts, and browser tests assert both source-quality
tokens and visible runtime signals.

**Lesson.** Prompt engineering for code is interface design. The prompt should
describe the contract that downstream extraction, rendering, and validation
can enforce.

**Remaining gap.** A valid contract can still produce mediocre art. Aesthetic
judgment remains partly subjective and needs structured review rather than a
fabricated universal score.

## 2. Higher retrieval coverage can be worse evidence

**Challenge.** An intermediate retrieval iteration reached 19/23 expected
source anchors, but lineage inspection showed that some “coverage” came from
title-only chunks and an API-name index.

**Response.** The retrieval loop removed non-substantive chunks, preserved
source and domain diversity, and added bounded candidate headroom. The final
report settled at 16/23 source anchors and 18/19 requested domains.

**Evidence.** The machine-readable report binds seven top-five result sets to a
1,445-record metadata fingerprint and records every selected source and domain.

**Lesson.** A lower honest metric can be a better result. Evaluation should
reward useful context, not labels that happen to match an expected list.

**Remaining gap.** One requested domain is still absent from the local index.
The seven-case current-product generated-answer evaluation is complete, but
broader or private-session evaluation remains outside the approved data
boundary until a separate reviewed benchmark is available.

## 3. Evaluation datasets are not interchangeable

**Challenge.** Retrieval coverage, synthetic RAGAS rows, redacted historical
rows, browser fixtures, and current product behavior answer different
questions. Combining them into a single “AI score” would be misleading.

**Response.** The Evaluation Lab separates Retrieval/RAGAS, Creative, Workflow,
and Reliability evidence. The current seven-case five-metric macro is 68.03%;
the approved-fixture 61.44% macro is labeled historical.

**Evidence.** All seven current-product RAG cases completed with zero skips or
metric failures. Context Precision is 51.96%, Faithfulness 64.90%, Answer
Relevancy 56.63%, Context Relevancy 85.71%, and Context Recall 80.95%.

**Lesson.** “Missing” and “not comparable” are valid engineering outcomes.
They are more useful than a confident number with the wrong denominator.

**Remaining gap.** Broader public-safe coverage, repeated evaluator runs, and
stronger context ordering are needed before generalizing beyond the seven-case
versioned protocol.

## 4. A preview must report what happened, not what was intended

**Challenge.** Generated code can exist while the selected browser runtime is
unsupported, unhealthy, or incapable of executing that artifact shape.

**Response.** Artifact routing distinguishes previewable code, code/export-only
handoff, unavailable renderer, provider fallback, and partial outcome. The
isolated preview host reports lifecycle and visible-output signals.

**Evidence.** The direct Three.js smoke checks runtime revision 176, a visible
canvas, frame energy above 80, a changing signature, nested scene-graph tokens,
and zero page errors. The GLSL and p5 fixtures have their own bounded checks.

**Lesson.** Source presence is not runtime success. UI state should be derived
from the strongest observed evidence available.

**Remaining gap.** Automated visual signals catch blank or static failures but
do not replace an accessibility review, performance profile, or aesthetic
critique.

## 5. Multimodal transport is not multimodal quality

**Challenge.** A file picker and image metadata can create the appearance of
multimodality even if no image pixels reach the model.

**Response.** The backend test follows real PNG bytes from the browser request
to an `input_image` block beside `input_text`. Raw image data is excluded from
string representations and diagnostics, cleared from the composer after
submission, and excluded from persisted session snapshots.

**Evidence.** Contract tests cover byte transport, text-only requests,
metadata-only requests, and sanitized diagnostics.

**Lesson.** Multimodal evidence needs two independent proofs: transport and
influence. Only the first is currently complete.

**Remaining gap.** A controlled study must compare outputs with and without a
synthetic reference image before the product can claim that visual properties
materially guided the result.

## 6. Workflow labels must match execution

**Challenge.** “Multi-Agent” can become marketing copy if the graph follows the
same path regardless of selection.

**Response.** The runtime records requested and resolved modes, role names,
researcher activation, rationale, and a refinement-loop bound. Single-Agent
skips planning, research, critique, and review nodes; Multi-Agent exposes a
five-role bounded plan; Auto resolves using request and route evidence.

**Evidence.** Backend integration tests assert node transitions and route
payloads; browser tests assert the selected mode in outgoing requests.

**Lesson.** The graph shown to a user should be a projection of execution
evidence, not an aspirational architecture diagram.

**Remaining gap.** The current roles are bounded steps inside one product
workflow. They do not prove parallel agents, autonomous external tools, or
unattended task completion.

## 7. Fullscreen revealed an ordinary interaction bug

**Challenge.** After fullscreen restore at a compact viewport, nested scrolling
and hover translation could move a refinement submit control under another
hit-test surface.

**Response.** The form received an explicit stacking context, the parameter
panel and submit control were layered deterministically, scroll margin was
added, and hover translation was removed.

**Evidence.** The regression retained an ordinary pointer click. All four
canonical refinement flows passed 4/4 in 19.2 seconds, and the complete
Playwright suite passed 28/28.

**Lesson.** Accessibility semantics and programmatic dispatch are not enough.
High-value paths need real pointer geometry after the same resize, fullscreen,
and restore sequence used in a presentation.

**Remaining gap.** Normal-browser rehearsal with the actual presentation
display, audio policy, and configured services is still a manual preflight.

## 8. A large repository can hide its own evidence

**Challenge.** The project grew to 1,627 tracked files at the audited baseline,
including very large metadata registries, a large workstation component, and
historical reviewer documents whose earlier claims can be mistaken for current
state.

**Response.** Current reviewer documents now point to canonical source files,
machine-readable evaluation artifacts, and dated validation snapshots. The
public-boundary audit labels historical material and private local state.

**Evidence.** The repository and history audits enumerate tracked-file hygiene,
large-file hotspots, documentation debt, and the limits of the scans.

**Lesson.** Documentation architecture is part of product architecture. One
clear current evidence path is more useful than many optimistic status files.

**Remaining gap.** Oversized modules should be split by stable responsibility,
and old reviewer material should eventually move to a clearly named archive or
be regenerated from canonical data.

## Summary

The recurring lesson is evidence alignment: request, route, data, artifact,
runtime, score, and claim must describe the same run. When they do not, the
product should show the mismatch instead of smoothing it over.
