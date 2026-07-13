# Future Work

This roadmap lists evidence gaps and maintainability work. None of the items
below should be read as a current capability or committed delivery date.

## Priority 0 — close reviewer-facing evidence gaps

### Current-product RAG evaluation

**Goal.** Produce one privacy-approved, versioned evaluation in which the query,
retrieved context, generated answer, and independently justified reference
answer all come from the same current product run.

**Acceptance evidence.** At least 20 public-safe cases across the canonical
technical domains; Context Precision, Faithfulness, Answer Relevancy, Context
Relevancy, and Context Recall; per-case results; zero private session text;
model and embedding configuration; dataset and code fingerprints.

**Why first.** The current 61.44% RAGAS result is a four-row approved fixture
and is explicitly non-comparable to the improved local retrieval report.

### Controlled image-influence study

**Goal.** Test whether a configured multimodal model materially uses a supplied
reference rather than merely receiving it.

**Acceptance evidence.** A synthetic image set with no private content;
matched text-only and image-plus-text prompts; blinded criteria for palette,
composition, and motif transfer; retained outputs and request metadata; no raw
image bytes in persisted session records.

**Why first.** Image transport is proven; image influence is not.

### Final browser and presentation preflight

**Goal.** Rehearse the exact 10-minute route on the presentation machine.

**Acceptance evidence.** Backend health, frontend load, configured-service
status, four fixture previews, pointer interaction, fullscreen restore,
refinement, reload, optional audio policy, Evaluation Lab, and offline fallback
checked immediately before presenting.

**Why first.** Automated browser evidence is strong, but it is not independent
or presentation-room acceptance.

## Priority 1 — improve quality and maintainability

### Resolve the remaining retrieval-domain gap

Identify an accessible, reviewable source for the missing shader-community
domain, sync it through the existing source registry, and rerun the unchanged
seven-case benchmark. Do not pin the source into results. Success means useful
substantive chunks appear through normal selection while existing domain
coverage does not regress.

### Split large modules by responsibility

The audited baseline contains a 460 KB stylesheet, several 300–380 KB metadata
modules, a 305 KB workstation component, and a 289 KB stream model. Extract
stable domain registries, view models, panels, and style layers while preserving
public contracts. Require unchanged unit/E2E behavior and a measurable drop in
the largest file sizes.

### Add accessibility evidence

Run keyboard-only navigation, focus restoration, screen-reader naming, reduced
motion, color-contrast, and zoom checks across the workstation, Dashboard,
Demo Mode, preview, and fullscreen session. Convert findings into repeatable
tests where the browser API is reliable.

### Add real performance profiles

Measure cold start, warm start, stream first-event latency, preview mount,
refinement, session restore, and memory behavior under defined hardware and
dataset conditions. Keep these separate from WebGL frame-energy checks and
from provider latency, which is environment-dependent.

### Consolidate current public documentation

Generate catalog and evaluation tables from canonical JSON/TypeScript exports,
label historical documents unmistakably, and fail CI when current reviewer
pages mention an obsolete scenario count or broken link.

## Priority 2 — expand product scope carefully

### Broader preview coverage

Add a runtime only when there is a sandbox contract, failure model, security
review, visible-output test, reload behavior, and explicit unsupported-source
boundary. External creative tools should remain export/handoff paths until an
actual controlled integration exists.

### Creative evaluation protocol

Develop a rubric that separates technical validity, prompt adherence,
composition, motion, interaction, originality, and accessibility. Use multiple
reviewers or preference comparisons where possible. Never collapse subjective
quality and runtime correctness into one number.

### Parameter-tuning experiment

Create the controlled dataset required for Capstone Case 7: identical tasks,
declared temperature/max-token/reasoning conditions, blinded output review,
latency/token records, and a reproducible CSV. Until this exists, creativity
profiles remain controls rather than performance-tuning evidence.

### Deployment hardening

If a public deployment is desired, add production identity, authorization,
rate limiting, encrypted secrets, data-retention controls, dependency/runtime
monitoring, abuse safeguards, and an explicit supported browser matrix. The
current local workstation should not be described as a public service.

## Decision order

1. Close evidence gaps before expanding claims.
2. Reduce module and documentation coupling before adding more domains.
3. Add accessibility and performance baselines before public deployment.
4. Expand integrations only with the same observable runtime and privacy
   boundaries used by the current browser-native paths.
## Suggested SMART checkpoint

Within one focused iteration, create a 20-case public-safe current-product RAG
dataset, execute all five justified metrics, publish per-case machine-readable
results, and update the Evaluation Lab without changing the existing retrieval
benchmark. Completion requires zero private text, reproducible fingerprints,
and explicit non-comparability to older fixtures.
