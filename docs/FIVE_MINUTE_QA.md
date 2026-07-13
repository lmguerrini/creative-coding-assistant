# Five-Minute Capstone Q&A

Use the answer that matches the question; do not try to recite every answer.
Each primary answer is designed for roughly 35–45 seconds, leaving time for the
reviewer’s question and one follow-up. If asked for evidence, open the source
named under the answer.

## Priority answer 1 — Why use RAG instead of relying on the model?

> Creative-coding APIs and runtime boundaries are specific and can change. RAG
> lets the product select registered technical material and expose the source
> IDs and domains used for a request. The current retrieval report contains
> seven queries with five results each, 18 of 19 requested domains, and 16 of 23
> substantive source anchors. RAG does not guarantee a correct answer; it makes
> grounding inspectable and gives me a selection layer I can measure and
> improve independently from generation.

**Open:** `demo/evaluation/canonical_retrieval_report.json`.

## Priority answer 2 — Is 61.44% a good score?

> I do not treat 61.44 percent as a global product score. It is the equal-weight
> macro across four RAGAS dimensions on four sanitized approved-fixture rows.
> Context Precision is nearly 100 percent, but Faithfulness is 29.58 percent and
> Answer Relevancy is 47.43 percent, so the fixture exposes real weaknesses.
> Context Recall is missing because there are no independent reference answers.
> It is historical and not comparable to the current retrieval report. The
> correct conclusion is that the evaluation path works and the answer fixture
> needs improvement—not that the product is 61.44 percent “good.”

**Open:** Dashboard → Evaluation → AI Engineering Lab.

## Priority answer 3 — What is actually Multi-Agent?

> Multi-Agent is a bounded route in the product workflow, not a claim of an
> autonomous swarm. It resolves to planner, researcher, generator, critic, and
> reviewer roles, with at most one refinement loop. Single-Agent resolves to a
> generator and skips the separate planning, retrieval, critique, and review
> route. Auto chooses one of those paths using the request and route. The stream
> records requested mode, resolved mode, roles, rationale, and node transitions,
> and integration tests assert the differences. It does not run external tools
> or unattended production workflows.

**Open:** Dashboard → Workflow and
`src/creative_coding_assistant/orchestration/runtime/execution.py`.

## Priority answer 4 — Is the app really multimodal?

> It has a real but partial multimodal contract. A valid image’s pixels are
> serialized beside the text as an image input for one provider request. Tests
> verify that exact payload, that diagnostics exclude the raw data URL, that the
> composer clears the image after submission, and that persisted session
> snapshots contain no image bytes. Those tests do not prove live provider
> receipt or use. What I cannot yet show is a captured
> configured-model comparison proving the image materially changed palette or
> composition. So I claim image-input transport and privacy, not complete
> text-image synergy.

**Open:** `tests/test_multimodal_provider_inputs.py` and the Reference-guided
palette study.

## Priority answer 5 — What works live, and what remains bounded?

> The controlled browser surfaces support p5.js, plain bounded Three.js, WebGL
> 1 GLSL, and silent-first Tone.js for the four canonical showcase fixtures.
> The exact E2E gate proves rendering, interaction, fullscreen restore,
> refinement, persistence, and reload. It uses deterministic streams, so it is
> not model-quality evidence. React components, standalone HTML, remote modules,
> and external creative tools remain code, export, or handoff paths. The current
> product is a local workstation, not a public hosted service.

**Open:** `clients/nextjs/e2e/demo-showcase-smoke.spec.js`.

## Alternate questions

### How do you prevent hallucinated APIs?

> I reduce the risk through registered-source retrieval, runtime-specific
> prompts, strict artifact extraction, source eligibility checks, renderer
> boundaries, and visible failure states. I do not claim hallucinations are
> eliminated. Retrieval can miss a source, a model can ignore context, and code
> can still be wrong. That is why source lineage, artifact inspection, browser
> execution, and fallback status remain separate checks.

### How is creative quality evaluated?

> Technical validity is automated through source contracts, runtime health,
> visible-frame signals, interaction, and persistence. Creative quality is
> supported by prompt-adherence tokens and review metadata, but it is not
> reduced to a universal objective score. A future protocol should separate
> composition, motion, originality, interaction, and accessibility and use
> multiple reviewers or preference comparisons. Today I label aesthetic
> judgment as reviewer-facing evidence, not measurement certainty.

### What data can leave the machine?

> Provider-bound prompt content, selected retrieved context, and a
> request-scoped image can leave only when that provider path is explicitly
> used. Real credentials, workspace sessions, local databases, raw evaluation
> rows, and image bytes are not committed. Public evaluator fixtures are
> synthetic or redacted. In the current environment, local retrieved excerpts
> are not sent to an external RAGAS evaluator, which is why current-product
> end-to-end scoring is unavailable.

### Why is Context Recall missing?

> Context Recall asks whether retrieval found the information needed for a
> justified reference answer. The approved fixture does not contain independent
> reference answers, so computing recall would manufacture a denominator. The
> next evaluation dataset must add those references before the metric can be
> reported.

### Does the test count prove the product is finished?

> No. The dated local run passed 2,688 backend tests plus 423 subtests, 553
> frontend tests, typecheck, Ruff, build, and 28 Playwright cases. That is broad
> regression evidence for one checkout. It does not prove independent user
> acceptance, accessibility, public deployment, configured-service
> availability, or aesthetic quality. Those remain separate validation gates.

### How would you deploy it?

> I would first add production identity, authorization, rate limits, encrypted
> secret handling, retention controls, abuse safeguards, runtime monitoring,
> and a supported-browser policy. The backend already has a WSGI production
> surface and CI security checks, but the reviewed product is intentionally a
> local workstation. I would not expose the local demo server directly as a
> public service.

### What was the most important technical lesson?

> Evidence must follow the same run. A source-coverage metric looked better
> until lineage showed title-only chunks. A preview can look successful even
> when only source exists. A multimodal file picker does not prove pixels were
> used. The project improved most when I kept retrieval, generation, renderer,
> evaluation, and acceptance evidence separate and allowed missing evidence to
> remain missing.

### What would you build next?

> First, a 20-case public-safe current-product RAG dataset with independent
> reference answers and all justified RAGAS dimensions. Second, a controlled
> image-influence comparison. Third, accessibility and presentation-machine
> validation. Then I would split the largest UI, stream, style, and metadata
> modules before adding more runtimes.

## Five-minute control

- Answer the question asked; stop after one claim, one evidence point, and one
  limitation.
- Keep the Evaluation Lab or relevant source open; do not search through old
  release documents during Q&A.
- If evidence is unavailable, say so directly and name the validation needed.
- Do not describe deterministic fixtures as provider generations.
- Do not call 61.44% a product score.
- Do not claim image influence, independent acceptance, public deployment, or
  external creative-tool execution.

## Closing line if time remains

> The product’s value is not that every output is automatically trustworthy.
> It is that the evidence needed to judge an output is visible and bounded.
