# Exact 10-Minute Capstone Presentation

Use this script with `outputs/creative-coding-assistant-capstone.pptx`.
Presenter actions are not spoken. Rehearse on the presentation machine and keep
one preflight-approved artifact session ready. If it is unavailable, use the
matching fallback wording instead of inventing a live provider result.

## 0:00–1:00 — Problem

**Presenter action:** Show slides 1–2, then the empty Creative Workstation.

**Spoken script:**

> Creative coding often starts as a feeling—an aurora garden, an orbital
> sculpture, or a score that becomes a visual system. Turning that idea into a
> working piece still means choosing a runtime, finding APIs, constraining the
> output, debugging it, and deciding whether it really ran.
>
> A generic chatbot can return code-shaped text, but it rarely preserves the
> creative session or answers the reviewer’s questions: which sources grounded
> the response, which workflow executed, whether the artifact matches the
> renderer, and what failed when a dependency was unavailable. Creative Coding
> Assistant addresses that continuity and evidence gap.

## 1:00–2:00 — Solution

**Presenter action:** Show slide 3 and open Demo Mode.

**Spoken script:**

> Creative Coding Assistant is a local AI workstation that turns audiovisual
> intent into an inspectable browser-native code artifact. The same surface
> keeps the prompt, workflow, sources, code, preview, refinement, session state,
> and evaluation boundary visible.
>
> Demo Mode offers ten curated flows. Four are canonical browser showcases for
> Tone.js, p5.js, Three.js, and GLSL. The others cover retrieval, Single and
> Multi routes, export handoff, a reference-image request, and recovery. A demo
> choice loads the normal product path; its expected output and recovery
> instructions are labeled before execution.

## 2:00–3:30 — Architecture and workflow

**Presenter action:** Show slides 4–5, then Dashboard → Architecture and
Dashboard → Workflow.

**Spoken script:**

> The Python backend exposes a local WSGI service. Its compiled workflow handles
> intake, routing, optional retrieval, prompt construction, generation,
> artifact extraction, preview preparation, critique, up to two bounded
> refinements, and finalization. SQLite stores sessions; local Chroma stores
> indexed technical knowledge; provider calls remain explicit and
> environment-dependent.
>
> The Next.js workstation consumes the stream and turns it into conversation,
> artifact, source, preview, workflow, recovery, and session state. Supported
> browser code runs through controlled runtime surfaces. Unsupported outputs
> stay code or export-only rather than being called live previews. Browser
> frame status and telemetry remain local after hydration; they do not feed
> backend critique or review.
>
> Workflow choice changes execution. Single uses one generator path and skips
> the separate research, planning, critique, and review stages. Multi uses those
> roles with up to two refinement attempts, for at most three generation calls
> including the initial call. Auto resolves Single only for Explain or Debug
> with no attachments and no resolved domains; every other Auto request resolves
> Multi. Successful Explain generation goes directly to finalization without
> the artifact and review path. Requested mode, resolved mode, executed nodes,
> and rationale are emitted in the stream, so the graph describes the run
> instead of decorating it. The published maximum-refinement field still says
> one, a documented drift from the executable two-attempt limit.

## 3:30–4:45 — Hardest challenge: retrieval without moving the goalposts

**Presenter action:** Show slide 6 and Dashboard → Evaluation → Retrieval.

**Spoken script:**

> The hardest engineering challenge was improving retrieval without changing
> the benchmark to flatter the result. The dated local snapshot contains 1,445
> chunks. Seven fixed queries each return five ranked results. Requested-domain
> coverage improved from 7 of 19 to 18 of 19, and substantive expected-source
> overlap improved from 9 of 23 to 16 of 23.
>
> An intermediate result looked better at 19 of 23, but it counted title-only
> and index-only chunks. I removed those false positives. Expected sources are
> diagnostic anchors; they are not injected into the top five. The final lower
> number is more defensible. These are local selection measures, not RAGAS,
> final-answer accuracy, or a product grade.

## 4:45–6:30 — Live product path

**Presenter action:** Open the preflight-approved Recursive aurora garden.
Show provenance, pointer interaction, fullscreen, one cold-palette refinement,
and reload. Briefly name the other three showcase runtimes.

**Spoken script:**

> I am using the artifact session prepared in preflight, and its visible
> provenance is the source of truth. The deterministic browser gate separately
> proves the workstation and renderer path; it is not a fresh model-quality
> result.
>
> Recursive aurora garden is a compact p5.js artifact with golden-angle seeds,
> slow motion, and pointer parallax. I can enter the fullscreen Creative
> Session, return without losing the sidebar or Inspector state, request a
> colder palette, and reload the refined artifact.
>
> The other fixtures test distinct contracts. Polyrhythmic constellation uses
> Tone.js and starts silent until explicit user action. Kinetic orbit sculpture
> uses the locally bundled Three.js revision 176. Fractal solar bloom is a
> bounded WebGL 1 shader. The browser suite checks exact request and artifact
> identity, runtime signal, fullscreen restoration, follow-up payload,
> persistence, and source-specific tokens. It does not assign an objective
> aesthetic score.

## 6:30–8:00 — Evaluation and ethics

**Presenter action:** Show slides 7–8 and Dashboard → Evaluation.

**Spoken script:**

> Evaluation stays in separate lanes: current retrieval/RAGAS, creative review,
> workflow, and reliability. The current seven-case public RAG run completed
> every eligible case with no skips or metric failures. Context Precision is
> 51.96 percent, Faithfulness 64.90 percent, Answer Relevancy 56.63 percent,
> Context Relevancy 85.71 percent, and Context Recall 80.95 percent. Their
> equal-weight Retrieval Quality macro is 68.03 percent.
>
> The former 61.44 percent four-row fixture is retained in History. It was
> obsolete as a primary score because it did not execute the current retrieval,
> prompt, generation, and benchmark paths. The current score is still not a
> project grade or human artistic-quality judgment. Dashboard current-product
> runs do not overwrite the committed canonical summary; publication is a
> separate explicitly gated CLI action, and private diagnostics are CLI-only.
>
> Reference images are also bounded. The backend tests prove that accepted
> pixels are included in the configured-provider request payload, then removed
> from the composer and omitted from session persistence. They do not prove
> live provider receipt, use, or image influence. Private prompts, raw sessions,
> local retrieval stores, and secrets remain outside public evidence.

## 8:00–9:15 — Validation and learning

**Presenter action:** Return to slide 7, then show slide 9.

**Spoken script:**

> The dated V9.8 release gates passed 2,688 backend tests plus 423 subtests, 553
> frontend tests, TypeScript checking, Ruff, a production build, and 28 of 28
> browser tests. One useful regression came from an ordinary pointer click that
> could miss the refinement button after fullscreen restoration. The fix
> established a deterministic stacking context and kept the real-click test.
>
> The larger lesson is that automated breadth is not independent human
> acceptance. The demo fixtures prove their asserted product and runtime
> contracts; they do not prove universal accessibility, aesthetic quality, or
> deployment reliability.

## 9:15–10:00 — Next steps and close

**Presenter action:** Show slide 10.

**Spoken script:**

> The next evidence priorities are a new, broader version of the retained
> seven-case public current-product benchmark, improvement of its weaker
> context-precision and answer-relevancy metrics, a controlled image-influence
> study, a parameter experiment dataset for the incomplete tuning case,
> accessibility validation, and independent human acceptance before release.
>
> The project’s main learning is simple: an AI coding product is trustworthy
> only when the request, route, data, artifact, runtime, score, and limitation
> describe the same run. Creative Coding Assistant makes that agreement visible.

## Exact fallback substitutions

Use only the sentence that matches the observed failure.

**Configured generation unavailable:**

> The configured generation service is unavailable, so I am using the
> deterministic local artifact. This is renderer and product-path evidence,
> not a new provider result.

**Retrieval unavailable:**

> Current retrieval is unavailable. I will show the dated fingerprinted report
> and will not claim current citations.

**Preview unavailable:**

> The renderer is unavailable. I am showing source and tested boundaries, not
> calling this a successful live preview.

**Time overrun at 8:00:**

> The dated local backend, frontend, build, and browser gates pass; their claims
> remain narrower than human acceptance.

Then continue with the final 45-second close.

The [Architecture Diagram Guide](../architecture/README.md) is the visual
companion for the architecture, workflow, preview, recovery, and evaluation
claims in this script.
