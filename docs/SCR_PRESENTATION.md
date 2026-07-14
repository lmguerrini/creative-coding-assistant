# SCR Presentation

This reviewer-facing narrative uses **Situation–Complication–Resolution** to
explain why Creative Coding Assistant exists, what is demonstrably working, and
where its boundaries remain. It is designed as the opening three minutes of a
ten-minute capstone presentation, followed by the live product and evaluation
evidence.

## One-slide version

| Situation | Complication | Resolution |
|---|---|---|
| Creative technologists move between ideas, official API documentation, code editors, and browser runtimes. They need rapid exploration without losing control of the artifact. | A generic assistant can return plausible code while hiding weak sources, route choices, unsupported runtimes, or image handling. That makes a polished answer difficult to review and reproduce. | Creative Coding Assistant turns text or bounded image references into an inspectable code artifact through a published Single/Multi route, optional official-source retrieval, a bounded LangGraph workflow, browser preview metadata, persistent sessions, and explicit evaluation evidence. |

**Claim in one sentence:** the product does not try to replace the creative
coder; it makes AI-assisted creative-code exploration more visible, grounded,
and reviewable.

## Sixty-second opening

> Creative coders often begin with an aesthetic idea but must immediately cross
> several technical boundaries: choose a library, find reliable documentation,
> translate intent into code, and verify the result in a browser. A general
> chatbot can accelerate the first draft, but a convincing answer is not the
> same as a traceable working artifact. Creative Coding Assistant addresses that
> gap with a Next.js workstation and a bounded backend workflow. The user can
> submit a prompt or a validated image reference, see whether Single, Multi, or
> Auto was requested and which route actually ran, inspect official-source
> retrieval and node events, extract the generated code, and preview the four
> canonical live domains: p5.js, Three.js, GLSL, and Tone.js. The point is not an
> invisible agent swarm. It is a compact creative loop whose route, evidence,
> artifact, and limitations remain available for review.

## Narrated structure with demo cues

### Situation — 0:00–0:50

Show the Creative Session rather than a slide full of feature names.

- Start from a visual or sonic intention, such as a Three.js orbit sculpture.
- Point to the domain and workflow selectors.
- Explain that the desired outcome is an editable artifact and a runtime result,
  not prose alone.

Transition: “The hard part is not producing code-shaped text; it is knowing what
happened between the idea and the result.”

### Complication — 0:50–1:35

Use the Inspector and Evaluation surfaces to make the risk concrete.

- Source registration, indexing, retrieval, and citation are different states.
- Single and Multi are different execution paths; Multi is sequential
  application logic, not five parallel provider-backed agents.
- Auto selects Single only for an Explain or Debug route with no attachments
  and no resolved domains; every other Auto request resolves Multi.
- A listed runtime is not automatically a live internal preview. The canonical
  browser-preview contract is p5.js, Three.js, GLSL, and Tone.js.
- Browser frame status and telemetry are local post-hydration evidence; they do
  not feed the backend artifact critic or workflow reviewer.
- An uploaded image matters only if its validated bytes enter the provider
  payload; filename metadata alone would not be multimodal generation.

Transition: “So the product has to expose those boundaries instead of asking a
reviewer to trust the interface.”

### Resolution — 1:35–3:00

Run one request and narrate only evidence visible in the product or repository.

1. The Next.js client sends the request to the exact-path WSGI API.
2. The compiled LangGraph emits lifecycle and node events.
3. Multi retrieves official-source context, prepares typed planning guidance,
   calls the generation provider, extracts an artifact, prepares preview
   metadata, reviews the result, and allows up to two refinement passes—at most
   three generation calls including the initial call.
4. The response preserves the resolved route, executed nodes, source context,
   artifact, and terminal state for inspection.
5. A supported domain opens through its browser-focused preview path; other
   domains remain code/export or external-tool handoffs unless separately
   validated.

The public `execution.max_refinement_loops` field still reports `1` while the
executable reviewer permits two attempts. Treat that as a documented
contract/runtime drift. Explain is also a deliberate short path: after
generation it goes directly to finalization without artifact, preview, critique,
or review nodes.

Then show the evaluation boundary:

- The canonical fixed retrieval run completed all **7/7** cases.
- It covered **16/23 expected source anchors (69.57%)**.
- It covered **18/19 requested domains (94.74%)**.
- The current seven-case RAGAS run reported context precision
  `0.5196428571169692`, faithfulness `0.648989898989899`, answer relevancy
  `0.5662963631284655`, context relevancy `0.8571428571428571`, and context
  recall `0.8095238095238094`.

The five current means form the 68.03% Retrieval Quality macro. Do not present
it as a whole-product score, project grade, or human artistic judgment. The old
61.44% four-row fixture remains historical and has no context-recall result.

## Evidence ledger for the slide

| Statement | Evidence to open | Boundary to say aloud |
|---|---|---|
| Single and Multi execute different paths | [Runtime workflow graph](../architecture/workflow_graph.md) and streamed node events | Multi responsibilities are sequential; only generation crosses the text-generation boundary |
| Image-guided input is genuine | [Architecture Walkthrough](ARCHITECTURE_WALKTHROUGH.md) and multimodal request tests | At most four PNG/JPEG/WebP/GIF files, 1 MiB each; request bytes are not restored with sessions |
| Retrieval is evaluated | [`canonical_retrieval_report.json`](../demo/evaluation/canonical_retrieval_report.json) | Coverage is not grounded-answer quality |
| Current-product RAGAS components exist | [Evaluation Metrics Summary](EVALUATION_METRICS_SUMMARY.md) | Seven public RAG cases; five metrics; no whole-product claim |
| Browser previews are bounded | [Capability Matrix](CAPABILITY_MATRIX.md) | Four canonical live domains; other adapters/handoffs are not the same claim |
| Privacy is inspectable | [Ethics & Privacy Assessment](ETHICS_PRIVACY_ASSESSMENT.md) | Provider and embedding calls can send user/source text off-device; no production auth claim |

The Dashboard queues and polls only current-product evaluation runs. Dry-runs
are unscored, canonical publication is a separate explicit CLI action, and the
four-row historical fixture remains a direct historical-API lane. See the
[evaluation workflow](../architecture/evaluation_workflow.md).

## Closing line

> The resolution is not “AI generates perfect art.” It is that a creative coder
> can move from intention to an inspectable artifact while the system keeps its
> route, sources, runtime boundary, and uncertainty visible enough to challenge.

## Presentation guardrails

- Keep the architecture terms tied to observable behavior.
- Treat automated checks as engineering evidence, not human aesthetic judgment.
- If the provider or network is unavailable, show the truthful failure state and
  switch to committed evidence; do not present a prerecorded success as live.
- State when a metric belongs to a fixed retrieval report or synthetic fixture.
- Do not describe metadata registries as runtime agents, model routing, telemetry,
  or autonomous policy execution.

Continue with the [SMART Presentation](SMART_PRESENTATION.md) for measurable
scope, the [Ten-Minute Presentation](TEN_MINUTE_PRESENTATION.md) for timing, and
the [Evaluation Criteria Mapping](EVALUATION_CRITERIA_MAPPING.md) for the
reviewer evidence path. Use the
[Architecture Diagram Guide](../architecture/README.md) for the complete visual
suite.
