# Evaluation Criteria Mapping

The official AI Engineering Capstone evaluates four areas: **Outcome Quality**,
**Learning Application**, **Ethical Considerations**, and **Presentation**.
This document maps each criterion to evidence a reviewer can inspect now. It
does not invent rubric weights or turn repository metrics into a predicted
grade.

## Evidence hierarchy

When two claims conflict, use this order:

1. current behavior observed in the running product;
2. current automated validation;
3. current evaluation output and machine-readable evidence;
4. current product and architecture documentation;
5. historical evidence, explicitly dated and labelled;
6. planned work, explicitly labelled.

`BLOCKED`, `MISSING_EVIDENCE`, and `NOT_COMPARABLE` are valid findings. None is
automatically a zero, and none is a pass.

## Capstone case alignment

The project combines several official case patterns rather than pretending to
be a perfect implementation of each one.

| Official case | Alignment | Inspectable implementation | Important boundary |
|---|---|---|---|
| Case 1 — RAG-powered knowledge assistant | **Primary** | Approved official-source registry, explicit sync, OpenAI embeddings, Chroma search, request provenance, retrieval evaluation | Single Agent skips retrieval; current retrieval quality is not identical to RAGAS answer quality |
| Case 2 — AI agent for task automation | **Primary but bounded** | LangGraph automates intake, routing, memory, research, planning, generation, critique, review, refinement, and finalization | Sequential application responsibilities; no arbitrary tool executor or parallel provider-backed swarm |
| Case 3 — Smart document search | **Supporting** | Semantic search over an official creative-coding documentation index with domain/source metadata | No user PDF upload or general enterprise document repository |
| Case 4 — Multimodal AI | **Supporting and partial** | Validated image pixels accompany prompt text as `input_image` content in the configured-provider payload | Payload construction is tested; live provider receipt, use, image influence, local vision, caption indexing, and audio input are not established |
| Case 5 — Code generation and debugging | **Primary** | Domain-aware prompts, code artifacts, Debug mode, source inspection, bounded preview, targeted refinement | Not a general-purpose IDE or unrestricted code-execution service |
| Case 6 — Advanced topic/tool | **Supporting** | Typed orchestration, evaluation separation, provenance, preview contracts, failure truthfulness | Advanced metadata registries do not imply active executors |
| Case 7 — LLM performance tuning | **Supporting / incomplete** | Generation controls, retrieval improvement history, provider telemetry, RAGAS components, repeatable fixed-query report | No controlled parameter-experiment or broad model-comparison dataset; requested temperature is not proven applied by every model |

## 1. Outcome Quality

### Official reviewer question

Is the final product complete and functional, is the interaction coherent, and
is its purpose and achieved outcome clear?

### Evidence map

| Expected outcome | Current evidence | Live review action | Residual limitation |
|---|---|---|---|
| Clear purpose | The first README paragraph states goal, problem, and mechanism | Ask the presenter for the same one-sentence explanation | A strong statement is not proof of usability |
| Interactive application | Next.js workstation with composer, workspace, Dashboard, Inspector, sessions, Demo Mode, and fullscreen | Start both services and submit one flagship prompt | Provider-backed behavior depends on credentials/network |
| End-to-end artifact | Stream → artifact extraction → source → preview contract → browser renderer → saved session | Generate p5.js or Three.js, inspect source, interact, reload session | A generated artifact can fail preview validation |
| Workflow choice | Published Single, Multi, and Auto request/resolution payloads | Compare explicit Single and Multi; let Auto publish its resolution | Current default Auto resolves Multi for normal requests |
| Multimodal request | Image validation at browser/backend plus configured-provider `input_image` payload construction | Run Reference-guided palette study with a public image | No proof of live provider receipt/use/influence, durable image restoration, or local pixel analysis |
| Failure handling | Typed subsystem errors, terminal workflow failure, retry/recovery UI | Inspect a controlled failure fixture or disconnect dependency deliberately in test | Recovery does not make an unavailable preview successful |
| Persistent creative session | SQLite lifecycle and browser fallback | Create, rename, switch, reload, and delete a session | Local single-user storage posture |
| Professional documentation | README, reviewer guide, architecture, capability, ethics/evaluation, and presentation artifacts | Follow links without reading source first | Documentation ranks below live behavior |

### Automated evidence

Focused deterministic modules include:

- [`tests/test_langgraph_workflow_integration.py`](../tests/test_langgraph_workflow_integration.py)
  for compiled graph execution, modes, transitions, review, and final events;
- [`tests/test_nextjs_streaming_bridge.py`](../tests/test_nextjs_streaming_bridge.py)
  for the WSGI request/stream bridge;
- [`tests/test_multimodal_provider_inputs.py`](../tests/test_multimodal_provider_inputs.py)
  for redaction, binary validation, and OpenAI image payloads;
- [`tests/test_workspace_session_persistence.py`](../tests/test_workspace_session_persistence.py)
  for session lifecycle;
- [`clients/nextjs/e2e/workstation-smoke.spec.js`](../clients/nextjs/e2e/workstation-smoke.spec.js)
  and [`demo-showcase-smoke.spec.js`](../clients/nextjs/e2e/demo-showcase-smoke.spec.js)
  for browser journeys.

Run the relevant subset rather than presenting one suite as proof of every
claim:

```bash
.venv/bin/python -m pytest -q \
  tests/test_langgraph_workflow_integration.py \
  tests/test_nextjs_streaming_bridge.py \
  tests/test_multimodal_provider_inputs.py \
  tests/test_workspace_session_persistence.py

npm run typecheck --prefix clients/nextjs
npm run test:e2e:smoke --prefix clients/nextjs
```

Browser smoke establishes the asserted interaction in its controlled setup. It
does not establish universal aesthetic quality, accessibility with real users,
or production reliability.

### Outcome Quality conclusion

The strongest claim is a functional local creative-code workstation with
inspectable workflow, artifacts, four canonical live preview domains, sessions,
and evidence surfaces. The project does **not** claim a public hosted service,
multi-user SaaS readiness, universal runtime support, or a completed human
usability study.

## 2. Learning Application

### Official reviewer question

Are AI Engineering tools and application-design practices used effectively,
and was system quality measured and improved deliberately?

### Architecture evidence

| Engineering topic | Applied learning | Evidence |
|---|---|---|
| LLM API | Provider-neutral generation contracts translated to OpenAI Responses; streaming, typed errors, usage, and image content | [OpenAI adapter](../src/creative_coding_assistant/llm/openai_adapter.py) |
| Orchestration | Compiled LangGraph, explicit state, registered nodes, conditional edges, one bounded refinement loop, terminal failures | [workflow graph](../architecture/workflow_graph.md) |
| RAG | Source governance, normalization/chunking, embeddings, Chroma, intent/domain filters, diversity, lineage, recoverable empty context | [System Overview](SYSTEM_OVERVIEW.md) |
| Prompt engineering | Structured inputs, Jinja sections, untrusted-context isolation, domain/runtime/artifact constraints | [Architecture Walkthrough](ARCHITECTURE_WALKTHROUGH.md) |
| Agents | Direct and sequential team routes with visible roles and no hidden concurrency claim | [Capability Matrix](CAPABILITY_MATRIX.md) |
| Evaluation | Fixed-query retrieval report, RAGAS components, privacy classes, missing/blocked states, fingerprints | [evaluation fixture notes](../demo/evaluation/README.md) |
| Application design | WSGI/Next separation, exact API contracts, local stores, browser execution boundary, recovery state | [System Overview](SYSTEM_OVERVIEW.md) |
| Testing | Deterministic providers, WSGI integration, frontend unit tests, Playwright smoke, docs/Mermaid gates | Repository tests and CI workflows |

### Measured retrieval engineering loop

The fixed seven-query, top-five retrieval pack was kept constant. Engineering
changes improved selection rather than changing queries, anchors, weights, or
the score rule:

| Stage | Source anchors | Requested domains | Engineering finding |
|---|---:|---:|---|
| Baseline | 9/23 | 7/19 | Intent ambiguity and one relevance pool hid requested domains |
| Intent + result diversity | 12/23 | 11/19 | Bounded domain intent and diversified selection |
| Balanced per-domain candidates | 15/23 | 17/19 | Strong domains stopped monopolizing the pool |
| Requested-domain fallback | 15/23 | 18/19 | Best available requested-domain evidence survived generic filtering |
| Source diversity | 17/23 | 18/19 | Two-chunk source cap opened space for distinct evidence |
| Apparent peak | 19/23 | 18/19 | Lineage exposed title-only and API-index false positives |
| Substantive-only correction | 15/23 | 18/19 | False coverage was removed even though the metric fell |
| Final bounded headroom | **16/23** | **18/19** | Substantive Three.js manual evidence recovered without pinning |

The final result is more defensible than the apparent 19/23 peak because it
removes weak structural content. The committed
[canonical report](../demo/evaluation/canonical_retrieval_report.json) binds
the result to 1,445 local KB metadata records and a selection fingerprint.

### Current-product RAGAS evidence

The committed canonical summary reports the retained current retrieval,
prompt, generation, and evaluator paths over seven public RAG cases:

| Metric | Mean | Interpretation |
|---|---:|---|
| Context precision | `0.5196428571169692` | The weakest retrieval-ordering dimension; improvement remains justified |
| Faithfulness | `0.648989898989899` | Most answer claims were supported, with material gaps remaining |
| Answer relevancy | `0.5662963631284655` | Partial answer alignment; evaluator variability and terminal caveats remain limits |
| Context relevancy | `0.8571428571428571` | Most selected contexts were useful to their questions |
| Context recall | `0.8095238095238094` | Reference-aware evidence coverage across the seven cases |
| Equal-weight macro | `0.6803191571804` (68.03191571804%) | Current Retrieval Quality, not a Capstone grade or human-quality score |

All seven rows were eligible and scored; none were skipped and no metric
failed. The dataset fingerprint is
`sha256:b5fbc0e7cc9a523658eee8b0fc5cd7c417aa10540f8919e10bc2c4e10a40705f`.
The former 61.44% four-row fixture remains historical evidence with no
context-recall denominator. Showing it as primary was classified
`EVALUATION_PIPELINE_DEFECT` and has been corrected.

### Learning Application conclusion

The best learning evidence is the separation of concerns and the willingness
to lower a headline metric after provenance revealed false positives. The
largest remaining technical learning opportunities are stronger grounded
answer construction, genuinely reachable Auto route diversity, broader but
privacy-approved end-to-end evaluation, and disciplined renderer expansion.

## 3. Ethical Considerations

### Official reviewer question

Does the project show awareness of privacy, bias, unsupported claims, provider
boundaries, and the risks of generated outputs?

### Data-boundary evidence

| Data | Local handling | External handling | Reviewer control |
|---|---|---|---|
| Prompt and generated answer | Stored in workspace/session and optional local memory/eval rows | Prompt/context go to OpenAI generation; successful prompt/answer go to OpenAI embeddings when memory recording is configured | Choose content, credentials, and whether to retain/delete local session data |
| Image reference | Browser-local until submit; stripped from durable session attachment state | Pixels enter the configured-provider payload only on explicit submit; live receipt/use/influence need separate evidence | Use public non-sensitive images; remove before submit or do not submit |
| Official KB source text | Chunks and vectors stored in local Chroma | Source is fetched from approved URL; chunk text goes to OpenAI embeddings; selected excerpts may enter generation | Select/check/update sources explicitly |
| Raw local evaluation sessions | Stored under ignored `data/eval/` | External evaluation remains excluded | Use only reviewed committed public/sanitized/redacted datasets |
| Evaluation fixture | Committed current-product public benchmark or historical synthetic/redacted rows | Goes to evaluator only after explicit live-call authorization | Dry-run first; inspect dataset, fingerprints, and cost warning |
| Trace metadata | Local stream contains safe status/lineage | Optional LangSmith metadata only when tracing and key are enabled | Keep tracing disabled by default |
| Secrets | Local `.env`, secret-bearing Pydantic settings | Used only to authenticate selected services | Never commit or display `.env`/keys |

### Ethical design choices

- External calls are named; “local-first” is not presented as “offline.”
- Images are byte/signature validated and held in secret-bearing contracts so
  ordinary serialization does not reveal data URLs.
- Retrieved and remembered text is isolated as untrusted context rather than
  treated as system policy.
- Retrieval errors produce empty evidence with an explicit recoverable error;
  citations are not fabricated.
- Weak RAGAS components and retrieval gaps remain visible.
- Browser-preview claims are limited to canonical validated contracts; external
  tools are code/export handoffs.
- Generated code, creative quality, and provider outputs still require human
  review.
- No religious, medical, psychological, or scientific authority is inferred
  from aesthetic geometry, audio, or generative-system metaphors.

### Residual risks

- The model can hallucinate APIs, copy insecure patterns, reproduce bias, or
  generate code outside the supported preview contract.
- Local storage is not a substitute for authentication, encryption-at-rest,
  retention policy, or protected backups.
- Optional provider and trace services have their own data-processing terms.
- Official documentation may change after the recorded local index timestamp.
- Automated creative scores may create false objectivity; no completed human
  study is claimed.

### Ethical Considerations conclusion

The defensible claim is transparent risk reduction, not perfect safety. The
reviewer should inspect provider settings, one multimodal request, local
persistence boundaries, evaluation dataset class, and at least one blocked or
missing state.

## 4. Presentation

### Official reviewer question

Can the presenter explain the problem, solution, architecture, data,
evaluation, hardest challenge, and next steps clearly within 10 minutes, then
defend trade-offs in 5 minutes of questions?

### Presentation evidence

| Requirement | Support artifact | What to show |
|---|---|---|
| Concise goal/problem/how | [README](../README.md) opening paragraph | Say it without reading a feature list |
| Problem → solution narrative | [SCR Presentation](SCR_PRESENTATION.md) | Fragmented tools → unverifiable generated code → evidence-aware workstation |
| Specific measurable framing | [SMART Presentation](SMART_PRESENTATION.md) | Four live runtimes, exact route evidence, current retrieval/RAGAS components, explicit limits |
| Architecture and models | [Architecture Walkthrough](ARCHITECTURE_WALKTHROUGH.md) | Next → WSGI → LangGraph → Chroma/OpenAI → artifact → browser preview |
| Data explanation | [System Overview](SYSTEM_OVERVIEW.md) | Official sources, local stores, embedding/generation/evaluator boundaries |
| Evaluation | This document and Dashboard Evaluation | Separate current retrieval, fixture RAGAS, product smoke, and human evidence |
| Live outcome | Demo Mode flagship | Generate, inspect source, preview, fullscreen, refine, reload |
| Hardest challenge | Retrieval evolution or genuine multimodal boundary | Show diagnosis, measured iteration, and narrower truthful claim |
| Limitations and next steps | [Capability Matrix](CAPABILITY_MATRIX.md) | Name a few high-value gaps, not a speculative feature catalogue |

### Recommended ten-minute evidence flow

1. **0:00–1:00 — Problem:** creative technologists cross documentation,
   prompting, coding, runtimes, and evaluation with little shared evidence.
2. **1:00–2:00 — Product:** show the workstation and state its bounded value.
3. **2:00–4:00 — Architecture:** show route resolution and the real
   Next/WSGI/LangGraph/Chroma/OpenAI/browser boundaries.
4. **4:00–5:00 — Data:** explain official sources, embeddings, local Chroma,
   memory, and external-call consent.
5. **5:00–7:00 — Demo:** run one flagship artifact, inspect source/preview,
   then make one refinement or reload the session.
6. **7:00–8:30 — Evaluation:** show 7/7, 16/23, and 18/19 as retrieval-only
   coverage, then show the five current-product RAGAS means and 68.03% macro.
7. **8:30–9:30 — Challenge:** explain why false-positive retrieval coverage was
   removed or how image bytes were made genuine, bounded provider input.
8. **9:30–10:00 — Next:** grounded-answer quality, Auto diversity, evaluated
   runtime expansion, and production security.

### Likely five-minute questions

- Why does Single Agent skip RAG?
- Is Multi Agent really multiple LLMs or parallel work?
- Why does Auto currently resolve Multi?
- What does 68.03% mean—and what does it not mean?
- Why is 61.44% historical, and why did the Evaluation pipeline show it before?
- Why are context precision and answer relevancy still weak?
- What leaves the machine during generation, embeddings, evaluation, and
  tracing?
- How does an image actually reach the model, and is it persisted?
- Which domains run live and which are code/export-only?
- How would authentication, scale, and provider choice change the architecture?

### Presentation conclusion

The strongest presentation is not the one with the most features. It is the
one in which every visible claim has a nearby evidence source and a clear
boundary.

## Reviewer acceptance checklist

- [ ] The README opens with one concise goal/problem/how paragraph.
- [ ] Both local services start and health/readiness are interpreted correctly.
- [ ] One live prompt produces an inspectable route, answer, artifact, and
      supported preview or a truthful failure.
- [ ] Single and Multi show different executed nodes.
- [ ] Auto displays its published resolution.
- [ ] The multimodal demo sends real image content, not metadata alone.
- [ ] Session save/restore works without restoring queued image bytes.
- [ ] Registered, indexed, retrieved, and cited source states remain distinct.
- [ ] Retrieval coverage and RAGAS fixture scores are presented separately.
- [ ] Weak, missing, and blocked metrics remain visible.
- [ ] External-tool handoffs are not described as internal execution.
- [ ] Automated tests are not presented as human artistic evaluation.
- [ ] Privacy boundaries and residual risks are explained before sensitive use.

If an item cannot be demonstrated, label the exact state and use the evidence
hierarchy above rather than substituting a historical pass.
