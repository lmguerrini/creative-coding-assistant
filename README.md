# Creative Coding Assistant

Creative Coding Assistant is an AI-native creative translation system and creative coding platform: a Creative Workstation that helps creative coders and technical artists turn ideas, constraints, or image references into inspectable creative-code artifacts. It closes the gap between plausible chatbot code and reviewable results through a Next.js workstation, a bounded LangGraph workflow, official-source Chroma retrieval, OpenAI generation, persistent artifacts and sessions, and browser-focused preview paths that keep evidence and limitations visible.

![Creative Coding Assistant workstation](assets/preview_current.png)

## Installation and run

### Prerequisites

- Python 3.11 or newer.
- Node.js and npm compatible with Next.js 14.
- An OpenAI API key for live generation, image-guided generation, query
  embeddings, memory embeddings, or a knowledge-base sync.
- Network access only for the provider-backed or official-source operations you
  explicitly choose to run.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,evaluation,server]"
npm ci --prefix clients/nextjs
cp .env.example .env
```

Edit `.env`, replace the placeholder `OPENAI_API_KEY`, and set
`LANGSMITH_TRACING=false` unless you intentionally want optional external
trace metadata. Never commit `.env`.

If this is a fresh workspace, populate Chroma only when you are ready to fetch
and embed the approved official sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all --continue-on-error
```

That command makes network and embedding-provider calls and can incur cost.
The application still starts without a populated index, but retrieval evidence
will correctly appear empty or unavailable.

Start the exact-path WSGI development API:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server --host 127.0.0.1 --port 8000
```

In a second terminal, start the workstation:

```bash
cd clients/nextjs
npm run dev
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). Confirm that the API is
live and ready:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/health/ready
```

Readiness can be guarded when provider credentials or production-safe settings
are missing even though liveness succeeds. See the
[Reviewer Guide](docs/REVIEWER_GUIDE.md) for the shortest inspection path.

## Product tour

The canonical product is the Next.js workstation. A normal session follows
this visible path:

1. Choose a creative domain, task mode, and **Single Agent**, **Multi Agent**,
   or **Auto** workflow.
2. Write a prompt or load a curated Demo Mode scenario. Optionally attach up to
   four PNG, JPEG, WebP, or GIF references of at most 1 MiB each.
3. Submit once. The workstation streams route, workflow, retrieval, generation,
   artifact, preview, review, and terminal events from the local API.
4. Inspect the answer, extracted source, runtime route, preview, provenance,
   and diagnostics instead of treating generated prose as execution evidence.
5. Refine the selected artifact, compare outputs, save the creative session,
   enter Session Fullscreen, or export an inspectable handoff package.

The Dashboard explains the same published product state at reviewer depth:
Overview, Architecture, Workflow, Workspace, Runtime, Preview, Artifacts,
Domains, Knowledge Base, AI & agents, Memory, Sessions, Telemetry, Evaluation,
User Guide, and Settings. The compact Inspector remains tied to the active run;
the Dashboard is the deeper evidence surface.

Demo Mode currently defines ten curated scenarios. Four are the canonical live
browser showcase sequence—Tone.js, p5.js, Three.js, and GLSL—while the remaining
scenarios exercise retrieval, workflow choice, multimodal input, export
boundaries, or failure recovery. Loading a scenario does not bypass the normal
assistant path or turn a prepared artifact into a new provider result.

## Architecture

```mermaid
flowchart LR
    user["Reviewer or creative coder"] --> ui["Next.js 14 workstation"]
    ui -->|"NDJSON + JSON over HTTP"| api["Local WSGI API"]
    api --> graph["Compiled LangGraph workflow"]
    graph --> memory["Local Chroma memory"]
    graph --> retrieval["Official-source retrieval"]
    retrieval --> embeddings["OpenAI embeddings"]
    retrieval --> kb["Chroma official-doc index"]
    graph --> generation["OpenAI Responses generation"]
    graph --> artifacts["Artifact, critique, and final event contracts"]
    artifacts --> ui
    ui --> preview["Controlled browser preview"]
    ui --> sessions["Workspace session API"]
    sessions --> sqlite["Local SQLite"]
    graph -. "optional trace metadata" .-> langsmith["LangSmith"]
```

The browser never calls the model directly. The Python backend validates the
request, resolves a workflow, assembles local memory and retrieval context,
renders provider-neutral messages, and crosses the OpenAI boundary only inside
the generation or embedding adapters. The backend prepares preview contracts;
the Next.js client executes accepted artifacts in controlled browser runtime
surfaces. Workspace snapshots use SQLite, while official knowledge and
conversation memory use separate Chroma collections.

The runtime node registry order is shown below for auditability. It is a
registry order, not a claim that every node executes on every route:

`intake -> routing -> memory -> retrieval -> context_assembly -> prompt_input -> planning -> director -> reasoning -> prompt_rendering -> generation -> artifact_extraction -> preview_preparation -> artifact_critique -> review -> refinement -> finalization -> failure`

### Workflow selection

| Choice | Actual route | Retrieval | Planning and review | Provider calls |
|---|---|---:|---|---|
| **Single Agent** | Direct generator path | Skipped | Skips planning, Director, reasoning, critique, review, and refinement | One generation pass when configured |
| **Multi Agent** | Sequential planner → researcher → generator → critic → reviewer responsibilities | Requested, with recoverable empty context on failure | Typed planning plus deterministic critique/review; at most one bounded refinement loop | One generation pass, plus one more only if refinement is requested |
| **Auto** | Resolves to one of the two routes after routing | Follows the resolved route | Follows the resolved route | Follows the resolved route |

Auto's Single branch requires a lightweight Explain or Debug request with no
official-document capability, no image attachment, and at most one domain.
The current default route map grants the official-document capability to every
task mode, so ordinary Auto requests currently resolve to Multi Agent. The UI
publishes the resolved route instead of guessing it. Auto is therefore a
selector, not a third hidden graph.

The role labels do not mean five parallel LLM workers. Planning, creative
direction, reasoning, artifact critique, and review are typed deterministic
application stages around the generation adapter. OpenAI is the only configured
generation provider in this repository.

For the complete source-aligned view, use:

- [System Overview](docs/SYSTEM_OVERVIEW.md)
- [Architecture Walkthrough](docs/ARCHITECTURE_WALKTHROUGH.md)
- [Runtime workflow graph](architecture/workflow_graph.md)
- [Standalone Mermaid source](architecture/workflow_graph.mmd)

## Capability Scope, evidence, and limitations

| Capability | Current product evidence | Boundary |
|---|---|---|
| Streaming text generation | OpenAI Responses adapter and NDJSON stream events | Requires credentials and network; no offline model is bundled |
| Image-bearing request construction | Validated image bytes become `input_image` beside user text in the configured-provider payload | Request-scoped; current evidence does not prove live provider receipt, use, or image influence, and attachments are not restored with a session |
| Official-source RAG | Explicit source registry, sync pipeline, embeddings, Chroma search, ranked lineage | Retrieval runs only on the Multi Agent route; an indexed or registered source is not automatically a cited source |
| Workflow automation | Compiled LangGraph with Single, Multi, and Auto route evidence | Sequential and bounded; no hidden parallel agent swarm or arbitrary tool execution |
| Creative artifacts | Source extraction, preview preparation, critique, one refinement loop, export | A generated artifact is not a successful preview until its runtime surface validates it |
| Live browser preview | Validated contracts for p5.js, Three.js, GLSL, and Tone.js | React Three Fiber and Hydra remain code/export; external creative tools are handoffs |
| Sessions and memory | SQLite workspace snapshots plus local Chroma conversation/project memory | Local single-user posture; memory embeddings send successful prompt/answer text to OpenAI when configured |
| Evaluation | Explicit Dashboard evaluation action, committed public fixtures, machine-readable retrieval report | RAGAS fixture quality and current retrieval coverage are separate measurements |
| Observability | Published workflow events, provider usage metadata, optional LangSmith adapter | LangSmith is off unless explicitly configured; telemetry is not model reasoning |

The [Capability Matrix](docs/CAPABILITY_MATRIX.md) gives a fuller implemented,
bounded, optional, and unsupported inventory. Domain delivery contracts live in
[Domain Experience](docs/DOMAIN_EXPERIENCE.md).

Important limitations:

- This repository does not prove a hosted public deployment, multi-user
  authorization, rate limiting, load/soak performance, managed backup, or
  enterprise isolation.
- It does not execute TouchDesigner, Blender, Houdini, Unreal, Unity, or other
  external creative tools. It can generate inspectable source and handoff
  packages for compatible runtimes.
- Generated code and creative quality still require human judgment. A browser
  smoke test proves the asserted interaction, not artistic merit.
- Retrieval can return weak or incomplete evidence. The UI must preserve empty,
  failed, blocked, and missing-evidence states rather than manufacture a pass.
- The current provider factory supports OpenAI only; provider/model matrices
  elsewhere in the codebase are advisory metadata unless the active adapter
  publishes execution evidence.

## Evaluation

Evaluation is split into lanes so unrelated measurements are not collapsed
into a single product score.

### Current local retrieval report

The committed report at
[`demo/evaluation/canonical_retrieval_report.json`](demo/evaluation/canonical_retrieval_report.json)
is bound to a 1,445-record local KB metadata snapshot and a fixed seven-query,
top-five benchmark. At `2026-07-13T05:05:33.306298+00:00` it reported:

- 7/7 cases with results;
- 16/23 substantive expected-source anchors covered (69.57%);
- 18/19 requested domains covered (94.74%).

Expected source IDs are coverage anchors, not forced top-k results. Heading-only
and verified index-only chunks were removed even when that lowered the headline
ratio. The remaining domain gap is documented rather than converted into a
false pass. Reproduce the read-only selection report with:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5
```

This still sends the public benchmark queries to the configured embedding
provider. Retrieved excerpt text remains local.

### Approved public RAGAS fixture

The product includes a committed, transcribed summary of an approved
provider-scored fixture that used four synthetic, public-safe rows, RAGAS
0.4.3, `gpt-4o-mini`, and `text-embedding-3-small`. All four rows were eligible,
none were skipped, and no metric call failed. The repository does not claim the
untracked raw result JSONL or run manifest as a committed artifact:

| Metric | Mean | What it says here |
|---|---:|---|
| Context precision | 1.0000 | The fixture's useful contexts were highly ranked |
| Faithfulness | 0.2958 | Many answer claims were not supported tightly enough by supplied context |
| Answer relevancy | 0.4743 | Answers only partly matched evaluator expectations for the questions |
| Context relevancy | 0.6875 | Retrieved excerpts were moderately useful to the questions |

The equal-weight display macro across those four means is 61.44%. It is not a
canonical RAGAS metric, a project grade, a current-product score, or a golden
case pass rate. Context recall is missing because the fixture has no
independently justified reference answers. Current local Chroma excerpts are
not approved for an external end-to-end evaluator run, so current-product
provider-assisted RAGAS remains blocked rather than silently substituted with
the synthetic score.

See [`demo/evaluation/README.md`](demo/evaluation/README.md) and
[`docs/eval.md`](docs/eval.md) for the fixtures, privacy decision, commands, and
weak-row analysis. The [Evaluation Criteria Mapping](docs/EVALUATION_CRITERIA_MAPPING.md)
maps this evidence to the official rubric without treating one metric as proof
of the whole product.

## Ethics and privacy

The default product posture is local, but local does not mean no data leaves
the machine.

| Operation | External data boundary |
|---|---|
| Generation | The configured OpenAI request can contain rendered system/user/context messages and explicitly submitted image pixels | Tests prove payload construction, not live provider receipt, use, or image influence for the current review |
| Retrieval | The query text goes to OpenAI embeddings; ranked knowledge excerpts are read from local Chroma and may enter the generation prompt |
| KB sync | Approved official-source text goes to OpenAI embeddings before local Chroma storage |
| Conversation memory | Successful user and assistant text goes to OpenAI embeddings, then the text and vectors are stored locally |
| RAGAS | Only an explicitly selected committed sanitized/redacted dataset may cross the evaluator boundary, and live calls require opt-in |
| LangSmith | Optional trace metadata is sent only when tracing and credentials are deliberately enabled |

Raw local evaluation rows, workspace snapshots, local Chroma excerpts, `.env`,
and secrets are not public evidence. Image MIME type, decoded size, and file
signature are validated at both browser and backend boundaries; image values
use secret-bearing contracts so ordinary serialization and logs do not expose
the data URL. Session persistence strips queued image references after the
request boundary.

The application isolates untrusted context in prompts, applies bounded request
safety checks, validates supported preview source shapes, and uses explicit
failure states. These reduce risk; they do not eliminate model hallucination,
bias, unsafe generated code, or privacy mistakes. Review prompts, outputs,
exports, and provider settings before using private or sensitive material.

## Demos

For a reliable live review:

1. Start both services and verify health/readiness.
2. Open Demo Mode and choose one canonical live scenario:
   **Polyrhythmic constellation** (Tone.js), **Recursive aurora garden**
   (p5.js), **Kinetic orbit sculpture** (Three.js), or **Fractal solar bloom**
   (GLSL).
3. Use the scenario's **Load prompt & run** action; do not present its fallback
   as a fresh provider result.
4. Inspect the route, streamed events, artifact source, live preview, fullscreen,
   and a small follow-up refinement.
5. If configured generation is unavailable, show a preflight-approved product
   artifact or the separately labelled deterministic browser fixture, and say
   explicitly that it is renderer/product-path evidence, not a fresh provider
   result. Demo Mode recovery instructions are not themselves artifact fixtures.

The image-guided **Reference-guided palette study** demonstrates the real
multimodal request boundary. Use only a public, non-sensitive reference and
show that the resulting p5.js artifact is self-contained rather than fetching
the original image at runtime.

## Reviewer path

If time is limited:

1. Read the [Reviewer Guide](docs/REVIEWER_GUIDE.md).
2. Run one flagship creative artifact and inspect its route, source, preview,
   and session persistence.
3. Compare Single Agent with Multi Agent, then let Auto publish its resolved
   route.
4. Open Dashboard → Knowledge Base and Dashboard → Evaluation; distinguish
   inventory, request retrieval, current retrieval coverage, and synthetic
   RAGAS evidence.
5. Review the [Capability Matrix](docs/CAPABILITY_MATRIX.md),
   [Evaluation Criteria Mapping](docs/EVALUATION_CRITERIA_MAPPING.md), and
   [Architecture Walkthrough](docs/ARCHITECTURE_WALKTHROUGH.md).

Minimum deterministic checks from the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_langgraph_workflow_integration.py tests/test_multimodal_provider_inputs.py tests/test_nextjs_streaming_bridge.py
.venv/bin/python scripts/v7_quality_gates.py docs-mermaid
npm run typecheck --prefix clients/nextjs
npm run test --prefix clients/nextjs -- src/lib/demo-mode.test.ts src/lib/workflow-graph.test.ts
```

The browser smoke requires a running local stack and an installed Playwright
browser:

```bash
npm run test:e2e:smoke --prefix clients/nextjs
```

## Documentation index

### Start, operate, and review

| Document | Reviewer use |
|---|---|
| [Reviewer Guide](docs/REVIEWER_GUIDE.md) | Fast, evidence-prioritized product inspection |
| [Installation Guide](docs/INSTALLATION_GUIDE.md) | Clean setup, services, and first health checks |
| [Configuration Guide](docs/CONFIGURATION_GUIDE.md) | Environment variables, provider settings, storage, and tracing |
| [User Manual](docs/USER_MANUAL.md) | Creative Session, Dashboard, Inspector, artifacts, previews, and sessions |
| [FAQ](docs/FAQ.md) | Short answers to likely product and evidence questions |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Startup, provider, retrieval, preview, session, and evaluation failures |

### Architecture, data, and capability boundaries

| Document | Reviewer use |
|---|---|
| [System Overview](docs/SYSTEM_OVERVIEW.md) | Components, data stores, APIs, and external boundaries |
| [Architecture Walkthrough](docs/ARCHITECTURE_WALKTHROUGH.md) | One request from browser input to final event |
| [Capability Matrix](docs/CAPABILITY_MATRIX.md) | Implemented, bounded, optional, and unsupported claims |
| [Runtime Workflow Graph](architecture/workflow_graph.md) | Exact Single/Multi/Auto topology and transitions |
| [Domain Experience](docs/DOMAIN_EXPERIENCE.md) | Canonical live-preview, code/export, and handoff boundaries |
| [Data & Knowledge Base](docs/DATA_AND_KB.md) | Source governance, Chroma collections, retrieval lineage, and memory |
| [KB Sync](docs/sync.md) | Approved-source ingestion operations |
| [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md) | Gunicorn/container foundation and production responsibilities |

### Evaluation, ethics, and public evidence

| Document | Reviewer use |
|---|---|
| [Evaluation Criteria Mapping](docs/EVALUATION_CRITERIA_MAPPING.md) | Official Capstone criteria mapped to inspectable evidence |
| [Evaluation Metrics Summary](docs/EVALUATION_METRICS_SUMMARY.md) | Retrieval and RAGAS values with comparison limits |
| [Evaluation Runner](docs/eval.md) | RAGAS and retrieval-report commands and privacy rules |
| [Ethics & Privacy Assessment](docs/ETHICS_PRIVACY_ASSESSMENT.md) | Data flows, external services, risks, mitigations, and reviewer controls |
| [Capstone Evaluation & Ethics](docs/CAPSTONE_EVALUATION_ETHICS.md) | Compact evaluation/ethics evidence card |
| [Challenges & Lessons](docs/CHALLENGES_AND_LESSONS.md) | Hard engineering problems, decisions, and learning |
| [Future Work](docs/FUTURE_WORK.md) | Prioritized next steps without presenting them as shipped |
| [Public Documentation Boundary Audit](docs/PUBLIC_DOCUMENTATION_BOUNDARY_AUDIT.md) | Claim, privacy, and public-safe evidence checks |
| [Repository Hygiene Audit](docs/REPOSITORY_HYGIENE_AUDIT.md) | Tracked-file, secret, generated-output, and release-boundary review |
| [Commit History Audit](docs/COMMIT_HISTORY_AUDIT.md) | Public commit-lineage and history hygiene evidence |
| [Portfolio Case Study](docs/PORTFOLIO_CASE_STUDY.md) | Problem, decisions, evidence, limitations, and outcomes in portfolio form |

### Presentation and showcase

| Document | Reviewer use |
|---|---|
| [SCR Presentation](docs/SCR_PRESENTATION.md) | Situation–Complication–Resolution reviewer narrative |
| [SMART Presentation](docs/SMART_PRESENTATION.md) | Specific and measurable presentation framing |
| [Demo Narrative](docs/DEMO_NARRATIVE.md) | Spoken product story and demo cues |
| [Capstone Demo Showcase](docs/CAPSTONE_DEMO_SHOWCASE.md) | Flagship scenarios, proof points, and fallback boundaries |
| [Ten-Minute Presentation](docs/TEN_MINUTE_PRESENTATION.md) | Time-boxed presentation script |
| [Five-Minute Q&A](docs/FIVE_MINUTE_QA.md) | Likely reviewer questions and evidence-backed answers |
| [Manual Demo Checklist](demo/manual_demo_checklist.md) | Preflight and per-scenario human validation |
| [Showcase Upload Preparation](demo/showcase_upload_preparation.md) | Public upload, capture, and evidence checklist |

## License

No repository license is currently declared. Do not assume redistribution or
production-use rights until the owner adds one.
