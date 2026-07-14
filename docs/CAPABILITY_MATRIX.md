# Capability Matrix

This matrix is the public claim boundary for the current repository. It uses
four statuses:

- **Implemented** — executable in the current product and backed by source or
  current validation.
- **Bounded** — executable only within the stated contract; broader variants
  are not claimed.
- **Optional** — implemented but inactive without credentials, data, or an
  explicit operator action.
- **Not supported** — metadata, a route label, a handoff, or future-compatible
  contract exists, but the behavior is not an active runtime capability.

Use live behavior first, then current automated validation, machine-readable
evidence, and documentation. A historical file or UI label cannot promote a
capability to Implemented.

## User-facing product

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Next.js creative workstation | **Implemented** | Root page loads the workstation; composer, workspace, Dashboard, Inspector, and sessions have component/integration coverage | Canonical UI is the pinned Next.js 15 backport; packaged Streamlit foundations are not the primary review path |
| Streaming assistant response | **Implemented** | `/api/assistant/stream` emits versioned NDJSON status, node, token, artifact, preview, error, and final events | Local WSGI endpoint; no WebSocket or public service claim |
| Task modes | **Implemented** | Generate, Explain, Debug, Design, Review, and Preview map to explicit routes | A mode changes prompt/routing contracts; it does not install or call arbitrary tools |
| Domain selection | **Implemented** | Explicit and query-detected domains are published in route evidence | Selecting a domain does not prove its knowledge is indexed or its runtime is previewable |
| Creativity profiles | **Bounded** | Controlled, Balanced, and Exploratory map to requested temperature values | OpenAI adapter applies temperature only to known compatible GPT-4o/GPT-4.1 Responses models; the default GPT-5-family route does not claim it was applied |
| Curated Demo Mode | **Implemented** | Ten scenario contracts load prompts into the normal composer; four canonical live showcase recommendations | A loaded scenario or fallback is not a new provider result |
| Session Fullscreen | **Implemented** | Workstation focus mode collapses side surfaces and restores the session layout | Presentation control only; it does not enter browser-native fullscreen for every renderer automatically |
| Dashboard and Inspector | **Implemented** | Both project published run/session/artifact models; Dashboard exposes deeper categories | They must display unavailable or missing data rather than infer hidden runtime state |
| Project export/handoff | **Bounded** | Artifact bundles can include source, manifests, notes, and validation guidance | Export does not execute, deploy, or validate an external creative tool |

## Generation and multimodal input

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| OpenAI text generation | **Optional** | OpenAI Responses adapter supports streaming and non-streaming events with model/usage metadata | Only implemented generation provider; credentials, network, quota, and model availability are external dependencies |
| Image + text request construction | **Bounded** | User message payload contains `input_text` followed by validated `input_image` parts; provider-input tests inspect the payload | PNG/JPEG/WebP/GIF only, up to 4 files and 1 MiB each; live OpenAI receipt, use, and image influence are not established by current evidence |
| Image persistence | **Not supported** | Session persistence applies a request-scoped multimodal boundary | Queued image bytes are not restored with a session; review any explicit export made before submission |
| Image captioning/search library | **Not supported** | No local vision model, caption index, or cross-modal vector store is composed | The configured generation model may use an image as request context; that is not a general image-search system |
| Audio upload or microphone analysis | **Not supported** | Upload contract accepts images only | Tone.js browser playback is synthetic and opt-in; no microphone permission is requested |
| Provider failover | **Not supported** | Provider factory accepts only `openai` | Errors remain explicit; no hidden model/provider substitution |
| Function/tool calling | **Not supported** | Some routes publish a `tool_use` capability marker, but the runtime graph composes no callable tool executor | Do not describe route metadata as external API automation |

## Workflow and agents

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Single Agent | **Implemented** | Publishes `generator`; skips researcher/retrieval, planning, Director, reasoning, critique, review, and refinement | One direct generation pass when provider input is available |
| Multi Agent | **Bounded** | Publishes planner, researcher, generator, critic, reviewer; executes extra planning/retrieval/review nodes | Sequential responsibilities, not parallel independent LLM workers |
| Auto workflow | **Implemented** | UI waits for `resolved_mode` from the route payload and projects the selected graph | Selects Single only for Explain/Debug with no attachments and no resolved domains; every other Auto request resolves Multi |
| Bounded research | **Implemented** | Multi route embeds the query and searches official-doc Chroma, with explicit empty/error behavior | Single route skips retrieval; researcher is not a web-browsing agent |
| Creative planning, Director, and reasoning | **Implemented** | Typed deterministic planning events and final payloads | Not separate provider calls and not exposed chain-of-thought |
| Artifact critic and workflow reviewer | **Implemented** | Deterministic critique/review events gate Multi artifact routes using backend-prepared metadata | Rules-based application review; browser frame telemetry arrives later and is not an input |
| Refinement loop | **Bounded** | Review may return to generation for up to two refinement attempts; pass history and transition evidence are emitted | At most three generation calls including the initial call; the published `max_refinement_loops=1` remains contract/runtime drift |
| Parallel agents, debate, voting, or dynamic allocation | **Not supported** | Repository metadata registries describe contracts and simulations only | No active parallel scheduler or provider-backed agent swarm is in the runtime graph |
| Human approval gate inside the request graph | **Not supported** | Clarification can stop before generation; UI can ask for ordinary operator actions | No generalized approval engine controls provider or workflow execution |

### Executed topology

| Stage | Single | Multi | Auto |
|---|:---:|:---:|:---:|
| Intake and routing | ✓ | ✓ | Runs before resolution |
| Local memory read | When route/request IDs allow | When route/request IDs allow | Follows resolved route |
| Official-doc retrieval | — | ✓ when configured | Follows resolved route |
| Typed creative planning | — | ✓ | Follows resolved route |
| OpenAI generation | ✓ when configured | ✓ when configured | Follows resolved route |
| Artifact extraction + preview preparation | ✓ except Explain | ✓ except Explain | Follows resolved route; successful Explain goes directly to finalization |
| Deterministic critique + review | — | ✓ except Explain | Follows resolved route |
| Refinement attempts | — | At most two | Only when review requests refinement; up to three generation calls total |
| Finalization / terminal failure | ✓ | ✓ | Follows resolved route |

See the [Architecture Diagram Guide](../architecture/README.md) and
[runtime workflow graph](../architecture/workflow_graph.md) for exact branches.

## Knowledge and retrieval

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Approved official-source registry | **Implemented** | Source IDs include publisher, URL, domain, type, priority, and tags | Registered is not fetched, indexed, retrieved, or cited |
| Explicit source sync | **Optional** | CLI and confirmed KB API actions fetch, normalize, chunk, embed, and upsert selected sources | Network/provider cost; source failures remain visible; no background auto-update |
| Persistent Chroma index | **Implemented** | `kb_official_docs` is separate from memory collections; Dashboard reads inventory without returning excerpts | A fresh workspace may be empty; local timestamp does not prove upstream freshness |
| Semantic query embedding | **Optional** | OpenAI `text-embedding-3-small` is the configurable default | Query text crosses the provider boundary; missing credentials skip retrieval rather than fabricate it |
| Multi-domain retrieval | **Implemented** | Explicit query-domain detection, per-domain candidate pools, bounded fallback, source diversity, and lineage | Final top-k is 5 by default; relevance remains imperfect |
| Request-level provenance | **Implemented** | Selected chunks include source ID, URL, document title, rank, score, and selection reason | Provenance says what context was selected, not that every answer claim is supported |
| Current canonical report | **Implemented** | 7/7 result cases, 16/23 substantive source anchors, 18/19 requested domains at top 5; KB/report fingerprints committed | Retrieval-only evidence; not answer quality, RAGAS, latency, or project score |
| Current upstream freshness verification | **Bounded** | Explicit check actions compare available source state; inventory exposes last local index time | Local “last indexed” is not proof the remote page is unchanged now |
| Autonomous KB enrichment or rollback orchestration | **Not supported** | Typed metadata foundations may exist | Updates require selected sources and confirmation; no autonomous mutation claim |

Current machine evidence:

- [canonical retrieval report](../demo/evaluation/canonical_retrieval_report.json)
- [evaluation fixture explanation](../demo/evaluation/README.md)
- [KB sync operations](sync.md)

## Memory and persistence

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Workspace sessions | **Implemented** | SQLite service supports list, restore, create/update, and delete; browser fallback exists | Local user identity and local-store posture, not multi-user authorization |
| Recent conversation memory | **Optional** | Successful user/assistant turns are embedded and stored in Chroma when configured | Prompt and answer cross OpenAI embeddings; errored runs are not recorded |
| Conversation summary read | **Bounded** | Runtime can read latest stored summary | The request path does not autonomously create every summary |
| Project memory read | **Bounded** | Runtime can load stored goals/preferences/decisions by project ID | No claim that the model autonomously learns or approves durable preferences |
| Explicit output feedback | **Implemented** | Local feedback signals can inform bounded personalization context | Not a trained recommender, fine-tuning system, or completed human study |
| Queued attachment restoration | **Not supported** | Persistence strips request-scoped multimodal data | Attach again for a new image-guided request |
| Managed backup, encryption-at-rest, retention policy | **Not supported** | Local paths are configurable | Operator/platform responsibility; protect and delete local data deliberately |

## Artifacts, previews, and domains

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Artifact extraction and registry | **Implemented** | Generated source becomes typed artifacts with IDs, versions, domain/runtime metadata, and provenance | Prose without an extractable deliverable may produce no artifact |
| Targeted artifact refinement | **Implemented** | Selected source and instruction return through the normal assistant request | Provider-dependent; generated changes still need review |
| Preview contract preparation | **Implemented** | Backend publishes preview target/status/provenance contracts | Preparation is not execution |
| p5.js live preview | **Bounded** | Global-mode `setup()`/`draw()` source passes controlled validator/renderer | No HTML documents, imports, or unrestricted browser access |
| Three.js live preview | **Bounded** | Bundled Three.js runtime executes compact self-contained scene source | No React wrapper, CDN import, standalone HTML, or remote module |
| GLSL live preview | **Bounded** | WebGL runtime compiles the supported fragment subset | Not a general GPU, texture, compute, or Shadertoy runtime |
| Tone.js live preview | **Bounded** | Parsed synth/oscillator/noise programs run only after explicit Start audio | No autoplay, microphone, uploaded audio, or scientific audio-analysis claim |
| React Three Fiber and Hydra | **Bounded** | Source artifacts and handoffs remain inspectable | Code/export only; canonical domain contracts do not claim internal live execution |
| Other browser creative domains | **Bounded** | Source and implementation notes can be generated | Code/export only; a client-side adapter or filename signal alone does not promote a domain to live preview |
| External DCC/game/live-visual tools | **Bounded** | Briefs, parameter schemas, manifests, source, and checklists can be exported | Handoff only; no install, remote control, project execution, rendering, or deployment inside the product |
| Visual/aesthetic quality score | **Not supported** | Deterministic artifact checks and optional human observation are separate | No objective-truth claim, completed blinded human study, or universal creativity metric |

The canonical per-domain delivery decision is
[Domain Experience](DOMAIN_EXPERIENCE.md), backed by
[`domains/experience.py`](../src/creative_coding_assistant/domains/experience.py).

## Evaluation and observability

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Dashboard current-product RAGAS action | **Optional** | Run Evaluation executes selected canonical public RAG cases, publishes progress, and refreshes comparable evidence; provider calls require explicit authorization | Provider cost; no raw local dataset selection through the public request contract |
| Current-product RAGAS evidence | **Implemented** | 7/7 eligible/scored cases, 0 skips/failures, five component means, 68.03% macro, benchmark/run/model/fingerprint provenance committed | Seven-case RAG evidence, not a project grade, artistic-quality score, or broad statistical sample |
| Approved synthetic RAGAS evidence | **Implemented** | 4/4 rows, 0 skips, 0 metric failures; four component means committed | Fixture evidence, not current-product score or project grade |
| Redacted latest-live fixture | **Implemented** | Latest-live structure with public-safe replacement text and committed results | Evidence fixture; p5.js-only redacted subset, not raw-session scoring |
| Context recall | **Implemented for the current benchmark** | Seven public cases include independently authored reference answers/context; current mean is 80.95% | Historical four-row fixture still has no defensible recall denominator |
| Evaluation contract catalog | **Implemented** | 35 stable deduplicated product-authored prompt contracts and deterministic fingerprint | Contract coverage only; Full does not generate or evaluator-score all 35 prompts |
| Full evaluation scope | **Implemented** | Seven canonical RAG cases plus current Creative, Workflow, and Reliability workspace snapshots | Snapshot lanes are not additional generated/RAGAS-scored cases and never enter Retrieval Quality |
| Workflow observability | **Implemented** | Node start/complete/skip, transition, retry, provider, error, and final events | Published events are system telemetry, not private model reasoning |
| Token usage | **Bounded** | Provider-returned usage is published when present | Missing provider metadata remains missing |
| Cost estimate | **Bounded** | Static reference pricing exists for recognized model prefixes | Not live billing, not guaranteed current pricing, and not a budget enforcer |
| LangSmith tracing | **Optional** | Safe run metadata adapter with sampling and explicit configuration | Disabled without tracing request and API key; external service boundary |
| Human usability/aesthetic evaluation | **Not supported as completed evidence** | Reviewer can perform and record one | Automated smoke tests are not human evaluation |

## Deployment and operations

| Capability | Status | Observable evidence | Boundary |
|---|---|---|---|
| Local development stack | **Implemented** | Loopback WSGI and Next.js commands, health/readiness probes | Two processes; provider/KB dependencies can be guarded |
| Production WSGI packaging | **Implemented** | Gunicorn entrypoint, Dockerfile, Compose service, CORS and readiness guards | Packaging foundation only; no hosted endpoint is evidenced |
| Public deployment | **Not supported** | None | Deployment foundations do not prove an operating service |
| Authentication and authorization | **Not supported** | None in application API | Required at the platform edge before networked use |
| Rate limiting/WAF/TLS | **Not supported** | None in WSGI application | Platform responsibility |
| Multi-tenant isolation | **Not supported** | None | Local single-workstation design |
| Broad load/soak/FPS benchmark | **Not supported** | Focused deterministic and browser smoke coverage exists | No current broad benchmark; do not generalize local frame timing to display or production performance |

## Known gaps worth discussing

1. Auto's tested Single condition is deliberately narrow: Explain or Debug,
   with no attachments and no resolved domains. Every other Auto request
   resolves Multi; broader route diversity remains future work.
2. Context precision (`0.5196`) and answer relevancy (`0.5663`) are the weakest
   means in the retained current-product run. They remain visible rather than
   being hidden behind the 68.03% macro. The historical 61.44% fixture remains
   comparison evidence only.
3. Canonical retrieval covers 18/19 requested domains and 16/23 source anchors;
   the remaining gaps and filtered false positives are documented.
4. Canonical live preview is intentionally limited to four runtime contracts.
   Extending it requires source validation, sandbox behavior, recovery, and
   browser evidence—not just a renderer label.
5. The deployment foundation needs auth, rate limits, secret management,
   protected persistence, backup, and operational testing before public use.

For how these claims map to the official rubric, continue with the
[Evaluation Criteria Mapping](EVALUATION_CRITERIA_MAPPING.md) or start from the
[Architecture Diagram Guide](../architecture/README.md).
