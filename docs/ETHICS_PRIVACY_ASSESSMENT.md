# Ethics and Privacy Assessment

Creative Coding Assistant (CCA) combines user prompts, optional reference
images, local retrieval, generated code, workflow routing, persistence, and
optional external providers. This assessment documents current controls and
residual risks; it is not a legal certification, security audit, accessibility
conformance report, or completed human-subject study.

## Data-flow summary

| Stage | Data involved | Default/local behavior | Possible external boundary |
|---|---|---|---|
| Prompt composition | User prompt and UI choices | Browser state until submission | Submitted content can be sent to the configured generation provider |
| Reference images | Up to four queued images and metadata | Browser-local until explicit submit; cleared afterward; not restored by session persistence | Accepted pixels can enter the configured-provider request payload; current evidence does not prove live receipt/use/influence; a pre-submit export may include them |
| Retrieval | Query, local Chroma chunks, source lineage | Retrieval and ranked excerpts are local | Query embeddings and selected contexts may reach configured providers depending on the operation |
| Generation | Prompt, workflow context, selected retrieval, optional image | Request is orchestrated by the local API | Configured provider can receive the request payload |
| Conversation memory | Successful prompt/answer, conversation identity, embeddings | Vectors are stored in local Chroma when configured | Prompt and answer are sent to the embedding provider before local storage |
| Workspace/session | Session, artifact, workflow, and UI state | Configured local SQLite plus compact browser localStorage fallback | No implicit cloud sync; browser profile or workstation backup/software may create a separate boundary |
| Knowledge refresh | Registered URLs, downloaded text, embeddings | Local registry/index mutation with explicit confirmation | Source download and provider embeddings use network services |
| Evaluation | Approved fixture or private local session records | Dry run and local manifests can remain local | `--allow-provider-calls` sends the selected approved evaluation payload to evaluator services |
| Optional tracing | Prompts, events, metadata depending on integration | Off by default in application settings | Enabling LangSmith creates an additional external telemetry boundary |
| Export | Code, metadata, attribution, and possible queued image | User-created local bundle | User decides where it is shared or executed |

## Current controls

### Consent and minimization

- An image enters the backend generation request only when the user explicitly
  submits it; live provider receipt, use, and influence require separate evidence.
- The backend validates image size, declared media type, and signature.
- The composer clears the image after submission, and normal session restore
  does not rehydrate it.
- Provider-backed evaluation requires the explicit `--allow-provider-calls`
  flag.
- Knowledge update/rebuild requires confirmation and selected source IDs.
- Optional external tracing is off by default in application settings and
  should remain off unless separately authorized.

### Evidence honesty

- Provider failure, missing evidence, and blocked measurement remain distinct
  from success and zero.
- The 61.44% RAGAS macro is labeled as an approved synthetic/public fixture,
  not current-product quality.
- Current local retrieval coverage is reported separately from RAGAS.
- No completed human evaluation, external DCC execution, public deployment, or
  autonomous agent swarm is claimed.

### Secret and private-data handling

- API keys are loaded as secret settings and must not be printed, committed, or
  included in support bundles.
- Raw `data/eval/live_sessions.jsonl` is private by default and must not be sent
  to an evaluator without a new, documented privacy approval.
- Local Chroma excerpts are not approved for the blocked current-product RAGAS
  rerun.
- Public evaluation evidence uses an explicitly approved synthetic fixture or
  reviewed redacted fixture rather than raw session text.

### Provenance and user control

- The product distinguishes a registered source from an indexed, retrieved, or
  cited source.
- Retrieval reports retain source/chunk lineage and collection fingerprints
  without publishing local excerpt text.
- Exports are user-mediated and must be inspected before transfer.
- Live preview claims are limited to p5.js, Three.js, GLSL, and Tone.js;
  external-tool packages remain handoffs.

## Risk and control register

| Risk | Current mitigation | Residual risk / required human action |
|---|---|---|
| Hallucinated or weakly grounded code/advice | Retrieval lineage, visible evidence states, artifact inspection, tests | A model can still misread sources or produce unsafe/incorrect code; review and test output |
| Cultural, spiritual, or geometric overclaim | Product language frames geometry, symbolism, and narrative as creative inspiration | Users and models can reproduce stereotypes or assign false authority; use culturally informed human review |
| Bias in sources and evaluator models | Named source registry, bounded fixture, explicit metric scope | Registry and evaluator coverage are incomplete and can privilege dominant ecosystems/languages |
| Creative ownership and imitation | User-directed ideation, attribution/provenance surfaces, export review | Generated work can resemble existing styles or code; check licenses, attribution, and originality before publication |
| Malicious or unsafe generated code | Bounded in-app runtimes, code inspection, tests | Generated code can still consume resources, make network requests, or behave differently externally; sandbox and review it |
| Image privacy or rights | Explicit request submission, validation, nonrestoration | Potential provider processing and a pre-submit export remain boundaries; submit only images the user may lawfully process |
| Prompt/session disclosure | Local persistence and private evaluation classification | Local files, logs, backups, screenshots, or enabled tracing can disclose data; control access and redact diagnostics |
| Knowledge-source copyright/licensing | Registered official sources and source-level provenance | Indexing does not grant redistribution rights; do not publish the Chroma store or long excerpts by default |
| Provider cost and availability | Explicit provider operations, timeouts, visible failure state | Retries and evaluator metrics can incur unpredictable cost; authorize and bound each run |
| External-tool overclaim | Clear code/export-only and handoff labels | Compatibility must be verified in the target tool by a human |
| Agent autonomy misconception | Published route/graph and Inspector | A workflow label can be mistaken for independent agency or parallelism; describe observed orchestration only |
| Accessibility/usability gaps | Accessibility surface and automated checks where present | No complete human accessibility or usability study is claimed; manual assistive-technology review remains necessary |

## Reference-image lifecycle

The current image path proves a partial multimodal request contract when an
image is submitted and accepted pixels are included beside the prompt in the
configured-provider payload. It does not prove live provider receipt, use, or
image influence. Merely selecting, previewing, persisting a session, or
exporting a prompt proves none of those things.

Normal lifecycle:

1. the user selects an image locally;
2. the browser displays a queued preview;
3. the user explicitly submits;
4. the backend validates and includes accepted bytes in that request;
5. if a live provider call succeeds, the provider may process the payload under
   its own terms; that event needs run-specific evidence;
6. the UI clears the attachment and session restoration omits it.

A project bundle exported between steps 1 and 3 can include the queued image.
That exception must be disclosed because “not persisted in the session” does
not mean “cannot appear in an export.”

## Evaluation privacy decision

The approved four-row fixture is committed synthetic/public material and is the
only basis of the current 61.44% RAGAS macro. The local retrieval report
contains non-text lineage, ratios, fingerprints, and counts; local excerpt text
does not need to be published. The exact current-product provider-assisted
RAGAS path remains `BLOCKED_BY_EXECUTION_ENVIRONMENT` because local Chroma
excerpts are not approved for the external transfer required by that run.

Do not resolve that block by treating consent to generation as consent to
evaluation, uploading raw live-session rows, or copying private excerpts into a
new “sanitized” file without a content review. Data minimization and purpose
limitation apply independently to generation, embeddings, evaluation, tracing,
and public evidence.

## Retention and deletion

CCA's local paths do not by themselves supply encryption, deletion deadlines,
backup retention, or multi-user access control. Before processing personal or
confidential work, define:

- who can access the workstation and backups;
- which session, artifact, Chroma, log, and evaluation paths are retained;
- when and how those paths are deleted;
- whether an external provider or tracing service retains submitted data;
- how exports and screenshots are reviewed before sharing.

## Human review gate

Before a public demo, showcase upload, or real-world handoff, a human should:

- inspect prompts, images, outputs, citations, and exports for private data;
- run generated work in the intended environment;
- verify source licenses and attribution;
- review cultural framing, harmful content, and imitation risk;
- test keyboard, contrast, motion, audio activation, and assistive-technology
  behavior appropriate to the audience;
- describe unmeasured or blocked evidence plainly.

Related operational detail: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md),
[DATA_AND_KB.md](DATA_AND_KB.md),
[EVALUATION_METRICS_SUMMARY.md](EVALUATION_METRICS_SUMMARY.md), and
[DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).
