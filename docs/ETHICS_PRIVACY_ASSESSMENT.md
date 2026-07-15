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
| Knowledge refresh | Registered URLs, downloaded text, embeddings | In-product update/rebuild uses confirmation; the CLI is an explicit direct command | Source download and provider embeddings use network services |
| Evaluation | Versioned public benchmark or private local session records | Dry run and local manifests can remain local | `--allow-provider-calls` sends the selected approved evaluation payload to evaluator services |
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
- In-product knowledge update/rebuild requires confirmation and selected source
  IDs; the sync CLI is a direct operator command without a confirmation dialog.
- Optional external tracing is off by default in application settings and
  becomes an external boundary only when configured and enabled.

### Evidence honesty

- Provider failure, missing evidence, and blocked measurement remain distinct
  from success and zero.
- Current-product RAGAS is labeled as seven-case retrieval evidence, not a
  universal product or artistic-quality score; older fixtures remain historical.
- Current local retrieval coverage is reported separately from RAGAS.
- No completed human evaluation, external DCC execution, public deployment, or
  autonomous agent swarm is claimed.

### Secret and private-data handling

- API keys are loaded as secret settings; the public repository and diagnostic
  formats exclude their values.
- Raw `data/eval/live_sessions.jsonl` is classified as private local data and is
  excluded from approved public evaluation paths.
- Arbitrary local Chroma excerpts and raw session rows are not covered by the
  public current-product benchmark approval.
- Public evaluation evidence uses the reviewed current-product benchmark or an
  explicitly approved synthetic/redacted fixture rather than raw session text.

### Provenance and user control

- The product distinguishes a registered source from an indexed, retrieved, or
  cited source.
- Retrieval reports retain source/chunk lineage and collection fingerprints
  without publishing local excerpt text.
- Exports are user-mediated and must be inspected before transfer.
- Live preview claims are limited to p5.js, Three.js, GLSL, and Tone.js;
  external-tool packages remain handoffs.

## Risk and control register

| Risk | Current control | Residual boundary |
|---|---|---|
| Hallucinated or weakly grounded output | Retrieval lineage, visible evidence states, artifact inspection, tests | Incorrect or unsafe output remains possible; external use depends on human inspection and testing |
| Cultural or symbolic overclaim | Product language frames geometry, symbolism, and narrative as creative material | Stereotypes and false authority remain possible; culturally informed assessment is outside automation |
| Source and evaluator bias | Named source registry, bounded fixtures, explicit metric scope | Coverage can privilege dominant ecosystems, languages, and evaluator assumptions |
| Creative ownership and imitation | Attribution, provenance, and user-mediated export surfaces | Similarity, licensing, attribution, and originality still require human or legal judgment |
| Unsafe generated code | Bounded preview runtimes, source inspection, tests | Code can consume resources, use networks, or behave differently outside the preview boundary |
| Image privacy and rights | Explicit submission, signature validation, request-scoped handling, no session restoration | Provider processing and pre-submit exports remain possible; CCA does not verify rights or consent |
| Prompt or session disclosure | Local persistence and private evaluation classification | Files, logs, backups, screenshots, exports, and tracing can expose content; access control and redaction are not automatic |
| Knowledge-source licensing | Approved-source registry and source-level provenance | Indexing grants no redistribution rights; the Chroma store is not a public dataset |
| Provider cost and availability | Explicit provider operations, timeouts, visible failure states | Generation, embeddings, retries, and evaluator metrics can fail or incur variable cost |
| External-tool compatibility | Clear code/export and handoff labels | Compatibility remains unverified until the package is inspected in the target tool |
| Agent autonomy misconception | Published route graph and Inspector | Role labels can still be mistaken for independent agency or parallel workers |
| Accessibility and usability gaps | ARIA semantics and limited automated interaction checks | No complete human accessibility, assistive-technology, or usability study is claimed |

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
This is a distinct boundary: “not persisted in the session” does not mean
“cannot appear in an export.”

## Evaluation privacy decision

The canonical seven-case current-product benchmark is reviewed public material
and is the basis of the current RAGAS result shown in the Dashboard. Older
synthetic or redacted fixtures remain historical. The local retrieval report
contains non-text lineage, ratios, fingerprints, and counts; arbitrary local
excerpt text does not need to be published or externally evaluated.

Approval of the public benchmark does not extend to arbitrary local content,
raw live-session rows, or unreviewed copies of private excerpts. Generation,
embeddings, evaluation, tracing, and public evidence remain separate data-use
boundaries.

## Retention and deletion

CCA's local paths do not supply encryption, deletion deadlines, backup
retention, or multi-user access control. The product does not currently enforce
who can access workstation backups, how long sessions and vector data remain,
whether provider-side retention applies, or how exported media is governed.

## Human responsibility boundary

CCA exposes provenance, failure states, and bounded preview evidence, but it
does not certify privacy, licensing, cultural context, accessibility, security,
or target-tool compatibility. Public sharing and execution outside the bounded
preview remain user-controlled decisions.

Related operational detail: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md),
[DATA_AND_KB.md](DATA_AND_KB.md),
[eval.md](eval.md), and
[DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).
