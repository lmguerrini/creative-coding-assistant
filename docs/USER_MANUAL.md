# User Manual

Creative Coding Assistant (CCA) helps a user plan, generate, inspect, refine,
preview, and export creative-coding work. This manual describes the current
local application. Start it with [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
and open `http://127.0.0.1:3000`.

## Creative Session

### Create a request

1. Choose **Single Agent**, **Multi Agent**, or **Auto**.
2. Enter a concrete creative brief, including the intended runtime and desired
   interaction where relevant.
3. Optionally attach up to four supported reference images.
4. Submit the request and follow the streamed status.
5. Inspect the artifact, preview, retrieval evidence, and published workflow
   before accepting or exporting it.

**Single Agent** uses the single-agent route. **Multi Agent** uses the
multi-role workflow exposed by the current graph. **Auto** resolves a route for
the request and shows that decision in the interface. These labels describe
product routes; they do not by themselves prove parallel execution, provider
success, or output quality. Use the current workflow graph and Inspector as the
request-specific record.

If a provider is unavailable, CCA should show a bounded unavailable/fallback
state. A fallback must not be described as a provider response.

### Attach a reference image

The attachment is deliberately request-scoped:

1. Select the image in the composer.
2. Review its thumbnail and the prompt that explains how it should guide the
   result.
3. Submit the prompt. Until submission, the image remains browser-local.
4. The backend validates the declared media type, file size, and file signature.
5. On an accepted request, the backend includes the image pixels in the
   configured-provider request payload. Current evidence does not prove live
   provider receipt, use, or image influence.
6. The composer clears the attachment after submission. Restoring the session
   does not restore it.

Removing the thumbnail before submission prevents that request from sending
the image. A project export made while the image is still queued may include
the image in the bundle; inspect exports before sharing them. CCA has no audio
upload or audio-analysis path.

### Inspect and refine an artifact

The artifact area presents generated code and, for supported runtimes, a local
browser preview. Review the code before running or exporting it. Use the
artifact refinement form to request a bounded revision while retaining the
creative-session context.

Current live browser previews are:

| Runtime | Current behavior |
|---|---|
| p5.js | Executed in the in-app browser preview |
| Three.js | Executed in the in-app browser preview |
| GLSL | Executed in the in-app shader preview |
| Tone.js | Executed in the browser after explicit **Start**; no audio upload/analysis |
| Hydra | Code/export-only |
| React Three Fiber | Code/export-only |

TouchDesigner, Unreal Engine, Blender Geometry Nodes, and Houdini output is a
handoff package for an external tool. CCA does not execute, control, deploy to,
or validate those tools. See [DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).

### Use Fullscreen Creative Session

Fullscreen temporarily prioritizes the creative workspace. Enter it from the
session controls and leave it with the visible exit control or supported escape
interaction. Returning should restore the surrounding panels; it does not
create a separate session.

## Demo Mode

Demo Mode offers ten review scenarios. Choose a card, review its requirements,
then select **Load prompt & run**. Some paths need a configured provider; the
reference-guided scenario also requires an image.

| Scenario | What it demonstrates |
|---|---|
| Polyrhythmic constellation | Canonical Tone.js live showcase |
| Recursive aurora garden | Canonical p5.js live showcase |
| Kinetic orbit sculpture | Canonical Three.js live showcase |
| Fractal solar bloom | Canonical GLSL live showcase |
| Source-grounded design brief | Retrieval and cited-source inspection |
| Multi-agent production plan | Published multi-role workflow |
| Single-agent line study | Bounded single-agent route |
| Export handoff package | External-tool handoff boundary |
| Reference-guided palette study | Request-scoped image input |
| Failure-recovery rehearsal | Visible degraded/failure behavior |

A scenario card is a reproducible starting point, not proof of its outcome.
Observe the current run and its evidence.

## Inspector

Inspector panels expose request-specific state such as the selected session,
artifact, workflow, retrieval, evaluation, and telemetry posture. Use them to
answer:

- Which route actually handled this request?
- Which sources were retrieved, rather than merely registered?
- Did the provider path succeed, fail, or remain unavailable?
- Which evidence applies to this artifact or session?
- Is a value measured, historical, missing, or blocked?

The Inspector is evidence-oriented UI, not a guarantee that every optional
provider or metric exists. Telemetry stays bounded to collected local metadata
unless an external tracing integration is deliberately enabled.

## Dashboard

The Dashboard organizes product and review information into these current
destinations:

- Overview, Architecture, Workflow, Workspace, Runtime, and Preview
- Artifacts, Domains, Knowledge Base, AI & agents, and Memory
- Sessions, Telemetry, and Evaluation
- User Guide and Settings as secondary destinations

Use Dashboard > Artifacts to inspect saved artifact metadata and previews. Use
Dashboard > Evaluation to inspect the AI Engineering Lab and evidence status.
Dashboard summaries should link or resolve to current records; they do not
replace the underlying runtime, report, or test.

Settings includes Theme, Typography, Workspace layout/focus, workflow,
provider, and creativity defaults. Appearance and layout choices affect the
review surface; they do not change a generated artifact or its external-tool
support.

## Sessions and local persistence

Workspace/session snapshots are persisted to a configured local SQLite path.
The browser also keeps a compact localStorage fallback and workspace identity
so bounded state can recover if the session API is unavailable. Treat both
stores as user data: local placement does not imply encryption, automatic
deletion, or safe public distribution.

Normal session restoration can recover supported workspace and artifact state,
but intentionally does not restore an image into the composer. If a restored
session appears inconsistent, refresh the UI, confirm backend readiness, and
follow [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before changing storage.

## Knowledge Base

The Knowledge surface distinguishes four states:

- **registered**: the source exists in the source registry;
- **indexed**: compatible chunks exist in the active Chroma collection;
- **retrieved**: chunks were returned for the current query;
- **cited**: the answer or UI attributes a claim to a source.

Do not collapse these into “the app knows this source.” Knowledge Base updates
and rebuilds are explicit, source-selected operations. They require
confirmation and use backup/restore behavior on failure. See
[DATA_AND_KB.md](DATA_AND_KB.md) before refreshing or rebuilding.

## Evaluation

The Evaluation surface reports evidence by scope and status. The primary
68.03% Retrieval Quality is the equal-weight five-metric macro from seven
current-product public RAG cases, all eligible and scored with no failures or
skips. The 61.44% four-row synthetic fixture remains historical. Current local
retrieval coverage uses a different report and must not be blended with either
RAGAS macro. The 35-case catalog represents contract coverage; Full executes
seven RAG cases and records local Creative, Workflow, and Reliability snapshots
rather than generating 35 answers. No human evaluation is claimed. Full interpretation is in
[EVALUATION_METRICS_SUMMARY.md](EVALUATION_METRICS_SUMMARY.md).

## Export and handoff

Export packages are for user-mediated inspection and transfer. Depending on
the active project they can include prompt-derived code, metadata, attribution,
or a queued reference image. Before sharing:

1. inspect all files in the package;
2. remove secrets, personal data, private prompts, and unnecessary image data;
3. verify licenses and source attribution;
4. run the output in the intended external tool yourself;
5. record external validation separately.

CCA does not claim that export means execution, deployment, compatibility, or
validation in an external environment.

## Safe recovery

When a request fails, keep its failure state visible and retry only after
identifying the boundary: backend readiness, provider credentials, provider
timeout, retrieval/index state, preview runtime, image validation, or browser
state. Avoid repeated paid calls and do not publish private diagnostics. Use
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) and [FAQ.md](FAQ.md).
