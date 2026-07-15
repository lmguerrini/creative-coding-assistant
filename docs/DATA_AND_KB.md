# Data and Knowledge Base

Creative Coding Assistant (CCA) keeps several kinds of data with different
provenance, privacy, and freshness rules. This guide explains the local Chroma
knowledge base, source refresh, request retrieval, session records, evaluation
data, and exports.

## Data inventory

| Data class | Typical location/configuration | Purpose | Sharing boundary |
|---|---|---|---|
| Official source registry | committed application metadata | Defines supported source identities and sync policy | Public only when the registered source and metadata are public |
| Chroma collection | configured local Chroma path | Stores embedded chunks for retrieval | May contain copyrighted/source excerpts; do not publish by default |
| Creative/domain knowledge | committed source metadata and indexed chunks | Grounds creative-coding guidance | License, attribution, and freshness remain source-specific |
| Workspace/session state | configured local SQLite path | Restores supported session and artifact state | User data; not automatically public or encrypted |
| Browser workspace fallback | browser localStorage | Keeps compact fallback state and workspace identity | Browser-profile data; clear/share only deliberately |
| Raw evaluation sessions | configured local JSONL path, including `data/eval/live_sessions.jsonl` | Captures eligible evaluation structure | Private by default; not approved for external evaluation or commit |
| Approved evaluation fixtures | `demo/evaluation/` | Reproducible synthetic/public evaluation evidence | Only the explicitly approved fixture may cross the documented evaluator boundary |
| Evaluation results | configured results path and approved committed reports | Records metric output and status | Share only sanitized, scope-labeled evidence |
| Artifacts and exports | workspace snapshots or user-selected browser download | Generated code and handoff metadata | Inspect before sharing; may contain prompt-derived or queued image data |

Provider credentials are configuration secrets, not application data. Never
store them in a source registry, fixture, artifact, export, session record, or
support bundle.

The configured Chroma store can contain separate collections for official
documentation, conversation turns, conversation summaries, project memory,
evaluation-trace foundations, and preview-artifact index foundations. Official
documentation, conversation turns, summaries, and project memory are active
runtime read sources; the other collections are not automatically equivalent
to active evaluation or preview behavior.

A successful non-error conversation turn is recorded only when a conversation
ID and OpenAI embedding configuration are available. That recording step sends
the prompt and answer for external embedding, then stores their vectors locally.
Failure to record the memory does not fail the already generated response. This
is a distinct provider/privacy boundary from the generation request itself.

## Four different knowledge states

CCA deliberately distinguishes:

1. **Registered** — the source has a registry entry.
2. **Indexed** — chunks from a source exist in the active compatible collection.
3. **Retrieved** — chunks were returned for this query.
4. **Cited** — an answer or product surface attributes material to the source.

Registration does not prove a successful download or embedding. Indexing does
not prove relevance to a request. Retrieval does not prove that the generated
answer used a chunk correctly. A citation does not prove factual accuracy.

## Refresh the knowledge base

Review source terms, network access, embedding configuration, provider cost,
and the target storage path before a refresh.

Refresh all registered sources:

```bash
.venv/bin/python scripts/sync_official_kb.py --all
```

Refresh one source:

```bash
.venv/bin/python scripts/sync_official_kb.py --source-id SOURCE_ID
```

Continue a batch and request a machine-readable summary:

```bash
.venv/bin/python scripts/sync_official_kb.py --all --continue-on-error --summary-format json
```

The sync command uses these exit-code classes:

- `0`: requested work succeeded;
- `1`: the batch completed with one or more source failures;
- `2`: configuration or command usage prevented the run.

Do not convert exit code 1 into a blanket success. Preserve source-level
failures in review evidence. Network restrictions, source changes, HTTP
responses, parsing, provider credentials, embedding compatibility, and quotas
can all affect a refresh.

The official-source transport accepts registered absolute HTTPS targets only.
It requires every resolved address to be globally routable, caps each response
at 5 MiB, permits only same-host-and-port HTTPS redirects, and rejects a
cross-origin redirect before following it. These controls reduce local-network
and oversized-response risk; they do not turn arbitrary URLs into approved
sources.

A successful source update replaces that source's prior Chroma snapshot:
current records are upserted, then superseded record IDs are deleted. If the
new normalized source legitimately yields zero chunks, the prior snapshot is
cleared instead of leaving stale retrieval evidence behind. A failed higher-
level rebuild continues to use the documented backup/restore path.

## In-product update and rebuild

`GET /api/knowledge-base` exposes the current knowledge-base review state. The
corresponding mutating operation accepts an explicit selected-source set and
confirmation before update/rebuild. Its failure path uses backup/restore
behavior so a failed rebuild does not silently replace the last usable index.

Before confirming a mutation:

- record the current collection identity/fingerprint and chunk count;
- confirm the selected source IDs;
- confirm the embedding model and provider boundary;
- ensure sufficient local disk space;
- avoid concurrent update/rebuild operations;
- retain the source-level summary and any restore status.

Do not delete the active Chroma directory as a routine fix. Follow the backup
and explicit rebuild path or investigate the reported configuration failure.

## Verify current retrieval evidence

Generate the canonical local report without a provider evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5
```

The command prints a JSON report. It reads the index and does not overwrite the
committed report, benchmark, or KB data.

The checked-in `demo/evaluation/canonical_retrieval_report.json` is the
authoritative snapshot. It binds the fixed seven-query, top-five protocol to
the active collection fingerprint and records non-text result lineage. Exact
counts remain in that machine-readable artifact rather than duplicated in
prose, where they can drift after a new run.

These are deterministic local coverage indicators, not RAGAS scores and not a
human assessment of answer quality. The benchmark must not be reweighted or
changed after observing results. See [eval.md](eval.md).

## Freshness and provenance

A sync date only shows when a job ran; it does not prove every upstream page was
fresh, complete, or licensed for every downstream use. Preserve:

- canonical source identity and URL metadata;
- retrieval/sync timestamp and per-source outcome;
- embedding model and collection compatibility;
- chunk count and collection fingerprint;
- query/report version and top-k;
- citations shown to the user;
- known blocks, exclusions, or source-specific limitations.

When evidence changes, keep historical numbers labeled with their date and
scope rather than silently replacing their meaning.

## Privacy and external transfer

Knowledge-base material stays local at retrieval time, but source sync may use
the network and embedding generation may send text to the configured provider.
Provider-assisted generation can send selected retrieved contexts with the
prompt. Provider-scored evaluation can send its selected evaluation rows and
contexts. These are separate external transfers and require separate review.

The canonical current-product evaluation uses the reviewed committed public
benchmark. That approval does not extend to arbitrary local Chroma excerpts or
raw session JSONL. Do not broaden the result by uploading the index or private
rows; use only an explicitly reviewed public, synthetic, or redacted dataset
for an external evaluation path.

## Retention, export, and deletion

Local storage is not an automatic retention policy. Apply least-privilege file
access, workstation encryption where appropriate, backups suitable for the
data, and a deliberate deletion schedule. Before archiving or sharing a clone,
inspect ignored runtime paths as well as tracked files; before handing off a
browser profile, inspect its localStorage fallback too.

A project export can include code, metadata, attribution, and a reference image
that is still queued before submission. Inspect the bundle and remove private
or unnecessary material. Export is not proof of execution or validation in an
external tool.

Ethical and privacy risks are assessed in
[ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md).
