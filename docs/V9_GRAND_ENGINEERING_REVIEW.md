# V9 Grand Engineering Review

Date: 2026-07-13

Branch: `version-review/v9`

Starting checkpoint: `eb525ed5` (`feature/reviewer-experience`)

This record closes the deterministic engineering portion of the V9 Grand
Engineering Review. It is a local release-candidate assessment, not a public
deployment approval, merge, push, tag, release freeze, showcase upload, or
human acceptance decision.

> Historical checkpoint: the subsequent V9 Evaluation capability replaced this
> checkpoint's fixture-primary 61.44% Evaluation posture with a seven-case
> current-product RAGAS result of 68.03%. See
> [Evaluation Metrics Summary](EVALUATION_METRICS_SUMMARY.md). The dated findings
> below are preserved as the state reviewed on 2026-07-13.

## Decision

**Local engineering release candidate: ready for the independent-review and
human gates. Product Bug Gate: clear.**

The current tree passes the complete backend, frontend, browser, documentation,
repository, and private-ledger validation matrix described below. Reviewer-
visible defects reproduced during the review were corrected and regression
tested. The process stops at **Junie Independent Grand Engineering Review**;
that review has not been executed.

The repository is not approved for public/network deployment. It remains a
loopback-oriented capstone workstation with explicit security, dependency,
licensing, provider, evaluation, and human-review limits.

## V9 Lineage Reviewed

The review reconciled the accumulated V9 capability line rather than treating
the final documentation checkpoint as a standalone change:

- V9.1 — runtime and product reliability
- V9.2 — developer observability and agent transparency
- V9.3 — unified product UX
- V9.4 — knowledge and domain experience
- V9.5 — creative experience and ten-flow demo engine
- V9.6 — product UX finalization and demo readiness
- V9.7 — product control, safety, personalization, and evaluation workspace
- V9.8 — Dashboard intelligence, multimodal request boundary, runtime quality,
  canonical showcases, shared controls, and fullscreen session
- V9.9 — README, architecture, evaluation, ethics, reviewer, portfolio, demo,
  Q&A, and presentation material

The V9.8 checkpoint is `e817a8ad`. The V9.9 checkpoint and starting point for
this review is `eb525ed5`. No V9 release tag exists.

## Corrections Made During Grand Review

### API, configuration, and network safety

- Changed the default browser API CORS policy from a wildcard to the exact
  loopback origins `http://127.0.0.1:3000` and `http://localhost:3000`.
  Untrusted origins receive no `Access-Control-Allow-Origin` header. An
  operator may still configure an explicit origin policy.
- Rejected incoming request IDs containing non-printable characters or more
  than 128 characters and replaced them with generated UUIDs, preventing
  response-header injection and unbounded correlation values.
- Made development-server API exports lazy so module execution no longer emits
  an import-order warning.
- Removed private engineering-path references from the serialized public demo
  plan and replaced them with public reviewer artifacts.
- Disabled tracing by default in the example environment, left its key blank,
  and documented that identifiers leave the process only when tracing is
  explicitly enabled.

### Official Knowledge Base synchronization

- Added a 5 MiB response ceiling for each approved source fetch.
- Required absolute HTTPS targets with no embedded credentials.
- Added public-IP DNS checks for the initial target and every redirect.
- Restricted redirects to the same approved HTTPS origin and retained the
  final registered-host and allowed-path checks.
- Replaced each indexed source snapshot as a bounded single-operator sequence:
  new records are upserted and superseded chunk IDs are removed so changed
  source structure cannot leave stale searchable chunks.
- Added regression coverage for private/cross-origin redirects, oversized
  responses, source replacement, rollback behavior, and explicit deletion.

### Frontend, toolchain, and CI

- Upgraded Next.js from 14.2.35 to 15.5.20 and moved the client test stack to
  Vitest 4.1.10 and jsdom 29.1.1.
- Added a supported Node engine contract: Node 22.13+ on the 22.x line or
  Node 24+.
- Resolved the audited `js-yaml` and `postcss` dependency paths through lockfile
  overrides.
- Replaced the removed `next lint` command with direct ESLint coverage for
  `src` and `e2e`.
- Made Vitest's JSX transform and worker bound explicit and repaired writable
  animation-frame mocks for the new test runtime.
- Added exact accessible names to icon/quick actions exercised by reviewers.
- Expanded CI to review and freeze branches and added frontend audit, lint, and
  production-build gates before the full Chromium suite.

### Evidence and documentation authority

- Reconciled README, installation, configuration, deployment, troubleshooting,
  data/KB, synchronization, evaluation, capability, repository-hygiene, and
  commit-history guidance to the current V9 product.
- Marked retired V8 demo/evaluation documents and the static launcher as
  historical so they cannot override the current ten-flow in-app Demo Mode.
- Kept the 61.44% Retrieval Quality value scoped to the committed, transcribed,
  approved public-safe RAGAS fixture. It is not a current-product score, an
  overall score, or a tracked raw evaluation run.
- At this checkpoint, kept Context Recall as `MISSING_EVIDENCE`, current-product
  provider RAGAS as `BLOCKED_BY_EXECUTION_ENVIRONMENT`, and the fixture/current
  retrieval delta as `NOT_COMPARABLE`. The historical note above records the
  subsequently completed Evaluation run.
- Kept multimodal payload construction separate from provider receipt, use, or
  influence. Deterministic browser fixtures remain labeled as fixtures.
- Added a configurable public evaluation-run URL instead of implying a fixed
  repository or provider evidence location.

## Cross-Capability Coverage

The review traced current behavior through all sixteen Dashboard categories:
Overview, Architecture, Workflow, Workspace, Runtime, Preview, Artifacts,
Domains, Knowledge Base, AI & agents, Memory, Sessions, Telemetry, Evaluation,
User Guide, and Settings. Shared workspace state, absence labels, renderer
routes, source inventory, current-run retrieval, persistence, and evaluation
evidence remain separate rather than being inferred across categories.

The complete ten-flow Demo Mode catalog was retained and audited:

1. Polyrhythmic constellation
2. Recursive aurora garden
3. Kinetic orbit sculpture
4. Fractal solar bloom
5. Source-grounded design brief
6. Multi-agent production plan
7. Single-agent line study
8. Export handoff package
9. Reference-guided palette study
10. Failure-recovery rehearsal

The four homepage recommendations—Physarum drift, Kinetic orbit sculpture,
Chladni light field, and Cymatic audio study—remain concise entry points rather
than extra demo-authority records.

Runtime and delivery boundaries were rechecked for p5.js, Three.js, GLSL,
silent-ready Tone/audio-visual work, React Three Fiber, Hydra, external-tool
handoffs, export packages, and unsupported external execution. Only supported
browser-native contracts claim a live preview. Code/export-only and external-
handoff artifacts retain explicit non-execution boundaries.

Auto, Single-Agent, and Multi-Agent selection, workflow graph projection,
stream completion/recovery, artifact extraction and selection, Preview and
Runtime state, refinement, session isolation, local persistence/reload,
fullscreen restoration, and explicit operator handoffs remain covered by the
combined unit, integration, browser, and documentation matrix.

## Validation Evidence

### Backend and repository

- Full Python suite: 2,700 tests and 423 subtests passed on the final backend
  tree in 1,370.79 seconds. Its 481 warnings were limited to the known
  third-party Chroma/Pydantic Python 3.14 deprecations.
- Changed security, API, CORS, Knowledge Base, evaluation, and documentation
  matrix: 65 focused tests passed before the full-suite replay.
- Ruff, Python byte-compilation, `git diff --check`, repository-hygiene, and
  requirement-coverage gates passed.
- Runtime Pack requirement coverage includes all 240 declared requirements;
  the private pack remains ignored and untracked.
- A tracked-file secret scan covered 1,651 files, skipped four binary artifacts,
  and found no high-confidence committed credential. Twelve generic matches
  were reviewed as identifiers, arguments, or test values.
- The repository has no tracked symlinks, tracked private runtime-ledger files,
  local databases, environment secrets, or generated test-output trees.

### Frontend and browser

- Clean Node 24.11.0 / npm 11.6.1 install: 495 packages installed, 496 audited,
  and zero vulnerabilities.
- Vitest: 77 files and 553 tests passed.
- TypeScript, ESLint, and the Next.js 15.5.20 production build passed. The build
  emitted six static pages; the main route reports a 377 kB first-load bundle.
- Playwright: all 28 Chromium tests passed after the clean install.
- In-app browser rehearsal exercised Demo Mode, the reference-image gate,
  Dashboard navigation, Evaluation evidence, live artifact preview, User Guide,
  all nine themes, Settings, Fullscreen Creative Session, User/Developer
  Inspector modes, and Telemetry. The 1280×720 workspace had no horizontal
  overflow and the browser warning/error log was empty.
- The live loopback backend returned successful health, domain-experience,
  session retrieval, session persistence, and CORS-preflight responses during
  that rehearsal. No provider generation or image upload was attempted.

### Documentation and presentation

- Public Markdown/link audit: 65 Markdown files and 301 relative links/anchors
  were checked on the final review tree, with zero issues.
- Mermaid and Dashboard quality gates passed; all 23 Dashboard roadmap gates
  were covered.
- Forty focused CORS/documentation tests passed with one third-party Chroma
  deprecation warning.
- The 10-slide PowerPoint passed ZIP integrity, relationship, macro/OLE,
  external-link, font, overflow, full render, and visual inspection checks.
  All 10 slides rendered; the main spoken script contains 899 lexical words.

### Dependency audit

- A clean 142-pin `.[server]` environment passed `pip check` and `pip-audit`
  with no known vulnerabilities.
- A clean 163-pin `.[server,evaluation]` environment passed `pip check` and
  retained two no-fix advisories: DiskCache 5.6.3 unsafe deserialization when
  an attacker can write its cache, and RAGAS 0.4.3 multimodal-context SSRF.
  The product uses text-only metrics, allow-lists committed local evaluation
  datasets, and requires explicit provider authorization, but optional
  evaluation installation remains a declared residual risk.

## Security, Privacy, and Public Boundary

The application is suitable for its documented local demonstration boundary,
not an exposed service. Exact CORS improves browser-origin isolation but is not
authentication. The repository does not implement authentication,
authorization, CSRF protection, rate limiting, abuse controls, or TLS
termination. A network deployment requires a separate trusted gateway and a
new security review.

Knowledge Base fetches remain limited to fixed registry entries, HTTPS,
public-address DNS checks, same-origin redirects, response-size bounds, and
approved final paths. DNS resolution is repeated by the underlying HTTP stack,
so an inherent DNS time-of-check/time-of-use window remains. A same-host
redirect outside the registered path may be fetched before the final path check
rejects it. Cross-host and private-address redirects are rejected by the custom
handler before redirected HTTP I/O.

Knowledge Base backup/restore protects the supported single-operator update
flow, but no cross-process update lock is implemented. Conversation or project
identifiers may leave the local process if LangSmith is deliberately enabled;
tracing remains off by default and emits bounded summaries rather than hidden
provider reasoning.

The repository has no root license. Third-party components retain their own
licenses, but no redistribution or production-use right is granted for the
project as a whole. Media, source, generated-output, and showcase rights still
require human approval.

## Residual Risks and Deferred Gates

- At this historical checkpoint, no current-product provider RAGAS rerun
  existed. The completed 2026-07-14 run supersedes this residual item; the
  current evidence authority is `EVALUATION_METRICS_SUMMARY.md`.
- No live configured-provider result proves that an attached image was
  received, used, or influenced output; only the request/persistence contract
  is proven.
- The two optional evaluation advisories described above have no published fix
  version in the audited dependency set.
- Authentication, rate limiting, CSRF protection, TLS termination, and a
  network-deployment threat model are not implemented.
- Official Knowledge Base replacement has no cross-process lock, and DNS
  prechecking cannot remove the resolver TOCTOU window.
- Human browser, screen-reader, projector, timed presentation, configured-
  provider, rights/licensing, and final product-acceptance protocols remain
  deferred.
- Docker was unavailable in the review environment, so no Docker build/runtime
  result is claimed.
- Maintainability debt remains in the 9,414-line workstation shell and
  19,066-line stylesheet. The main client route is 377 kB first load, and the
  ESLint 8 toolchain is deprecated even though its gate passes.
- Package metadata remains `0.1.0`; a V9 version change belongs to an approved
  release gate.
- Local Git object maintenance reports unreachable residue, while reachable
  connectivity is intact. No destructive object cleanup was performed.

## Gate Disposition

Deterministic engineering audit, bug remediation, validation, release-candidate
documentation, demo rehearsal, reviewer Q&A preparation, and final Codex
evidence are complete locally. Human acceptance protocols are prepared but are
not represented as executed or passed.

The only permitted next review action is **Junie Independent Grand Engineering
Review**. Merge, push, `v9.0.0` tag creation, release freeze, public showcase
upload, and Runtime Pack publication remain prohibited until their explicit
human gates are satisfied.
