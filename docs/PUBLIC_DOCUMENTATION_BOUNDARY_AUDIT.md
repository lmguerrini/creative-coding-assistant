# Public Documentation Boundary Audit

## Audit statement

This audit defines which repository material can support current public claims
and which material is historical, local-only, conditional, or future work. The
baseline inspected was commit `e817a8ad` on 2026-07-13; current-branch reviewer
documents created after that baseline supersede earlier presentation wording.

The central rule is simple: use the strongest evidence that describes the same
code, data, input condition, and execution path as the claim.

## Current reviewer path

These files form the intended public narrative after the current documentation
update:

| Artifact | Public role |
|---|---|
| `README.md` | Concise product purpose, problem, operation, setup, and reviewer start |
| `docs/CAPSTONE_DEMO_SHOWCASE.md` | Current ten-flow product and evidence guide |
| `docs/PORTFOLIO_CASE_STUDY.md` | Problem-to-outcome portfolio narrative |
| `docs/DEMO_NARRATIVE.md` | Six-part presentation story |
| `docs/TEN_MINUTE_PRESENTATION.md` | Exact timed talk and presenter actions |
| `docs/FIVE_MINUTE_QA.md` | Timed reviewer-answer bank |
| `docs/CHALLENGES_AND_LESSONS.md` | Technical decisions and residual gaps |
| `docs/FUTURE_WORK.md` | Explicitly non-current roadmap |
| `docs/COMMIT_HISTORY_AUDIT.md` | Bounded history-quality audit |
| `docs/REPOSITORY_HYGIENE_AUDIT.md` | Bounded tracked-tree hygiene audit |
| `demo/manual_demo_checklist.md` | Current local rehearsal protocol |
| `demo/showcase_upload_preparation.md` | Manual public-upload gate |

The presentation deck is `outputs/creative-coding-assistant-capstone.pptx`.
Its claims must remain consistent with the files above and the canonical
machine-readable evidence below.

## Canonical current product sources

| Question | Canonical source |
|---|---|
| What Demo Mode scenarios exist? | `clients/nextjs/src/lib/demo-mode.ts` |
| Which ten scenario IDs and four showcase IDs are the fallback contract? | `demo/v9_5_golden_demo_dataset.json` |
| What browser artifacts are exercised? | `clients/nextjs/e2e/support/demo-fixtures.js` |
| What does the exact showcase smoke prove? | `clients/nextjs/e2e/demo-showcase-smoke.spec.js` |
| What does the direct Three.js runtime smoke prove? | `clients/nextjs/e2e/preview-sandbox-three.spec.js` |
| What are the current retrieval selections and fingerprints? | `demo/evaluation/canonical_retrieval_report.json` |
| What does the approved RAGAS fixture measure? | `clients/nextjs/src/lib/current-ragas-evidence.ts` and `demo/evaluation/README.md` |
| How are Single, Multi, and Auto resolved? | `src/creative_coding_assistant/orchestration/runtime/execution.py` |
| How do image pixels reach the provider adapter? | `tests/test_multimodal_provider_inputs.py` |
| What does CI run? | `.github/workflows/ci.yml` and `.github/workflows/backend-release-verification.yml` |

Narrative documents should link to or restate these sources without changing
their evidence class.

## Current public machine-readable evidence

The following material is reviewer-safe within its stated boundary:

- `demo/evaluation/canonical_retrieval_report.json` contains ranks, source IDs,
  domains, distances, scores, counts, and fingerprints, but no retrieved
  excerpt text.
- `demo/evaluation/sanitized_ragas_live_sessions.jsonl` contains synthetic,
  public-safe evaluation inputs.
- `demo/evaluation/redacted_live_session_ragas_latest4*.jsonl*` preserves a
  bounded historical structure while replacing private text.
- `demo/v9_5_golden_demo_dataset.json` is rehearsal metadata. It is not a
  provider response, retrieval result, or renderer execution record.
- Browser E2E fixtures are deterministic test inputs. They prove product and
  renderer behavior, not fresh configured-provider generation.

## Historical public material

At the audited baseline, the following tracked families retain earlier release
framing or earlier demo catalogs:

- `docs/V8_*.md`
- `demo/final_demo_suite.json`
- `demo/golden_demo_dataset.json`
- `demo/final_demo_launcher.html`
- earlier preview QA manifests and screenshots whose runtimes or scenario names
  differ from the current product

They may be kept for provenance, but they must not be the default current
reviewer route. An older “pass,” scenario count, provider timing, runtime
version, or release-ready statement does not override current source or current
evaluation evidence.

At baseline, `README.md`, `docs/CAPSTONE_DEMO_SHOWCASE.md`, and related demo
checklists still described an eight-scenario catalog. This current-branch
documentation update replaces that presentation path with the canonical ten
flows and four browser-validated showcase fixtures. The mismatch should be
checked again before public upload.

## Local-only material

The following classes remain outside public evidence:

- real environment files and credentials;
- raw workspace sessions and private prompts;
- local Chroma contents and retrieved excerpt text unless separately reviewed;
- locally generated artifacts that have not passed a public claim review;
- raw live-session evaluation rows;
- local engineering planning, task, and acceptance records;
- browser profiles, screenshots, traces, or logs containing personal data or
  machine paths.

The tracked tree retains only environment placeholders, local-data directory
placeholders, sanitized/redacted fixtures, and explicitly selected public
artifacts.

## Claim-boundary rules

### Provider and fixtures

- Say “configured-provider run” only for a captured run from that provider.
- Say “deterministic browser fixture” for the canonical E2E streams.
- A fixture may prove request serialization, parsing, rendering, interaction,
  persistence, and recovery; it does not prove model quality.
- Provider availability must be checked in the presentation environment and
  must never be inferred from an API-key field or an older result.

### Retrieval and RAGAS

- Report 18/19 requested-domain coverage and 16/23 source-anchor overlap only
  as the dated current retrieval report.
- Report 61.44% only as the equal-weight macro across four measured dimensions
  on four sanitized approved-fixture rows.
- Do not compare those values as before/after scores; they use different data
  and measure different stages.
- Context Recall is missing, and current-product external evaluation is
  unavailable under the present privacy boundary.

### Multimodal input

- Claim that valid image pixels can be serialized as a request-scoped
  `input_image` beside text and excluded from persisted session snapshots.
- Do not claim visual influence, image understanding quality, or Case 4
  completion until a controlled configured-provider outcome is captured.

### Runtime and artifact support

- Claim live browser preview only when source eligibility, selected renderer,
  runtime health, and visible output agree.
- Tone.js is silent-ready until an explicit user action starts audio.
- Plain bounded Three.js uses the bundled r176 runtime.
- React components, standalone HTML, remote modules, and external creative
  tools remain code/export/handoff paths unless separately validated.
- A nonblank changing canvas is runtime evidence, not a frame-rate,
  accessibility, or aesthetic benchmark.

### Tests and acceptance

- Test counts are dated local evidence for one checkout and environment.
- Automated checks do not imply independent review, user research, public
  deployment, or presentation-room acceptance.
- A manual checklist may state “pending,” “passed on this machine,” or
  “unavailable”; it must not pre-fill a pass.

## Publication checklist

- [ ] README and current guides say ten Demo Mode flows.
- [ ] The four showcase names match `demo-mode.ts` exactly.
- [ ] Deterministic fixtures are labeled non-provider evidence.
- [ ] Retrieval and RAGAS results remain separate and dated.
- [ ] Missing Context Recall and multimodal influence evidence are visible.
- [ ] No private prompt, raw session, credential, personal path, or internal
      task identifier appears in public media or text.
- [ ] Historical release documents are not linked as current authority.
- [ ] The PowerPoint, upload copy, and spoken script use the same claim matrix.
- [ ] Internal Markdown links and referenced files resolve.
- [ ] A final manual browser rehearsal is recorded without implying
      independent acceptance.

## Audit limitation

This is a source-boundary and pattern audit, not legal review, a complete secret
scan, dependency certification, privacy impact assessment, or guarantee that
all historical Git objects are safe to publish. Run dedicated security and
media reviews before making the repository or showcase public.
