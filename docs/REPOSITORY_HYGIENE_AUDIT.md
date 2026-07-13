# Repository Hygiene Audit

## Scope and method

This audit inspected the tracked tree at baseline commit `e817a8ad`, Git status,
ignore rules, file modes, largest blobs, path categories, public-path strings,
and a high-confidence credential-pattern scan. It did not inspect remote
systems, deleted history, private local files, or the full dependency supply
chain.

The working tree was clean at the start of the audit. The reviewer documents
created by this task are intentionally outside that baseline count.

## Tracked-tree snapshot

| Area | Tracked files |
|---|---:|
| `src/` | 880 |
| `tests/` | 458 |
| `clients/` | 202 |
| `demo/` | 30 |
| `architecture/` | 19 |
| `docs/` | 17 |
| Root files | 7 |
| `scripts/` | 6 |
| `data/` | 3 |
| `assets/` | 3 |
| `.github/` | 2 |
| **Total** | **1,627** |

The tree contains no tracked symlinks.

## Pass findings

- No tracked `node_modules`, virtual environment, Python cache, test-results,
  coverage, Next.js build, or macOS metadata path was found.
- `.env` and environment-specific variants are ignored; only
  `.env.example` is tracked.
- Local Chroma, artifact, and evaluation directories retain only three
  `.gitkeep` placeholders in Git.
- No high-confidence private-key, OpenAI-style secret, AWS access-key, GitHub
  token, or Slack token pattern was found in the tracked tree by the audit
  regexes.
- No machine-specific macOS, Linux, or Windows user-home path was found in
  README, docs, demo, scripts, source, tests, or client files.
- The frontend dependency lockfile is tracked, and CI uses `npm ci`.
- CI separates backend static/tests, dependency audit, runtime documentation,
  frontend unit/type, and Playwright jobs.

These are bounded scan results, not a guarantee that no secret or vulnerable
dependency exists.

## Large tracked artifacts

| File | Approximate size | Assessment |
|---|---:|---|
| `assets/preview_v2.png` | 2.47 MB | Reviewer media; acceptable but should be optimized if replaced |
| `assets/preview_current.png` | 2.19 MB | Primary README image; acceptable with size awareness |
| `clients/nextjs/public/vendor/three-r176.min.js` | 692 KB | Intentional local runtime dependency; version and license must remain traceable |
| `clients/nextjs/src/app/globals.css` | 460 KB | Maintainability hotspot |
| `assets/preview_v1.png` | 406 KB | Historical media; archive policy should be explicit |

No individual tracked blob exceeds 2.5 MB, but the screenshots should not grow
without an image-size budget.

## Source maintainability hotspots

| File | Approximate size | Risk |
|---|---:|---|
| `src/creative_coding_assistant/orchestration/metadata/multimodal_studio.py` | 381 KB | Registry and policy concepts are difficult to review as one unit |
| `src/creative_coding_assistant/orchestration/metadata/hybrid_studio.py` | 365 KB | Large declarative surface increases change coupling |
| `src/creative_coding_assistant/orchestration/metadata/hybrid_agentic_workflow.py` | 321 KB | Workflow metadata and policy boundaries are dense |
| `clients/nextjs/src/components/workstation-shell.tsx` | 305 KB | UI state, commands, persistence, and streaming share one component |
| `clients/nextjs/src/lib/assistant-stream.ts` | 289 KB | Stream parsing and derived product state have a large review surface |

Large tests accompany several of these files, which reduces regression risk but
does not remove the cost of understanding or changing them. Extraction should
follow stable responsibilities, not arbitrary line-count targets.

## Public-documentation findings

At the audited baseline, current and historical evidence are mixed in the same
public folders. Several documents and demo assets retain earlier release names
or an eight-scenario catalog. They are valuable history but are not safe as the
default reviewer path for the current ten-scenario product.

The public-boundary audit identifies the canonical current sources and labels
historical material. Before showcase upload, rerun the link and stale-catalog
checks after all README and reviewer-document updates have landed.

## Environment and data boundary

The tracked environment example contains placeholders and local-relative data
paths. Real credentials, local database records, generated artifacts, raw
evaluation sessions, and workspace-session storage are ignored. Public
evaluation evidence is limited to sanitized/redacted fixtures and a retrieval
report that contains metadata and rankings rather than retrieved excerpt text.

## Recommended actions

| Priority | Action | Completion signal |
|---|---|---|
| High | Make the ten-flow guide the only default reviewer path | README, demo guide, and source catalog agree in CI |
| High | Keep current evidence separate from historical release documents | Historical files carry an unmistakable banner or move to an archive |
| Medium | Split workstation and stream modules by stable responsibility | Smaller modules with unchanged unit/E2E contracts |
| Medium | Split large metadata registries by domain and policy family | Registry API remains stable; review scope decreases |
| Medium | Add Markdown link and stale-count checks to CI | Broken internal links and obsolete catalog counts fail CI |
| Low | Establish screenshot and bundled-asset size budgets | New media/runtime blobs exceed budgets only with review |

## Reproduction commands

```bash
git status --short
git ls-files | wc -l
git ls-files -s
git ls-tree -r -l HEAD | sort -k4,4nr
git ls-files '.env*' 'data/**'
git check-ignore -v .env data
git diff --check
```

Credential and local-path scans should be run with a dedicated secret scanner
in CI as well; the pattern scan used for this audit is intentionally described
as a first-pass check.
