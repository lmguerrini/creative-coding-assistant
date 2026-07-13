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

## Current V9 reviewer checkpoint

A follow-up at commit `eb525ed569e6` on 2026-07-13 counted 1,651 tracked
files: 880 under `src/`, 458 under `tests/`, 202 under `clients/`, 40 under
`docs/`, 30 under `demo/`, 19 under `architecture/`, 6 under `scripts/`, 3
each under `data/` and `assets/`, 2 under `.github/`, 1 under `outputs/`, and 7
root files. These counts describe the committed tree, not concurrent
uncommitted review work.

The current tracked tree still has no symlinks, tracked runtime databases, real
environment files, or high-confidence secret matches under the scan patterns
described below. The internal Markdown audit checked 301 relative links across
64 files without finding a missing target or heading anchor.

The local clone has maintenance residue: `git count-objects -vH` reported 4,484
loose objects (33.40 MiB), 29,814 packed objects in 3 packs (202.95 MiB), and
61 garbage files (1.57 MiB). A connectivity-only `git fsck` found no missing or
broken reachable object, while `--no-reflogs` reported 92 dangling commits,
6,136 dangling trees, and 2 dangling blobs. This is local `.git` state, not
tracked product content. No prune, garbage-collection, or history rewrite was
performed during review.

### Dependency audit boundary

A fresh local Python 3.14.0 virtual environment was audited with `pip-audit`
2.10.1 against the PyPI advisory service on 2026-07-13. CI documents Python
3.11, so this is a dated local reproduction rather than a claim about every
supported resolver result:

- `.[server]`: 142 packages; no known vulnerabilities reported.
- `.[server,evaluation]`: 163 packages; two no-fix advisories were reported:
  `diskcache` 5.6.3,
  [CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)
  (unsafe pickle deserialization when an attacker can write the cache
  directory), and `ragas` 0.4.3,
  [CVE-2026-6587 / GHSA-95ww-475f-pr4f](https://github.com/advisories/GHSA-95ww-475f-pr4f)
  (SSRF through multimodal URL/local-file context processing).

CCA's current evaluator imports text metrics and selects committed local
evaluation datasets. It does not expose the affected RAGAS multimodal context
processor as a product route. That reduces current exposure but does not remove
the vulnerable optional packages. Keep evaluation local and trusted, protect
cache directories, avoid attacker-controlled URL/file contexts, and re-audit
before use or release. “No known vulnerabilities” is a dated tool result, not
a general security guarantee.

The local reproduction created `/tmp/cca-v9-security-venv`, installed
`.[server]` plus `pip-audit>=2.7`, froze non-editable packages to a 142-line
requirements file, and audited that exact freeze with `--disable-pip`,
`--no-deps`, `--strict`, and `--desc`. It then installed `.[evaluation]`, froze
again, and repeated the audit with JSON output. `pip-audit` warns that hashes
are preferable for this requirements-file mode; the temporary environments and
diagnostics are not committed evidence.

The frontend lockfile was installed and audited separately with Node 24.11.0
and npm 11.6.1. A lock-strict `npm ci` completed with 495 packages and reported
zero known vulnerabilities; a separate npm audit at the `low` threshold also
reported zero. A targeted dependency-tree check resolved the security
overrides to `js-yaml` 4.3.0 and `postcss` 8.5.18. Raw `npm ls --all` is not a
lock-integrity gate here because npm labels platform-skipped optional WASM
fallback children as extraneous immediately after the clean install. These are
dated ecosystem results, not a guarantee about future advisories or other
resolver/runtime combinations.

### Licensing and publication boundary

The repository has no root license; the README explicitly tells readers not to
assume redistribution or production-use rights. The bundled Three.js r176 file
has a tracked `three.LICENSE.txt`, but that does not license the repository as a
whole. Source excerpts, generated artifacts, screenshots, reference images,
and showcase media still require owner review for copyright, attribution, and
publication rights before any public upload.

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
| `outputs/creative-coding-assistant-capstone.pptx` | 2.44 MB | Ten-slide reviewer deck; ZIP integrity, overflow, and rendered layout were verified |
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

Current-branch follow-up: the production and evaluation runbooks no longer
route reviewers through the retired eight-flow launcher, and the human-readable
V8 demo/evidence files now carry explicit historical banners. Dated JSON and QA
records remain unchanged for provenance.

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
