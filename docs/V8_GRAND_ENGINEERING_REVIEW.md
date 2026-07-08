# V8 Grand Engineering Review

Date: 2026-07-08
Branch: `version-review/v8`

This record captures the V8 Grand Engineering Review release-candidate pass.
It is not a merge, push, tag, release freeze, or V9 start.

## Review Standard

V8 Grand Engineering Review is a release-candidate validation program. The
standard is broader than a read-only audit: local safe blockers may be fixed,
validated, integrated, and committed before the final freeze HITL decision.

Required HITL boundaries remain active for merge, push, tag, final freeze,
destructive actions, provider/source blockers, security/privacy risks, and
product or architecture decisions.

## Runtime Pack Alignment

Private Runtime Pack files under `.runtime_pack/active/` were updated locally to
use the V8 Grand Engineering Review definition as the active source of truth.
The private pack remains ignored and is not committed to the public repository.

Validation:

- Runtime Pack JSON manifests parsed successfully.
- `.runtime_pack/active/scripts/runtime_pack_hygiene.py` passed.
- Runtime Pack roadmap, active state, validation policy, capability manifest,
  and capability spec now describe the release-candidate review posture.

## Implemented Fixes

- Replaced stale documentation claims that treated V8 Grand Review as out of
  scope after the review branch had already started.
- Kept merge, push, tag, release freeze, showcase upload, and final release
  decisions behind HITL.
- Expanded golden demo coverage for p5.js, Hydra-if-supported, GLSL, Sacred
  Geometry, Sacred Architecture, Mythopoetic Narrative, and HoloGenesis Planner
  as bounded review evidence.
- Preserved unsupported-runtime boundaries for HoloMind, HOLOiVERSE, live
  external DCC/MCP execution, autonomous delivery, certification, and
  metaphysical or medical truth claims.

## Validation Evidence

Backend:

- `.venv/bin/ruff check src tests scripts clients/streamlit` passed.
- `.venv/bin/python -m compileall -q src tests scripts clients/streamlit` passed.
- `.venv/bin/pytest` passed: 2604 tests, 1 warning.
- Focused post-fix demo/readiness tests passed: 20 tests, 1 warning.
- Focused Chroma and retrieval stabilization tests passed: 52 tests, 266
  warnings from Chroma telemetry and Pydantic deprecations.

Frontend:

- `npm run typecheck` passed.
- `npm run test` passed: 58 files, 391 tests.
- `npm run build` passed after refreshing local dependencies with `npm ci`.
- `npm run test:e2e:smoke` passed: 2 tests.
- `npm run test:e2e` passed: 5 tests.

Product smoke:

- In-process API smoke passed for `/api/health`, `/api/health/live`,
  `/api/health/ready`, missing-route 404 behavior, invalid assistant request
  validation, and workspace session retrieval.
- Minimal live OpenAI provider smoke passed with the configured local
  `OPENAI_API_KEY` and `gpt-5-mini`: the API returned a response id, model
  `gpt-5-mini-2025-08-07`, expected content `CCA_V8_PROVIDER_OK`, and usage
  metadata for a 107-token bounded request.
- Quality dashboard generation passed and reported complete roadmap coverage for
  23 quality gates.

Evaluation:

- Capstone retrieval smoke passed against local Chroma with OpenAI query
  embeddings over committed demo scenario queries: 7 of 7 scenarios returned
  results, with 9 expected-source overlaps.
- Latest RAGAs dry-run regeneration passed for
  `data/eval/live_sessions.jsonl`: 60 total samples, 1 latest eligible sample,
  59 skipped by selection/eligibility, metric `context_precision`, and no
  evaluator provider calls.
- The latest eligible RAGAs sample was p5.js-only, with 126 user characters,
  379 response characters, 5 retrieved contexts, and no provider metadata or
  ground truth.
- Secret-pattern scan over `data/eval` found no obvious API key, bearer token,
  password, or cloud credential patterns.
- Live RAGAs evaluator execution was not rerun because it would send recorded
  local sample text and retrieved contexts to an external provider. That remains
  a privacy/HITL boundary rather than a technical failure.
- Existing completed local RAGAs evidence remains available:
  `ragas_latest2_after_kb_quality.jsonl` scored 2 rows with average context
  precision 0.8604; `ragas_latest4_context_precision_after_glsl_fix.jsonl`
  scored 4 rows with average context precision 0.5986.

Security:

- Secret scan found only test fixtures and documented placeholders, not real
  committed credentials.
- Clean server dependency environment audit passed with `pip-audit`: no known
  vulnerabilities found.
- Local runtime data, Chroma state, Runtime Pack material, environment files,
  and generated test output remain ignored.

## V8.1 Through V8.8 Coverage

The review validated cross-capability coverage rather than only one milestone:

- Retrieval and source-pack evidence through retrieval foundation, Chroma
  foundation, Capstone KB, and demo-pack tests.
- API/runtime stabilization through health, readiness, workspace, and assistant
  stream smoke paths.
- Workstation UX through typecheck, unit, build, smoke E2E, resilience E2E, and
  creative journey E2E coverage.
- Demo and Capstone readiness through production demo assets, demo showcase
  experience, production creative readiness review, manual checklist, prompt
  library, and golden dataset evidence.
- Runtime-family prompt evidence through Three.js, p5.js, GLSL,
  Hydra-if-supported, retrieval/RAG, symbolic translation, Sacred Geometry,
  Sacred Architecture, Mythopoetic Narrative, Immersive Composer, and
  HoloGenesis Planner flows.

## Capstone And Golden Demo Status

Capstone criteria are aligned to conservative, reviewable claims: problem,
solution, data sources, evaluation, ethical considerations, limitations,
fallbacks, and future work are documented without overstating live execution.

Golden demo coverage is release-candidate ready as a bounded rehearsal and
evidence set. Live showcase upload and final public claims approval still need
HITL.

## Readiness Assessment

Detailed engineering scorecard:
`docs/V8_CAPSTONE_EXCELLENCE_SCORECARD.md`.

The Grand Review no longer uses a single overall score. Production readiness is
one category in the detailed scorecard and remains 94/100; the higher-level
release-candidate judgment is based on category evidence, remaining risks, and
HITL boundaries.

AI review readiness: ready for final reviewer evaluation with bounded claims.

Senior reviewer readiness: ready for final release-candidate review, subject to
HITL approval for freeze and public release actions.

Deprecation status:

- Current runtime versions: Python 3.14.0, `chromadb` 0.6.3, `pydantic` 2.13.3,
  and `pydantic-settings` 2.14.0.
- Focused Chroma/retrieval tests pass, but Chroma emits third-party warnings
  from `.venv/lib/python3.14/site-packages/chromadb/telemetry/opentelemetry`
  for `asyncio.iscoroutinefunction` and from
  `.venv/lib/python3.14/site-packages/chromadb/types.py` for instance-level
  Pydantic `model_fields` access.
- No low-risk local production-code fix is available because both warning sites
  are inside Chroma. Do not monkeypatch or silence them as a release-quality
  substitute.
- Upgrade path: evaluate a newer Chroma release that removes those warning
  sites while keeping `chromadb>=0.6.3,<1.0.0`; run `pip-audit`,
  `pip check`, `tests/test_chroma_foundation.py`,
  `tests/test_retrieval_foundation.py`, and
  `tests/test_v7_5_production_api_runtime_stabilization.py`; then refresh the
  clean server dependency audit before changing the pin.

Remaining risks:

- Live provider smoke has passed, but it is a minimal connectivity/content
  check and not a broad provider benchmark.
- Live RAGAs scoring still requires explicit HITL because it sends recorded eval
  content to an external provider. The safe dry-run and existing historical
  scores are documented.
- Fresh generated-output artifact scoring was not performed because no persisted
  generated artifacts are available under `data/artifacts`; the golden demo
  benchmark remains prepared-flow evidence until approved artifacts are
  generated and QA reviewed.
- External DCC/MCP execution, HoloMind, and HOLOiVERSE remain unsupported
  future-scope items.
- Chroma/Pydantic deprecation warnings are non-blocking dependency warnings with
  a documented upgrade-validation path.
- Final freeze, merge, push, tag, and showcase upload require explicit HITL.

Conclusion: V8 is ready for final freeze HITL review, not autonomously frozen.
