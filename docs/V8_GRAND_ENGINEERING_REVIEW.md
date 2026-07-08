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
- Expanded golden demo coverage for p5.js, Hydra, GLSL, retrieval-grounded
  answers, concept-to-visual translation, geometry/morphogenesis systems, and
  installation/immersive planning as bounded review evidence.
- Added integrated in-app Demo Mode inside Creative Coding Assistant so the
  Capstone presenter can choose curated scenarios, load prompts into the normal
  assistant composer, and keep expected behavior, fallback, evidence, source
  boundaries, and output guidance in the workstation.
- Preserved unsupported-runtime boundaries for live external DCC/MCP
  execution, autonomous delivery, certification, public cloud deployment, and
  metaphysical or medical truth claims.

## Validation Evidence

Backend:

- `.venv/bin/ruff check src tests scripts clients/streamlit` passed.
- `.venv/bin/python -m compileall -q src tests scripts clients/streamlit` passed.
- `.venv/bin/pytest` passed: 2604 tests, 1 warning.
- Focused post-fix demo/readiness tests passed: 19 tests, 1 warning.
- Focused Chroma and retrieval warning check passed: 42 tests, 266
  warnings from Chroma telemetry and Pydantic deprecations.

Frontend:

- `npm run typecheck` passed.
- `npm run test` passed: 59 files, 395 tests.
- `npm run build` passed after local `node_modules` dependency repair without
  package metadata changes; webpack cache snapshot warnings were non-blocking.
- `npm run test:e2e:smoke` passed: 3 tests, including integrated Demo
  Mode scenario selection and prompt prefill.
- Previous full `npm run test:e2e` evidence remains available: 5 tests.

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
- Internal advisory review surfaces ran as engineering reviewers, not
  automatic graders: production readiness status `guarded` with 6 records,
  13 guarded findings, and 0 blocking findings; creative readiness status
  `guarded` with 6 records, 3 guarded findings, and 0 blocking findings; demo
  showcase plan returned 8 flows, 5 fallbacks, and 8 presentation segments;
  retrieval demo pack returned 7 scenarios.

Evaluation:

- Capstone retrieval smoke passed against local Chroma with OpenAI query
  embeddings over committed demo scenario queries: 7 of 7 scenarios returned
  results, with 9 expected-source overlaps.
- Privacy-approved sanitized RAGAs execution passed against
  `demo/evaluation/sanitized_ragas_live_sessions.jsonl`: 4 total samples,
  4 eligible samples, 0 skipped samples, metric `context_precision`,
  provider calls explicitly allowed for synthetic/public fixture content, and
  0 metric failures.
- Exact sanitized RAGAs context precision values:
  `0.9999999999`, `0.9999999999`, `0.99999999995`, and `0.99999999995`;
  average `0.999999999925`, minimum `0.9999999999`, maximum
  `0.99999999995`.
- Final redacted latest-live RAGAs execution passed against
  `demo/evaluation/redacted_live_session_ragas_latest4.jsonl`: 4 total
  samples, 4 eligible samples, 0 skipped samples, metrics `context_precision`,
  `faithfulness`, and `answer_relevancy`, provider calls explicitly allowed for
  redacted reviewer-safe fixture content, and 0 metric failures.
- Exact redacted latest-live RAGAs averages after the public-safe wording
  refresh: context precision `0.7006944444230672`, faithfulness `0.6875`, and
  answer relevancy `0.4419141765019863`. Per-row scores are recorded in
  `demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl`.
- Raw private live-session dry-run over `data/eval/live_sessions.jsonl`
  selected 4 latest eligible samples from 60 total samples and skipped 56. Raw
  external scoring is still avoided because the rows contain recorded local
  questions, answers, and retrieved contexts; the redacted latest-live fixture
  above is the public reviewer evidence path.
- Secret-pattern scan over `data/eval` found no obvious API key, bearer token,
  password, or cloud credential patterns.
- Digital Morphogenesis audit: targeted repository scan found no explicit
  Jason Webb, `Digital Morphogenesis`, or `morphogenesis-resources` source
  coverage in the KB/demo registry. Generic morphogenesis techniques are
  represented through local knowledge/catalog and orchestration records
  (reaction diffusion, cellular automata, L-systems, flow fields, particle
  systems, self-organization, generative structures, and emergent form), so
  Demo Mode uses generic morphogenesis prompts without claiming Jason
  Webb-specific source coverage.
- Existing completed local RAGAs evidence remains available:
  `ragas_latest2_after_kb_quality.jsonl` scored 2 rows with average context
  precision 0.8604; `ragas_latest4_context_precision_after_glsl_fix.jsonl`
  scored 4 rows with average context precision 0.5986.
- Generated golden artifacts were added and QA checked:
  `demo/golden_artifacts/p5_generative_morphogenesis_sketch.js` and
  `demo/golden_artifacts/three_audio_reactive_scene.js` pass `node --check`;
  `demo/golden_artifacts/glsl_kaleidoscope_field.frag` passes static fragment
  shader structure checks; `demo/golden_artifacts/hydra_feedback_lattice.js`
  passes `node --check`.
- Final full-runtime browser/render QA loaded
  `demo/golden_artifacts/browser_full_runtime_qa.html` through the Codex
  in-app browser via local `127.0.0.1` static server rooted at the temporary QA
  workspace. `p5@2.3.0` rendered the p5 artifact nonblank; `three@0.185.1`
  rendered the Three.js artifact nonblank; GLSL compiled, linked, drew, and
  pixel-checked nonblank through WebGL; `hydra-synth@1.4.0` rendered the Hydra
  artifact nonblank with audio detection disabled. No display-FPS, load, soak,
  public deployment, or broad product-preview-runtime benchmark is claimed.
- Integrated Demo Mode is now the primary local demo path inside Creative
  Coding Assistant. The static launcher remains fallback/reviewer evidence.
- One-click local launcher validation passed for
  `demo/final_demo_launcher.html`: the page loaded from the local static QA
  server, rendered all 8 demo flows from `demo/final_demo_suite.json`, loaded
  the suite JSON with HTTP 200, and emitted no console or page errors.

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
  library, final demo suite, and golden dataset evidence.
- Runtime-family prompt evidence through Three.js, p5.js, GLSL, Hydra,
  retrieval/RAG, concept-to-visual translation, geometry/morphogenesis, and
  installation/immersive planning flows.

## Capstone And Golden Demo Status

Capstone criteria are aligned to conservative, reviewable claims: problem,
solution, data sources, evaluation, ethical considerations, limitations,
fallbacks, and future work are documented without overstating live execution.

Golden demo coverage is release-candidate ready as a bounded rehearsal and
evidence set. The public artifact set now includes generated p5.js, Three.js,
GLSL, and Hydra examples with full-runtime browser QA evidence for p5/Three/
Hydra through temporary QA dependencies and direct WebGL evidence for GLSL.
Live showcase upload and final public claims approval still need HITL.

## Readiness Assessment

Detailed engineering evidence matrix:
`docs/V8_CAPSTONE_EVIDENCE_MATRIX.md`.

The Grand Review now publishes evidence and residual risks. The
release-candidate judgment is based on category evidence, remaining risks, and
HITL boundaries: redacted provider-backed RAGAs, Hydra browser runtime QA,
generated artifact QA, final browser/render QA, integrated Demo Mode, static
launcher validation, local demo target review, public/private docs audit,
README evaluator path, and timed-demo evidence.

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
- Raw private live-session RAGAs scoring is still avoided because it would send
  recorded eval content to an external provider. Redacted latest-live and
  sanitized fixture runs are complete and exact scores are documented.
- Generated-output artifact QA is now available for p5.js, Three.js, GLSL, and
  Hydra, with full-runtime browser/render evidence through temporary QA
  packages for p5/Three/Hydra and WebGL for GLSL. Frame timing is uncapped
  local draw-loop timing, not display FPS, load, soak, or production
  performance benchmarking.
- External DCC/MCP execution, autonomous delivery, and future experience-engine
  execution remain unsupported future-scope items.
- Chroma/Pydantic deprecation warnings are non-blocking dependency warnings with
  a documented upgrade-validation path.
- Final freeze, merge, push, tag, and showcase upload require explicit HITL.

Conclusion: V8 is ready for final freeze HITL review, not autonomously frozen.
