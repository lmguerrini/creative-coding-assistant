# V5 Project Freeze Checklist

This checklist describes the accepted Version 5 freeze posture for the
Creative Coding Assistant / first HOLOiVERSE Engine foundation. It is a
human-controlled release checklist, not an automation script.

## Freeze Authority

- Freeze, merge, push, and tag actions require explicit HITL approval.
- Runtime Evolution remains HITL-gated and must not run automatically.
- Junie review is a separate human-requested gate and must not be simulated.
- Provider provisioning, model downloads, runtime installation, and provider
  switching must remain manual or explicitly configured by the user.

## Completed V5 Capability Scope

- V5.1 Execution Optimization Engine metadata and advisory execution surfaces.
- V5.2 Intelligent Model Routing Engine metadata and routing advice surfaces.
- V5.3 Performance Engine metadata and optimization planning surfaces.
- V5.4 Production Observability metadata and production-readiness reports.
- V5.5 Adaptive Execution Intelligence metadata and HITL advisory policy.
- V5.6 Production Release architecture freeze, audit, smoke-test, and release
  readiness metadata.

## Accepted Validation Gate

Run the current V5 Python validation gate before freeze:

```bash
git diff --check
.venv/bin/python -m compileall -q src tests
.venv/bin/pytest -q
```

Run focused tests for every touched runtime surface. When documentation,
workflow contracts, or public release guidance changes, also run the applicable
documentation and alignment tests.

Use scoped ruff checks on changed Python files as the accepted lint gate:

```bash
.venv/bin/python -m ruff check <changed-python-files>
```

Full-project ruff over `src clients tests scripts` is a documented technical
debt item. It is not a clean V5 release gate until the existing backlog is
resolved.

For frontend changes, run the client checks that match the touched surface:

```bash
cd clients/nextjs
npm run typecheck
npm run test
npm run build
```

## Smoke Test Gate

Before freeze, complete the local app smoke test and record the result in the
runtime reports:

- backend starts successfully with the intended local settings
- frontend starts successfully and reaches the assistant UI
- in-app browser can execute a representative user workflow
- final output, explanation, artifact surfaces, and telemetry are visible
- browser console and backend logs contain no new runtime errors
- V5 registries remain metadata/advisory surfaces only
- no unexpected long-running dev server remains after the smoke test

## Audit And Review Gate

- Codex Grand Engineering Audit must be completed and accepted by HITL.
- Any version-scoped fixes identified by the audit must be implemented,
  validated, committed, and reviewed before proceeding.
- A focused follow-up audit must confirm those fixes did not change provider
  routing, generated output mutation, runtime evolution, merge/push/tag, or
  safety boundaries.
- Junie review must be performed only after explicit HITL approval.
- Freeze may proceed only after HITL accepts the final validation and review
  posture.

## Safety Checklist

- no automatic model downloads
- no provider provisioning
- no runtime auto-installation
- no API key assumptions beyond explicit settings or environment fallback
- no uncontrolled provider switching
- no uncontrolled Runtime Evolution
- no generated output mutation outside intended artifact surfaces
- merge, push, tag, and freeze remain human-controlled

## Known Non-Blocking Risks

- Full-project ruff still has a backlog and remains technical debt.
- V5 registries are broad metadata surfaces; runtime behavior remains narrower
  than the advisory capability map.
- Demo readiness depends on local environment configuration and explicit API
  credentials when live OpenAI paths are exercised.
- Junie review and final freeze authorization remain pending HITL steps.

## Freeze Decision

V5 is freeze-eligible only when the validation gate, smoke test gate, Codex
audit gate, Junie gate, and final HITL freeze authorization are all complete.
Until then, the project remains review-ready but not frozen.
