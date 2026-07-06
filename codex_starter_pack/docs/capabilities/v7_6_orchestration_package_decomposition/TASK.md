# V7.6 Orchestration Package Decomposition - Task

## Task
Implement safe orchestration package decomposition before V7 freeze.

## Contractual Roadmap Item
Items 1 through 15 in the V7.6 capability plan.

## Scope
- Separate live runtime orchestration modules from passive metadata,
  governance, audit, contract, and advisory modules.
- Preserve legacy root module imports through compatibility shims.
- Preserve root package exports.
- Add architecture documentation and import compatibility regression tests.
- Update Runtime Pack ledgers.
- Validate and commit before Codex Engineering Audit / HITL.

## Non-Goals
- Do not continue the V7 Grand Engineering Audit.
- Do not change user-visible behavior.
- Do not remove passive metadata.
- Do not alter provider/model routing, workflow order, prompt rendering,
  generated output semantics, persistence ownership, retry behavior, telemetry
  emission, frontend UI behavior, merge, push, tag, freeze, or V8 start state.

## Required Files
- `src/creative_coding_assistant/orchestration/`
- `docs/ORCHESTRATION_PACKAGE_BOUNDARIES.md`
- `tests/test_orchestration_package_decomposition.py`
- `codex_starter_pack/docs/ROADMAP_DEFINITIVE_V7.md`
- `codex_starter_pack/docs/runtime/*.md`
- `codex_starter_pack/docs/capabilities/v7_6_orchestration_package_decomposition/*.md`

## Validation
- `git diff --check`
- Full-project Ruff over `src`, `tests`, and `scripts` if feasible
- `python -m compileall src tests scripts`
- Focused V7.6 import compatibility pytest
- Full backend pytest
- Frontend typecheck / Vitest only if frontend code is affected
- Local app smoke only if required by Runtime Pack after validation

## Stop Conditions
Stop for required HITL gates, validation failures, product bugs, Runtime
Evolution proposals, or the final human-controlled merge/push/tag gate.

## Progress Update Requirements
Update capability progress, central progress, roadmap coverage, validation
evidence, product bug status, Runtime Evolution status, and historical
consistency before Codex Engineering Audit / HITL.
