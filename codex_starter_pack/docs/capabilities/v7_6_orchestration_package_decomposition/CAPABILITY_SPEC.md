# V7.6 Orchestration Package Decomposition - Capability Spec

## Purpose
Resolve the pre-freeze architecture debt found by Junie: the flat
`orchestration/` package mixed live runtime code with passive metadata,
governance, audit, contract, and advisory modules. V7.6 separates package
ownership without changing user-visible behavior.

## Roadmap Contract
- Junie architecture finding intake.
- Phase -1 / Phase 0 Runtime Pack validation and release reconciliation.
- Orchestration module inventory and boundary classification.
- Runtime package extraction.
- Metadata package extraction.
- Governance package extraction.
- Audit package extraction.
- Contract package extraction.
- Advisory package extraction.
- Legacy import compatibility shims.
- Internal import rebinding.
- Architecture boundary documentation.
- Import compatibility regression tests.
- Runtime Pack ledger synchronization.
- Capability validation and Commit Readiness Gate.

## Architecture Boundaries
Canonical module ownership lives under:

- `creative_coding_assistant.orchestration.runtime`
- `creative_coding_assistant.orchestration.metadata`
- `creative_coding_assistant.orchestration.governance`
- `creative_coding_assistant.orchestration.audit`
- `creative_coding_assistant.orchestration.contracts`
- `creative_coding_assistant.orchestration.advisory`

The root `creative_coding_assistant.orchestration.<module>` import paths remain
public compatibility shims. Root package exports remain supported.

## Product Boundaries
V7.6 must not change workflow execution, provider/model routing, prompt
rendering, generated output semantics, persistence ownership, retry behavior,
stream subscriptions, telemetry emission, frontend UI behavior, merge, push,
tag, freeze, or V8 start state.

## Validation Contract
Required validation:

- `git diff --check`
- Full-project Ruff over `src`, `tests`, and `scripts` if feasible
- `python -m compileall src tests scripts`
- Focused V7.6 import compatibility pytest
- Full backend pytest
- Frontend typecheck / Vitest only if frontend code is affected
- Local app smoke only if Runtime Pack rules require it for affected behavior

## Runtime Evolution Contract
No product Runtime Evolution is expected. If decomposition requires changing the
Runtime Pack rules or live runtime behavior, stop for Runtime Evolution HITL.
