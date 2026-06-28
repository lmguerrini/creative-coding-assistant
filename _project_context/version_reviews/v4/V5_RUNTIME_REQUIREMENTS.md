# V5 Runtime Requirements

This file records runtime-evolution decisions approved during the V4 Grand
Review. It is an engineering input for designing the V5 Version Runtime Pack.
It is not product documentation, a runtime file, or an implementation of V5.

## Core Decision

V5 must use one Version Runtime Pack for the entire V5, not a separate Runtime
Pack per capability. Capability-specific work should live inside that version
pack through capability subfolders and capability-level progress artifacts.

## Required V5 Runtime Pack Structure

The V5 Version Runtime Pack must include:

- `VERSION_RUNTIME_PROMPT.md`
- `VERSION_PROGRESS.md`
- `CAPABILITY_PROGRESS.md`
- `VERSION_HISTORY.md`
- Technical Debt Ledger
- Engineering Metrics
- Architectural Drift Review
- Complexity Budget
- Runtime Failure Path Audit
- capability subfolders for capability-specific specifications, audits,
  validation evidence, and handoff notes

`CAPABILITY_PROMPT.md` may remain only as fallback or single-capability
regeneration support. It should not be the primary cross-version continuity
mechanism.

## VERSION_RUNTIME_PROMPT Requirement

`VERSION_RUNTIME_PROMPT.md` will replace the version-transition workflow as the
stable prompt between versions. It should carry version-wide engineering
context, review rules, validation expectations, audit gates, runtime-evolution
decisions, and stop conditions across all V5 capabilities.

The V5 Runtime Pack should not be created during V4 freeze work unless a future
task explicitly requests Runtime Pack Evolution implementation.

## Progress And History Requirements

V5 must maintain version-level and capability-level progress separately:

- `VERSION_PROGRESS.md` records version state, gates, final validation, freeze
  readiness, merge/push/tag status, and cross-capability decisions.
- `CAPABILITY_PROGRESS.md` records each capability's implementation status,
  validation status, audit result, HITL outcome, and handoff state.
- `VERSION_HISTORY.md` records the evolution of the version in a durable format
  suitable for future version planning.

## Required Review Artifacts

V5 must keep the following review artifacts current:

- Engineering Metrics
- Architectural Drift Review
- Complexity Budget
- Technical Debt Ledger
- Runtime Failure Path Audit

These artifacts must be treated as engineering review evidence, not product
documentation.

## Validation Requirements

V5 must keep cumulative local app smoke tests for every capability. Each
cumulative smoke run must verify, as applicable:

- backend starts
- frontend starts
- localhost loads
- main app renders
- real UI workflow reaches finalization
- browser console has no fatal errors
- backend logs have no fatal runtime errors
- new capability surfaces are visible, exposed, or inspectable
- backend-metadata-only surfaces are verified through registry/API/test/source
  inspection paths
- Runtime Failure Path Audit invariants remain true
- provider/model routing remains unchanged unless explicitly approved
- passive registries remain passive unless explicitly promoted by HITL
- generated outputs are not mutated unless explicitly in scope
- servers stop cleanly and ports close

## Audit And HITL Requirements

V5 must keep:

- Codex Engineering Audit for every capability
- HITL after every audit
- HITL before any Runtime Evolution change
- human-controlled merge gates
- human-controlled push gates
- human-controlled freeze/tag gates

Automation must support quality. It must not replace HITL quality judgment.

## Runtime Evolution Process

Runtime Evolution changes must be explicit, scoped, and human-approved before
implementation. The expected process is:

1. Record the proposed Runtime Evolution change.
2. Complete engineering review and validation evidence.
3. Ask for HITL approval before changing runtime files or runtime process.
4. Apply the approved change only within the approved scope.
5. Revalidate affected runtime and documentation surfaces.
6. Record the decision in version progress, history, and relevant ledgers.

Runtime Evolution notes may be recorded during V4 freeze preparation, but V5
runtime files must not be generated until a future task explicitly requests the
V5 Runtime Pack.

## Quality Principle

Quality is always higher priority than automation. Automated validation,
runtime prompts, generated checklists, and review templates are useful only when
they improve engineering judgment, evidence quality, and human-controlled
decision making.
