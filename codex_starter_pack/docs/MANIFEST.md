# V7 Version Runtime Pack Manifest

## Directories

- `runtime/`: version-level active state, ledgers, audit records, release state, and smoke evidence.
- `capabilities/`: one directory per V7.1 through V7.11 capability plus V7 Grand Engineering Audit.
- `templates/`: reusable task, audit, progress, HITL, product-bug, release, and Junie handoff templates.
- `knowledge/`: engineering philosophy, process rules, policies, first principles, and V6 lessons.

## Final Reconciliation State

- V7.1 through V7.11 are represented in capability ledgers and runtime ledgers.
- Release state, Git tags, GitHub CI evidence, branch history, roadmap coverage, and capability coverage are recorded in `runtime/`.
- The Runtime Pack stops at Codex Engineering Audit HITL and does not record a V7 freeze, V8 start, merge, push, or tag operation.
- `architecture/engine_matrix.md` reflects V7.1 through V7.11 implementation boundaries.
- `scripts/v7_quality_gates.py runtime-hygiene` is the canonical lightweight Runtime Pack consistency helper.

## Design Goal

The pack should let Codex run V7 with a short continuation prompt while stopping only at real HITL gates, validation failures, product bugs, Runtime Evolution proposals, fix approvals, freeze, and human release operations.
