# V7 Version Runtime Pack Manifest

## Directories

- `runtime/`: version-level active state, ledgers, audit records, release state, and smoke evidence.
- `capabilities/`: one directory per V7.1 through V7.11 capability plus V7 Grand Engineering Audit.
- `templates/`: reusable task, audit, progress, HITL, product-bug, release, and Junie handoff templates.
- `knowledge/`: engineering philosophy, process rules, policies, first principles, and V6 lessons.

## Final Reconciliation State

- V7.1 through V7.11 are represented in capability ledgers and runtime ledgers.
- Release state, Git tags, GitHub CI evidence, branch history, roadmap coverage, and capability coverage are recorded in `runtime/`.
- The Runtime Pack records V7 Freeze after final Codex and Junie audits reported
  no V7 blockers.
- `architecture/engine_matrix.md` reflects V7.1 through V7.11 implementation boundaries.
- `scripts/v7_quality_gates.py runtime-hygiene` is the canonical lightweight Runtime Pack consistency helper.

## Freeze Artifacts

- `runtime/FINAL_VERSION_SUMMARY.md`
- `runtime/GENERATION_V7_CONTEXT_PACK.md`
- `runtime/RUNTIME_LEDGER_INTEGRITY_FINAL_VERIFICATION.md`
- `runtime/FINAL_RELEASE_RECONCILIATION.md`
- `runtime/FINAL_ARCHITECTURE_SNAPSHOT.md`
- `runtime/ENGINEERING_WORKFLOW_SNAPSHOT.md`
- `runtime/RUNTIME_PACK_EVOLUTION_REPORT.md`
- `runtime/ENGINEERING_WORKFLOW_EVOLUTION_REPORT.md`

V8 has not started. The freeze does not merge, push, tag, or create V8
capabilities.

## Design Goal

The pack should let Codex run V7 with a short continuation prompt while stopping only at real HITL gates, validation failures, product bugs, Runtime Evolution proposals, fix approvals, freeze, and human release operations.
