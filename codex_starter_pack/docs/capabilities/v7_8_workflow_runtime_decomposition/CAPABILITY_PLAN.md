# V7.8 Workflow Runtime Decomposition - Capability Plan

## Capability
V7.8 Workflow Runtime Decomposition.

## Branch
`feature/workflow-runtime-decomposition`

## Tag
Pending human-controlled release tag `v7.8.0`.

## Goal
Decompose workflow runtime graph responsibilities into a clearer module
architecture while preserving behavior, topology, state transitions, contracts,
provider routing, streaming, workspace behavior, and generated outputs.

## Contractual Roadmap Items
1. Workflow Graph Decomposition.
2. Runtime Node Extraction.
3. `runtime/nodes/` architecture.
4. LangGraph Node Handler Split.
5. Node Registration Layer.
6. Graph Builder Refactor.
7. State Transition Isolation.
8. Node Dependency Simplification.
9. Runtime Module Boundary Enforcement.
10. Import Graph Simplification.
11. Workflow Runtime Documentation.
12. Runtime Architecture Validation.

## Task Ordering
1. Phase -1 / Phase 0 Runtime Pack validation.
2. Runtime graph inventory and baseline topology capture.
3. Node handler extraction into `runtime/nodes/`.
4. Node registration and graph builder refactor.
5. State transition isolation and import boundary simplification.
6. Compatibility validation and focused regression tests.
7. Workflow runtime documentation and ledger synchronization.
8. Capability validation, commit readiness, and Codex Engineering Audit gate.

## Closure Workflow
Architecture Update -> Documentation Update -> Capability Validation -> Codex
Engineering Audit -> HITL -> Eventuali Capability-Scoped Fixes -> Final
Validation -> Cumulative Local App Smoke Test if required -> Human-controlled
merge/push/tag -> Remote GitHub CI green.
