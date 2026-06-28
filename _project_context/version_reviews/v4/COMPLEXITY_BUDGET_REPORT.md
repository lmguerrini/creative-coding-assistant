# V4 Complexity Budget Report

This file records engineering-review complexity findings for V5 runtime-pack
design. It is not product documentation, a runtime file, or an application
behavior contract.

## Complexity Introduced

V4 introduced complexity primarily as static metadata and validation surface:

- six cumulative capability layers from V4.1 through V4.6
- dozens of passive registries for agent, orchestration, hybrid workflow,
  hybrid studio, multimodal studio, and hardening concepts
- additional exported API surfaces through package `__all__` declarations
- architecture and documentation alignment requirements for every passive
  registry family
- cumulative smoke-test requirements that combine backend, frontend, UI,
  registry/API/source inspection, failure-path, routing, and cleanup checks
- review gates for Codex audit, HITL acceptance, Junie audit, final validation,
  runtime evolution decisions, and freeze readiness

This complexity was accepted because V4 needed to prepare future Agentic Studio
behavior without activating it in the current runtime.

## Complexity Removed Or Reduced

V4 reduced ambiguity in several areas:

- passive agent contracts now make agent identities, roles, authorities, memory
  boundaries, and metadata visible
- orchestration-readiness surfaces now separate future planning from active
  runtime execution
- hybrid workflow metadata now documents where escalation may eventually attach
  while preserving the V3 backbone
- Studio and Multimodal Studio surfaces now describe future UI/runtime concepts
  without silently implying behavior
- V4.6 audit registries converted many implicit invariants into inspectable
  records
- Runtime Failure Path Audit evidence confirms terminal failure normalization
  remains intact
- the p5 HTML preview guard fix removed a concrete mismatch where leading HTML
  comments could let a full HTML document reach the JavaScript sandbox

## Complexity Intentionally Deferred

The following complexity was intentionally deferred:

- active multi-agent orchestration
- active blackboard storage
- materialized shared context views
- active hybrid escalation or specialist loops
- local/cloud provider execution changes
- Auto Mode or Studio Mode runtime control
- active HITL interruption/request behavior
- preview/rendering execution changes beyond the scoped p5 classification fix
- artifact collaboration persistence
- artifact provenance or lineage recording
- workflow graph consolidation
- passive registry boilerplate consolidation
- V5 Version Runtime Pack implementation
- V7 runtime graph consolidation and handler extraction

## Maintainability Impact

The complexity budget is acceptable for V4 freeze because new concepts remain
passive, tested, inspectable, and documented. The budget becomes less acceptable
if future versions keep adding registry families without consolidating shared
patterns or centralizing version-level review artifacts.

V5 should therefore use a single Version Runtime Pack with stable version
progress, capability progress, history, metrics, drift, complexity, and debt
artifacts. That structure should lower review overhead while preserving HITL
quality gates.

## Budget Status

| Area | Status | Rationale |
| --- | --- | --- |
| Runtime behavior | Within budget | Provider/model routing, workflow control, retries, runtime selection, and output mutation remained unchanged |
| Metadata surface | High but accepted | Registry count grew intentionally to expose future architecture without activating it |
| Validation surface | High but valuable | Cumulative tests and smoke runs caught the preview classification defect before freeze |
| Documentation burden | High | V5 runtime artifacts should reduce reconstruction cost |
| Future maintainability | Watch | Registry boilerplate and `workflow_graph.py` size need future consolidation |
