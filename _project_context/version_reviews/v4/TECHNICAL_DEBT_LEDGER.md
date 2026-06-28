# V4 Technical Debt Ledger

This file records technical debt identified during the V4 Grand Engineering
Review. It is an engineering artifact for V5 runtime-pack design. It is not
product documentation, a runtime file, or an application behavior contract.

## Resolved

| Debt item | Resolution | Rationale |
| --- | --- | --- |
| Leading-comment HTML reached the p5 JavaScript preview path | Fixed by `3726eacd Stabilize p5 HTML preview guard` | HTML documents beginning with leading comments are now classified as HTML and rejected before sandbox execution |
| Runtime failure-path invariants were implicit across some review notes | Covered by V4.6 hardening and final validation | Terminal failure normalization and non-recovery behavior were rechecked during cumulative validation |
| Passive registry boundary ambiguity | Reduced by V4.6 audit registries and documentation alignment | Registry discoverability, blackboard, shared context, collaboration, reliability, determinism, and architecture consistency are now inspectable |
| Final V4 readiness uncertainty after Codex audit | Resolved by HITL approval, Junie audit, final validation, and smoke test | Both audits scored 9/10 with no blockers |

## Open

| Debt item | Status | Rationale |
| --- | --- | --- |
| `workflow_graph.py` is large | Open future work | The graph remains functional and validated, but maintainability would benefit from consolidation or module split |
| Passive registry boilerplate | Open future work | Repeated passive-boundary patterns increase maintenance cost |
| Dependency warnings | Open accepted warnings | Chroma, Vite CJS, Next.js dev cross-origin, and intentional backend shutdown warnings are non-blocking but should stay visible |
| Review evidence is distributed | Open process debt | V4 progress, audit, validation, and runtime-evolution evidence had to be reconstructed across several runtime files |

## Deferred To V5

| Debt item | Required V5 direction | Rationale |
| --- | --- | --- |
| Version runtime structure | Use one Version Runtime Pack for all of V5 | A single pack should reduce capability-to-capability drift |
| Version progress tracking | Add `VERSION_PROGRESS.md` | V4 showed that version-level state needs a stable home |
| Capability progress tracking | Add `CAPABILITY_PROGRESS.md` | Capability-level audit and validation evidence should remain close to each capability |
| Version history | Add `VERSION_HISTORY.md` | Version evolution should not need to be reconstructed from commit history and progress notes |
| Metrics, drift, complexity, and debt tracking | Add Engineering Metrics, Architectural Drift Review, Complexity Budget, and Technical Debt Ledger | These artifacts should become first-class version-review inputs |
| Runtime Failure Path Audit | Keep as required version evidence | Failure-path invariants must remain explicit for future runtime changes |
| Cumulative local app smoke tests | Keep for every capability | V4 smoke testing found a real preview classification defect |
| Audit and HITL gates | Keep Codex audit for every capability, HITL after every audit, HITL before Runtime Evolution | Automation should assist review, not replace human quality judgment |
| Version-transition workflow | Replace with `VERSION_RUNTIME_PROMPT.md` | A stable version prompt should carry decisions between capabilities and versions |

## Deferred To V6

| Debt item | Status | Rationale |
| --- | --- | --- |
| No V6-specific blocker assigned during V4 review | Deferred placeholder | V6 debt should be assigned only after V5 runtime-pack execution reveals concrete version-level needs |

## Deferred To V7

| Debt item | Required V7 direction | Rationale |
| --- | --- | --- |
| Runtime Graph Decomposition naming | Rename optional roadmap item to Runtime Graph Consolidation | Junie recommended the rename to match the actual architectural need |
| Workflow graph maintainability | Include Workflow Node Handler Extraction and Runtime Graph Module Split in V7.1 | These are appropriate future consolidation tasks, not V4 blockers |
| Style and logging review | Keep Pydantic/Jinja2/Style Review, Code Style and Comment Quality Audit, and Logging Architecture Review in V7.3 | Loguru evaluation/adoption should occur only if justified |

## Debt Policy

Future dependency warnings and runtime warnings must be tracked in the Technical
Debt Ledger when they are accepted as non-blocking. Accepted warnings should not
disappear into smoke-test notes only.
