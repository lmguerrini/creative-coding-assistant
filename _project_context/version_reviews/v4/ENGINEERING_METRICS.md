# V4 Engineering Metrics

This file records engineering-review metrics for the completed V4 effort. It is
an engineering artifact for V5 runtime-pack design. It is not product
documentation, a runtime file, or an application behavior contract.

## Scope

V4 completed six cumulative capability tracks:

- V4.1 Multi-Agent Core
- V4.2 Agent Orchestration
- V4.3 Hybrid Agentic Workflow
- V4.4 Hybrid Studio
- V4.5 Multimodal Studio
- V4.6 Agentic Studio Hardening

The version remained intentionally passive. V4 added metadata, inspection,
audit, and documentation surfaces without changing provider/model routing,
workflow behavior, runtime selection, retry behavior, artifact execution, or
generated output mutation.

## Version Metrics

| Metric | Value | Source / Notes |
| --- | ---: | --- |
| Completed V4 capabilities | 6 | V4.1 through V4.6 |
| Post-review commits on `version-review/v4` before these artifacts | 1 | `3726eacd Stabilize p5 HTML preview guard` |
| Backend Python source files | 203 | `src/creative_coding_assistant/**/*.py` |
| Frontend TypeScript / TSX source files | 134 | `clients/nextjs/src/**/*.ts(x)` |
| Backend test files | 191 | Top-level `tests/test_*.py` |
| Frontend test files | 58 | `clients/nextjs/src/**/*.test.ts(x)` |
| Architecture files | 11 | `architecture/` |
| Product docs plus architecture files | 24 | `docs/` and `architecture/` |
| Exported backend API modules | 28 | Modules declaring `__all__` |
| Registry-specific source/test files by path scan | 10 | Files with `registry` or `registries` in source/test/doc paths |
| V4.1 registry rows | 6 | Multi-Agent Core registry table |
| V4.2 registry rows | 14 | Agent Orchestration registry table |
| V4.3 source registries | 43 | Hybrid Workflow Integration Registry |
| V4.4 source registries | 17 | Hybrid Studio Integration Registry |
| V4.5 source registries | 14 | Multimodal Studio Integration Registry |
| V4.6 hardening/audit registry rows | 17 | Agentic Studio Hardening registry table |
| V4.6 audited agent registry records | 20 | Agent Registry Audit Registry |

## Validation Metrics

| Validation surface | Result |
| --- | --- |
| Backend full pytest | 1296 passed, 1 warning, 382 subtests passed |
| Frontend full Vitest | 390 passed |
| Frontend typecheck | Passed |
| Documentation alignment tests | 13 passed, 1 warning |
| Focused preview regression tests | 33 passed |
| Focused V4 smoke registry/failure validation | 191 passed, 1 warning, 20 subtests passed |
| Full V4 local app smoke test | Passed |
| Post-preview-fix local app smoke test | Passed |

## Smoke-Test Coverage

The real local app was exercised through the cumulative V4 smoke workflow:

- backend started on `127.0.0.1:8000`
- frontend started on `127.0.0.1:3000`
- localhost loaded successfully
- main app rendered
- a real UI workflow reached finalization
- browser console had no fatal errors
- backend logs had no fatal runtime errors
- V4.1 through V4.6 surfaces were visible, exposed, or inspectable through UI,
  registry, API, test, or source-inspection paths as applicable
- Runtime Failure Path Audit invariants remained true
- provider/model routing remained unchanged
- passive registries remained passive
- generated outputs were not mutated
- servers stopped cleanly and ports closed

## Audit Metrics

| Audit | Score | Blocking issues | Version-scoped fixes |
| --- | ---: | ---: | --- |
| Codex Grand Engineering Audit | 9/10 | 0 | 1 post-smoke fix approved by HITL |
| Junie Grand Engineering Audit | 9/10 | 0 | 0 |

The approved post-smoke Version-Scoped Fix was the p5 HTML preview guard
stabilization. It classified HTML documents beginning with leading HTML comments
as HTML and rejected them before p5 JavaScript sandbox execution.

## Accepted Non-Blocking Warnings

- Chroma deprecation warning
- Vite CJS deprecation warning
- Next.js dev cross-origin warning
- backend `KeyboardInterrupt` during intentional Ctrl-C shutdown

## Notable Engineering Statistics

- V4 expanded the passive registry and audit surface across six capabilities
  while preserving the V3 workflow backbone.
- V4.3 through V4.5 integration registries made 74 cumulative source-registry
  references inspectable across hybrid workflow, hybrid studio, and multimodal
  studio layers.
- V4.6 added hardening/audit coverage for agent contracts, escalation policy,
  hybrid workflow readiness, registry discoverability, blackboard boundaries,
  shared context boundaries, collaboration, creative diversity, explainability,
  reliability, determinism, telemetry, cost, performance, architecture
  consistency, final hardening, and LangGraph error-path audit surfaces.
- The only approved V4 fix after the Grand Engineering Audit was localized to
  preview source classification and did not change provider/model routing,
  workflow behavior, generated output, or registry behavior.
