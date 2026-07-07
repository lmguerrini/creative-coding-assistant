# Orchestration Package Boundaries

V7.6 decomposes `creative_coding_assistant.orchestration` into explicit boundary
packages while preserving the existing public import surface. The change is a
package navigation refactor only: it does not change workflow order, provider
routing, prompt rendering, persistence ownership, telemetry emission, retries,
or generated output semantics.

## Canonical Packages

- `orchestration.runtime`: live runtime adapters and direct execution surfaces,
  including service composition, routing, workflow state, prompt rendering,
  retrieval, memory adapters, event models, and cache helpers.
- `orchestration.metadata`: passive registries, profiles, descriptors, creative
  metadata, cognitive metadata, and studio/workspace metadata models.
- `orchestration.governance`: passive policy, HITL, safety, budget, evolution,
  and governance posture models.
- `orchestration.audit`: passive audit, architecture consistency, release
  review, and failure-path verification models.
- `orchestration.contracts`: public contract, compatibility, taxonomy, and
  consolidation records.
- `orchestration.advisory`: passive planning, diagnostics, analytics,
  optimization, learning, recommendation, prediction, and research models.

## Compatibility Contract

All legacy imports of the form
`creative_coding_assistant.orchestration.<module>` remain valid. Each legacy
module is now a small shim that aliases the corresponding canonical module in
`sys.modules`, so direct module imports, root package exports, monkeypatch paths,
and existing tests continue to resolve through the same public names.

New code should import implementation modules from their canonical boundary
package when ownership matters. Public root exports from
`creative_coding_assistant.orchestration` remain supported for compatibility and
for stable API consumption.

## Runtime Boundary

Only modules in `orchestration.runtime` are allowed to own live orchestration
behavior. Passive packages can derive, describe, validate, or recommend, but
must not mutate runtime graph execution, provider/model routing, persistence,
streaming contracts, or generated output behavior as part of import or metadata
construction.

## Validation Expectations

V7.6 must keep the compatibility shims importable, keep root package exports
stable, compile all moved modules, and pass the backend test suite. Frontend
typecheck and Vitest are required only if frontend code changes.
