# V7.8 Workflow Runtime Decomposition - Capability Spec

## Purpose
Reduce the remaining maintainability risk concentrated in
`workflow_graph.py` by decomposing runtime workflow execution into clearer
runtime modules while preserving 100% user-visible and runtime behavior.

## Roadmap Contract
- Workflow Graph Decomposition.
- Runtime Node Extraction.
- `runtime/nodes/` architecture.
- LangGraph Node Handler Split.
- Node Registration Layer.
- Graph Builder Refactor.
- State Transition Isolation.
- Node Dependency Simplification.
- Runtime Module Boundary Enforcement.
- Import Graph Simplification.
- Workflow Runtime Documentation.
- Runtime Architecture Validation.

## Architecture Boundaries
- Runtime execution logic may move from
  `creative_coding_assistant.orchestration.runtime.workflow_graph` into
  `creative_coding_assistant.orchestration.runtime.nodes`.
- Compatibility imports and public workflow graph entry points must remain
  stable.
- The graph topology, node ordering, state transitions, contracts, provider
  routing, streaming behavior, workspace behavior, and generated outputs must
  remain unchanged.
- Compatibility shims may remain when they preserve stable public imports.

## Product Boundaries
- No user-visible behavior changes.
- No provider/model routing changes.
- No prompt, generation, streaming payload, workspace persistence, or output
  contract changes except for mechanical import relocation.
- Do not implement V7.9, V8, merge, push, tag, or freeze.

## Validation Contract
- Prove graph builds correctly.
- Prove topology and node ordering are unchanged.
- Prove state transition mappings are unchanged.
- Prove node registration works.
- Run `git diff --check`, Ruff, compileall, focused workflow/runtime tests,
  runtime graph tests, and streaming/workspace tests if affected.

## Runtime Evolution Contract
Runtime Evolution may be proposed if the decomposition exposes product-level
runtime design changes, but it must not be applied automatically during V7.8.
