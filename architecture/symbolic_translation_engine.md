# Symbolic Translation Engine

The Symbolic Translation catalog defines a bounded guidance engine. It
translates user-visible symbolic, artistic, mythopoetic, geometric, ritual,
aesthetic, and conceptual intent into structured creative coding guidance.

> **Reference notice:** This is a deterministic guidance-contract reference,
> not a separate runtime subsystem or delivery plan. For executable
> behavior, see [System Architecture Overview](system_architecture_overview.md)
> and [Single and Multi Runtime Routes](workflow_graph.md).

The implementation lives in
`src/creative_coding_assistant/knowledge/symbolic_translation.py`.

## Runtime Boundary

- Translation is deterministic, local, and inspectable.
- Motifs map to visual structure, motion behavior, audio mapping, runtime
  families, parameter names, composition guidance, and safety boundaries.
- V3 Creative Translation, Semantic Motif, and Symbolic Narrative outputs can
  be reused as source evidence when available.
- Creative Knowledge Distillation records are reused as provenance and
  confidence signals where relevant.
- The module does not mutate the LangGraph workflow, route providers/models,
  change prompt rendering, write storage, mutate preview runtime, create UI, or
  execute generated artifacts.
- The module does not implement HoloMind, HOLOiVERSE, or the Sacred Geometry
  engine.

## Cataloged Symbolic Translation Surfaces

- Typed motif, operation, provenance, confidence, capability-assessment, and
  report contracts.
- Bounded motif-to-creative-coding mappings for supported symbols such as
  spiral, mandala, phoenix, labyrinth, threshold, mirror, seed, wave, pulse,
  grid, network, void, flame, tree, and related operational motifs.
- Operational guidance for visual structure, motion, audio, runtime families,
  parameter mapping, composition, and interpretation boundaries.
- Creative Knowledge Distillation provenance/confidence integration for symbolic and operational
  creative-knowledge records.
- Prompt-line rendering that exposes boundaries, evidence, guidance, risks, and
  HITL questions.
- Capability-boundary classification for all Symbolic Translation items.

## Reused Systems

- V3 Creative Translation remains the base request-to-guidance surface.
- V3 Semantic Motif remains the richer motif hierarchy and recurrence surface.
- V3 Symbolic Narrative remains the phased symbolic-arc surface.
- Creative Knowledge Distillation remains the provenance and confidence source
  for repository and demo-domain knowledge records.

Symbolic Translation composes these systems into an operational translation report instead of
duplicating their responsibilities.

## Excluded Boundaries

- Comparative tradition interpretation requires HITL and explicit scoped
  sources.
- Universal symbol dictionaries are intentionally not implemented because they
  would overclaim symbolic authority.
- Mystical correspondence, hermeneutic reasoning, and initiatory consistency
  validation remain excluded unless a separate scoped decision defines evidence,
  wording, review ownership, and safety constraints.
- Symbol evolution and historical interpretation are not implemented.
- Frontend surfacing is outside this contract and requires a product-scoped UI
  decision.
