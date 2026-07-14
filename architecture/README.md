# Architecture Diagram Guide

This is the reviewer entry point for the current V9 architecture. Start with the
two overview diagrams, then open only the zoom that answers the question under
review. The primary suite describes executable product behavior; the reference
section describes passive metadata inventories and must not be read as extra
runtime agents, provider routes, or workflow nodes.

## Primary reviewer sequence

| View | Purpose | What to notice | Truth boundary |
|---|---|---|---|
| [System Architecture Overview](system_architecture_overview.md) | Locate the browser, backend, stores, providers, preview runtime, and evidence surfaces | Product paths have distinct owners and persistence boundaries | Dashboard and Inspector are projections; evaluation is not a LangGraph node |
| [End-to-End Product Workflow](end_to_end_product_workflow.md) | Follow one request from validation to final browser state | Auto resolves to Single or Multi; only generation calls the text provider | Browser preview and telemetry begin after backend finalization |
| [Single and Multi Runtime Routes](workflow_graph.md) | Compare the exact compiled graph branches | Multi roles are sequential deterministic responsibilities | Executable review permits two refinement attempts while the published plan still reports one |
| [Multi-Agent Role Flows](multi_agent_roles.md) | Zoom into Planner, Researcher, Generator, Critic, and Reviewer handoffs | Only Generator owns an OpenAI generation call | Role names do not imply five autonomous LLM workers |
| [Artifact and Preview Runtime](artifact_preview_runtime.md) | Separate backend artifact metadata from browser execution | Prepared preview metadata precedes finalization; live frame telemetry follows it | A prepared backend preview is not proof of a rendered browser frame |
| [Error and Recovery Paths](error_and_recovery_paths.md) | Distinguish recoverable, retry, terminal, and post-final failures | Retrieval can recover explicitly; refinement is bounded; local preview failure preserves the result | Browser failure does not reopen the completed backend review loop |
| [Evaluation Workflow](evaluation_workflow.md) | Trace current-product, local snapshot, canonical publication, and historical lanes | Dashboard runs and canonical publication are separate paths | Public-safe evidence excludes private questions, answers, references, and excerpts |

## Diagram notation

- Blue: browser/client input.
- Green: executable backend runtime.
- Yellow: local state or bounded source evidence.
- Orange: explicit external provider boundary.
- Purple: published evidence or terminal projection.
- Gray: decision or non-executing boundary.
- Red: rejected or failed state.
- Dashed edges: optional, advisory, or non-owning relationships.

Each primary page states its purpose, the reviewer takeaway, the implementation
truth boundary, and deeper source links. Standalone `.mmd` files are retained
for one-to-one diagrams rendered outside Markdown, including selected
reference-only inventories; each mirrors its corresponding Mermaid fence.

## Reference-only architecture inventories

The remaining architecture documents preserve typed capability, design, and
historical metadata inventories. They are useful for code navigation and future
planning, but they do not extend the compiled V9 workflow described above.

- [Engine and registry matrix](engine_matrix.md)
- [Creative-intelligence metadata](creative_intelligence_graph.md)
- [Generative-design metadata](generative_design_graph.md)
- [Artifact-intelligence metadata](artifact_intelligence_graph.md)
- [Workstation-surface metadata](workstation_surface_graph.md)
- [Creative knowledge distillation](creative_knowledge_distillation.md)
- [Demo showcase experience](demo_showcase_experience.md)
- [Hologenesis core](hologenesis_core.md)
- [Immersive audiovisual composer](immersive_audiovisual_composer.md)
- [Mythopoetic narrative engine](mythopoetic_narrative_engine.md)
- [Sacred architecture engine](sacred_architecture_engine.md)
- [Sacred geometry engine](sacred_geometry_engine.md)
- [Symbolic translation engine](symbolic_translation_engine.md)

For prose context, continue with the
[System Overview](../docs/SYSTEM_OVERVIEW.md) and
[Architecture Walkthrough](../docs/ARCHITECTURE_WALKTHROUGH.md).
