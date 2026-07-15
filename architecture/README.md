# Architecture Diagram Guide

This index covers the executable product architecture: browser request and
preview surfaces, Python orchestration, local stores, provider boundaries,
evaluation, and the sequential Multi Agent roles.

## Primary diagrams

| View | Purpose | Important boundary |
|---|---|---|
| [System Architecture Overview](system_architecture_overview.md) | Locate the browser, backend, stores, providers, preview runtime, and evidence surfaces | Dashboard and Inspector are projections; evaluation is not a LangGraph node |
| [End-to-End Product Workflow](end_to_end_product_workflow.md) | Follow one request from validation to final browser state | Auto resolves to Single or Multi; browser preview begins after backend finalization |
| [Multi-Agent Role Flows](multi_agent_roles.md) | Inspect Planner, Researcher, Generator, Critic, and Reviewer responsibilities | Roles are sequential; only Generator owns a text-generation provider call |
| [Artifact and Preview Runtime](artifact_preview_runtime.md) | Separate backend artifact metadata from browser execution | Prepared preview metadata is not proof of a rendered frame |
| [Evaluation Workflow](evaluation_workflow.md) | Trace current-product, local snapshot, canonical, and historical lanes | Public-safe evidence excludes private questions, answers, references, and excerpts |

## Diagram notation

- Blue: browser or client input.
- Green: executable backend runtime.
- Yellow: local state or bounded source evidence.
- Orange: explicit external provider boundary.
- Purple: published evidence or terminal projection.
- Gray: decision or non-executing boundary.
- Red: rejected or failed state.
- Dashed edges: optional, advisory, or non-owning relationships.

The linked pages distinguish runtime behavior from metadata, projections, and
future-compatible contracts. Selected diagrams also link their standalone
`.mmd` source for full-size rendering outside Markdown.
