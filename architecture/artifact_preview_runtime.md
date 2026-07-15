# Artifact and Preview Runtime Workflow

## Purpose

This diagram separates artifact handling inside the backend request from actual
preview execution in the browser. That split is essential for interpreting
“preview prepared,” runtime telemetry, critique, persistence, and export
truthfully. The [standalone Mermaid source](artifact_preview_runtime.mmd)
contains the same diagram for full-size rendering.

```mermaid
flowchart TB
    classDef client fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef store fill:#FEF3C7,stroke:#A16207,color:#713F12,stroke-width:1.5px
    classDef evidence fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px

    subgraph server["Backend lane — before final event"]
        direction LR
        response["Generated response"]:::runtime
        explain{"Explain route?"}:::decision
        extract["Artifact extraction"]:::runtime
        prepare["Classify runtime +<br/>prepare preview metadata"]:::runtime
        mode{"Resolved mode?"}:::decision
        critique["Deterministic artifact<br/>critique + review"]:::runtime
        final["Finalization + artifact /<br/>preview contracts"]:::runtime

        response --> explain
        explain -->|"no"| extract -->|"artifact or explicit skip"| prepare --> mode
        explain -->|"yes"| final
        mode -->|"Single"| final
        mode -->|"Multi"| critique --> final
    end

    subgraph browser["Browser lane — after stream hydration"]
        direction LR
        hydrate["Hydrate artifact +<br/>preview state"]:::client
        preflight{"Renderer + source<br/>contract?"}:::decision
        preview["Sandboxed iframe →<br/>live preview"]:::client
        telemetry["Status +<br/>frame telemetry"]:::evidence
        fallback["Visible source /<br/>fallback"]:::evidence
        inspect["User inspection"]:::evidence

        hydrate --> preflight
        preflight -->|"runnable"| preview --> telemetry --> inspect
        preflight -->|"unavailable"| fallback --> inspect
    end

    subgraph outputs["Preserved outputs"]
        direction LR
        projections["Runtime · Preview<br/>Dashboard · Inspector"]:::evidence
        persistence[("Workspace persistence")]:::store
        export["Copy / download /<br/>handoff export"]:::evidence

        projections ~~~ persistence ~~~ export
    end

    final -->|"final event"| hydrate
    inspect --> projections
    inspect --> persistence
    inspect --> export
```

## Key properties

- Explain finalizes without entering artifact handling. Other routes extract
  code, infer artifact/runtime metadata, and prepare a `PreviewResult` or an
  explicit skip.
- Single finalizes after preview preparation. Only Multi continues through
  deterministic artifact critique/review before finalization.
- The client rechecks the renderer route and source contract before mounting an
  iframe with `sandbox="allow-scripts"`. Runtime status and frame samples then
  become local evidence for Runtime, Preview, Dashboard, and Inspector views.
- The user can preserve the workspace or export the source whether preview
  runs or falls back.

## Truth boundary

Backend “preview succeeded” means preview metadata was prepared; it is not
proof that the browser rendered a frame. On Multi, backend critique consumes
that prepared metadata and therefore occurs before live runtime telemetry.
Telemetry does not reopen the same LangGraph review loop. The canonical generated
live-preview contracts are p5.js, Three.js, GLSL, and Tone.js. The client
contains additional bounded adapters, but their existence alone does not turn
every code-export domain into a validated generation-to-preview product claim.

Browser preflight or runtime failure keeps the answer and extracted source
inspectable, publishes an explicit local error/fallback state, and does not
rewrite the already-finalized backend result.

## Deeper documentation

- [Domain Experience](../docs/DOMAIN_EXPERIENCE.md)
- [End-to-End Product Workflow](end_to_end_product_workflow.md)
