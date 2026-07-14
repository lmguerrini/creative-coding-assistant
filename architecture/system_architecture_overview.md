# System Architecture Overview

## Purpose

This is the primary reviewer-facing map of the current V9 product. It shows the
browser workspace, Python boundary, compiled request workflow, local stores,
provider calls, preview runtime, evaluation path, and evidence surfaces without
expanding every internal node.

```mermaid
flowchart TB
    classDef client fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef store fill:#FEF3C7,stroke:#A16207,color:#713F12,stroke-width:1.5px
    classDef external fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px
    classDef evidence fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px

    user["Creative coder / reviewer"]:::client

    subgraph request["Next.js workspace — request"]
        direction LR
        composer["Creative Session<br/>prompt · mode · attachments"]:::client
    end

    subgraph backend["Python API and orchestration"]
        direction TB
        api["Exact-path WSGI API"]:::runtime
        workflow["Compiled LangGraph"]:::runtime
        artifact["Artifact + preview contracts"]:::runtime
        services["Session + KB APIs"]:::runtime
        evaluation["Evaluation + RAGAS pipeline"]:::runtime

        api --> workflow --> artifact
        api --> services
        api --> evaluation
    end

    subgraph result["Next.js workspace — finalized result"]
        direction LR
        hydrate["Hydrated answer + artifacts"]:::client
        preview["Controlled preview"]:::client
        surfaces["Dashboard + Inspector"]:::evidence

        hydrate --> preview --> surfaces
        hydrate --> surfaces
    end

    subgraph state["Local state — separate persistence boundaries"]
        direction LR
        official_chroma[("Chroma<br/>official docs")]:::store
        memory_chroma[("Chroma<br/>memory collections")]:::store
        sqlite[("SQLite<br/>workspace sessions")]:::store
        browser_cache[("localStorage<br/>workspace fallback")]:::store
        artifact_files[("Files<br/>artifacts")]:::store
        eval_files[("JSON / JSONL<br/>eval evidence")]:::store

        official_chroma ~~~ memory_chroma ~~~ sqlite ~~~ browser_cache ~~~ artifact_files ~~~ eval_files
    end

    subgraph providers["Explicit external boundaries"]
        direction LR
        openai["OpenAI<br/>Responses + embeddings"]:::external
        official["Approved official URLs"]:::external
        langsmith["LangSmith<br/>optional trace metadata"]:::external

        openai ~~~ official ~~~ langsmith
    end

    user --> composer
    composer -->|"JSON request"| api
    api -->|"NDJSON event stream"| hydrate

    workflow -. "retrieval" .-> official_chroma
    workflow -. "memory" .-> memory_chroma
    services -.-> official_chroma
    services -.-> sqlite
    hydrate -. "session JSON API" .-> api
    hydrate -.-> browser_cache
    hydrate -. "browser export" .-> artifact_files
    evaluation -.-> eval_files

    workflow --> openai
    services --> openai
    services --> official
    evaluation --> openai
    workflow -. "when enabled" .-> langsmith

    eval_files ~~~ openai
```

## What the reviewer should notice

- The browser uses the local HTTP API; it never calls a model or embedding
  provider directly.
- The request graph, KB/evaluation actions, browser preview, and workspace
  persistence are related product paths with different owners.
- Dashboard and Inspector project published run, preview, persistence, and
  evaluation evidence; they are not additional workflow agents.

Blue nodes are browser surfaces, green nodes are backend runtime boundaries,
yellow cylinders are local stores, orange nodes are external systems, and
purple nodes are evidence projections. Solid arrows show calls or data flow;
the dashed LangSmith edge is optional.

## Truth boundary

The backend prepares artifact and preview contracts, but executable preview
source runs later inside the controlled browser surface. Evaluation uses a
separate API pipeline and is not a hidden LangGraph node. OpenAI is the only
configured generation-provider route; model/provider matrices elsewhere in the
repository remain advisory unless runtime evidence says otherwise.

## Deeper documentation

- [End-to-End Product Workflow](end_to_end_product_workflow.md)
- [Exact Runtime Routes](workflow_graph.md)
- [Artifact and Preview Runtime](artifact_preview_runtime.md)
- [Evaluation / RAGAS Workflow](evaluation_workflow.md)
- [System Overview](../docs/SYSTEM_OVERVIEW.md)
