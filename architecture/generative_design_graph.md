# Generative Design Dependency Graph

This document is the developer inspection view for the V3.2 Generative Design
Core. It focuses on the six V3.2 capabilities and how they extend stored V3.1
Creative Cognition metadata inside the single `planning` runtime node.

It is the dense companion to:

- [workflow_graph.md](workflow_graph.md) and
  [workflow_graph.mmd](workflow_graph.mmd), which document the real LangGraph
  runtime graph
- [creative_intelligence_graph.md](creative_intelligence_graph.md) and
  [creative_intelligence_graph.mmd](creative_intelligence_graph.mmd), which
  provide the readable capability pipeline
- [artifact_intelligence_graph.md](artifact_intelligence_graph.md) and
  [artifact_intelligence_graph.mmd](artifact_intelligence_graph.mmd), which
  document the downstream V3.3 Artifact Intelligence pipeline

## Scope And Boundary

- The graph below shows important internal dependencies, not every possible edge
- The dependency matrix below is the preferred way to show dense dependencies
- V3.2 capabilities are internal deterministic helpers, not separate LangGraph
  nodes with their own retries or failure routing
- V3.2 remains metadata and design guidance, not code generation execution,
  runtime mutation, provider routing, or preview behavior changes
- The current structure feeds the V3.3 Artifact Intelligence stack, but it is
  still a future V4 multi-agent blueprint rather than an implemented
  multi-agent runtime

```mermaid
flowchart LR
    classDef upstream fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef bundle fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px;
    classDef design fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:1.5px;
    classDef store fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px;
    classDef consumer fill:#F3E8FF,stroke:#7E22CE,color:#4C1D95,stroke-width:1.5px;
    classDef note fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px,stroke-dasharray: 6 4;

    subgraph upstream_core["Upstream V3.1 Creative Cognition outputs"]
        direction TB
        brief["Brief layer<br/>Creative Intent Decomposer<br/>Creative Hierarchy Planner<br/>Creative Strategy Engine<br/>Creative Technique Selector"]:::upstream
        execution["Execution layer<br/>Creative Execution Plan<br/>Creative Constraint Solver<br/>Runtime Capability Reasoner<br/>Creative Trade-off Explorer<br/>Creative Constraint Prioritizer<br/>Creative Quality Predictor"]:::bundle
        story["Structure layer<br/>Symbolic Narrative Planner<br/>Creative Composition Planner"]:::upstream
    end

    subgraph design_core["V3.2 Generative Design Core"]
        direction TB
        procedural["Procedural Structure Planner"]:::design
        generative["Generative Structure Engine"]:::design
        motif["Semantic Motif Engine"]:::design
        emotion["Emotional Consistency Engine"]:::design
        cross["Cross-Modality Composer"]:::design
        scene["Audio-Visual Scene System"]:::design
    end

    metadata_store["Metadata Store<br/>AssistantWorkflowState + PromptInputResponse"]:::store
    director["Creative Assistant Director runtime node"]:::consumer
    reasoning["Creative Reasoning Engine runtime node"]:::consumer
    prompt_rendering["Prompt rendering runtime node"]:::consumer
    note["Selective graph only<br/>Use the matrix below for exhaustive reads"]:::note

    brief --> procedural
    execution --> procedural
    story --> procedural

    execution --> generative
    story --> generative
    procedural --> generative

    story --> motif
    procedural --> motif
    generative --> motif

    execution --> emotion
    story --> emotion
    procedural --> emotion
    generative --> emotion
    motif --> emotion

    execution --> cross
    story --> cross
    procedural --> cross
    generative --> cross
    motif --> cross
    emotion --> cross

    execution --> scene
    story --> scene
    procedural --> scene
    generative --> scene
    motif --> scene
    emotion --> scene
    cross --> scene

    procedural -.-> metadata_store
    generative -.-> metadata_store
    motif -.-> metadata_store
    emotion -.-> metadata_store
    cross -.-> metadata_store
    scene -.-> metadata_store

    metadata_store --> director
    metadata_store --> reasoning
    metadata_store --> prompt_rendering
    director --> reasoning --> prompt_rendering
    note -.-> design_core
```

The raw Mermaid source for this detailed dependency view is available in
[generative_design_graph.mmd](generative_design_graph.mmd).

## Why The Graph Is Selective

- The graph emphasizes major shaping edges instead of every argument in every
  function call
- Important shared inputs such as `request`, `route_decision`, and
  `creative_translation` are omitted from the drawing to preserve readability
- The exact planning-time read sets are listed in the matrix below

## Dependency Matrix

The dependency matrix is the preferred way to show dense dependencies.

| Capability | Reads | Produces | Used by |
| --- | --- | --- | --- |
| `Procedural Structure Planner` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`<br/>`runtime_capabilities`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition` | `procedural_structure` | `Generative Structure Engine`, `Semantic Motif Engine`, `Emotional Consistency Engine`, `Cross-Modality Composer`, `Audio-Visual Scene System`, metadata store |
| `Generative Structure Engine` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`<br/>`runtime_capabilities`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition`, `procedural_structure` | `generative_structure` | `Semantic Motif Engine`, `Emotional Consistency Engine`, `Cross-Modality Composer`, `Audio-Visual Scene System`, metadata store |
| `Semantic Motif Engine` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition`, `procedural_structure`, `generative_structure` | `semantic_motif` | `Emotional Consistency Engine`, `Cross-Modality Composer`, `Audio-Visual Scene System`, metadata store |
| `Emotional Consistency Engine` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`<br/>`runtime_capabilities`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition`, `procedural_structure`, `generative_structure`, `semantic_motif` | `emotional_consistency` | `Cross-Modality Composer`, `Audio-Visual Scene System`, metadata store |
| `Cross-Modality Composer` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`<br/>`runtime_capabilities`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition`, `procedural_structure`, `generative_structure`, `semantic_motif`, `emotional_consistency` | `cross_modality` | `Audio-Visual Scene System`, metadata store |
| `Audio-Visual Scene System` | `request`, `route_decision`, `creative_translation`<br/>`creative_intent`, `creative_hierarchy`, `creative_plan`<br/>`creative_constraints`, `creative_constraint_priorities`<br/>`creative_strategy`, `creative_techniques`<br/>`runtime_capabilities`, `creative_tradeoffs`, `creative_quality_prediction`<br/>`symbolic_narrative`, `creative_composition`, `procedural_structure`, `generative_structure`, `semantic_motif`, `emotional_consistency`, `cross_modality` | `audio_visual_scene` | V3.3 Artifact Intelligence stack, metadata store, `Creative Assistant Director runtime node`, `Creative Reasoning Engine runtime node`, `prompt rendering runtime node` |

## Downstream Consumption

- All six V3.2 outputs are stored on `AssistantWorkflowState` and mirrored into
  `PromptInputResponse`
- `Creative Assistant Director runtime node` reads the stored V3.2 metadata
  after `planning` completes
- `Creative Reasoning Engine runtime node` reads the stored V3.2 metadata after
  the Director brief is available
- `prompt rendering runtime node` serializes the stored V3.2 metadata into
  dedicated prompt sections alongside the V3.1 cognition metadata
- The V3.3 Artifact Intelligence stack reads the stored V3.1 and V3.2
  metadata inside the same `planning` runtime node, then contributes artifact
  planning, compatibility, critique/refinement, merge, export, and engine
  contract metadata to workflow serialization and stream hydration
