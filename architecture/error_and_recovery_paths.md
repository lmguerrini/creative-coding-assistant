# Error and Recovery Paths

## Purpose

These diagrams separate recoverable conditions, bounded Multi-Agent refinement,
terminal backend failures, and post-final browser runtime failures. It shows
which conditions continue with explicit evidence and which conditions end the
workflow without hiding or upgrading the outcome.

### Backend validation and recoverable retrieval

```mermaid
flowchart TB
    classDef client fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef evidence fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    submit["Request + attachments"]:::client
    validate{"Validation + safety<br/>passes?"}:::decision
    rejected["HTTP 4xx<br/>visible request error"]:::failure
    retrieval{"Graph entered:<br/>retrieval evidence?"}:::decision
    recover["Recoverable event<br/>empty context"]:::evidence
    produce["Generation → extraction<br/>→ preview metadata or explicit skip"]:::runtime
    normalized["Provider or graph exception<br/>→ normalized terminal failure"]:::failure

    submit --> validate
    validate -->|"no"| rejected
    validate -->|"yes"| retrieval
    retrieval -->|"available"| produce
    retrieval -->|"missing / failed"| recover --> produce
    retrieval -. "uncaught graph exception" .-> normalized
    produce -. "generation / extraction / prep exception" .-> normalized
```

### Review, refinement, and exhaustion

```mermaid
flowchart TB
    classDef runtime fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef evidence fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    mode{"After preview metadata:<br/>resolved mode?"}:::decision
    review["Artifact critique + review"]:::runtime
    outcome{"Review outcome"}:::decision
    budget{"Retry allowed?<br/>under 2 actual passes"}:::decision
    refinement["Next pass: append guidance → regenerate<br/>→ extract → prepare → review"]:::runtime
    next_review(["↺ next review outcome<br/>same loop above"]):::decision
    exhausted{"At stop / exhaustion:<br/>required output absent?"}:::decision
    finalized{"Finalization status"}:::decision
    success["Completed final"]:::evidence
    partial["Partial / needs-refinement final"]:::evidence
    terminal["Terminal failed final"]:::failure

    mode -->|"Single"| finalized
    mode -->|"Multi"| review --> outcome
    outcome -->|"pass"| finalized
    outcome -->|"other quality stop"| partial
    outcome -->|"refine"| budget
    budget -->|"continue"| refinement
    refinement --> next_review
    budget -->|"stop"| exhausted
    exhausted -->|"no"| partial
    exhausted -->|"yes"| terminal
    finalized -->|"completed"| success
    finalized -->|"partial"| partial
    finalized -->|"semantic failure"| terminal
    review -. "critique / review exception" .-> terminal
```

### Post-final browser recovery

```mermaid
flowchart TB
    classDef client fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E,stroke-width:1.5px
    classDef decision fill:#F4F4F5,stroke:#52525B,color:#18181B,stroke-width:1.5px
    classDef store fill:#FEF3C7,stroke:#A16207,color:#713F12,stroke-width:1.5px
    classDef evidence fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px
    classDef failure fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-width:1.5px

    stream["NDJSON final stream"]:::evidence
    rejected["HTTP 4xx or<br/>unexpected stream break"]:::failure

    subgraph browser["After a successful or partial final"]
        direction LR
        hydrate["Hydrate answer<br/>+ artifact"]:::client
        preflight{"Preflight + runtime<br/>healthy?"}:::decision
        preview["Controlled preview"]:::client
        local_status["Running / complete<br/>local telemetry"]:::store
        local_failure["Preflight or runtime<br/>unavailable / failed"]:::failure
        preserved["Preserve result +<br/>truthful local status"]:::evidence

        hydrate --> preflight
        preflight -->|"yes"| preview --> local_status --> preserved
        preflight -->|"no"| local_failure --> preserved
    end

    stream --> hydrate
    rejected --> preserved
    preserved --> surfaces["Dashboard + Inspector<br/>truthful status"]:::evidence
```

## What the reviewer should notice

- Invalid request, attachment, or safety input returns an HTTP 4xx response
  before the LangGraph workflow starts.
- Retrieval failure and missing evidence are explicit recoverable states. They
  emit an event, use empty retrieval context, and continue without invented
  sources.
- No extracted artifact is an explicit skip, not an extraction exception.
  Single finalizes with the resulting semantic status; Multi review can still
  request a bounded regeneration.
- Review may run up to two actual refinement passes and may stop earlier when
  another pass would not help.
- Exhausting refinement for a required deliverable produces a terminal failed
  final. Exhausting a non-deliverable quality refinement preserves a partial,
  needs-refinement final.
- Browser preflight and runtime failures occur after the backend final. They
  preserve the finalized answer and artifact while publishing a truthful local
  runtime status.

## Truth boundary

The backend prepares preview metadata; it does not execute browser creative
code. Browser telemetry therefore does not reopen or automatically feed the
backend artifact-critique and review loop. A later user refinement is a new
explicit request. Likewise, an unexpected stream break is a recoverable client
transport condition, while a delivered terminal failed final is authoritative
backend workflow evidence.

## Failure classification

| Condition | Classification | Published result |
|---|---|---|
| Invalid request, attachment, or safety input | Pre-graph rejection | HTTP 4xx and visible request error |
| Retrieval gateway failure or missing evidence | Recoverable backend condition | Explicit event, empty context, continued workflow |
| No extracted artifact | Explicit skip | Preview preparation records the skip; Single finalizes truthfully and Multi may refine |
| Generation/provider or graph-node exception | Normalized terminal failure | Failure node and terminal failed final |
| Review requests another useful pass | Bounded retry | At most two actual refinements, with possible early stop |
| Required deliverable still absent at exhaustion | Terminal failure | Failed final with preserved evidence |
| Non-deliverable refinement stops or exhausts | Partial completion | Partial / needs-refinement final |
| Browser preflight or runtime failure | Post-final local runtime failure | Preserved answer/artifact and truthful local status |
| Unexpected stream break | Recoverable client condition | Client error state without a fabricated backend final |

## Deeper documentation

- [Runtime Workflow Graph](workflow_graph.md) documents registered nodes,
  route branches, and transition rules.
- [Architecture Walkthrough](../docs/ARCHITECTURE_WALKTHROUGH.md) follows one
  request from browser validation through recording and evaluation boundaries.
- [System Overview](../docs/SYSTEM_OVERVIEW.md) separates browser, backend,
  persistence, provider, and evaluation responsibilities.
- [Troubleshooting](../docs/TROUBLESHOOTING.md) provides operator-facing
  recovery guidance.
