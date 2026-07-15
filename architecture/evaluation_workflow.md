# Evaluation Workflow

## Purpose

This map shows how evaluation evidence is produced, validated, stored, and
displayed. It separates a Dashboard runtime run from the explicit CLI
publication path so neither can be mistaken for the other. Blue is client work,
green is application runtime, yellow is storage, orange is an external provider
boundary, purple is evidence, and red is a blocked or rejected path.
The [standalone Mermaid source](evaluation_workflow.mmd) contains the same
diagram for slide and README reuse.

```mermaid
flowchart TB
    classDef client fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,stroke-width:1.5px
    classDef runtime fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20,stroke-width:1.5px
    classDef store fill:#fefce8,stroke:#a16207,color:#713f12,stroke-width:1.5px
    classDef external fill:#fff7ed,stroke:#c2410c,color:#7c2d12,stroke-width:1.5px
    classDef evidence fill:#ede9fe,stroke:#6d28d9,color:#4c1d95,stroke-width:1.5px
    classDef failure fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d,stroke-width:1.5px

    subgraph current_lane["1 · Dashboard current-product"]
        direction TB
        subgraph current_a[" "]
            direction LR
            dashboard["Dashboard<br/>Run Evaluation"]:::client
            api_gate["Preflight / privacy / provider gate<br/>live calls require explicit opt-in"]:::runtime
            frozen["Frozen benchmark<br/>7 canonical cases<br/>subsets stay unscored"]:::runtime
        end
        subgraph current_b[" "]
            direction LR
            dry["Prepared / unscored<br/>no provider-bound stack"]:::evidence
            live["Current AssistantService<br/>official retrieval ↔ Chroma<br/>context + Jinja prompt"]:::runtime
            evaluation["OpenAI generation + evaluator LLM / embeddings<br/>Explain final skips artifact / preview / review<br/>→ five RAGAS metrics"]:::external
            outcome["Public-safe projection → strict client validation<br/>exact valid → current primary; else unscored<br/>rolling last-24 session history"]:::evidence
        end
        stopped["Blocked / failed<br/>no fabricated score"]:::failure

        dashboard --> api_gate --> frozen
        frozen -->|"dry"| dry --> outcome
        frozen -->|"authorized live"| live --> evaluation --> outcome
        api_gate -. "invalid request" .-> stopped
        live -. "source / privacy" .-> stopped
        evaluation -. "provider / evaluator" .-> stopped
        stopped -. "terminal attempt" .-> outcome
    end

    subgraph local_lane["2 · Full scope local snapshots — no provider or RAGAS calls"]
        direction LR
        full["Full or local-only scope"]:::client
        creative["Creative artifact<br/>snapshot"]:::evidence
        workflow["Workflow<br/>snapshot"]:::evidence
        reliability["Product reliability<br/>snapshot"]:::evidence
        local_history["Benchmark wrapper → rolling history<br/>no cross-category total"]:::store

        full --> creative --> local_history
        full --> workflow --> local_history
        full --> reliability --> local_history
    end

    subgraph canonical_lane["3 · Explicit CLI canonical publication — never triggered by Dashboard"]
        direction LR
        cli["CLI<br/>--publish-canonical"]:::client
        cli_result["Same current-product evaluator<br/>raw result process-local by default"]:::runtime
        provenance["Deep provenance + arithmetic<br/>+ public-safe schema gate"]:::runtime
        canonical["Committed public-safe<br/>canonical JSON"]:::store
        static_evidence["Static Dashboard evidence<br/>on build / reload"]:::evidence
        publish_rejected["Rejected<br/>canonical unchanged"]:::failure
        private["Manual private diagnostics<br/>data/eval/ · gitignored"]:::store

        cli --> cli_result --> provenance -->|"pass"| canonical --> static_evidence
        provenance -. "reject" .-> publish_rejected
        cli_result -. "explicit private output only" .-> private
    end

    subgraph historical_lane["4 · Direct / manual historical fixture — not selectable in Dashboard UI"]
        direction LR
        historical_entry["historical_fixture<br/>direct API / manual"]:::client
        fixtures["Approved synthetic / redacted<br/>recorded public rows"]:::store
        historical_metrics["Four RAGAS metrics<br/>recorded answer + context<br/>no current product stack"]:::external
        historical_evidence["Historical comparison only<br/>history-only · never current primary"]:::evidence

        historical_entry --> fixtures --> historical_metrics --> historical_evidence
    end

    current_lane ~~~ local_lane
    local_lane ~~~ canonical_lane
    canonical_lane ~~~ historical_lane

    style current_a fill:none,stroke:none
    style current_b fill:none,stroke:none
```

## Key properties

- The Dashboard live lane uses the current `AssistantService`, real official-doc
  retrieval, local Chroma, the current Jinja prompt renderer, and OpenAI
  generation. Its Explain requests finalize before artifact extraction,
  preview, critique, review, or refinement.
- The five current-product metrics are context precision, faithfulness, answer
  relevancy, context relevancy, and context recall. Only context precision and
  context recall consume the authored reference.
- A safe runtime result can become the current Dashboard score after strict
  client validation. It is also added to a rolling 24-entry session history;
  this is not canonical publication or an append-only audit log.
- Full scope adds three local workspace snapshots. Those lanes do not invoke
  the provider or RAGAS, and the wrapper deliberately has no cross-category
  product score.
- Historical fixtures evaluate recorded public rows with four metrics and are
  comparison-only. They use the same metric set without context recall and do
  not exercise the current retrieval, prompt, or generation stack.

## Truth boundary

- The Dashboard UI submits only `current_product`; `historical_fixture` is an
  explicit API or manual lane.
- A dry run fingerprints the frozen selection but stops before constructing or
  invoking the provider-bound retrieval, generation, embedding, and evaluator
  stack.
- Questions, references, generated answers, and retrieved excerpts remain out
  of the public-safe API projection and committed canonical artifact.
- Only the explicit CLI `--publish-canonical` path can replace the committed
  canonical JSON. The Dashboard reads that file as static evidence on build or
  reload; a Dashboard run never writes it.
- Exact diagnostics require a separate manual option below gitignored
  `data/eval/` and are never Dashboard-default or public evidence.

## Deeper links

- [Evaluation API and async job registry](../src/creative_coding_assistant/api/evaluation.py)
- [Current-product runner](../src/creative_coding_assistant/eval/current_product.py)
- [Explicit publication and private-diagnostic CLI](../src/creative_coding_assistant/eval/current_product_cli.py)
- [RAGAS metric wiring](../src/creative_coding_assistant/eval/ragas_runner.py)
- [Dashboard run and polling client](../clients/nextjs/src/components/workstation-shell.tsx)
- [Strict client evidence validation](../clients/nextjs/src/lib/evaluation-benchmark.ts)
- [Static canonical evidence import](../clients/nextjs/src/lib/current-ragas-evidence.ts)
- [Committed canonical evidence](../demo/evaluation/current_product_ragas_evidence.json)
- [Canonical public-safe schema](../demo/evaluation/current_product_ragas_evidence.schema.json)
- [End-to-End Product Workflow](end_to_end_product_workflow.md)
