# Creative Knowledge Distillation

The V8.1 catalog defines a bounded creative knowledge distillation surface. It
converts existing repository contracts, approved source registry
entries, committed retrieval demo scenarios, and read-only local Chroma inventory
into typed creative knowledge records.

> **Reference notice:** This is a deterministic knowledge-contract reference,
> not the live retrieval path or a roadmap. For executable architecture, see
> [System Architecture Overview](system_architecture_overview.md) and
> [End-to-End Product Workflow](end_to_end_product_workflow.md).

The implementation lives in
`src/creative_coding_assistant/knowledge/creative_distillation.py`.

## Runtime Boundary

- Distillation is local and read-only.
- Registry coverage is reported separately from indexed KB coverage.
- Provenance records identify demo, repository, documentation, official-source,
  and indexed-KB evidence.
- Confidence scoring is deterministic and based on indexed-source ratio,
  provenance count, domain breadth, and extracted creative signals.
- The module does not fetch external sources, rebuild indexes, write Chroma,
  mutate source registries, change retrieval configuration, route providers,
  execute workflows, or implement HoloMind or HOLOiVERSE.

## Cataloged V8.1 Surfaces

- Creative technique, workflow, pattern, taxonomy, and best-practice records.
- KB reality snapshots for registry-vs-indexed coverage.
- Demo KB hardening manifests for focused sync readiness and blocked sync
  evidence.
- Repository/documentation relationship graph support.
- Creative knowledge relationships across shared sources and domains.
- Demo-domain hardening recommendations for focused sync and evaluation.

## Excluded Boundaries

- PDF and paper distillation are outside this contract unless a separate scoped
  ingestion path is approved.
- Demo KB expansion and domain sync remain recommendations unless a focused
  sync/reindex action is explicitly authorized.
- Registry entries are not treated as indexed KB evidence.
- No repository-standard `kb_manifest.json` path exists yet, so V8.1 records
  manifest state through typed code instead of writing ignored local data.
