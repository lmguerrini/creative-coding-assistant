# Creative Knowledge Distillation

V8.1 adds a bounded creative knowledge distillation surface for Creative Coding
Assistant. It converts existing repository contracts, approved source registry
entries, capstone retrieval demo scenarios, and read-only local Chroma inventory
into typed creative knowledge records.

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

## Implemented V8.1 Surfaces

- Creative technique, workflow, pattern, taxonomy, and best-practice records.
- KB reality snapshots for registry-vs-indexed coverage.
- Repository/documentation relationship graph support.
- Creative knowledge relationships across shared sources and domains.
- Demo-domain hardening recommendations for focused sync and evaluation.

## Deferred Boundaries

- PDF and paper distillation remain deferred until a scoped ingestion path is
  approved.
- Demo KB expansion and domain sync remain recommendations unless a focused
  sync/reindex action is explicitly authorized.
- Registry entries are not treated as indexed KB evidence.
