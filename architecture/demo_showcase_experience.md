# V8.8 Demo Showcase Experience

V8.8 adds a capstone-facing demo and showcase preparation layer over existing
Creative Coding Assistant surfaces.

## Scope

- Golden demo flows
- Demo prompt library
- Capstone case alignment
- Manual demo checklist
- Showcase upload preparation
- Evaluation and metrics summary
- Ethical AI summary
- Offline/provider/retrieval/preview fallback runbook

## Implementation Boundary

The implementation is metadata and documentation only. It does not:

- execute providers
- execute retrieval
- render or repair previews
- route providers or models
- control workflows
- mutate prompts, artifacts, generated output, memory, or storage
- run external DCC or MCP tools
- implement HoloMind or HOLOiVERSE
- merge, push, tag, freeze, or start V8 Grand Review

## Source Surfaces

- `build_production_demo_asset_plan()`
- `build_production_creative_readiness_review()`
- `build_capstone_retrieval_demo_pack()`
- `docs/eval_pipeline.md`
- `assets/preview_current.png`
- `demo/golden_demo_dataset.json`

## Capstone Boundary

Primary alignment:

- Case 5: AI coding assistant for creative coding
- Case 1: RAG-powered knowledge assistant
- Case 6: advanced LLM tools

Guarded support:

- Case 2: bounded workflow explanation only
- Case 3: registered-source KB search only
