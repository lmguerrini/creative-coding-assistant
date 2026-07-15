# Product Demo Guide

Creative Coding Assistant includes an in-product Demo Mode and a static local
launcher for reproducible fallback inspection. Both use committed prompts,
artifacts, and sanitized evidence; neither bypasses the normal product
boundaries.

## Start the product

From the repository root, start the local API:

```bash
.venv/bin/python -m creative_coding_assistant.api.dev_server \
  --host 127.0.0.1 \
  --port 8000
```

Start the workstation in a second terminal:

```bash
npm run dev --prefix clients/nextjs
```

Open `http://127.0.0.1:3000`, then choose Demo Mode from the workstation.
Live generation and retrieval require configured providers and a populated
local knowledge index; provider-unavailable states remain visible.

## Demo paths

The committed scenario set covers:

- p5.js, Three.js, GLSL, and Tone.js creative-code requests;
- Single Agent, Multi Agent, and Auto routing;
- official-source retrieval and provenance;
- optional image-reference transport;
- artifact inspection, bounded preview, refinement, and export; and
- explicit provider, retrieval, preview, and persistence failures.

Loading a scenario prepares inputs only. A scenario card, fixture, or expected
result does not prove that a provider call, retrieval, or browser runtime has
succeeded in the current session.

## Reproducible local assets

- [Demo pack](../demo/README.md)
- [Committed scenario dataset](../demo/golden_demo_dataset.json)
- [Prompt library](../demo/demo_prompt_library.md)
- [Golden runtime artifacts](../demo/golden_artifacts/README.md)
- [Evaluation methodology](eval.md)

The static launcher at `demo/final_demo_launcher.html` is a local fallback for
opening committed assets. It is not a hosted deployment and does not substitute
for current-product execution.

## Evidence boundaries

- Browser QA records describe the named local harness and dependency versions;
  they are not production performance guarantees.
- Sanitized evaluation fixtures are historical compatibility evidence unless
  explicitly identified as current-product evidence.
- Raw sessions, private evaluator payloads, local Chroma data, and workspace
  databases are excluded from the public demo pack.
- TouchDesigner, Houdini, Blender, Unreal Engine, and Unity remain continuation
  targets; CCA does not execute those applications.

See the [User Manual](USER_MANUAL.md),
[Architecture Diagram Guide](../architecture/README.md), and
[Ethics and Privacy Assessment](ETHICS_PRIVACY_ASSESSMENT.md) for the product
workflow and trust boundaries.
