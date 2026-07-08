# Project Context

Creative Coding Assistant is a creative coding workstation that turns
conceptual, geometric, stylistic, audiovisual, and multimodal intent into
structured creative guidance, generated artifacts, previewable outputs,
critique, and refinement context.

## Product Surface

- Python backend for retrieval, memory, planning, provider execution,
  artifact extraction, preview records, critique, refinement, and final
  response assembly.
- Next.js workstation for chat, preview, inspection, retrieval visibility,
  workflow visibility, parameter review, and export-oriented artifact surfaces.
- Streamlit client retained as a lightweight local client.
- Chroma-backed local retrieval and memory storage under `data/`.

## Public Repository Scope

The public repository contains runnable product code, tests, CI, deployment
guidance, architecture diagrams, assets, and user/developer documentation.

Private engineering records are intentionally excluded from public tracking and
live in the local ignored `.runtime_pack/` directory.

## Runtime Boundary

The runtime is a bounded LangGraph workflow exposed through the backend service
and browser-facing API bridge. Product documentation should describe observable
behavior, deployment expectations, validation surfaces, and operational
boundaries without depending on private engineering ledgers.

## Local Data Boundary

Runtime data remains local and ignored:

- `data/chroma/`
- `data/artifacts/`
- `data/eval/`
- `data/workspace_sessions.sqlite3`

These paths are not part of the public source history.
