# Creative Coding Assistant

Production-minded assistant for creative coding and generative visuals.

The implementation root is this outer `creative_coding_assistant/` directory.
`codex_starter_pack/` is retained only as local starter/reference material.

## Bootstrap Scope

This branch establishes the project foundation only:

- root project metadata
- canonical project docs in `docs/`
- backend package skeleton under `src/`
- Streamlit V1 client placeholder under `clients/streamlit/`
- tests, scripts, and runtime data directories
- exclusions for starter/reference material

Product features such as Chroma repositories, memory behavior, RAG sync,
assistant modes, preview rendering, live evaluation, and analytics are planned
for follow-up feature branches.

## Reference Material

The old implementation remains under `codex_starter_pack/old_project/` as
reference-only material. It is excluded from package discovery, test collection,
linting, and the application import path.
