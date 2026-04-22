# Architecture Decisions

## Core architecture
- Keep backend logic frontend-agnostic
- V1 client: Streamlit
- Future client: Next.js/TypeScript
- Core modules live outside the UI layer

## Persistence
Use Chroma only, but separate concerns by collection:
- kb_official_docs
- conversation_turns
- conversation_summaries
- project_memory
- eval_traces
- preview_artifacts_index

## UI principle
Streamlit is only the first client, not the application core.

## Key backend modules
- backend/rag
- backend/memory
- backend/tools
- backend/preview
- backend/eval
- backend/analytics
- backend/vectorstore

## Memory model
Three layers:
1. Recent turn memory
2. Running conversation summary
3. Persistent project memory

## Preview model
Controlled preview pipeline:
- generate preview-safe assets
- build HTML bundle
- render preview
- capture image
- export/download assets

## Code quality rules
- small modules
- one responsibility per file
- no giant hub files
- tests for every feature branch before merge
