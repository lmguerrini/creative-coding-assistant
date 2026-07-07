# Architecture Decisions

## Backend Ownership

The Python backend is the source of truth for request handling, retrieval,
memory assembly, planning metadata, provider calls, artifact metadata, critique,
bounded refinement, finalization, and failure handling.

## Workflow Boundary

The application uses a compact LangGraph workflow for the user-facing creative
pipeline. Helper modules may provide metadata, validation, diagnostics, or
contract surfaces, but they should not create hidden workflow execution paths
or change provider routing without an explicit product change.

## Frontend Ownership

The Next.js workstation owns browser interaction, preview surfaces, artifact
inspection, comparison, export preparation, workflow visibility, telemetry
display, and operator controls. The frontend should consume backend contracts
rather than reimplement backend planning or retrieval behavior.

## Persistence

Chroma is the retrieval and memory database. Workspace session persistence uses
the backend workspace-session boundary. New persistent stores require an
explicit product architecture decision and migration path.

## API And Deployment

The production backend surface is WSGI. Local development may use the packaged
development server, while production deployment should use a WSGI host such as
Gunicorn with explicit production CORS configuration.

## Public And Private Separation

Public architecture docs describe product behavior and operator expectations.
Private engineering ledgers, audits, prompts, historical review material, and
version-review drafts belong in `.runtime_pack/` and must remain ignored by Git.
