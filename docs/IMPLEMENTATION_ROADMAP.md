# Implementation Roadmap

This roadmap summarizes public product capability areas. It is not a private
engineering ledger and does not start future version work.

## Delivered Product Foundation

- LangGraph-backed creative assistant workflow.
- Retrieval and memory context assembly.
- OpenAI provider integration through backend adapters.
- Artifact extraction and preview record generation.
- Critique, review, and bounded refinement surfaces.
- Browser workstation with streaming conversation, retrieval visibility,
  preview surfaces, artifact inspection, workflow visibility, and local
  persistence.
- WSGI backend bridge for local and production-style deployment.
- Deterministic backend, runtime, and frontend validation coverage.

## Public Maintenance Priorities

- Keep backend API, stream events, workspace sessions, and preview contracts
  stable.
- Keep product documentation focused on setup, operation, validation, and
  deployment.
- Keep CI focused on product quality gates: backend checks, selected runtime
  integration tests, dependency security, frontend type/unit checks, and
  Playwright smoke coverage.
- Keep local data, generated artifacts, evaluation outputs, and private
  engineering records out of Git.

## Future Product Candidates

Future capabilities should be described in public docs only after they are
approved for product work. Private planning, audits, prompts, engineering
ledgers, and version-review drafts belong in `.runtime_pack/`.
