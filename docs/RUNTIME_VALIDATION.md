# Runtime Validation

The validation strategy separates deterministic application checks from
provider-backed, networked, and human evaluation.

## Validation layers

Contract tests verify registry completeness, payload shapes, and passive
metadata without executing the user-facing runtime.

Runtime integration validation executes the compiled LangGraph workflow through
`AssistantService` with controlled memory, retrieval, prompt, generation, and
provider dependencies. It verifies node order, transitions, streamed events,
final payloads, and failure recovery without OpenAI calls, network access, or a
live vector database.

Browser and frontend checks cover request serialization, NDJSON hydration,
session persistence, supported preview paths, explicit runtime failures, and
the main workstation flows.

Release validation remains the broader release gate: static checks, Ruff,
compileall, focused and full backend tests, frontend lint/type/unit checks,
documentation and Mermaid validation, browser smoke tests, and CI.

## Primary deterministic coverage

- `tests/test_langgraph_workflow_integration.py`
- `tests/test_v7_9_runtime_validation_integration.py`
- `tests/test_workspace_session_persistence.py`
- `tests/test_nextjs_streaming_bridge.py`
- `tests/test_current_product_evaluation.py`
- `tests/test_multimodal_provider_inputs.py`
- `clients/nextjs/src/lib/assistant-stream.test.ts`
- `clients/nextjs/e2e/workstation-smoke.spec.js`

Together these cover graph construction, route-specific node order, prompt and
context assembly, controlled provider success/failure, WSGI streaming,
workspace lifecycle, multimodal payload transport, evaluation contracts, and
browser state transitions.

## External validation boundaries

Live generation, embedding refresh, official-source synchronization, RAGAS
scoring, optional tracing, load testing, and hosted security controls require
separate authorization and environment-specific evidence. Deterministic test
success does not establish provider availability, artistic quality, human
usability, accessibility conformance, or production deployment readiness.
