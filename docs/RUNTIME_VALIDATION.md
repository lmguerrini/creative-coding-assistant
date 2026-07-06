# Runtime Validation

V7.9 adds deterministic runtime integration validation for the real
HoloGenesis / Creative Coding Assistant execution path after the V7.8 workflow
decomposition. These checks are intentionally separate from passive
metadata/model validation.

## Validation Layers

Metadata/model validation proves registry completeness, contract shapes, and
advisory model payload stability without executing the user-facing runtime.

Runtime integration validation executes the compiled LangGraph workflow through
`AssistantService` with controlled memory, retrieval, prompt, generation, and
provider dependencies. It verifies real node order, state transitions, stream
events, final payloads, and failure recovery without OpenAI calls, network
access, vector databases, or external services.

Release validation remains the broader release gate: static checks, Ruff,
compileall, focused runtime tests, full backend pytest when feasible, frontend
validation when touched, smoke tests, and remote CI/tag verification.

Future production validation remains outside V7.9: live provider calls, live
network retrieval, external vector DB health, multi-user authorization, load
testing, managed backups, WAF/TLS controls, and enterprise SaaS hardening.

## V7.9 Runtime Risk Coverage

The dedicated integration layer is
`tests/test_v7_9_runtime_validation_integration.py`.

It covers:

- Full `AssistantService` execution through the compiled LangGraph workflow.
- Deterministic mock provider streaming with token deltas and completion
  metadata.
- Node execution order through memory, retrieval, context assembly, prompt
  input, planning, rendering, generation, artifact extraction, preview,
  critique, review, and finalization.
- Runtime state transitions and final workflow metadata.
- Memory and retrieval context assembly into provider-ready prompt messages.
- Retrieval gateway failure recovery with a recoverable empty-context event.
- WSGI NDJSON streaming over `/api/assistant/stream`.
- Terminal provider failure surfaced as stream error, generation node failure,
  failure workflow status, and final failure answer.

Workspace lifecycle coverage is expanded in
`tests/test_workspace_session_persistence.py` to verify create, update with
`PUT`, restore, missing session, invalid payload, and persistence round trip
behavior.

Existing supporting runtime coverage remains in:

- `tests/test_langgraph_workflow_integration.py`
- `tests/test_workflow_runtime_decomposition.py`
- `tests/test_planning_runtime_decomposition.py`
- `tests/test_nextjs_streaming_bridge.py`
- `tests/test_v7_5_production_api_runtime_stabilization.py`

V7.11 planning runtime decomposition coverage verifies that planning, Director,
and reasoning handlers are registered from focused modules, the legacy
`runtime.nodes.planning` facade remains compatible, and planning state/event
payload field order remains stable.

## Coverage Threshold

Before V7 freeze, runtime-risk coverage must include at least one deterministic
integration assertion for each of these surfaces:

- Compiled LangGraph workflow construction and execution.
- Node ordering and transition metadata.
- Success stream contract through the backend API/WSGI layer.
- Error stream contract through the backend API/WSGI layer.
- Workspace session create, restore, update, and missing-session paths.
- Memory or retrieval context feeding prompt assembly.
- Controlled provider success behavior with no external API call.
- Controlled provider or subsystem failure recovery.

Fast push CI includes the representative deterministic V7.9 integration module.
Full backend pytest remains the release/manual/tag verification tier.
