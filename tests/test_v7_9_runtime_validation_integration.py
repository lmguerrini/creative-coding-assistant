from __future__ import annotations

import io
import json
import unittest
from collections.abc import Iterable
from datetime import UTC, datetime

from creative_coding_assistant.api import AssistantStreamingApplication
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.llm.generation import (
    GeneratedOutput,
    GenerationDelta,
    GenerationError,
    GenerationEventType,
    GenerationFinishReason,
    GenerationInput,
    GenerationResponse,
    GenerationStreamEvent,
    GenerationTokenUsage,
)
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    AssistantService,
    ConversationSummaryContext,
    JinjaPromptRenderer,
    LlmGenerationAdapter,
    MemoryContextRequest,
    MemoryContextResponse,
    OrchestrationContextAssembler,
    ProjectMemoryContext,
    RecentConversationTurn,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievedKnowledgeChunk,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class V79RuntimeValidationIntegrationTests(unittest.TestCase):
    def test_assistant_service_runs_full_pipeline_with_mock_provider_contexts(
        self,
    ) -> None:
        provider = _MockGenerationProvider(_P5_ANSWER)
        memory_gateway = _ControlledMemoryGateway()
        retrieval_gateway = _ControlledRetrievalGateway()
        service = _service(
            provider=provider,
            memory_gateway=memory_gateway,
            retrieval_gateway=retrieval_gateway,
        )

        events = tuple(service.stream(_request()))

        final = events[-1]
        event_types = [event.event_type for event in events]
        completed_nodes = [
            event.payload["node"]
            for event in events
            if event.event_type is StreamEventType.NODE_COMPLETED
        ]

        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual(memory_gateway.requests[0].conversation_id, "conversation-v79")
        self.assertEqual(
            retrieval_gateway.requests[0].filters.domain,
            CreativeCodingDomain.P5_JS,
        )
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(
            completed_nodes,
            [
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "reasoning",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "finalization",
            ],
        )
        self.assertIn(StreamEventType.MEMORY, event_types)
        self.assertIn(StreamEventType.RETRIEVAL, event_types)
        self.assertIn(StreamEventType.CONTEXT, event_types)
        self.assertIn(StreamEventType.PROMPT_INPUT, event_types)
        self.assertIn(StreamEventType.PROMPT_RENDERED, event_types)
        self.assertIn(StreamEventType.GENERATION_INPUT, event_types)
        self.assertIn(StreamEventType.TOKEN_DELTA, event_types)
        self.assertIn(StreamEventType.ARTIFACT_EXTRACTED, event_types)
        self.assertIn(StreamEventType.PREVIEW_ARTIFACT, event_types)
        self.assertIn(StreamEventType.REVIEW_PASSED, event_types)
        self.assertEqual(final.event_type, StreamEventType.FINAL)
        self.assertEqual(final.payload["answer"], _P5_ANSWER)
        self.assertEqual(final.payload["workflow"]["status"], "completed")
        self.assertEqual(final.payload["workflow"]["skipped_steps"], [])
        self.assertEqual(final.payload["workflow"]["artifact_count"], 1)
        self.assertEqual(final.payload["workflow"]["preview_artifact_count"], 1)
        self.assertEqual(final.payload["telemetry"]["provider"]["name"], "mock")
        self.assertEqual(final.payload["telemetry"]["provider"]["model"], "mock-v7")
        self.assertEqual(final.payload["telemetry"]["finish_reason"], "stop")

        message_names = [message.name.value for message in provider.requests[0].messages]
        message_text = "\n".join(message.content for message in provider.requests[0].messages)
        self.assertIn("memory", message_names)
        self.assertIn("retrieval", message_names)
        self.assertIn("favors teal and violet gradients", message_text)
        self.assertIn("createCanvas", message_text)

    def test_retrieval_gateway_failure_recovers_with_empty_context(
        self,
    ) -> None:
        provider = _MockGenerationProvider(_P5_ANSWER)
        service = _service(
            provider=provider,
            memory_gateway=_ControlledMemoryGateway(),
            retrieval_gateway=_FailingRetrievalGateway(),
        )

        events = tuple(service.stream(_request()))

        retrieval_completed = _first_event(
            events,
            StreamEventType.RETRIEVAL,
            "retrieval_completed",
        )
        final = events[-1]

        self.assertEqual(retrieval_completed.payload["context"]["chunks"], [])
        self.assertEqual(
            retrieval_completed.payload["error"]["type"],
            "retrieval_gateway_failed",
        )
        self.assertEqual(final.event_type, StreamEventType.FINAL)
        self.assertEqual(final.payload["workflow"]["status"], "completed")
        self.assertEqual(provider.call_count, 1)

    def test_wsgi_stream_runs_runtime_pipeline_as_ndjson(self) -> None:
        provider = _MockGenerationProvider(_P5_ANSWER)
        app = AssistantStreamingApplication(
            service=_service(
                provider=provider,
                memory_gateway=_ControlledMemoryGateway(),
                retrieval_gateway=_ControlledRetrievalGateway(),
            ),
            settings_factory=lambda: Settings(_env_file=None),
        )
        status_headers: dict[str, object] = {}
        payload = json.dumps(
            {
                "query": "Write a p5.js sketch.",
                "conversationId": "conversation-v79",
                "projectId": "project-v79",
                "domain": "p5_js",
                "domains": ["p5_js"],
                "mode": "generate",
            }
        ).encode("utf-8")

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]

        self.assertEqual(status_headers["status"], "200 OK")
        self.assertIn(
            ("Content-Type", "application/x-ndjson; charset=utf-8"),
            status_headers["headers"],
        )
        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(len(events))),
        )
        self.assertIn("node_started", [event["event_type"] for event in events])
        self.assertIn("token_delta", [event["event_type"] for event in events])
        self.assertEqual(events[-1]["event_type"], "final")
        self.assertEqual(events[-1]["payload"]["answer"], _P5_ANSWER)
        self.assertEqual(events[-1]["payload"]["workflow"]["status"], "completed")
        self.assertEqual(provider.call_count, 1)

    def test_wsgi_stream_preserves_terminal_runtime_failure_events(self) -> None:
        app = AssistantStreamingApplication(
            service=_service(
                provider=_MockGenerationProvider(
                    "partial response",
                    error_code="mock_provider_failed",
                    error_message="Mock provider failed.",
                ),
                memory_gateway=_ControlledMemoryGateway(),
                retrieval_gateway=_ControlledRetrievalGateway(),
            ),
            settings_factory=lambda: Settings(_env_file=None),
        )
        status_headers: dict[str, object] = {}
        payload = json.dumps(
            {
                "query": "Write a p5.js sketch.",
                "conversationId": "conversation-v79",
                "projectId": "project-v79",
                "domain": "p5_js",
                "domains": ["p5_js"],
                "mode": "generate",
            }
        ).encode("utf-8")

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]
        error_events = [
            event
            for event in events
            if event["event_type"] == StreamEventType.ERROR.value
        ]
        generation_failures = [
            event
            for event in events
            if event["event_type"] == StreamEventType.NODE_FAILED.value
            and event["payload"].get("node") == "generation"
        ]

        self.assertEqual(status_headers["status"], "200 OK")
        self.assertEqual(error_events[-1]["payload"]["code"], "mock_provider_failed")
        self.assertEqual(
            generation_failures[-1]["payload"]["transition_target"],
            "failure",
        )
        self.assertEqual(events[-1]["event_type"], "final")
        self.assertEqual(events[-1]["payload"]["workflow"]["status"], "failed")
        self.assertEqual(
            events[-1]["payload"]["answer"],
            "Generation failed (mock_provider_failed): Mock provider failed.",
        )


_P5_ANSWER = "\n".join(
    [
        "```javascript filename=runtime-validation.p5.js",
        "function setup() {",
        "  createCanvas(640, 360);",
        "}",
        "function draw() {",
        "  background(12);",
        "  circle(width / 2, height / 2, 120);",
        "}",
        "```",
    ]
)
_NOW = datetime(2026, 7, 6, 10, 0, tzinfo=UTC)


def _request() -> AssistantRequest:
    return AssistantRequest(
        query="Write a p5.js sketch.",
        conversation_id="conversation-v79",
        project_id="project-v79",
        domain=CreativeCodingDomain.P5_JS,
        domains=(CreativeCodingDomain.P5_JS,),
        mode=AssistantMode.GENERATE,
    )


def _route_with_runtime_dependencies(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        domain=request.domain,
        domains=request.domains,
        capabilities=(
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
            RouteCapability.TOOL_USE,
            RouteCapability.LIVE_EVALUATION,
        ),
    )


def _service(
    *,
    provider: _MockGenerationProvider,
    memory_gateway: _ControlledMemoryGateway,
    retrieval_gateway: _ControlledRetrievalGateway | _FailingRetrievalGateway,
) -> AssistantService:
    return AssistantService(
        settings=Settings(_env_file=None),
        route_fn=_route_with_runtime_dependencies,
        memory_gateway=memory_gateway,
        retrieval_gateway=retrieval_gateway,
        context_assembler=OrchestrationContextAssembler(),
        prompt_input_builder=StructuredPromptInputBuilder(),
        prompt_renderer=JinjaPromptRenderer(),
        generation_gateway=LlmGenerationAdapter(),
        generation_provider=provider,
    )


class _ControlledMemoryGateway:
    def __init__(self) -> None:
        self.requests: list[MemoryContextRequest] = []

    def retrieve_context(self, request: MemoryContextRequest) -> MemoryContextResponse:
        self.requests.append(request)
        return MemoryContextResponse(
            request=request,
            recent_turns=(
                RecentConversationTurn(
                    turn_index=0,
                    role=ConversationRole.USER,
                    content="Use a calm particle field.",
                    created_at=_NOW,
                    mode=AssistantMode.GENERATE,
                ),
            ),
            running_summary=ConversationSummaryContext(
                content="The session favors teal and violet gradients.",
                created_at=_NOW,
                covered_turn_count=2,
            ),
            project_memories=(
                ProjectMemoryContext(
                    content="Keep generated sketches browser-previewable.",
                    created_at=_NOW,
                    memory_kind=ProjectMemoryKind.CONSTRAINT,
                    source="runtime-test",
                ),
            ),
        )


class _ControlledRetrievalGateway:
    def __init__(self) -> None:
        self.requests: list[RetrievalContextRequest] = []

    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        self.requests.append(request)
        return RetrievalContextResponse(
            request=request,
            chunks=(
                RetrievedKnowledgeChunk(
                    source_id="p5_reference",
                    domain=CreativeCodingDomain.P5_JS,
                    source_type=OfficialSourceType.EXAMPLES,
                    publisher="p5.js",
                    registry_title="p5.js Reference",
                    document_title="createCanvas example",
                    source_url="https://p5js.org/reference/p5/createCanvas/",
                    chunk_index=0,
                    excerpt="Use createCanvas(width, height) inside setup().",
                    score=0.97,
                    rank=1,
                    domain_match=True,
                    selection_reason="Selected by deterministic runtime test.",
                ),
            ),
        )


class _FailingRetrievalGateway(_ControlledRetrievalGateway):
    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        self.requests.append(request)
        raise RuntimeError("retrieval offline")


class _MockGenerationProvider:
    def __init__(
        self,
        answer: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._answer = answer
        self._error_code = error_code
        self._error_message = error_message
        self.requests: list[GenerationInput] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def stream(self, request: GenerationInput) -> Iterable[GenerationStreamEvent]:
        self.requests.append(request)
        if self._error_code is not None and self._error_message is not None:
            yield GenerationStreamEvent(
                event_type=GenerationEventType.ERROR,
                error=GenerationError(
                    code=self._error_code,
                    message=self._error_message,
                ),
            )
            return

        midpoint = max(1, len(self._answer) // 2)
        for index, chunk in enumerate((self._answer[:midpoint], self._answer[midpoint:])):
            if chunk:
                yield GenerationStreamEvent(
                    event_type=GenerationEventType.DELTA,
                    delta=GenerationDelta(
                        index=index,
                        content=chunk,
                        provider="mock",
                        model="mock-v7",
                    ),
                )
        yield GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=GenerationResponse(
                request=request,
                output=GeneratedOutput(
                    content=self._answer,
                    finish_reason=GenerationFinishReason.STOP,
                    provider="mock",
                    model="mock-v7",
                    response_id="mock-response-v79",
                    usage=GenerationTokenUsage(
                        input_tokens=12,
                        output_tokens=24,
                        total_tokens=36,
                    ),
                ),
            ),
        )


def _first_event(
    events,
    event_type: StreamEventType,
    code: str | None = None,
):
    for event in events:
        if event.event_type is not event_type:
            continue
        if code is not None and event.payload.get("code") != code:
            continue
        return event
    raise AssertionError(f"Missing event {event_type} with code {code!r}.")


def _capture_start_response(target: dict[str, object]):
    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: object | None = None,
    ) -> None:
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


if __name__ == "__main__":
    unittest.main()
