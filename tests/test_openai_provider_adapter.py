import sys
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import creative_coding_assistant.llm.openai_adapter as openai_adapter_module
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.llm import (
    GenerationEventType,
    GenerationFinishReason,
    OpenAIGenerationProvider,
)
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    AssembledContextResponse,
    AssembledContextSummary,
    AssistantService,
    ConversationSummaryContext,
    JinjaPromptRenderer,
    LlmGenerationAdapter,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
    OrchestrationContextAssembler,
    ProjectMemoryContext,
    RecentConversationTurn,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalContextSource,
    RetrievedKnowledgeChunk,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_assembled_context_request,
    build_prompt_input_request,
    build_provider_generation_request,
    build_rendered_prompt_request,
)
from creative_coding_assistant.rag.sources import OfficialSourceType
from event_assertions import first_event, legacy_events


class OpenAIProviderAdapterTests(unittest.TestCase):
    def test_live_client_uses_settings_api_key_without_reloading_environment(
        self,
    ) -> None:
        response = SimpleNamespace(
            output_text="Use a restrained camera drift.",
            status="completed",
        )
        constructor = _FakeOpenAIConstructor(response=response)
        settings = Settings(
            openai_api_key="sk-settings-secret",
            openai_model="gpt-5-mini",
        )

        with patch.dict(
            sys.modules,
            {"openai": SimpleNamespace(OpenAI=constructor)},
        ):
            with patch(
                "creative_coding_assistant.llm.openai_adapter.load_settings",
                side_effect=AssertionError("settings should not reload"),
            ):
                provider = OpenAIGenerationProvider(settings=settings)
                events = tuple(provider.stream(_generation_input(stream=False)))

        self.assertEqual(constructor.api_keys, ["sk-settings-secret"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, GenerationEventType.COMPLETED)
        self.assertEqual(
            events[0].response.output.content,
            "Use a restrained camera drift.",
        )

    def test_live_client_falls_back_to_environment_settings(self) -> None:
        response = SimpleNamespace(
            output_text="Use a restrained camera drift.",
            status="completed",
        )
        constructor = _FakeOpenAIConstructor(response=response)
        fallback_settings = Settings(openai_api_key="sk-env-secret")

        with patch.dict(
            sys.modules,
            {"openai": SimpleNamespace(OpenAI=constructor)},
        ):
            with patch(
                "creative_coding_assistant.llm.openai_adapter.load_settings",
                return_value=fallback_settings,
            ):
                client = openai_adapter_module._build_openai_client()

        self.assertIsInstance(client, _FakeOpenAIClient)
        self.assertEqual(constructor.api_keys, ["sk-env-secret"])

    def test_non_streaming_request_maps_to_openai_payload(self) -> None:
        response = SimpleNamespace(
            output_text="Use a restrained camera drift.",
            status="completed",
        )
        client = _FakeOpenAIClient(response=response)
        provider = OpenAIGenerationProvider(client=client, model="gpt-5-mini")
        generation_input = _generation_input(stream=False)

        events = tuple(provider.stream(generation_input))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, GenerationEventType.COMPLETED)
        self.assertEqual(
            events[0].response.output.content,
            "Use a restrained camera drift.",
        )
        self.assertEqual(
            events[0].response.output.finish_reason,
            GenerationFinishReason.STOP,
        )
        self.assertEqual(client.last_kwargs["model"], "gpt-5-mini")
        self.assertEqual(client.last_kwargs["max_output_tokens"], 4000)
        self.assertFalse(client.last_kwargs["stream"])
        self.assertIn("Route: explain", client.last_kwargs["instructions"])
        self.assertEqual(len(client.last_kwargs["input"]), 3)
        self.assertEqual(client.last_kwargs["input"][0]["role"], "user")
        self.assertEqual(client.last_kwargs["input"][1]["role"], "user")
        self.assertEqual(client.last_kwargs["input"][2]["role"], "user")
        self.assertIn(
            "Memory Context:",
            client.last_kwargs["input"][1]["content"][0]["text"],
        )
        self.assertIn(
            "UNTRUSTED MEMORY",
            client.last_kwargs["input"][1]["content"][0]["text"],
        )
        self.assertIn(
            "Retrieval Context:",
            client.last_kwargs["input"][2]["content"][0]["text"],
        )
        self.assertIn(
            "UNTRUSTED RETRIEVED-DOCUMENT",
            client.last_kwargs["input"][2]["content"][0]["text"],
        )

    def test_streaming_request_emits_deltas_and_completion(self) -> None:
        stream_events = (
            SimpleNamespace(type="response.output_text.delta", delta="Use "),
            SimpleNamespace(type="response.output_text.delta", delta="documented "),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="resp_123",
                    model="gpt-5-mini",
                    output_text="Use documented camera settings.",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=1200,
                        output_tokens=300,
                        total_tokens=1500,
                        output_tokens_details=SimpleNamespace(
                            reasoning_tokens=12,
                        ),
                    ),
                ),
            ),
        )
        client = _FakeOpenAIClient(stream_events=stream_events)
        provider = OpenAIGenerationProvider(
            settings=Settings(openai_model="gpt-5-mini"),
            client=client,
        )

        events = tuple(provider.stream(_generation_input(stream=True)))

        self.assertEqual(
            [event.event_type for event in events],
            [
                GenerationEventType.DELTA,
                GenerationEventType.DELTA,
                GenerationEventType.COMPLETED,
            ],
        )
        self.assertEqual(events[0].delta.content, "Use ")
        self.assertEqual(events[0].delta.provider, "openai")
        self.assertEqual(events[0].delta.model, "gpt-5-mini")
        self.assertEqual(events[1].delta.content, "documented ")
        self.assertEqual(
            events[2].response.output.content,
            "Use documented camera settings.",
        )
        self.assertEqual(events[2].response.output.provider, "openai")
        self.assertEqual(events[2].response.output.model, "gpt-5-mini")
        self.assertEqual(events[2].response.output.response_id, "resp_123")
        self.assertEqual(events[2].response.output.usage.input_tokens, 1200)
        self.assertEqual(events[2].response.output.usage.output_tokens, 300)
        self.assertEqual(events[2].response.output.usage.total_tokens, 1500)
        self.assertEqual(events[2].response.output.usage.reasoning_tokens, 12)
        self.assertTrue(client.last_kwargs["stream"])
        self.assertEqual(client.last_kwargs["model"], "gpt-5-mini")
        self.assertEqual(client.last_kwargs["max_output_tokens"], 4000)

    def test_length_limited_empty_response_returns_recoverable_boundary(
        self,
    ) -> None:
        stream_events = (
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    id="resp_length",
                    model="gpt-5-mini",
                    output_text="",
                    status="incomplete",
                    incomplete_details=SimpleNamespace(
                        reason="max_output_tokens",
                    ),
                ),
            ),
        )
        client = _FakeOpenAIClient(stream_events=stream_events)
        provider = OpenAIGenerationProvider(
            settings=Settings(openai_model="gpt-5-mini"),
            client=client,
        )

        events = tuple(provider.stream(_generation_input(stream=True)))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, GenerationEventType.COMPLETED)
        self.assertEqual(
            events[0].response.output.finish_reason,
            GenerationFinishReason.LENGTH,
        )
        self.assertIn(
            "configured output limit was reached",
            events[0].response.output.content,
        )

    def test_provider_maps_error_events(self) -> None:
        stream_events = (
            SimpleNamespace(
                type="response.failed",
                error=SimpleNamespace(
                    code="rate_limit_exceeded",
                    message="Rate limit exceeded.",
                ),
            ),
        )
        provider = OpenAIGenerationProvider(
            client=_FakeOpenAIClient(stream_events=stream_events)
        )

        events = tuple(provider.stream(_generation_input(stream=True)))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, GenerationEventType.ERROR)
        self.assertEqual(events[0].error.code, "rate_limit_exceeded")
        self.assertEqual(events[0].error.message, "Rate limit exceeded.")

    def test_provider_maps_raised_exceptions(self) -> None:
        provider = OpenAIGenerationProvider(
            client=_FakeOpenAIClient(raised_error=RuntimeError("boom"))
        )

        events = tuple(provider.stream(_generation_input(stream=False)))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, GenerationEventType.ERROR)
        self.assertEqual(events[0].error.code, "openai_error")
        self.assertEqual(events[0].error.message, "boom")

    def test_service_streams_provider_output_when_openai_adapter_present(self) -> None:
        stream_events = (
            SimpleNamespace(type="response.output_text.delta", delta="Use "),
            SimpleNamespace(type="response.output_text.delta", delta="soft motion."),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="resp_service",
                    model="gpt-5-mini",
                    output_text="Use soft motion.",
                    status="completed",
                    usage=SimpleNamespace(
                        input_tokens=90,
                        output_tokens=15,
                        total_tokens=105,
                    ),
                ),
            ),
        )
        provider = OpenAIGenerationProvider(
            client=_FakeOpenAIClient(stream_events=stream_events)
        )
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=provider,
        )

        events = tuple(service.stream(_assistant_request()))
        legacy = legacy_events(events)

        self.assertEqual(
            [event.event_type for event in legacy],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.MEMORY,
                StreamEventType.MEMORY,
                StreamEventType.RETRIEVAL,
                StreamEventType.RETRIEVAL,
                StreamEventType.CONTEXT,
                StreamEventType.PROMPT_INPUT,
                StreamEventType.PLANNING,
                StreamEventType.PLANNING,
                StreamEventType.PLANNING,
                StreamEventType.PROMPT_RENDERED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.TOKEN_DELTA,
                StreamEventType.FINAL,
            ],
        )
        token_events = [
            event for event in events if event.event_type is StreamEventType.TOKEN_DELTA
        ]
        final_event = first_event(events, StreamEventType.FINAL)

        self.assertEqual(token_events[0].payload["text"], "Use ")
        self.assertEqual(token_events[1].payload["text"], "soft motion.")
        self.assertEqual(final_event.payload["answer"], "Use soft motion.")
        self.assertEqual(
            token_events[0].payload["telemetry"]["provider"]["name"],
            "openai",
        )
        self.assertEqual(
            token_events[0].payload["telemetry"]["execution"]["generation_mode"],
            "streaming",
        )
        self.assertTrue(
            token_events[0].payload["telemetry"]["execution"]["streaming"],
        )
        self.assertEqual(
            token_events[0].payload["telemetry"]["execution"]["streaming_status"],
            "active",
        )
        self.assertEqual(
            final_event.payload["telemetry"]["provider"]["model"],
            "gpt-5-mini",
        )
        self.assertEqual(
            final_event.payload["telemetry"]["token_usage"]["total_tokens"],
            105,
        )
        self.assertEqual(
            final_event.payload["telemetry"]["execution"]["streaming_status"],
            "completed",
        )
        self.assertEqual(
            final_event.payload["telemetry"]["execution"]["retry_count"],
            0,
        )
        self.assertGreaterEqual(
            final_event.payload["telemetry"]["execution"]["request_duration_ms"],
            0,
        )
        self.assertIn(
            "request_started_at",
            final_event.payload["telemetry"]["execution"],
        )
        self.assertIn(
            "request_completed_at",
            final_event.payload["telemetry"]["execution"],
        )

    def test_service_emits_error_and_failure_answer_on_provider_error(self) -> None:
        stream_events = (
            SimpleNamespace(
                type="response.failed",
                error=SimpleNamespace(
                    code="provider_unavailable",
                    message="Provider unavailable.",
                ),
            ),
        )
        provider = OpenAIGenerationProvider(
            client=_FakeOpenAIClient(stream_events=stream_events)
        )
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=provider,
        )

        events = tuple(service.stream(_assistant_request()))
        error_event = first_event(events, StreamEventType.ERROR)
        final_event = first_event(events, StreamEventType.FINAL)

        self.assertEqual(error_event.payload["code"], "provider_unavailable")
        self.assertEqual(
            error_event.payload["telemetry"]["execution"]["generation_mode"],
            "streaming",
        )
        self.assertEqual(
            error_event.payload["telemetry"]["execution"]["streaming_status"],
            "failed",
        )
        self.assertEqual(
            error_event.payload["telemetry"]["execution"]["errors"],
            [
                {
                    "code": "provider_unavailable",
                    "message": "Provider unavailable.",
                }
            ],
        )
        self.assertIn("Generation failed", final_event.payload["answer"])
        self.assertIn("provider_unavailable", final_event.payload["answer"])


def _assistant_request() -> AssistantRequest:
    return AssistantRequest(
        query="Explain the scene setup.",
        conversation_id="conversation-1",
        project_id="project-1",
        domain=CreativeCodingDomain.THREE_JS,
        mode=AssistantMode.EXPLAIN,
    )


def _route_decision() -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=AssistantMode.EXPLAIN,
        capabilities=(
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
        ),
    )


def _route_with_memory_and_docs(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=request.mode,
        domain=request.domain,
        capabilities=(
            RouteCapability.MEMORY_CONTEXT,
            RouteCapability.OFFICIAL_DOCS,
        ),
    )


def _generation_input(*, stream: bool):
    prompt_input_request = build_prompt_input_request(
        assistant_request=_assistant_request(),
        route_decision=_route_decision(),
        assembled_context=_assembled_context(),
    )
    prompt_input = StructuredPromptInputBuilder().build(prompt_input_request)
    rendered_prompt = JinjaPromptRenderer().render(
        build_rendered_prompt_request(
            route_decision=_route_decision(),
            prompt_input=prompt_input,
        )
    )
    provider_request = build_provider_generation_request(
        route_decision=_route_decision(),
        rendered_prompt=rendered_prompt,
        stream=stream,
    )
    return LlmGenerationAdapter().prepare_generation(provider_request)


def _assembled_context() -> AssembledContextResponse:
    request = build_assembled_context_request(
        route_decision=_route_decision(),
        memory_context=_memory_context(),
        retrieval_context=_retrieval_context(),
    )
    assert request is not None
    return AssembledContextResponse(
        request=request,
        summary=AssembledContextSummary(
            recent_turn_count=2,
            has_running_summary=True,
            project_memory_count=2,
            retrieval_chunk_count=1,
        ),
        memory_context=_memory_context(),
        retrieval_context=_retrieval_context(),
    )


def _memory_context() -> MemoryContextResponse:
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.EXPLAIN,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        source=MemoryContextSource.CHROMA_MEMORY,
        recent_turns=(
            RecentConversationTurn(
                turn_index=0,
                role=ConversationRole.USER,
                content="Keep the motion restrained.",
                created_at=_time(),
                mode=AssistantMode.EXPLAIN,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content="We can keep the camera drift subtle.",
                created_at=_time(),
                mode=AssistantMode.EXPLAIN,
            ),
        ),
        running_summary=ConversationSummaryContext(
            content="The user prefers restrained motion and calm palettes.",
            created_at=_time(),
            covered_turn_count=2,
        ),
        project_memories=(
            ProjectMemoryContext(
                content="Prefer restrained palettes.",
                created_at=_time(),
                memory_kind=ProjectMemoryKind.STYLE,
                source="user",
            ),
            ProjectMemoryContext(
                content="Build atmospheric shader studies.",
                created_at=_time(),
                memory_kind=ProjectMemoryKind.GOAL,
                source="user",
            ),
        ),
    )


def _retrieval_context() -> RetrievalContextResponse:
    return RetrievalContextResponse(
        request=RetrievalContextRequest(
            query="Explain the scene setup.",
            route=RouteName.EXPLAIN,
        ),
        source=RetrievalContextSource.OFFICIAL_KB,
        chunks=(
            RetrievedKnowledgeChunk(
                source_id="three_docs",
                domain=CreativeCodingDomain.THREE_JS,
                source_type=OfficialSourceType.API_REFERENCE,
                publisher="three.js",
                registry_title="three.js Documentation",
                document_title="PerspectiveCamera",
                source_url="https://threejs.org/docs/",
                resolved_url="https://threejs.org/docs/",
                chunk_index=0,
                excerpt="PerspectiveCamera controls field of view and aspect ratio.",
                score=0.83,
            ),
        ),
    )


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeMemoryGateway:
    def __init__(self, *, response: MemoryContextResponse) -> None:
        self.response = response

    def retrieve_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        del request
        return self.response


class _FakeRetrievalGateway:
    def __init__(self, *, response: RetrievalContextResponse) -> None:
        self.response = response

    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        del request
        return self.response


class _FakeResponsesApi:
    def __init__(
        self,
        *,
        response: object | None = None,
        stream_events: tuple[object, ...] = (),
        raised_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.stream_events = stream_events
        self.raised_error = raised_error
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        if self.raised_error is not None:
            raise self.raised_error
        if kwargs.get("stream"):
            return iter(self.stream_events)
        return self.response


class _FakeOpenAIClient:
    def __init__(
        self,
        *,
        response: object | None = None,
        stream_events: tuple[object, ...] = (),
        raised_error: Exception | None = None,
    ) -> None:
        self.responses = _FakeResponsesApi(
            response=response,
            stream_events=stream_events,
            raised_error=raised_error,
        )

    @property
    def last_kwargs(self) -> dict[str, object]:
        assert self.responses.last_kwargs is not None
        return self.responses.last_kwargs


class _FakeOpenAIConstructor:
    def __init__(self, *, response: object) -> None:
        self._response = response
        self.api_keys: list[str] = []

    def __call__(self, *, api_key: str) -> _FakeOpenAIClient:
        self.api_keys.append(api_key)
        return _FakeOpenAIClient(response=self._response)


if __name__ == "__main__":
    unittest.main()
