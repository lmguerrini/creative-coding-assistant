import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.llm import (
    GenerationMessageName,
    GenerationMessageRole,
    GenerationProvider,
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
    ProviderGenerationRequest,
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


class RenderedPromptToProviderBoundaryTests(unittest.TestCase):
    def test_provider_generation_request_rejects_misaligned_route(self) -> None:
        rendered_prompt = _rendered_prompt()

        with self.assertRaisesRegex(ValueError, "provider boundary route"):
            ProviderGenerationRequest(
                route=RouteName.GENERATE,
                rendered_prompt=rendered_prompt,
            )

    def test_llm_generation_adapter_transports_rendered_sections(self) -> None:
        request = build_provider_generation_request(
            route_decision=_route_decision(),
            rendered_prompt=_rendered_prompt(),
        )

        generation_input = LlmGenerationAdapter().prepare_generation(request)

        self.assertTrue(generation_input.request.stream)
        self.assertEqual(
            [message.name for message in generation_input.messages],
            [
                GenerationMessageName.SYSTEM,
                GenerationMessageName.USER,
                GenerationMessageName.MEMORY,
                GenerationMessageName.RETRIEVAL,
            ],
        )
        self.assertEqual(
            [message.role for message in generation_input.messages],
            [
                GenerationMessageRole.SYSTEM,
                GenerationMessageRole.USER,
                GenerationMessageRole.CONTEXT,
                GenerationMessageRole.CONTEXT,
            ],
        )
        self.assertIn("Route: explain", generation_input.messages[0].content)
        self.assertIn(
            "Explain the scene setup.",
            generation_input.messages[1].content,
        )
        self.assertIn(
            "The user prefers restrained motion and calm palettes.",
            generation_input.messages[2].content,
        )
        self.assertIn(
            "Prefer restrained palettes.",
            generation_input.messages[2].content,
        )
        self.assertIn(
            "PerspectiveCamera controls field of view and aspect ratio.",
            generation_input.messages[3].content,
        )

    def test_service_emits_generation_input_event_when_gateway_present(self) -> None:
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=_IdleGenerationProvider(),
        )
        request = AssistantRequest(
            query="Explain the scene setup.",
            conversation_id="conversation-1",
            project_id="project-1",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.MEMORY,
                StreamEventType.MEMORY,
                StreamEventType.RETRIEVAL,
                StreamEventType.RETRIEVAL,
                StreamEventType.CONTEXT,
                StreamEventType.PROMPT_INPUT,
                StreamEventType.PROMPT_RENDERED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[9].payload["code"], "generation_input_prepared")
        generation_input = events[9].payload["generation_input"]
        self.assertEqual(len(generation_input["messages"]), 4)
        self.assertEqual(generation_input["messages"][0]["name"], "system")
        self.assertEqual(generation_input["messages"][1]["name"], "user")
        self.assertEqual(generation_input["messages"][2]["name"], "memory")
        self.assertEqual(generation_input["messages"][3]["name"], "retrieval")

    def test_service_continues_generation_when_retrieval_fails(self) -> None:
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FailingRetrievalGateway(),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=_IdleGenerationProvider(),
        )
        request = AssistantRequest(
            query="Explain the scene setup.",
            conversation_id="conversation-1",
            project_id="project-1",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.MEMORY,
                StreamEventType.MEMORY,
                StreamEventType.RETRIEVAL,
                StreamEventType.RETRIEVAL,
                StreamEventType.CONTEXT,
                StreamEventType.PROMPT_INPUT,
                StreamEventType.PROMPT_RENDERED,
                StreamEventType.GENERATION_INPUT,
                StreamEventType.FINAL,
            ],
        )
        retrieval_context = events[5].payload["context"]
        self.assertEqual(retrieval_context["chunks"], [])
        assembled_context = events[6].payload["context"]
        self.assertEqual(assembled_context["summary"]["retrieval_chunk_count"], 0)
        self.assertEqual(events[9].payload["code"], "generation_input_prepared")


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


def _rendered_prompt():
    prompt_input_request = build_prompt_input_request(
        assistant_request=AssistantRequest(
            query="Explain the scene setup.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        ),
        route_decision=_route_decision(),
        assembled_context=_assembled_context(),
    )
    prompt_input = StructuredPromptInputBuilder().build(prompt_input_request)
    return JinjaPromptRenderer().render(
        build_rendered_prompt_request(
            route_decision=_route_decision(),
            prompt_input=prompt_input,
        )
    )


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


class _FailingRetrievalGateway:
    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        del request
        raise RuntimeError("retrieval unavailable")


class _IdleGenerationProvider(GenerationProvider):
    def stream(self, request: object):
        del request
        return iter(())


if __name__ == "__main__":
    unittest.main()
