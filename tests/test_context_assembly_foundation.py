import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    AssembledContextRequest,
    AssembledContextResponse,
    AssembledContextSummary,
    AssistantService,
    ConversationSummaryContext,
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
    build_assembled_context_request,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class ContextAssemblyFoundationTests(unittest.TestCase):
    def test_build_assembled_request_skips_when_no_context_is_available(self) -> None:
        decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            capabilities=(RouteCapability.MEMORY_CONTEXT,),
        )

        request = build_assembled_context_request(
            route_decision=decision,
            memory_context=None,
            retrieval_context=None,
        )

        self.assertIsNone(request)

    def test_build_assembled_request_keeps_route_and_nested_contexts(self) -> None:
        decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            capabilities=(
                RouteCapability.MEMORY_CONTEXT,
                RouteCapability.OFFICIAL_DOCS,
            ),
        )
        memory_context = _memory_context()
        retrieval_context = _retrieval_context()

        request = build_assembled_context_request(
            route_decision=decision,
            memory_context=memory_context,
            retrieval_context=retrieval_context,
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.route, RouteName.EXPLAIN)
        self.assertEqual(request.memory_context, memory_context)
        self.assertEqual(request.retrieval_context, retrieval_context)

    def test_context_assembler_computes_summary_without_reformatting(
        self,
    ) -> None:
        assembler = OrchestrationContextAssembler()
        request = AssembledContextRequest(
            route=RouteName.EXPLAIN,
            memory_context=_memory_context(),
            retrieval_context=_retrieval_context(),
        )

        response = assembler.assemble(request)

        self.assertEqual(
            response.summary,
            AssembledContextSummary(
                recent_turn_count=2,
                has_running_summary=True,
                project_memory_count=2,
                retrieval_chunk_count=1,
            ),
        )
        self.assertIsNotNone(response.memory_context)
        self.assertIsNotNone(response.retrieval_context)
        assert response.memory_context is not None
        assert response.retrieval_context is not None
        self.assertEqual(
            response.memory_context.recent_turns[0].content,
            "Keep the motion restrained.",
        )
        self.assertEqual(
            response.retrieval_context.chunks[0].excerpt,
            "PerspectiveCamera controls field of view and aspect ratio.",
        )

    def test_service_emits_context_event_when_assembler_is_present(self) -> None:
        memory_gateway = _FakeMemoryGateway(response=_memory_context())
        retrieval_gateway = _FakeRetrievalGateway(response=_retrieval_context())
        assembler = _FakeContextAssembler()
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=memory_gateway,
            retrieval_gateway=retrieval_gateway,
            context_assembler=assembler,
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
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[6].payload["code"], "context_assembled")
        self.assertEqual(len(assembler.requests), 1)
        self.assertIsNotNone(assembler.requests[0].memory_context)
        self.assertIsNotNone(assembler.requests[0].retrieval_context)

    def test_service_skips_context_event_without_assembler(self) -> None:
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
        )
        request = AssistantRequest(
            query="Explain the scene setup.",
            conversation_id="conversation-1",
            project_id="project-1",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )

        events = tuple(service.stream(request))

        self.assertNotIn(
            StreamEventType.CONTEXT,
            [event.event_type for event in events],
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
            content="The user prefers a restrained visual language.",
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


class _FakeContextAssembler:
    def __init__(self) -> None:
        self.requests: list[AssembledContextRequest] = []

    def assemble(
        self,
        request: AssembledContextRequest,
    ) -> AssembledContextResponse:
        self.requests.append(request)
        return AssembledContextResponse(
            request=request,
            summary=AssembledContextSummary(
                recent_turn_count=2,
                has_running_summary=True,
                project_memory_count=2,
                retrieval_chunk_count=1,
            ),
            memory_context=request.memory_context,
            retrieval_context=request.retrieval_context,
        )


if __name__ == "__main__":
    unittest.main()
