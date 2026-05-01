import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.llm import (
    GeneratedOutput,
    GenerationDelta,
    GenerationError,
    GenerationEventType,
    GenerationFinishReason,
    GenerationMessageName,
    GenerationMessageRole,
    GenerationRequest,
    GenerationResponse,
    GenerationStreamEvent,
    RenderedPromptGenerationBuilder,
    build_generation_request,
)
from creative_coding_assistant.memory import ConversationRole, ProjectMemoryKind
from creative_coding_assistant.orchestration import (
    AssembledContextResponse,
    AssembledContextSummary,
    ConversationSummaryContext,
    JinjaPromptRenderer,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
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
    build_rendered_prompt_request,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class ProviderGenerationContractsTests(unittest.TestCase):
    def test_generation_request_rejects_misaligned_rendered_route(self) -> None:
        rendered_prompt = _rendered_prompt()

        with self.assertRaisesRegex(ValueError, "generation route"):
            GenerationRequest(
                route=RouteName.GENERATE,
                rendered_prompt=rendered_prompt,
            )

    def test_generation_builder_transports_rendered_prompt_sections(self) -> None:
        request = build_generation_request(
            route_decision=_route_decision(),
            rendered_prompt=_rendered_prompt(),
        )

        generation_input = RenderedPromptGenerationBuilder().build(request)

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

    def test_generation_stream_event_accepts_delta_payload(self) -> None:
        event = GenerationStreamEvent(
            event_type=GenerationEventType.DELTA,
            delta=GenerationDelta(index=0, content="Camera drift stays subtle."),
        )

        self.assertEqual(event.delta.content, "Camera drift stays subtle.")
        self.assertEqual(event.delta.role, GenerationMessageRole.ASSISTANT)

    def test_generation_stream_event_requires_completed_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "response payload"):
            GenerationStreamEvent(
                event_type=GenerationEventType.COMPLETED,
            )

    def test_generation_stream_event_requires_error_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "error payload"):
            GenerationStreamEvent(
                event_type=GenerationEventType.ERROR,
                delta=GenerationDelta(index=0, content="Unexpected delta."),
            )

    def test_generation_response_tracks_final_generated_output(self) -> None:
        request = build_generation_request(
            route_decision=_route_decision(),
            rendered_prompt=_rendered_prompt(),
            stream=False,
        )
        generation_input = RenderedPromptGenerationBuilder().build(request)

        response = GenerationResponse(
            request=generation_input,
            output=GeneratedOutput(
                content="Use a restrained camera drift and a calm palette.",
                finish_reason=GenerationFinishReason.STOP,
            ),
        )

        self.assertFalse(response.request.request.stream)
        self.assertEqual(response.output.role, GenerationMessageRole.ASSISTANT)
        self.assertEqual(response.output.finish_reason, GenerationFinishReason.STOP)
        self.assertIn("restrained camera drift", response.output.content)

    def test_generation_stream_event_accepts_completed_response(self) -> None:
        request = build_generation_request(
            route_decision=_route_decision(),
            rendered_prompt=_rendered_prompt(),
            stream=False,
        )
        generation_input = RenderedPromptGenerationBuilder().build(request)
        response = GenerationResponse(
            request=generation_input,
            output=GeneratedOutput(
                content="Use calm motion and documented camera settings.",
                finish_reason=GenerationFinishReason.STOP,
            ),
        )

        event = GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=response,
        )

        self.assertEqual(
            event.response.output.content,
            "Use calm motion and documented camera settings.",
        )

    def test_generation_stream_event_accepts_error_payload(self) -> None:
        event = GenerationStreamEvent(
            event_type=GenerationEventType.ERROR,
            error=GenerationError(
                code="provider_unavailable",
                message="The provider adapter is unavailable.",
            ),
        )

        self.assertEqual(event.error.code, "provider_unavailable")
        self.assertIn("unavailable", event.error.message)


def _route_decision() -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=AssistantMode.EXPLAIN,
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


if __name__ == "__main__":
    unittest.main()
