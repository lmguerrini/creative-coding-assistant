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
    AssembledContextResponse,
    AssembledContextSummary,
    AssistantService,
    ConversationSummaryContext,
    JinjaPromptRenderer,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
    OrchestrationContextAssembler,
    ProjectMemoryContext,
    RecentConversationTurn,
    RenderedPromptRequest,
    RenderedPromptRole,
    RenderedPromptSectionName,
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


class PromptTemplateFoundationTests(unittest.TestCase):
    def test_rendered_prompt_request_rejects_misaligned_route(self) -> None:
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=AssistantRequest(query="Explain the scene."),
                route_decision=RouteDecision(
                    route=RouteName.EXPLAIN,
                    mode=AssistantMode.EXPLAIN,
                    capabilities=(RouteCapability.OFFICIAL_DOCS,),
                ),
                assembled_context=None,
            )
        )

        with self.assertRaisesRegex(ValueError, "rendered route"):
            RenderedPromptRequest(
                route=RouteName.GENERATE,
                prompt_input=prompt_input,
            )

    def test_jinja_prompt_renderer_renders_structured_sections(self) -> None:
        renderer = JinjaPromptRenderer()
        prompt_input = _prompt_input()
        request = build_rendered_prompt_request(
            route_decision=_route_decision(),
            prompt_input=prompt_input,
        )

        response = renderer.render(request)

        self.assertEqual(
            [section.name for section in response.sections],
            [
                RenderedPromptSectionName.SYSTEM,
                RenderedPromptSectionName.USER,
                RenderedPromptSectionName.MEMORY,
                RenderedPromptSectionName.RETRIEVAL,
            ],
        )
        self.assertEqual(response.sections[0].role, RenderedPromptRole.SYSTEM)
        self.assertIn("Route: explain", response.sections[0].content)
        self.assertIn("User Request:", response.sections[1].content)
        self.assertIn(
            "Keep the motion restrained.",
            response.sections[2].content,
        )
        self.assertIn(
            "PerspectiveCamera controls field of view and aspect ratio.",
            response.sections[3].content,
        )

    def test_renderer_supports_user_only_prompt_inputs(self) -> None:
        renderer = JinjaPromptRenderer()
        prompt_input = build_prompt_input_request(
            assistant_request=AssistantRequest(query="Start a shader study."),
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                capabilities=(RouteCapability.TOOL_USE,),
            ),
            assembled_context=None,
        )
        prompt_input_response = StructuredPromptInputBuilder().build(prompt_input)

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.GENERATE,
                prompt_input=prompt_input_response,
            )
        )

        self.assertEqual(len(rendered.sections), 2)
        self.assertEqual(rendered.sections[0].name, RenderedPromptSectionName.SYSTEM)
        self.assertEqual(rendered.sections[1].name, RenderedPromptSectionName.USER)

    def test_service_emits_rendered_prompt_event_when_renderer_present(self) -> None:
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
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
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[8].payload["code"], "prompt_rendered")
        rendered_prompt = events[8].payload["rendered_prompt"]
        self.assertEqual(len(rendered_prompt["sections"]), 4)
        self.assertEqual(rendered_prompt["sections"][0]["name"], "system")
        self.assertEqual(rendered_prompt["sections"][1]["name"], "user")


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


def _prompt_input():
    assistant_request = AssistantRequest(
        query="Explain the scene setup.",
        domain=CreativeCodingDomain.THREE_JS,
        mode=AssistantMode.EXPLAIN,
    )
    prompt_input_request = build_prompt_input_request(
        assistant_request=assistant_request,
        route_decision=_route_decision(),
        assembled_context=_assembled_context(),
    )
    return StructuredPromptInputBuilder().build(prompt_input_request)


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


if __name__ == "__main__":
    unittest.main()
