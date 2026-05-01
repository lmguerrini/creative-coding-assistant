import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.llm.generation import (
    GeneratedOutput,
    GenerationEventType,
    GenerationResponse,
    GenerationStreamEvent,
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
        self.assertIn("fenced code block", response.sections[0].content)
        self.assertIn(
            "Do not leave runnable code unfenced.",
            response.sections[0].content,
        )
        self.assertIn(
            "Keep the answer focused on the user's request",
            response.sections[0].content,
        )
        self.assertIn(
            "Prefer practical creative-coding examples",
            response.sections[0].content,
        )
        self.assertIn("```python", response.sections[0].content)
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
        self.assertIn(
            "Lead with runnable code first",
            rendered.sections[0].content,
        )
        self.assertIn(
            (
                "Keep explanation short and add setup or run notes only when "
                "they are useful."
            ),
            rendered.sections[0].content,
        )
        self.assertIn(
            "Avoid long conceptual sections unless the user explicitly asks for them.",
            rendered.sections[0].content,
        )

    def test_renderer_uses_explanation_first_guidance_for_explain_mode(self) -> None:
        renderer = JinjaPromptRenderer()
        prompt_input = build_prompt_input_request(
            assistant_request=AssistantRequest(
                query="Explain how fog works in Three.js.",
                mode=AssistantMode.EXPLAIN,
            ),
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            assembled_context=None,
        )
        prompt_input_response = StructuredPromptInputBuilder().build(prompt_input)

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.EXPLAIN,
                prompt_input=prompt_input_response,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn(
            "Lead with conceptual clarity and explain the cause-and-effect first.",
            system_section,
        )
        self.assertIn(
            "Use concise code snippets only when they sharpen the explanation.",
            system_section,
        )
        self.assertIn(
            "Avoid full runnable projects unless the user explicitly asks for them.",
            system_section,
        )

    def test_renderer_uses_issue_fix_why_guidance_for_debug_mode(self) -> None:
        renderer = JinjaPromptRenderer()
        prompt_input = build_prompt_input_request(
            assistant_request=AssistantRequest(
                query="Why is my shader black?",
                mode=AssistantMode.DEBUG,
            ),
            route_decision=RouteDecision(
                route=RouteName.DEBUG,
                mode=AssistantMode.DEBUG,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            assembled_context=None,
        )
        prompt_input_response = StructuredPromptInputBuilder().build(prompt_input)

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.DEBUG,
                prompt_input=prompt_input_response,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn(
            "Lead with the most likely issue before proposing changes.",
            system_section,
        )
        self.assertIn(
            "Structure the response as Issue, Fix, and Why it works.",
            system_section,
        )
        self.assertIn(
            "briefly ask for the missing code or error",
            system_section,
        )

    def test_renderer_adds_multi_domain_discipline_guidance(self) -> None:
        renderer = JinjaPromptRenderer()
        assistant_request = AssistantRequest(
            query="Explain how React Three Fiber and GLSL fit together.",
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=RouteDecision(
                    route=RouteName.EXPLAIN,
                    mode=AssistantMode.EXPLAIN,
                    domains=assistant_request.domains,
                    capabilities=(RouteCapability.OFFICIAL_DOCS,),
                ),
                assembled_context=None,
            )
        )

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.EXPLAIN,
                prompt_input=prompt_input,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn("Domain Scope: multi-domain selection", system_section)
        self.assertIn("Effective Domains:", system_section)
        self.assertIn("- react_three_fiber", system_section)
        self.assertIn("- glsl", system_section)
        self.assertIn(
            "Bridge domains only when the request actually spans them",
            system_section,
        )
        self.assertIn("Prefer React Three Fiber components and hooks", system_section)
        self.assertIn("Prefer concrete shader snippets", system_section)

    def test_renderer_prioritizes_explicit_query_domain_over_ui_selection(self) -> None:
        renderer = JinjaPromptRenderer()
        assistant_request = AssistantRequest(
            query="Create a p5.js sketch with a bouncing ball.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=RouteDecision(
                    route=RouteName.GENERATE,
                    mode=AssistantMode.GENERATE,
                    domains=assistant_request.domains,
                    capabilities=(RouteCapability.OFFICIAL_DOCS,),
                ),
                assembled_context=None,
            )
        )

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.GENERATE,
                prompt_input=prompt_input,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn("Domain Scope: p5_js", system_section)
        self.assertIn("Effective Domains:", system_section)
        self.assertIn("Detected Query Domains:", system_section)
        self.assertIn("UI Selected Domains:", system_section)
        self.assertIn("- p5_js", system_section)
        self.assertIn("- three_js", system_section)
        self.assertIn("- react_three_fiber", system_section)
        self.assertIn(
            "Prioritize the explicitly detected query domains",
            system_section,
        )
        self.assertIn(
            "Prefer p5.js sketch structure such as setup(), draw()",
            system_section,
        )
        self.assertNotIn(
            "Prefer plain Three.js patterns over React wrappers",
            system_section,
        )
        self.assertNotIn(
            "Prefer React Three Fiber components and hooks",
            system_section,
        )

    def test_renderer_uses_ui_selected_domains_when_query_is_ambiguous(self) -> None:
        renderer = JinjaPromptRenderer()
        assistant_request = AssistantRequest(
            query="Create a rotating cube.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=RouteDecision(
                    route=RouteName.GENERATE,
                    mode=AssistantMode.GENERATE,
                    domains=assistant_request.domains,
                    capabilities=(RouteCapability.OFFICIAL_DOCS,),
                ),
                assembled_context=None,
            )
        )

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.GENERATE,
                prompt_input=prompt_input,
            )
        )

        system_section = rendered.sections[0].content
        self.assertIn("Domain Scope: multi-domain selection", system_section)
        self.assertIn("Effective Domains:", system_section)
        self.assertIn("- three_js", system_section)
        self.assertIn("- react_three_fiber", system_section)
        self.assertNotIn("Detected Query Domains:", system_section)
        self.assertNotIn("UI Selected Domains:", system_section)

    def test_renderer_skips_empty_memory_section(self) -> None:
        renderer = JinjaPromptRenderer()
        assistant_request = AssistantRequest(
            query="Create a simple rotating cube in three.js",
            conversation_id="conversation-1",
            mode=AssistantMode.GENERATE,
        )
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=RouteDecision(
                    route=RouteName.GENERATE,
                    mode=AssistantMode.GENERATE,
                    capabilities=(RouteCapability.MEMORY_CONTEXT,),
                ),
                assembled_context=AssembledContextResponse(
                    request=build_assembled_context_request(
                        route_decision=RouteDecision(
                            route=RouteName.GENERATE,
                            mode=AssistantMode.GENERATE,
                            capabilities=(RouteCapability.MEMORY_CONTEXT,),
                        ),
                        memory_context=_empty_memory_context(),
                        retrieval_context=None,
                    ),
                    summary=AssembledContextSummary(
                        recent_turn_count=0,
                        has_running_summary=False,
                        project_memory_count=0,
                        retrieval_chunk_count=0,
                    ),
                    memory_context=_empty_memory_context(),
                    retrieval_context=None,
                ),
            )
        )

        rendered = renderer.render(
            build_rendered_prompt_request(
                route_decision=RouteName.GENERATE,
                prompt_input=prompt_input,
            )
        )

        self.assertEqual(
            [section.name for section in rendered.sections],
            [
                RenderedPromptSectionName.SYSTEM,
                RenderedPromptSectionName.USER,
            ],
        )

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

    def test_service_generates_answer_with_empty_memory_and_retrieval(self) -> None:
        service = AssistantService(
            route_fn=_route_generate_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_empty_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_empty_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
            prompt_renderer=JinjaPromptRenderer(),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=_CompletedGenerationProvider(
                answer="Use `requestAnimationFrame` and rotate the mesh on the y-axis."
            ),
        )
        request = AssistantRequest(
            query="Create a simple rotating cube in three.js",
            conversation_id="conversation-1",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )

        events = tuple(service.stream(request))
        event_types = [event.event_type for event in events]

        self.assertIn(StreamEventType.PROMPT_RENDERED, event_types)
        self.assertIn(StreamEventType.GENERATION_INPUT, event_types)
        self.assertEqual(
            events[-1].payload["answer"],
            "Use `requestAnimationFrame` and rotate the mesh on the y-axis.",
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


def _route_generate_with_memory_and_docs(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
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


def _empty_memory_context() -> MemoryContextResponse:
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.GENERATE,
            conversation_id="conversation-1",
        ),
        source=MemoryContextSource.CHROMA_MEMORY,
        recent_turns=(),
        running_summary=None,
        project_memories=(),
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


def _empty_retrieval_context() -> RetrievalContextResponse:
    return RetrievalContextResponse(
        request=RetrievalContextRequest(
            query="Create a simple rotating cube in three.js",
            route=RouteName.GENERATE,
        ),
        source=RetrievalContextSource.OFFICIAL_KB,
        chunks=(),
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


class _CompletedGenerationProvider:
    def __init__(self, *, answer: str) -> None:
        self._answer = answer

    def stream(self, request):
        yield GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=GenerationResponse(
                request=request,
                output=GeneratedOutput(
                    content=self._answer,
                    finish_reason="stop",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
