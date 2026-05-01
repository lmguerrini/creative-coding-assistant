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
    DomainSelectionShape,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSource,
    OrchestrationContextAssembler,
    ProjectMemoryContext,
    PromptInputRequest,
    PromptUserInput,
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
)
from creative_coding_assistant.rag.sources import OfficialSourceType

_UNSET = object()


class PromptInputContractsTests(unittest.TestCase):
    def test_prompt_input_request_rejects_misaligned_context_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt-input route"):
            PromptInputRequest(
                route=RouteName.GENERATE,
                assistant_request=AssistantRequest(query="Explain the scene."),
                assembled_context=_assembled_context(route=RouteName.EXPLAIN),
            )

    def test_prompt_input_builder_transforms_assembled_context(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the scene setup.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=_route_decision(),
            assembled_context=_assembled_context(route=RouteName.EXPLAIN),
        )

        response = builder.build(request)

        self.assertEqual(
            response.user_input,
            PromptUserInput(
                query="Explain the scene setup.",
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.THREE_JS,
            ),
        )
        self.assertIsNotNone(response.memory_input)
        self.assertIsNotNone(response.retrieval_input)
        assert response.memory_input is not None
        assert response.retrieval_input is not None
        self.assertFalse(response.user_input.is_follow_up)
        self.assertEqual(len(response.memory_input.recent_turns), 0)
        self.assertEqual(len(response.memory_input.session_summaries), 2)
        self.assertEqual(
            response.memory_input.session_summaries[0].summary,
            "User continued the project.",
        )
        self.assertEqual(
            response.memory_input.session_summaries[1].summary,
            "Assistant responded with implementation guidance.",
        )
        self.assertEqual(
            response.memory_input.running_summary.content,
            "The user prefers restrained motion and calm palettes.",
        )
        self.assertEqual(
            response.retrieval_input.chunks[0].document_title,
            "PerspectiveCamera",
        )

    def test_prompt_input_builder_keeps_compact_prior_pair_for_follow_up(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Now make it rotate faster and add a blue material.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                domain=CreativeCodingDomain.THREE_JS,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.GENERATE,
                memory_context=_memory_context_with_code_answer(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        self.assertTrue(response.user_input.is_follow_up)
        self.assertEqual(len(response.memory_input.recent_turns), 2)
        self.assertEqual(
            response.memory_input.recent_turns[0].content,
            "Create a simple rotating cube in three.js.",
        )
        assistant_turn = response.memory_input.recent_turns[1].content
        self.assertIn("Relevant code excerpt:", assistant_turn)
        self.assertIn("```html", assistant_turn)
        self.assertIn("requestAnimationFrame", assistant_turn)
        self.assertEqual(len(response.memory_input.session_summaries), 2)
        self.assertLess(
            len(assistant_turn),
            len(_LONG_CODE_ANSWER),
        )

    def test_prompt_input_builder_caps_session_memory_summaries_at_five(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the current project direction.",
            domain=CreativeCodingDomain.P5_JS,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.P5_JS,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.EXPLAIN,
                memory_context=_memory_context_with_many_turns(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        self.assertEqual(len(response.memory_input.session_summaries), 5)
        summaries = tuple(
            item.summary for item in response.memory_input.session_summaries
        )
        self.assertNotIn("User requested Three.js rotating cube.", summaries)
        self.assertIn("Assistant generated p5.js sketch.", summaries)

    def test_prompt_input_builder_supports_user_only_flow(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(query="Start a shader study.")
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                capabilities=(RouteCapability.TOOL_USE,),
            ),
            assembled_context=None,
        )

        response = builder.build(request)

        self.assertEqual(response.user_input.query, "Start a shader study.")
        self.assertIsNone(response.memory_input)
        self.assertIsNone(response.retrieval_input)

    def test_prompt_input_builder_preserves_multi_domain_selection(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain how React Three Fiber and GLSL fit together.",
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domains=assistant_request.domains,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            assembled_context=None,
        )

        response = builder.build(request)

        self.assertIsNone(response.user_input.domain)
        self.assertEqual(
            response.user_input.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )
        self.assertEqual(
            response.user_input.domain_selection,
            DomainSelectionShape.MULTI,
        )
        self.assertEqual(
            response.user_input.ui_selected_domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )
        self.assertEqual(
            response.user_input.effective_domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_prompt_input_builder_prefers_explicit_query_domain_over_ui_selection(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Create a p5.js sketch with a bouncing ball.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                domains=assistant_request.domains,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            assembled_context=None,
        )

        response = builder.build(request)

        self.assertEqual(response.user_input.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(
            response.user_input.ui_selected_domains,
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
        )
        self.assertEqual(
            response.user_input.detected_domains,
            (CreativeCodingDomain.P5_JS,),
        )
        self.assertEqual(
            response.user_input.effective_domains,
            (CreativeCodingDomain.P5_JS,),
        )

    def test_prompt_input_builder_uses_ui_selection_when_query_is_ambiguous(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Create a rotating cube.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                domains=assistant_request.domains,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            assembled_context=None,
        )

        response = builder.build(request)

        self.assertEqual(response.user_input.detected_domains, ())
        self.assertEqual(
            response.user_input.effective_domains,
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
        )

    def test_prompt_input_builder_keeps_explicit_new_domain_over_previous_memory(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Convert this to p5.js.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                domains=assistant_request.domains,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.GENERATE,
                memory_context=_memory_context_with_code_answer(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        self.assertTrue(response.user_input.is_follow_up)
        self.assertEqual(response.user_input.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(
            response.user_input.effective_domains,
            (CreativeCodingDomain.P5_JS,),
        )
        assert response.memory_input is not None
        self.assertEqual(len(response.memory_input.recent_turns), 2)

    def test_service_emits_prompt_input_event_when_builder_present(self) -> None:
        service = AssistantService(
            route_fn=_route_with_memory_and_docs,
            memory_gateway=_FakeMemoryGateway(response=_memory_context()),
            retrieval_gateway=_FakeRetrievalGateway(response=_retrieval_context()),
            context_assembler=OrchestrationContextAssembler(),
            prompt_input_builder=StructuredPromptInputBuilder(),
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
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[7].payload["code"], "prompt_inputs_prepared")
        prompt_input = events[7].payload["prompt_input"]
        self.assertEqual(
            prompt_input["user_input"]["query"],
            "Explain the scene setup.",
        )
        self.assertEqual(len(prompt_input["memory_input"]["recent_turns"]), 0)
        self.assertEqual(len(prompt_input["retrieval_input"]["chunks"]), 1)


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


def _assembled_context(
    *,
    route: RouteName,
    memory_context: MemoryContextResponse | None = None,
    retrieval_context: RetrievalContextResponse | None | object = _UNSET,
) -> AssembledContextResponse:
    memory_context = _memory_context() if memory_context is None else memory_context
    retrieval_context = (
        _retrieval_context()
        if retrieval_context is _UNSET
        else retrieval_context
    )
    request = build_assembled_context_request(
        route_decision=RouteDecision(
            route=route,
            mode=AssistantMode.EXPLAIN,
            capabilities=(
                RouteCapability.MEMORY_CONTEXT,
                RouteCapability.OFFICIAL_DOCS,
            ),
        ),
        memory_context=memory_context,
        retrieval_context=retrieval_context,
    )
    assert request is not None
    return AssembledContextResponse(
        request=request,
        summary=AssembledContextSummary(
            recent_turn_count=(
                len(memory_context.recent_turns) if memory_context is not None else 0
            ),
            has_running_summary=(
                memory_context.running_summary is not None
                if memory_context is not None
                else False
            ),
            project_memory_count=(
                len(memory_context.project_memories)
                if memory_context is not None
                else 0
            ),
            retrieval_chunk_count=(
                len(retrieval_context.chunks)
                if retrieval_context is not None
                else 0
            ),
        ),
        memory_context=memory_context,
        retrieval_context=retrieval_context,
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


def _memory_context_with_code_answer() -> MemoryContextResponse:
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.GENERATE,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        source=MemoryContextSource.CHROMA_MEMORY,
        recent_turns=(
            RecentConversationTurn(
                turn_index=0,
                role=ConversationRole.USER,
                content="Create a simple rotating cube in three.js.",
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content=_LONG_CODE_ANSWER,
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=ConversationSummaryContext(
            content="The conversation is building a basic Three.js cube scene.",
            created_at=_time(),
            covered_turn_count=2,
        ),
        project_memories=(),
    )


def _memory_context_with_many_turns() -> MemoryContextResponse:
    turns = (
        RecentConversationTurn(
            turn_index=0,
            role=ConversationRole.USER,
            content="Create a rotating cube in three.js.",
            created_at=_time(),
            mode=AssistantMode.GENERATE,
        ),
        RecentConversationTurn(
            turn_index=1,
            role=ConversationRole.ASSISTANT,
            content="Use a Three.js scene with a box mesh.",
            created_at=_time(),
            mode=AssistantMode.GENERATE,
        ),
        RecentConversationTurn(
            turn_index=2,
            role=ConversationRole.USER,
            content="Convert it to p5.js.",
            created_at=_time(),
            mode=AssistantMode.GENERATE,
        ),
        RecentConversationTurn(
            turn_index=3,
            role=ConversationRole.ASSISTANT,
            content=(
                "```javascript\nfunction setup() { createCanvas(400, 400); }\n"
                "function draw() { background(220); }\n```"
            ),
            created_at=_time(),
            mode=AssistantMode.GENERATE,
        ),
        RecentConversationTurn(
            turn_index=4,
            role=ConversationRole.USER,
            content="Explain the shader idea in GLSL.",
            created_at=_time(),
            mode=AssistantMode.EXPLAIN,
        ),
        RecentConversationTurn(
            turn_index=5,
            role=ConversationRole.ASSISTANT,
            content="GLSL shaders run on the GPU and compute fragment color.",
            created_at=_time(),
            mode=AssistantMode.EXPLAIN,
        ),
    )
    return MemoryContextResponse(
        request=MemoryContextRequest(
            route=RouteName.EXPLAIN,
            conversation_id="conversation-1",
            project_id="project-1",
        ),
        source=MemoryContextSource.CHROMA_MEMORY,
        recent_turns=turns,
        running_summary=ConversationSummaryContext(
            content="The conversation spans Three.js, p5.js, and GLSL experiments.",
            created_at=_time(),
            covered_turn_count=6,
        ),
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


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


_LONG_CODE_ANSWER = """
Use this HTML file as the starting point for the rotating cube.

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Rotating cube</title>
    <style>
      body { margin: 0; overflow: hidden; }
      canvas { display: block; }
    </style>
  </head>
  <body>
    <script type="module">
      import * as THREE from "three";

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000,
      );
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(renderer.domElement);

      const geometry = new THREE.BoxGeometry();
      const material = new THREE.MeshStandardMaterial({ color: 0x44aa88 });
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);

      const light = new THREE.DirectionalLight(0xffffff, 1.5);
      light.position.set(2, 3, 4);
      scene.add(light);

      camera.position.z = 4;

      function animate() {
        requestAnimationFrame(animate);
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.015;
        renderer.render(scene, camera);
      }

      animate();
    </script>
  </body>
</html>
```

You can extend it with controls later.
""".strip()


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
