import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantArtifactRefinement,
    AssistantImageReference,
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
    PromptImageReferenceInput,
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
from event_assertions import first_event, legacy_events

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

    def test_prompt_input_builder_preserves_artifact_refinement_context(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Make this calmer.",
            domain=CreativeCodingDomain.P5_JS,
            mode=AssistantMode.GENERATE,
            artifact_refinement=AssistantArtifactRefinement(
                artifactId="source-sketch",
                title="aurora-field.p5.js",
                language="p5.js",
                content="function draw() { background(0); }",
                instruction="Make this calmer.",
                domain=CreativeCodingDomain.P5_JS,
                runtime="p5",
                rendererId="surface.p5",
                previewEligible=True,
                qualityScore=0.88,
                qualityRank=2,
                critiqueRationale="Stable sketch with useful motion.",
                refinementGuidance="Reduce visual density.",
                creativeTranslation={
                    "output_modality": "visual",
                    "creative_intent": "Create a calm cyan particle field.",
                    "symbolic_references": [],
                    "geometric_references": [],
                    "musical_references": [],
                    "mood_atmosphere": ["calm"],
                    "movement_language": ["drift"],
                    "color_material_direction": ["cyan"],
                    "runtime_recommendations": ["p5.js"],
                    "structure_direction": [],
                    "generation_constraints": [],
                    "refinement_targets": ["Preserve atmosphere: calm"],
                    "sacred_geometry": {
                        "concepts": ["mandala"],
                        "geometric_structure": [
                            "Build nested rings around a clear center."
                        ],
                        "symmetry_type": ["Use radial symmetry."],
                        "movement_behavior": [],
                        "visual_composition": [],
                        "color_material_direction": [],
                        "runtime_recommendations": ["p5.js"],
                        "audio_implications": [],
                        "generation_constraints": [
                            "Do not add unsupported symbolic claims."
                        ],
                    },
                    "shader_presets": {
                        "presets": ["glow"],
                        "color_behavior": ["Use a bright core color."],
                        "light_material_behavior": [
                            "Use bounded emission layers."
                        ],
                        "motion_behavior": ["Pulse intensity slowly."],
                        "shader_structure": ["Separate an emission mask."],
                        "runtime_suitability": [
                            "Use the selected compatible runtime: p5.js."
                        ],
                        "performance_constraints": [
                            "Use a bounded number of glow layers."
                        ],
                    },
                    "visual_style": {
                        "styles": ["minimal"],
                        "palette_behavior": [
                            "Use one dominant tone plus one restrained accent."
                        ],
                        "contrast_behavior": [
                            "Create hierarchy through spacing and value."
                        ],
                        "composition_tendencies": [
                            "Use deliberate negative space."
                        ],
                        "motion_tendencies": ["Use slow, readable transitions."],
                        "texture_tendencies": ["Keep surfaces clean."],
                        "spatial_organization": ["Favor a stable focal point."],
                        "runtime_suitability": [
                            "Use the selected compatible runtime: p5.js."
                        ],
                    },
                },
            ),
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                domain=CreativeCodingDomain.P5_JS,
                capabilities=(RouteCapability.TOOL_USE,),
            ),
            assembled_context=None,
        )

        response = builder.build(request)

        refinement = response.user_input.artifact_refinement
        self.assertIsNotNone(refinement)
        assert refinement is not None
        self.assertEqual(refinement.artifact_id, "source-sketch")
        self.assertEqual(refinement.title, "aurora-field.p5.js")
        self.assertEqual(refinement.content, "function draw() { background(0); }")
        self.assertEqual(refinement.runtime, "p5")
        self.assertEqual(refinement.quality_score, 0.88)
        self.assertEqual(
            refinement.critique_rationale,
            "Stable sketch with useful motion.",
        )
        self.assertIsNotNone(refinement.creative_translation)
        assert refinement.creative_translation is not None
        self.assertEqual(
            refinement.creative_translation.creative_intent,
            "Create a calm cyan particle field.",
        )
        self.assertEqual(
            response.creative_translation.mood_atmosphere,
            ("calm",),
        )
        self.assertIsNotNone(response.creative_translation.sacred_geometry)
        assert response.creative_translation.sacred_geometry is not None
        self.assertEqual(
            response.creative_translation.sacred_geometry.concepts,
            ("mandala",),
        )
        self.assertIsNotNone(response.creative_translation.shader_presets)
        assert response.creative_translation.shader_presets is not None
        self.assertEqual(
            tuple(
                preset.value
                for preset in response.creative_translation.shader_presets.presets
            ),
            ("glow", "kaleidoscopic symmetry"),
        )
        self.assertIsNotNone(response.creative_translation.visual_style)
        assert response.creative_translation.visual_style is not None
        self.assertEqual(
            tuple(
                style.value
                for style in response.creative_translation.visual_style.styles
            ),
            ("minimal", "sacred geometry", "psychedelic"),
        )

    def test_prompt_input_uses_bounded_dynamic_parameters_for_mapping(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Apply the selected artifact parameter changes.",
            domains=(
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.TONE_JS,
            ),
            artifact_refinement=AssistantArtifactRefinement(
                artifactId="reactive-field",
                title="reactive-field.p5.js",
                language="p5.js",
                content=(
                    "const fft = new Tone.FFT(); "
                    "Tone.Transport.bpm.value = 96;"
                ),
                instruction=(
                    "Rotation speed: 1.2 x\n"
                    "Bloom intensity: 1.4 x\n"
                    "Treat these values as refinement guidance."
                ),
                runtime="p5",
                creativeTranslation={
                    "outputModality": "audiovisual",
                    "creativeIntent": "Create an audio-reactive field.",
                    "runtimeRecommendations": ["p5.js", "Tone.js"],
                    "audioReactive": {
                        "mappings": [
                            {
                                "source": "amplitude",
                                "targets": ["scale", "brightness"],
                                "behavior": "Smooth short peaks.",
                            }
                        ],
                        "audioRuntime": "Tone.js",
                        "visualRuntime": "p5.js",
                        "summary": "amplitude -> scale / brightness",
                    },
                },
            ),
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            domains=assistant_request.domains,
            capabilities=(RouteCapability.TOOL_USE,),
        )

        response = builder.build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=route_decision,
                assembled_context=None,
            )
        )

        mapping = response.creative_translation.audio_reactive
        self.assertIsNotNone(mapping)
        assert mapping is not None
        sources = tuple(item.source.value for item in mapping.mappings)
        self.assertIn("rhythm", sources)
        self.assertIn("envelope", sources)
        self.assertTrue(
            all("dynamic parameters" in item.evidence for item in mapping.mappings)
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

    def test_prompt_input_builder_labels_second_v2_session_memory_domains(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the current animation project.",
            domain=CreativeCodingDomain.GSAP,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.GSAP,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.EXPLAIN,
                memory_context=_memory_context_with_gsap_turns(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        summaries = tuple(
            item.summary for item in response.memory_input.session_summaries
        )
        self.assertIn("User requested GSAP project work.", summaries)
        self.assertIn("Assistant generated GSAP animation code.", summaries)

    def test_prompt_input_builder_labels_third_v2_session_memory_domains(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the current TouchDesigner network.",
            domain=CreativeCodingDomain.TOUCHDESIGNER,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.TOUCHDESIGNER,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.EXPLAIN,
                memory_context=_memory_context_with_touchdesigner_turns(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        summaries = tuple(
            item.summary for item in response.memory_input.session_summaries
        )
        self.assertIn("User requested TouchDesigner project work.", summaries)
        self.assertIn(
            "Assistant generated TouchDesigner operator network guidance.",
            summaries,
        )

    def test_prompt_input_builder_labels_fourth_v2_session_memory_domains(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the current ComfyUI workflow.",
            domain=CreativeCodingDomain.COMFYUI,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.COMFYUI,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.EXPLAIN,
                memory_context=_memory_context_with_comfyui_turns(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        summaries = tuple(
            item.summary for item in response.memory_input.session_summaries
        )
        self.assertIn("User requested ComfyUI project work.", summaries)
        self.assertIn(
            "Assistant generated ComfyUI node workflow guidance.",
            summaries,
        )

    def test_prompt_input_builder_labels_fifth_v2_session_memory_domains(
        self,
    ) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Explain the current Ableton Live set.",
            domain=CreativeCodingDomain.ABLETON_LIVE,
            mode=AssistantMode.EXPLAIN,
        )
        request = build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=AssistantMode.EXPLAIN,
                domain=CreativeCodingDomain.ABLETON_LIVE,
                capabilities=(RouteCapability.MEMORY_CONTEXT,),
            ),
            assembled_context=_assembled_context(
                route=RouteName.EXPLAIN,
                memory_context=_memory_context_with_ableton_turns(),
                retrieval_context=None,
            ),
        )

        response = builder.build(request)

        assert response.memory_input is not None
        summaries = tuple(
            item.summary for item in response.memory_input.session_summaries
        )
        self.assertIn("User requested Ableton Live project work.", summaries)
        self.assertIn(
            "Assistant generated Ableton Live DAW workflow guidance.",
            summaries,
        )

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

    def test_prompt_input_builder_preserves_image_references(self) -> None:
        builder = StructuredPromptInputBuilder()
        assistant_request = AssistantRequest(
            query="Use this palette reference for the particle field.",
            attachments=(
                AssistantImageReference(
                    id="image-reference-1",
                    name="palette.png",
                    mimeType="image/png",
                    sizeBytes=128,
                    dataUrl="data:image/png;base64,cGFsZXR0ZQ==",
                ),
            ),
        )
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

        self.assertEqual(
            response.user_input.image_references,
            (
                PromptImageReferenceInput(
                    id="image-reference-1",
                    name="palette.png",
                    mime_type="image/png",
                    size_bytes=128,
                ),
            ),
        )
        self.assertIsNotNone(response.creative_translation.reference_fusion)
        assert response.creative_translation.reference_fusion is not None
        self.assertEqual(
            response.creative_translation.reference_fusion.source_count,
            1,
        )

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
                StreamEventType.FINAL,
            ],
        )
        prompt_event = first_event(
            events,
            StreamEventType.PROMPT_INPUT,
            "prompt_inputs_prepared",
        )

        self.assertEqual(prompt_event.payload["code"], "prompt_inputs_prepared")
        prompt_input = prompt_event.payload["prompt_input"]
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


def _memory_context_with_gsap_turns() -> MemoryContextResponse:
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
                content="Create a GSAP timeline animation.",
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content=(
                    "```javascript\n"
                    "const tl = gsap.timeline();\n"
                    "tl.to('.box', { x: 100, duration: 1 });\n"
                    "```"
                ),
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=None,
        project_memories=(),
    )


def _memory_context_with_touchdesigner_turns() -> MemoryContextResponse:
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
                content="Create a TouchDesigner TOP operator network.",
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content=(
                    "```text\n"
                    "TouchDesigner network: Movie File In TOP -> Level TOP -> "
                    "Composite TOP\n"
                    "```"
                ),
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=None,
        project_memories=(),
    )


def _memory_context_with_comfyui_turns() -> MemoryContextResponse:
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
                content="Create a ComfyUI node workflow.",
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content=(
                    "```text\n"
                    "ComfyUI workflow: Load Checkpoint -> KSampler -> "
                    "VAE Decode -> Save Image\n"
                    "```"
                ),
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=None,
        project_memories=(),
    )


def _memory_context_with_ableton_turns() -> MemoryContextResponse:
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
                content="Create an Ableton Live Session View workflow.",
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.ASSISTANT,
                content=(
                    "```text\n"
                    "Ableton Live workflow: clips -> scenes -> racks -> "
                    "automation lanes\n"
                    "```"
                ),
                created_at=_time(),
                mode=AssistantMode.GENERATE,
            ),
        ),
        running_summary=None,
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
