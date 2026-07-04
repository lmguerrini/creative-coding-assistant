import unittest

from creative_coding_assistant.clients import (
    ChatHistoryEntry,
    DomainCategoryDisplayGroup,
    RetrievalDisplayItem,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    default_domain,
    default_domain_selection,
    default_mode,
    domain_category_groups,
    domain_selection_summary,
    mode_selection_summary,
    ordered_domain_selection,
    reduce_stream_event,
    resolve_request_domain,
    resolve_request_domains,
    resolve_session_domain_selection,
    resolve_session_mode,
    retrieval_empty_message,
    retrieval_expander_label,
    user_safe_assistant_error_message,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.domains import DomainCategory


class StreamlitChatGenerationTests(unittest.TestCase):
    def test_default_domain_and_mode_follow_settings(self) -> None:
        settings = Settings(
            default_domain=CreativeCodingDomain.GLSL,
            default_mode=AssistantMode.DEBUG,
        )

        self.assertEqual(default_domain(settings), CreativeCodingDomain.GLSL)
        self.assertEqual(default_mode(settings), AssistantMode.DEBUG)

    def test_default_domain_and_mode_fall_back_safely(self) -> None:
        settings = Settings(
            default_domain="not-a-domain",
            default_mode="not-a-mode",
        )

        self.assertEqual(default_domain(settings), CreativeCodingDomain.THREE_JS)
        self.assertEqual(default_mode(settings), AssistantMode.GENERATE)

    def test_build_provider_warning_requires_openai_api_key(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_api_key=None,
        )

        warning = build_provider_warning(settings)

        self.assertEqual(
            warning,
            "Set OPENAI_API_KEY or CCA_OPENAI_API_KEY to enable live generation.",
        )

    def test_build_provider_warning_is_clear_when_key_exists(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_api_key="sk-test-secret",
        )

        self.assertIsNone(build_provider_warning(settings))

    def test_user_safe_error_message_handles_network_failures(self) -> None:
        class APIConnectionError(Exception):
            pass

        APIConnectionError.__module__ = "openai"

        message = user_safe_assistant_error_message(APIConnectionError("secret-url"))

        self.assertEqual(
            message,
            "Connection issue: unable to reach the model. Please check your "
            "internet or API configuration.",
        )
        self.assertNotIn("secret-url", message)

    def test_user_safe_error_message_handles_retrieval_failures(self) -> None:
        class RetrievalFailure(Exception):
            pass

        RetrievalFailure.__module__ = "creative_coding_assistant.rag.retrieval.search"

        message = user_safe_assistant_error_message(RetrievalFailure("raw details"))

        self.assertEqual(
            message,
            "Knowledge base temporarily unavailable. Generating response without "
            "retrieval.",
        )
        self.assertNotIn("raw details", message)

    def test_user_safe_error_message_handles_unknown_failures(self) -> None:
        message = user_safe_assistant_error_message(ValueError("sk-secret-never-show"))

        self.assertEqual(
            message,
            "Something went wrong while generating the response. Please try again.",
        )
        self.assertNotIn("sk-secret-never-show", message)

    def test_build_chat_request_uses_ui_selection_and_conversation_id(self) -> None:
        settings = Settings()

        request = build_chat_request(
            query="Explain the shader noise pattern.",
            conversation_id="conversation-123",
            settings=settings,
            domains=(CreativeCodingDomain.GLSL,),
            mode=AssistantMode.EXPLAIN,
        )

        self.assertEqual(request.query, "Explain the shader noise pattern.")
        self.assertEqual(request.conversation_id, "conversation-123")
        self.assertEqual(request.domain, CreativeCodingDomain.GLSL)
        self.assertEqual(request.domains, (CreativeCodingDomain.GLSL,))
        self.assertEqual(request.mode, AssistantMode.EXPLAIN)

    def test_default_domain_selection_uses_small_core_domain_set(self) -> None:
        self.assertEqual(
            default_domain_selection(),
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_resolve_session_domain_selection_defaults_to_core_domains(self) -> None:
        self.assertEqual(
            resolve_session_domain_selection(None),
            default_domain_selection(),
        )

    def test_resolve_session_domain_selection_preserves_empty_selection(self) -> None:
        self.assertEqual(resolve_session_domain_selection(()), ())

    def test_resolve_session_domain_selection_filters_invalid_values(self) -> None:
        self.assertEqual(
            resolve_session_domain_selection(
                (
                    CreativeCodingDomain.GLSL,
                    "invalid_domain",
                    "three_js",
                )
            ),
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_resolve_session_domain_selection_falls_back_when_all_are_invalid(
        self,
    ) -> None:
        self.assertEqual(
            resolve_session_domain_selection(("invalid_domain", "still_invalid")),
            default_domain_selection(),
        )

    def test_domain_category_groups_cover_all_domains_in_registry_order(self) -> None:
        groups = domain_category_groups()

        self.assertEqual(
            groups[0],
            DomainCategoryDisplayGroup(
                category=DomainCategory.WEB_CREATIVE_CODING,
                label="Web Creative Coding",
                domains=(
                    CreativeCodingDomain.THREE_JS,
                    CreativeCodingDomain.REACT_THREE_FIBER,
                    CreativeCodingDomain.P5_JS,
                    CreativeCodingDomain.PROCESSING,
                    CreativeCodingDomain.CANVAS_2D,
                    CreativeCodingDomain.PIXI_JS,
                    CreativeCodingDomain.OPENFRAMEWORKS,
                    CreativeCodingDomain.OPENRNDR,
                ),
            ),
        )
        self.assertEqual(
            groups[1],
            DomainCategoryDisplayGroup(
                category=DomainCategory.SHADERS_GPU,
                label="Shader / GPU",
                domains=(
                    CreativeCodingDomain.GLSL,
                    CreativeCodingDomain.WEBGPU_WGSL,
                    CreativeCodingDomain.SHADERTOY,
                ),
            ),
        )
        grouped_domains = tuple(domain for group in groups for domain in group.domains)
        self.assertEqual(len(grouped_domains), len(CreativeCodingDomain))
        self.assertEqual(set(grouped_domains), set(CreativeCodingDomain))

    def test_ordered_domain_selection_uses_registry_order(self) -> None:
        self.assertEqual(
            ordered_domain_selection(
                (
                    CreativeCodingDomain.PURE_DATA,
                    CreativeCodingDomain.GLSL,
                    CreativeCodingDomain.THREE_JS,
                    CreativeCodingDomain.GLSL,
                )
            ),
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.PURE_DATA,
            ),
        )

    def test_resolve_session_mode_defaults_and_falls_back_safely(self) -> None:
        settings = Settings(default_mode=AssistantMode.DEBUG)

        self.assertEqual(
            resolve_session_mode(None, settings=settings),
            AssistantMode.DEBUG,
        )
        self.assertEqual(
            resolve_session_mode("explain", settings=settings),
            AssistantMode.EXPLAIN,
        )
        self.assertEqual(
            resolve_session_mode("invalid_mode", settings=settings),
            AssistantMode.DEBUG,
        )

    def test_sidebar_selection_summaries_are_readable(self) -> None:
        self.assertEqual(
            domain_selection_summary(tuple(CreativeCodingDomain)),
            "All 43 domains selected across 12 categories",
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.REACT_THREE_FIBER,
                    CreativeCodingDomain.GLSL,
                )
            ),
            "2 selected: React Three Fiber, GLSL (Web Creative Coding, Shader / GPU)",
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.PROCESSING,
                    CreativeCodingDomain.CANVAS_2D,
                    CreativeCodingDomain.WEBGPU_WGSL,
                )
            ),
            (
                "3 selected: Processing, Canvas 2D, WebGPU/WGSL "
                "(Web Creative Coding, Shader / GPU)"
            ),
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.GSAP,
                    CreativeCodingDomain.TONE_JS,
                    CreativeCodingDomain.PIXI_JS,
                )
            ),
            (
                "3 selected: GSAP, Tone.js, PixiJS "
                "(Animation, Audio / Music Tech, Web Creative Coding)"
            ),
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.TOUCHDESIGNER,
                    CreativeCodingDomain.HOUDINI,
                    CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
                )
            ),
            (
                "3 selected: TouchDesigner, Houdini, Blender / Geometry Nodes "
                "(Visual Patching, Procedural / DCC)"
            ),
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.OPENFRAMEWORKS,
                    CreativeCodingDomain.WEB_AUDIO_API,
                    CreativeCodingDomain.COMFYUI,
                )
            ),
            (
                "3 selected: openFrameworks, Web Audio API, ComfyUI "
                "(Web Creative Coding, Audio / Music Tech, AI Creative Tools)"
            ),
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.ABLETON_LIVE,
                    CreativeCodingDomain.VCV_RACK,
                    CreativeCodingDomain.PURE_DATA,
                )
            ),
            (
                "3 selected: Ableton Live, VCV Rack, Pure Data "
                "(Audio / Music Tech, Modular Synthesis, Visual Patching)"
            ),
        )
        self.assertEqual(
            domain_selection_summary(()),
            "No domain filter (all domains eligible)",
        )
        self.assertEqual(
            mode_selection_summary(AssistantMode.EXPLAIN),
            "Explain",
        )

    def test_resolve_request_domain_returns_none_for_multiple_domains(self) -> None:
        resolved = resolve_request_domain(
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            )
        )

        self.assertIsNone(resolved)

    def test_resolve_request_domain_returns_none_for_empty_selection(self) -> None:
        resolved = resolve_request_domain(())

        self.assertIsNone(resolved)

    def test_resolve_request_domains_defaults_to_configured_domain(self) -> None:
        settings = Settings(default_domain=CreativeCodingDomain.GLSL)

        resolved = resolve_request_domains(None, settings=settings)

        self.assertEqual(resolved, (CreativeCodingDomain.GLSL,))

    def test_resolve_request_domains_preserves_empty_selection(self) -> None:
        settings = Settings(default_domain=CreativeCodingDomain.THREE_JS)

        resolved = resolve_request_domains((), settings=settings)

        self.assertEqual(resolved, ())

    def test_build_chat_request_leaves_domain_unconstrained_for_multi_select(
        self,
    ) -> None:
        settings = Settings()

        request = build_chat_request(
            query="Explain how R3F and GLSL fit together.",
            conversation_id="conversation-456",
            settings=settings,
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )

        self.assertIsNone(request.domain)
        self.assertEqual(
            request.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_reduce_stream_event_tracks_pipeline_progress_and_tokens(self) -> None:
        state = StreamRenderState()

        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.STATUS,
                sequence=0,
                payload={"message": "Request accepted."},
            ),
        )
        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.TOKEN_DELTA,
                sequence=1,
                payload={"text": "Hello"},
            ),
        )
        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.TOKEN_DELTA,
                sequence=2,
                payload={"text": " world"},
            ),
        )

        self.assertEqual(state.status_message, "Receiving response...")
        self.assertEqual(state.answer_text, "Hello world")
        self.assertIsNone(state.final_answer)
        self.assertTrue(state.is_streaming_answer)

    def test_reduce_stream_event_shows_generation_progress_status(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.GENERATION_INPUT,
                sequence=0,
                payload={
                    "code": "generation_input_prepared",
                    "message": "Provider generation input prepared.",
                    "generation_input": {
                        "request": {"route": "generate", "stream": True},
                        "messages": [],
                    },
                },
            ),
        )

        self.assertEqual(state.status_message, "Preparing response...")

    def test_reduce_stream_event_prefers_final_answer(self) -> None:
        state = StreamRenderState(streamed_text="Partial answer")

        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.FINAL,
                sequence=3,
                payload={"answer": "Completed answer"},
            ),
        )

        self.assertEqual(state.final_answer, "Completed answer")
        self.assertEqual(state.answer_text, "Completed answer")
        self.assertIsNone(state.status_message)
        self.assertFalse(state.is_streaming_answer)

    def test_reduce_stream_event_records_error_message(self) -> None:
        state = StreamRenderState()

        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.ERROR,
                sequence=4,
                payload={"message": "Provider request failed."},
            ),
        )

        self.assertEqual(state.error_message, "Provider request failed.")

    def test_reduce_stream_event_extracts_retrieval_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.RETRIEVAL,
                sequence=5,
                payload={
                    "code": "retrieval_completed",
                    "message": "Retrieval context prepared.",
                    "context": {
                        "chunks": [
                            {
                                "source_id": "three_docs",
                                "domain": "three_js",
                                "registry_title": "three.js Documentation",
                                "document_title": "PerspectiveCamera",
                                "excerpt": (
                                    "PerspectiveCamera defines a viewing frustum "
                                    "that is widely used for scene rendering."
                                ),
                                "score": 0.82,
                            }
                        ]
                    },
                },
            ),
        )

        self.assertEqual(state.retrieval_state, "available")
        self.assertEqual(
            state.retrieval_items,
            (
                RetrievalDisplayItem(
                    source_id="three_docs",
                    title="PerspectiveCamera",
                    domain=CreativeCodingDomain.THREE_JS,
                    score=0.82,
                    snippet=(
                        "PerspectiveCamera defines a viewing frustum that is "
                        "widely used for scene rendering."
                    ),
                ),
            ),
        )

    def test_reduce_stream_event_tracks_trace_domains_from_route_and_retrieval(
        self,
    ) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.STATUS,
                sequence=4,
                payload={
                    "code": "route_selected",
                    "message": "Route selected.",
                    "route": {
                        "route": "generate",
                        "domains": ["three_js", "react_three_fiber"],
                    },
                },
            ),
        )
        state = reduce_stream_event(
            state,
            StreamEvent(
                event_type=StreamEventType.RETRIEVAL,
                sequence=5,
                payload={
                    "code": "retrieval_requested",
                    "message": "Retrieval context requested.",
                    "request": {
                        "query": "Create a p5.js sketch with a bouncing ball.",
                        "filters": {
                            "domain": "p5_js",
                            "domains": ["p5_js"],
                        },
                    },
                },
            ),
        )

        self.assertEqual(
            state.ui_domains,
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
        )
        self.assertEqual(
            state.detected_domains,
            (CreativeCodingDomain.P5_JS,),
        )
        self.assertEqual(
            state.retrieval_domains,
            (CreativeCodingDomain.P5_JS,),
        )

    def test_reduce_stream_event_tracks_multi_domain_query_detection(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.RETRIEVAL,
                sequence=5,
                payload={
                    "code": "retrieval_requested",
                    "message": "Retrieval context requested.",
                    "request": {
                        "query": (
                            "Create a shader material in react three fiber using GLSL."
                        ),
                        "filters": {
                            "domains": ["react_three_fiber", "glsl"],
                        },
                    },
                },
            ),
        )

        self.assertEqual(
            state.detected_domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )
        self.assertEqual(
            state.retrieval_domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_reduce_stream_event_marks_empty_retrieval_context(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.RETRIEVAL,
                sequence=6,
                payload={
                    "code": "retrieval_completed",
                    "message": "Retrieval context prepared.",
                    "context": {"chunks": []},
                },
            ),
        )

        self.assertEqual(state.retrieval_state, "empty")
        self.assertEqual(state.retrieval_items, ())

    def test_assistant_history_entry_prefers_final_answer(self) -> None:
        state = StreamRenderState(
            streamed_text="Partial answer",
            final_answer="Final answer",
            error_message="Provider request failed.",
        )

        entry = assistant_history_entry(state)

        self.assertEqual(
            entry,
            ChatHistoryEntry(role="assistant", content="Final answer"),
        )

    def test_assistant_history_entry_preserves_retrieval_visibility(self) -> None:
        state = StreamRenderState(
            final_answer="Final answer",
            ui_domains=(CreativeCodingDomain.THREE_JS,),
            detected_domains=(CreativeCodingDomain.P5_JS,),
            retrieval_domains=(CreativeCodingDomain.P5_JS,),
            retrieval_state="available",
            retrieval_items=(
                RetrievalDisplayItem(
                    source_id="three_docs",
                    title="PerspectiveCamera",
                    domain=CreativeCodingDomain.THREE_JS,
                    score=0.82,
                    snippet="Camera setup details.",
                ),
            ),
        )

        entry = assistant_history_entry(state)

        self.assertEqual(entry.ui_domains, (CreativeCodingDomain.THREE_JS,))
        self.assertEqual(entry.detected_domains, (CreativeCodingDomain.P5_JS,))
        self.assertEqual(entry.retrieval_domains, (CreativeCodingDomain.P5_JS,))
        self.assertEqual(entry.retrieval_state, "available")
        self.assertEqual(len(entry.retrieval_items), 1)

    def test_retrieval_expander_helpers_handle_empty_and_available_states(self) -> None:
        label = retrieval_expander_label(
            (
                RetrievalDisplayItem(
                    source_id="three_docs",
                    title="PerspectiveCamera",
                    domain=CreativeCodingDomain.THREE_JS,
                    score=0.82,
                    snippet="Camera setup details.",
                ),
            ),
            retrieval_state="available",
        )

        self.assertEqual(label, "Retrieval context (1 chunk)")
        self.assertEqual(
            retrieval_empty_message("empty"),
            "No retrieval context was found for this response.",
        )
        self.assertIsNone(retrieval_empty_message("unknown"))

    def test_assistant_history_entry_falls_back_to_error(self) -> None:
        entry = assistant_history_entry(
            StreamRenderState(error_message="Provider request failed.")
        )

        self.assertEqual(
            entry,
            ChatHistoryEntry(role="assistant", content="Provider request failed."),
        )


if __name__ == "__main__":
    unittest.main()
