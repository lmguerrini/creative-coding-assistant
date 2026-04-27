import unittest

from creative_coding_assistant.clients import (
    ChatHistoryEntry,
    RetrievalDisplayItem,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    default_domain,
    default_domain_selection,
    default_mode,
    domain_selection_summary,
    mode_selection_summary,
    reduce_stream_event,
    resolve_request_domain,
    resolve_request_domains,
    resolve_session_domain_selection,
    resolve_session_mode,
    retrieval_empty_message,
    retrieval_expander_label,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import GenerationProviderName, Settings


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

    def test_default_domain_selection_includes_all_domains(self) -> None:
        self.assertEqual(
            default_domain_selection(),
            tuple(CreativeCodingDomain),
        )

    def test_resolve_session_domain_selection_defaults_to_all_domains(self) -> None:
        self.assertEqual(
            resolve_session_domain_selection(None),
            tuple(CreativeCodingDomain),
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
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.THREE_JS,
            ),
        )

    def test_resolve_session_domain_selection_falls_back_when_all_are_invalid(
        self,
    ) -> None:
        self.assertEqual(
            resolve_session_domain_selection(("invalid_domain", "still_invalid")),
            tuple(CreativeCodingDomain),
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
            "All 4 domains selected",
        )
        self.assertEqual(
            domain_selection_summary(
                (
                    CreativeCodingDomain.REACT_THREE_FIBER,
                    CreativeCodingDomain.GLSL,
                )
            ),
            "2 selected: React Three Fiber, GLSL",
        )
        self.assertEqual(
            domain_selection_summary(()),
            "No domain filter",
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

        self.assertEqual(state.status_message, "Streaming response...")
        self.assertEqual(state.answer_text, "Hello world")
        self.assertIsNone(state.final_answer)

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

        self.assertEqual(state.status_message, "Generating response...")

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
