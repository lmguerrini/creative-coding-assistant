import unittest

from clients.streamlit.helpers import (
    ChatHistoryEntry,
    StreamRenderState,
    assistant_history_entry,
    build_chat_request,
    build_provider_warning,
    default_domain,
    default_mode,
    reduce_stream_event,
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
            domain=CreativeCodingDomain.GLSL,
            mode=AssistantMode.EXPLAIN,
        )

        self.assertEqual(request.query, "Explain the shader noise pattern.")
        self.assertEqual(request.conversation_id, "conversation-123")
        self.assertEqual(request.domain, CreativeCodingDomain.GLSL)
        self.assertEqual(request.mode, AssistantMode.EXPLAIN)

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

        self.assertEqual(state.status_message, "Request accepted.")
        self.assertEqual(state.answer_text, "Hello world")
        self.assertIsNone(state.final_answer)

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
