import unittest

from creative_coding_assistant.clients import (
    GenerationInputDisplayItem,
    GenerationInputVisibilitySummary,
    StreamRenderState,
    assistant_history_entry,
    generation_input_empty_message,
    generation_input_expander_label,
    generation_input_meta,
    reduce_stream_event,
)
from creative_coding_assistant.contracts import StreamEvent, StreamEventType


class StreamlitGenerationVisibilityTests(unittest.TestCase):
    def test_reduce_stream_event_extracts_generation_input_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.GENERATION_INPUT,
                sequence=1,
                payload={
                    "code": "generation_input_prepared",
                    "message": "Provider generation input prepared.",
                    "generation_input": {
                        "request": {
                            "route": "docs_grounded",
                            "stream": True,
                        },
                        "messages": [
                            {
                                "role": "system",
                                "name": "system",
                                "content": (
                                    "Route: docs_grounded. Use the provided "
                                    "context sections."
                                ),
                            },
                            {
                                "role": "user",
                                "name": "user",
                                "content": "Explain how fog works.",
                            },
                            {
                                "role": "context",
                                "name": "memory",
                                "content": (
                                    "Running summary: Keep the scene quiet."
                                ),
                            },
                            {
                                "role": "context",
                                "name": "retrieval",
                                "content": (
                                    "Fog defines linear fog for distant "
                                    "scene depth."
                                ),
                            },
                        ],
                    },
                },
            ),
        )

        self.assertEqual(state.generation_input_state, "available")
        assert state.generation_input_summary is not None
        self.assertEqual(state.generation_input_summary.route, "docs_grounded")
        self.assertTrue(state.generation_input_summary.stream)
        self.assertEqual(state.generation_input_summary.message_count, 4)
        self.assertEqual(
            tuple(item.label for item in state.generation_input_summary.items),
            ("System", "User", "Memory", "Retrieval"),
        )

    def test_reduce_stream_event_marks_empty_generation_input_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.GENERATION_INPUT,
                sequence=2,
                payload={
                    "code": "generation_input_prepared",
                    "message": "Provider generation input prepared.",
                    "generation_input": {
                        "request": {"route": "docs_grounded"},
                        "messages": [],
                    },
                },
            ),
        )

        self.assertEqual(state.generation_input_state, "empty")
        self.assertIsNone(state.generation_input_summary)

    def test_assistant_history_entry_preserves_generation_input_visibility(
        self,
    ) -> None:
        state = StreamRenderState(
            final_answer="Final answer",
            generation_input_state="available",
            generation_input_summary=GenerationInputVisibilitySummary(
                route="docs_grounded",
                stream=True,
                message_count=2,
                items=(
                    GenerationInputDisplayItem(
                        label="System",
                        role="system",
                        snippet="Route: docs_grounded Mode: explain ...",
                    ),
                    GenerationInputDisplayItem(
                        label="User",
                        role="user",
                        snippet="Explain how fog works.",
                    ),
                ),
            ),
        )

        entry = assistant_history_entry(state)

        self.assertEqual(entry.generation_input_state, "available")
        assert entry.generation_input_summary is not None
        self.assertEqual(entry.generation_input_summary.message_count, 2)
        self.assertEqual(len(entry.generation_input_summary.items), 2)

    def test_generation_input_visibility_helpers_handle_meta_and_empty_state(
        self,
    ) -> None:
        summary = GenerationInputVisibilitySummary(
            route="docs_grounded",
            stream=True,
            message_count=2,
            items=(
                GenerationInputDisplayItem(
                    label="System",
                    role="system",
                    snippet="Route: docs_grounded Mode: explain ...",
                ),
            ),
        )

        self.assertEqual(
            generation_input_expander_label(
                visibility_state="available",
                summary=summary,
            ),
            "Generation input summary (2 messages)",
        )
        self.assertEqual(
            generation_input_meta(summary),
            "docs_grounded | stream",
        )
        self.assertEqual(
            generation_input_empty_message(visibility_state="empty"),
            "No generation input summary was available for this response.",
        )
        self.assertIsNone(
            generation_input_empty_message(visibility_state="unknown")
        )


if __name__ == "__main__":
    unittest.main()
