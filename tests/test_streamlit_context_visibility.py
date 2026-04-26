import unittest

from creative_coding_assistant.clients import (
    ContextDisplayItem,
    StreamRenderState,
    assistant_history_entry,
    context_empty_message,
    context_expander_label,
    reduce_stream_event,
)
from creative_coding_assistant.contracts import (
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)


class StreamlitContextVisibilityTests(unittest.TestCase):
    def test_reduce_stream_event_extracts_memory_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.MEMORY,
                sequence=1,
                payload={
                    "code": "memory_completed",
                    "message": "Memory context prepared.",
                    "context": {
                        "recent_turns": [
                            {
                                "turn_index": 3,
                                "role": "user",
                                "content": (
                                    "Please keep the lighting moody and cinematic."
                                ),
                            }
                        ],
                        "running_summary": {
                            "content": (
                                "The project uses a dark gallery scene "
                                "with soft highlights."
                            )
                        },
                        "project_memories": [
                            {
                                "memory_kind": "style_preference",
                                "source": "project-notes",
                                "content": (
                                    "Prefer slow camera motion and "
                                    "restrained bloom."
                                ),
                            }
                        ],
                    },
                },
            ),
        )

        self.assertEqual(state.memory_state, "available")
        self.assertEqual(len(state.memory_items), 3)
        self.assertEqual(state.memory_items[0].label, "Running summary")
        self.assertEqual(state.memory_items[1].label, "User turn 3")
        self.assertEqual(state.memory_items[2].source_id, "project-notes")

    def test_reduce_stream_event_marks_empty_memory_context(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.MEMORY,
                sequence=2,
                payload={
                    "code": "memory_completed",
                    "message": "Memory context prepared.",
                    "context": {
                        "recent_turns": [],
                        "running_summary": None,
                        "project_memories": [],
                    },
                },
            ),
        )

        self.assertEqual(state.memory_state, "empty")
        self.assertEqual(state.memory_items, ())

    def test_reduce_stream_event_extracts_assembled_context_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.CONTEXT,
                sequence=3,
                payload={
                    "code": "context_assembled",
                    "message": "Combined context prepared.",
                    "context": {
                        "memory_context": {
                            "recent_turns": [],
                            "running_summary": {
                                "content": (
                                    "The sketch should stay quiet, minimal, "
                                    "and slow."
                                )
                            },
                            "project_memories": [],
                        },
                        "retrieval_context": {
                            "chunks": [
                                {
                                    "source_id": "three_docs",
                                    "domain": "three_js",
                                    "document_title": "Fog",
                                    "excerpt": (
                                        "Fog defines linear fog for distant "
                                        "scene depth."
                                    ),
                                }
                            ]
                        },
                    },
                },
            ),
        )

        self.assertEqual(state.context_state, "available")
        self.assertEqual(len(state.context_items), 2)
        self.assertEqual(state.context_items[0].label, "Running summary")
        self.assertEqual(state.context_items[1].source_id, "three_docs")
        self.assertEqual(state.context_items[1].domain, CreativeCodingDomain.THREE_JS)

    def test_assistant_history_entry_preserves_memory_and_context_visibility(
        self,
    ) -> None:
        state = StreamRenderState(
            final_answer="Final answer",
            memory_state="available",
            memory_items=(
                ContextDisplayItem(
                    label="Running summary",
                    snippet="Project summary.",
                ),
            ),
            context_state="available",
            context_items=(
                ContextDisplayItem(
                    label="Fog",
                    source_id="three_docs",
                    domain=CreativeCodingDomain.THREE_JS,
                    snippet="Fog defines linear fog for distant scene depth.",
                ),
            ),
        )

        entry = assistant_history_entry(state)

        self.assertEqual(entry.memory_state, "available")
        self.assertEqual(entry.context_state, "available")
        self.assertEqual(len(entry.memory_items), 1)
        self.assertEqual(len(entry.context_items), 1)

    def test_context_expander_helpers_handle_empty_and_available_states(self) -> None:
        label = context_expander_label(
            kind="memory",
            items=(
                ContextDisplayItem(
                    label="Running summary",
                    snippet="Project summary.",
                ),
            ),
            visibility_state="available",
        )

        self.assertEqual(label, "Memory context (1 item)")
        self.assertEqual(
            context_empty_message(kind="memory", visibility_state="empty"),
            "No memory context was available for this response.",
        )
        self.assertEqual(
            context_empty_message(kind="context", visibility_state="empty"),
            "No assembled context was produced for this response.",
        )
        self.assertIsNone(
            context_empty_message(kind="context", visibility_state="unknown")
        )


if __name__ == "__main__":
    unittest.main()
