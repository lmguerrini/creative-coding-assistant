import unittest

from creative_coding_assistant.clients import (
    PromptDisplayItem,
    PromptVisibilitySummary,
    StreamRenderState,
    assistant_history_entry,
    prompt_visibility_empty_message,
    prompt_visibility_expander_label,
    prompt_visibility_meta,
    reduce_stream_event,
)
from creative_coding_assistant.contracts import (
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)


class StreamlitPromptVisibilityTests(unittest.TestCase):
    def test_reduce_stream_event_extracts_prompt_input_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.PROMPT_INPUT,
                sequence=1,
                payload={
                    "code": "prompt_inputs_prepared",
                    "message": "Prompt inputs prepared.",
                    "prompt_input": {
                        "request": {"route": "docs_grounded"},
                        "user_input": {
                            "query": "Explain how fog works in Three.js scenes.",
                            "mode": "explain",
                            "domain": "three_js",
                        },
                        "memory_input": {
                            "recent_turns": [
                                {
                                    "turn_index": 2,
                                    "role": "user",
                                    "content": (
                                        "Keep the scene quiet and atmospheric."
                                    ),
                                }
                            ],
                            "running_summary": {
                                "content": (
                                    "The scene uses minimal lighting and "
                                    "slow motion."
                                )
                            },
                            "project_memories": [
                                {
                                    "memory_kind": "style_preference",
                                    "source": "project-notes",
                                    "content": (
                                        "Prefer subtle fog and restrained bloom."
                                    ),
                                }
                            ],
                        },
                        "retrieval_input": {
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

        self.assertEqual(state.prompt_input_state, "available")
        assert state.prompt_input_summary is not None
        self.assertEqual(state.prompt_input_summary.route, "docs_grounded")
        self.assertEqual(state.prompt_input_summary.mode, "explain")
        self.assertEqual(state.prompt_input_summary.items[0].label, "User request")
        self.assertEqual(
            state.prompt_input_summary.items[-1].source_id,
            "three_docs",
        )

    def test_reduce_stream_event_extracts_rendered_prompt_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.PROMPT_RENDERED,
                sequence=2,
                payload={
                    "code": "prompt_rendered",
                    "message": "Rendered prompt prepared.",
                    "rendered_prompt": {
                        "request": {
                            "route": "docs_grounded",
                            "prompt_input": {
                                "user_input": {
                                    "mode": "explain",
                                }
                            },
                        },
                        "sections": [
                            {
                                "role": "system",
                                "name": "system",
                                "content": (
                                    "Route: docs_grounded Mode: explain "
                                    "Use the provided context sections."
                                ),
                            },
                            {
                                "role": "user",
                                "name": "user",
                                "content": "User Request: Explain how fog works.",
                            },
                        ],
                    },
                },
            ),
        )

        self.assertEqual(state.rendered_prompt_state, "available")
        assert state.rendered_prompt_summary is not None
        self.assertEqual(state.rendered_prompt_summary.route, "docs_grounded")
        self.assertEqual(state.rendered_prompt_summary.mode, "explain")
        self.assertEqual(
            tuple(item.label for item in state.rendered_prompt_summary.items),
            ("System (system)", "User (user)"),
        )

    def test_reduce_stream_event_marks_empty_prompt_visibility(self) -> None:
        state = reduce_stream_event(
            StreamRenderState(),
            StreamEvent(
                event_type=StreamEventType.PROMPT_INPUT,
                sequence=3,
                payload={
                    "code": "prompt_inputs_prepared",
                    "message": "Prompt inputs prepared.",
                    "prompt_input": {
                        "request": {"route": "docs_grounded"},
                        "user_input": {},
                    },
                },
            ),
        )

        self.assertEqual(state.prompt_input_state, "empty")
        self.assertIsNone(state.prompt_input_summary)

    def test_assistant_history_entry_preserves_prompt_visibility(self) -> None:
        state = StreamRenderState(
            final_answer="Final answer",
            prompt_input_state="available",
            prompt_input_summary=PromptVisibilitySummary(
                route="docs_grounded",
                mode="explain",
                items=(
                    PromptDisplayItem(
                        label="User request",
                        domain=CreativeCodingDomain.THREE_JS,
                        snippet="Explain how fog works.",
                    ),
                ),
            ),
            rendered_prompt_state="available",
            rendered_prompt_summary=PromptVisibilitySummary(
                route="docs_grounded",
                mode="explain",
                items=(
                    PromptDisplayItem(
                        label="System (system)",
                        snippet="Route: docs_grounded Mode: explain ...",
                    ),
                ),
            ),
        )

        entry = assistant_history_entry(state)

        self.assertEqual(entry.prompt_input_state, "available")
        self.assertEqual(entry.rendered_prompt_state, "available")
        assert entry.prompt_input_summary is not None
        assert entry.rendered_prompt_summary is not None
        self.assertEqual(len(entry.prompt_input_summary.items), 1)
        self.assertEqual(len(entry.rendered_prompt_summary.items), 1)

    def test_prompt_visibility_helpers_handle_labels_meta_and_empty_state(
        self,
    ) -> None:
        summary = PromptVisibilitySummary(
            route="docs_grounded",
            mode="explain",
            items=(
                PromptDisplayItem(
                    label="User request",
                    domain=CreativeCodingDomain.THREE_JS,
                    snippet="Explain how fog works.",
                ),
            ),
        )

        self.assertEqual(
            prompt_visibility_expander_label(
                kind="prompt_input",
                visibility_state="available",
                summary=summary,
            ),
            "Prompt inputs (1 item)",
        )
        self.assertEqual(
            prompt_visibility_meta(summary),
            "docs_grounded | explain",
        )
        self.assertEqual(
            prompt_visibility_empty_message(
                kind="rendered_prompt",
                visibility_state="empty",
            ),
            "No rendered prompt was available for this response.",
        )
        self.assertIsNone(
            prompt_visibility_empty_message(
                kind="prompt_input",
                visibility_state="unknown",
            )
        )


if __name__ == "__main__":
    unittest.main()
