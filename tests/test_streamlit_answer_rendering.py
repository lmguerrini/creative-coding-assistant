import unittest

from creative_coding_assistant.clients import (
    answer_working_message,
    split_answer_segments,
)


class StreamlitAnswerRenderingTests(unittest.TestCase):
    def test_split_answer_segments_extracts_fenced_code_blocks(self) -> None:
        segments = split_answer_segments(
            "Intro note.\n\n```html\n<div>Hello</div>\n```\n\nNext steps."
        )

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].kind, "prose")
        self.assertEqual(segments[0].content, "Intro note.")
        self.assertEqual(segments[1].kind, "code")
        self.assertEqual(segments[1].language, "html")
        self.assertEqual(segments[1].content, "<div>Hello</div>")
        self.assertEqual(segments[2].kind, "prose")
        self.assertEqual(segments[2].content, "Next steps.")

    def test_split_answer_segments_handles_unclosed_streaming_code_block(self) -> None:
        segments = split_answer_segments(
            "Working draft.\n\n```javascript\nconst cube = new THREE.Mesh();",
            allow_unclosed_code_block=True,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].kind, "prose")
        self.assertEqual(segments[1].kind, "code")
        self.assertEqual(segments[1].language, "javascript")
        self.assertEqual(segments[1].content, "const cube = new THREE.Mesh();")

    def test_answer_working_message_is_clear_without_fake_progress(self) -> None:
        self.assertEqual(
            answer_working_message(
                status_message="Generating response...",
                has_content=False,
            ),
            "Waiting for model output...",
        )
        self.assertEqual(
            answer_working_message(
                status_message="Streaming response...",
                has_content=False,
            ),
            "Receiving output...",
        )
        self.assertIsNone(
            answer_working_message(
                status_message="Streaming response...",
                has_content=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
