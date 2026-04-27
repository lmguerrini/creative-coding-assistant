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

    def test_split_answer_segments_detects_unfenced_html_document(self) -> None:
        segments = split_answer_segments(
            "Use this full page example:\n"
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<body>\n"
            "<script type=\"module\">\n"
            "const scene = new THREE.Scene();\n"
            "</script>\n"
            "</body>\n"
            "</html>\n"
            "Notes:\n"
            "- Increase the rotation speed to 0.05.",
        )

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].kind, "prose")
        self.assertEqual(segments[0].content, "Use this full page example:")
        self.assertEqual(segments[1].kind, "code")
        self.assertEqual(segments[1].language, "html")
        self.assertIn("<!DOCTYPE html>", segments[1].content)
        self.assertIn("const scene = new THREE.Scene();", segments[1].content)
        self.assertEqual(segments[2].kind, "prose")
        self.assertEqual(
            segments[2].content,
            "Notes:\n- Increase the rotation speed to 0.05.",
        )

    def test_split_answer_segments_keeps_notes_as_prose_around_code(self) -> None:
        segments = split_answer_segments(
            "Explanation first.\n\n"
            "const speed = 0.05;\n"
            "cube.rotation.y += speed;\n"
            "renderer.render(scene, camera);\n\n"
            "Notes:\n"
            "- Keep OrbitControls optional.\n"
            "- Lower speed for calmer motion."
        )

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].kind, "prose")
        self.assertEqual(segments[1].kind, "code")
        self.assertEqual(segments[1].language, "javascript")
        self.assertEqual(segments[2].kind, "prose")
        self.assertTrue(segments[2].content.startswith("Notes:"))

    def test_answer_working_message_uses_clear_neutral_copy(self) -> None:
        self.assertEqual(
            answer_working_message(
                status_message="Preparing response...",
                has_content=False,
            ),
            "Preparing response...",
        )
        self.assertEqual(
            answer_working_message(
                status_message="Receiving response...",
                has_content=False,
            ),
            "Receiving response...",
        )
        self.assertIsNone(
            answer_working_message(
                status_message="Receiving response...",
                has_content=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
