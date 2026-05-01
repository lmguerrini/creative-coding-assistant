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
        self.assertEqual(segments[1].suggested_filename, "index.html")
        self.assertEqual(segments[1].mime_type, "text/html")
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
        self.assertEqual(segments[1].suggested_filename, "script.js")

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
        self.assertEqual(segments[1].suggested_filename, "index.html")
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
        self.assertEqual(segments[1].suggested_filename, "script.js")
        self.assertEqual(segments[2].kind, "prose")
        self.assertTrue(segments[2].content.startswith("Notes:"))

    def test_split_answer_segments_assigns_jsx_and_glsl_filenames(self) -> None:
        jsx_segments = split_answer_segments(
            "```jsx\nimport { Canvas } from '@react-three/fiber';\n"
            "function Scene() {\n  return <Canvas />;\n}\n```"
        )
        glsl_segments = split_answer_segments(
            "```glsl\nprecision mediump float;\nvoid main() {\n"
            "  gl_FragColor = vec4(1.0);\n}\n```"
        )

        self.assertEqual(jsx_segments[0].language, "jsx")
        self.assertEqual(jsx_segments[0].suggested_filename, "App.jsx")
        self.assertEqual(jsx_segments[0].mime_type, "text/javascript")
        self.assertEqual(glsl_segments[0].language, "glsl")
        self.assertEqual(glsl_segments[0].suggested_filename, "shader.glsl")
        self.assertEqual(glsl_segments[0].mime_type, "text/plain")

    def test_split_answer_segments_uses_txt_fallback_for_unknown_language(self) -> None:
        segments = split_answer_segments("```mermaid\ngraph TD;\nA-->B;\n```")

        self.assertEqual(segments[0].language, "mermaid")
        self.assertEqual(segments[0].suggested_filename, "snippet.txt")
        self.assertEqual(segments[0].mime_type, "text/plain")

    def test_split_answer_segments_uses_three_js_query_for_html_filename(self) -> None:
        segments = split_answer_segments(
            "```html\n<canvas></canvas>\n```",
            query="Create a simple rotating cube in three.js",
        )

        self.assertEqual(segments[0].suggested_filename, "index.html")

    def test_split_answer_segments_uses_p5_query_for_sketch_filename(self) -> None:
        segments = split_answer_segments(
            "```javascript\nfunction draw() {}\n```",
            query="Create a simple p5.js sketch with a moving circle",
        )

        self.assertEqual(segments[0].suggested_filename, "sketch.js")

    def test_split_answer_segments_uses_r3f_query_for_app_filename(self) -> None:
        segments = split_answer_segments(
            "```jsx\nexport default function Scene() { return null; }\n```",
            query="Create a scene in React Three Fiber",
        )

        self.assertEqual(segments[0].suggested_filename, "App.jsx")

    def test_split_answer_segments_uses_glsl_query_for_shader_filename(self) -> None:
        segments = split_answer_segments(
            "```glsl\nvoid main() { gl_FragColor = vec4(1.0); }\n```",
            query="Write a basic GLSL shader",
        )

        self.assertEqual(segments[0].suggested_filename, "shader.glsl")

    def test_split_answer_segments_uses_fragment_shader_query_override(self) -> None:
        segments = split_answer_segments(
            "```glsl\nvoid main() { gl_FragColor = vec4(1.0); }\n```",
            query="Write a fragment shader with a color gradient",
        )

        self.assertEqual(segments[0].suggested_filename, "fragment.glsl")

    def test_split_answer_segments_stabilizes_multiple_code_block_names(self) -> None:
        segments = split_answer_segments(
            "```glsl\nvoid main() { gl_FragColor = vec4(1.0); }\n```\n\n"
            "```glsl\nvoid main() { gl_FragColor = vec4(0.5); }\n```",
            query="Write a GLSL shader",
        )

        self.assertEqual(segments[0].suggested_filename, "shader.glsl")
        self.assertEqual(segments[1].suggested_filename, "shader-2.glsl")

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
