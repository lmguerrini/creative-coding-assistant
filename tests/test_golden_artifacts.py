import json
import subprocess
import unittest
from pathlib import Path

from creative_coding_assistant.eval.ragas_models import load_live_session_samples


class GoldenArtifactEvidenceTests(unittest.TestCase):
    def test_sanitized_ragas_fixture_uses_live_session_schema(self) -> None:
        samples = load_live_session_samples(Path("demo/evaluation/sanitized_ragas_live_sessions.jsonl"))

        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample.conversation_id == "sanitized-capstone-ragas" for sample in samples))
        self.assertTrue(all(sample.retrieved_contexts for sample in samples))
        self.assertFalse(
            any(
                private_marker in sample.model_dump_json().lower()
                for sample in samples
                for private_marker in ("/users/", "api_key", "bearer ", "sk-", "password")
            )
        )

    def test_javascript_artifacts_pass_node_syntax_check(self) -> None:
        for path in (
            Path("demo/golden_artifacts/p5_sacred_geometry_sketch.js"),
            Path("demo/golden_artifacts/three_audio_reactive_scene.js"),
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["node", "--check", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_glsl_artifact_has_required_fragment_shader_structure(self) -> None:
        shader = Path("demo/golden_artifacts/glsl_kaleidoscope_field.frag").read_text(encoding="utf-8")

        self.assertIn("precision highp float", shader)
        self.assertIn("uniform vec2 u_resolution", shader)
        self.assertIn("uniform float u_time", shader)
        self.assertIn("void main()", shader)
        self.assertIn("gl_FragColor", shader)
        self.assertEqual(shader.count("{"), shader.count("}"))

    def test_qa_manifest_keeps_hydra_and_future_scope_bounded(self) -> None:
        manifest = json.loads(Path("demo/golden_artifacts/qa_manifest.json").read_text(encoding="utf-8"))
        artifacts = {artifact["artifact_id"]: artifact for artifact in manifest["artifacts"]}

        self.assertEqual(artifacts["hydra"]["qa_status"], "not_generated")
        self.assertIn("guidance-only", artifacts["hydra"]["boundary"])
        self.assertIn("No HoloMind implementation claim.", manifest["claim_boundaries"])
        self.assertIn("No HOLOiVERSE implementation claim.", manifest["claim_boundaries"])
        self.assertIn("No live external DCC/MCP execution claim.", manifest["claim_boundaries"])


if __name__ == "__main__":
    unittest.main()
