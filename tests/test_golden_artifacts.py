import json
import subprocess
import unittest
from pathlib import Path

from creative_coding_assistant.eval.ragas_models import load_live_session_samples


class GoldenArtifactEvidenceTests(unittest.TestCase):
    def test_sanitized_ragas_fixture_uses_live_session_schema(self) -> None:
        samples = load_live_session_samples(Path("demo/evaluation/sanitized_ragas_live_sessions.jsonl"))

        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample.conversation_id == "sanitized-public-ragas" for sample in samples))
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
            Path("demo/golden_artifacts/p5_generative_morphogenesis_sketch.js"),
            Path("demo/golden_artifacts/three_audio_reactive_scene.js"),
            Path("demo/golden_artifacts/hydra_feedback_lattice.js"),
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

    def test_qa_manifest_records_bounded_hydra_runtime_evidence(self) -> None:
        manifest = json.loads(Path("demo/golden_artifacts/qa_manifest.json").read_text(encoding="utf-8"))
        artifacts = {artifact["artifact_id"]: artifact for artifact in manifest["artifacts"]}

        self.assertEqual(
            manifest["browser_render_qa_result"],
            "demo/golden_artifacts/browser_full_runtime_qa_results.json",
        )
        self.assertEqual(
            artifacts["p5_generative_morphogenesis_sketch"]["qa_status"],
            "full_runtime_browser_render_passed",
        )
        self.assertEqual(artifacts["glsl_kaleidoscope_field"]["qa_status"], "browser_webgl_render_passed")
        self.assertEqual(
            artifacts["three_audio_reactive_scene"]["qa_status"],
            "full_runtime_browser_render_passed",
        )
        self.assertEqual(artifacts["hydra_feedback_lattice"]["qa_status"], "full_runtime_browser_render_passed")
        self.assertIn("hydra-synth", artifacts["hydra_feedback_lattice"]["browser_render_check"])
        self.assertIn("no microphone", artifacts["hydra_feedback_lattice"]["boundary"])
        self.assertIn("No autonomous immersive platform implementation claim.", manifest["claim_boundaries"])
        self.assertIn("No future experience-engine runtime claim.", manifest["claim_boundaries"])
        self.assertIn("No live external DCC/MCP execution claim.", manifest["claim_boundaries"])

    def test_browser_render_qa_result_records_honest_runtime_boundaries(self) -> None:
        result = json.loads(Path("demo/golden_artifacts/browser_render_qa_results.json").read_text(encoding="utf-8"))
        by_artifact = {entry["artifact_id"]: entry for entry in result["results"]}
        harness = Path("demo/golden_artifacts/browser_render_qa.html").read_text(encoding="utf-8")

        self.assertEqual(result["browser"], "Chromium browser")
        self.assertEqual(by_artifact["p5_generative_morphogenesis_sketch"]["status"], "rendered_nonblank")
        self.assertEqual(by_artifact["glsl_kaleidoscope_field"]["status"], "rendered_nonblank")
        self.assertEqual(
            by_artifact["three_audio_reactive_scene"]["status"],
            "static_only_dependency_unavailable",
        )
        self.assertIn(
            "No Three.js render or FPS benchmark is claimed.",
            by_artifact["three_audio_reactive_scene"]["limitations"],
        )
        self.assertIn("p5_generative_morphogenesis_sketch.js", harness)
        self.assertIn("glsl_kaleidoscope_field.frag", harness)
        self.assertIn("three_audio_reactive_scene.js", harness)
        self.assertNotIn("https://", harness)

    def test_browser_full_runtime_qa_result_records_real_runtime_boundaries(self) -> None:
        result = json.loads(
            Path("demo/golden_artifacts/browser_full_runtime_qa_results.json").read_text(encoding="utf-8")
        )
        by_artifact = {entry["artifact_id"]: entry for entry in result["results"]}
        harness = Path("demo/golden_artifacts/browser_full_runtime_qa.html").read_text(encoding="utf-8")

        self.assertEqual(result["temporary_dependencies"]["p5"], "2.3.0")
        self.assertEqual(result["temporary_dependencies"]["three"], "0.185.1")
        self.assertEqual(result["temporary_dependencies"]["hydra-synth"], "1.4.0")
        self.assertEqual(
            by_artifact["p5_generative_morphogenesis_sketch"]["classification"],
            "FULLY VALIDATED WITH ACCEPTED BOUNDARY",
        )
        self.assertEqual(
            by_artifact["three_audio_reactive_scene"]["classification"],
            "FULLY VALIDATED WITH ACCEPTED BOUNDARY",
        )
        self.assertEqual(by_artifact["glsl_kaleidoscope_field"]["classification"], "FULLY VALIDATED")
        self.assertEqual(
            by_artifact["hydra_feedback_lattice"]["classification"],
            "FULLY VALIDATED WITH ACCEPTED BOUNDARY",
        )
        self.assertTrue(by_artifact["p5_generative_morphogenesis_sketch"]["pixel_check"]["nonblank"])
        self.assertTrue(by_artifact["three_audio_reactive_scene"]["pixel_check"]["nonblank"])
        self.assertTrue(by_artifact["glsl_kaleidoscope_field"]["pixel_check"]["nonblank"])
        self.assertTrue(by_artifact["hydra_feedback_lattice"]["pixel_check"]["nonblank"])
        self.assertIn("/node_modules/p5/lib/p5.min.js", harness)
        self.assertIn("/node_modules/three/build/three.module.js", harness)
        self.assertIn("/node_modules/hydra-synth/dist/hydra-synth.js", harness)
        self.assertIn("performance_boundary", result)
        self.assertEqual(result["launcher_validation"]["flow_count"], 8)

    def test_final_demo_launcher_loads_suite_and_evidence_links(self) -> None:
        launcher = Path("demo/final_demo_launcher.html").read_text(encoding="utf-8")

        self.assertIn("Creative Coding Assistant Demo Launcher", launcher)
        self.assertIn("./final_demo_suite.json", launcher)
        self.assertIn("./golden_artifacts/browser_full_runtime_qa.html", launcher)
        self.assertIn("./evaluation/current_product_ragas_evidence.json", launcher)
        self.assertIn("hydra_feedback_lattice.js", launcher)

    def test_final_demo_suite_has_eight_startable_flows_with_boundaries(self) -> None:
        suite = json.loads(Path("demo/final_demo_suite.json").read_text(encoding="utf-8"))

        self.assertEqual(suite["schema_version"], "v8.final_demo_suite.v1")
        self.assertEqual(len(suite["demos"]), 8)
        self.assertEqual(
            suite["ui_start_to_finish_status"],
            "integrated_in_app_demo_mode_available; static local launcher retained for compatibility",
        )
        self.assertIn("in_app_demo_mode", suite["demo_start_path"])
        self.assertIn("launcher_fallback", suite["demo_start_path"])
        required_keys = {
            "prompt",
            "expected_behavior",
            "fallback_path",
            "success_criteria",
            "validation_path",
            "evidence_note",
            "classification",
        }
        for demo in suite["demos"]:
            self.assertTrue(required_keys.issubset(demo), demo)
            self.assertIn(
                demo["classification"],
                {
                    "FULLY VALIDATED",
                    "FULLY VALIDATED WITH ACCEPTED BOUNDARY",
                    "BLOCKED BY HITL PRIVACY APPROVAL",
                    "UNSUPPORTED / DO NOT CLAIM",
                },
            )


if __name__ == "__main__":
    unittest.main()
