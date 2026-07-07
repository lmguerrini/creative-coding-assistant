import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    ImmersiveComposerValidationSeverity,
    ImmersiveCompositionRoadmapClassification,
    ImmersivePreviewImplementationStatus,
    build_v8_6_immersive_audiovisual_composer,
    immersive_audiovisual_composer_prompt_lines,
    immersive_audiovisual_composer_roadmap_assessment,
)


class ImmersiveAudiovisualComposerTests(unittest.TestCase):
    def test_builds_composer_by_reusing_v8_engines_and_preview_audit(self) -> None:
        report = build_v8_6_immersive_audiovisual_composer(
            "Compose an immersive audiovisual phoenix mandala temple installation: "
            "golden ratio geometry, sacred light, symbolic color, particle "
            "traces, planetary orbit motion, quadrivium mapping, ritual timing, "
            "spatial audio with Tone.js, GLSL glow, Hydra feedback, p5 sketch "
            "fallback, Three.js scene graph, audience journey, and emotional resonance.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.HYDRA,
                CreativeCodingDomain.TONE_JS,
            ),
        )

        self.assertEqual(report.capability_id, "v8_6_immersive_composer")
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn("audio_visual_scene_system", report.reused_surface_ids)
        self.assertIn("v6_style_profiles", report.reused_surface_ids)
        self.assertIn("v8_1_creative_knowledge_distillation", report.reused_surface_ids)
        self.assertIn("v8_2_symbolic_translation_engine", report.reused_surface_ids)
        self.assertIn("v8_3_sacred_geometry_engine", report.reused_surface_ids)
        self.assertIn("v8_4_sacred_architecture_engine", report.reused_surface_ids)
        self.assertIn("v8_5_mythopoetic_engine", report.reused_surface_ids)

        self.assertGreaterEqual(len(report.scene_graph), 3)
        self.assertEqual(len(report.scene_transitions), len(report.scene_graph) - 1)
        self.assertTrue(report.visual_language.visual_identity)
        self.assertTrue(report.geometry_animation.quadrivium_mapping)
        self.assertTrue(report.geometry_animation.planetary_motion_guidance)
        self.assertTrue(report.spatial_audio.sacred_music_mapping)
        self.assertEqual(report.spatial_audio.activation, "explicit_user_gesture")
        self.assertTrue(report.audience_journey.audience_flow_simulation)
        self.assertTrue(report.artistic_decisions)
        self.assertEqual(len(report.composition_audit_summary), 5)

        preview_by_item = {item.item: item for item in report.preview_audit}
        self.assertEqual(len(preview_by_item), 9)
        self.assertEqual(
            preview_by_item["Internal p5.js Preview"].implementation_status,
            ImmersivePreviewImplementationStatus.ALREADY_IMPLEMENTED,
        )
        self.assertEqual(
            preview_by_item["Internal Three.js Preview"].implementation_status,
            ImmersivePreviewImplementationStatus.PARTIALLY_IMPLEMENTED,
        )
        self.assertTrue(preview_by_item["Browser Preview Sandbox"].reusable_for_v8_6)
        self.assertTrue(preview_by_item["Preview Error Recovery"].reusable_for_v8_6)
        self.assertTrue(preview_by_item["Multi Preview Comparison"].reusable_for_v8_6)

        self.assertTrue(report.scene_graph_composer_implemented)
        self.assertTrue(report.visual_language_engine_implemented)
        self.assertTrue(report.sacred_lighting_engine_implemented)
        self.assertTrue(report.symbolic_color_engine_implemented)
        self.assertTrue(report.geometry_animation_engine_implemented)
        self.assertTrue(report.spatial_audio_planner_implemented)
        self.assertTrue(report.quadrivium_engine_implemented)
        self.assertTrue(report.audience_flow_simulation_implemented)
        self.assertTrue(report.preview_runtime_audit_implemented)
        self.assertFalse(report.preview_runtime_mutation_implemented)
        self.assertFalse(report.preview_runtime_implementation_added)
        self.assertFalse(report.workflow_control_implemented)
        self.assertFalse(report.storage_write_implemented)
        self.assertFalse(report.provider_model_routing_implemented)
        self.assertFalse(report.v8_7_hologenesis_os_started)
        self.assertFalse(report.v8_8_demo_showcase_started)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)

        rendered = "\n".join(immersive_audiovisual_composer_prompt_lines(report)).lower()
        self.assertIn("scene graph node", rendered)
        self.assertIn("preview audit", rendered)
        self.assertIn("quadrivium mapping", rendered)
        self.assertIn("audience flow simulation", rendered)

    def test_boundary_blocks_external_v8_7_and_dcc_claims(self) -> None:
        report = build_v8_6_immersive_audiovisual_composer(
            "Build the immersive composer as a HoloMind HOLOiVERSE HoloGenesis OS "
            "with Unity, Unreal, Blender, Houdini, TouchDesigner, MCP, and exact "
            "spiritual ritual efficacy.",
            domains=(CreativeCodingDomain.THREE_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertEqual(report.confidence.band.value, "guarded")
        self.assertTrue(report.hitl_questions)
        severities = {finding.severity for finding in report.validation_findings}
        self.assertIn(ImmersiveComposerValidationSeverity.HITL_REQUIRED, severities)
        self.assertFalse(report.external_dcc_integration_implemented)
        self.assertFalse(report.mcp_integration_implemented)
        self.assertFalse(report.v8_7_hologenesis_os_started)
        self.assertFalse(report.v8_8_demo_showcase_started)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)
        self.assertFalse(report.preview_runtime_mutation_implemented)
        self.assertFalse(report.provider_model_routing_implemented)
        self.assertFalse(report.workflow_control_implemented)
        self.assertFalse(report.storage_write_implemented)

        rendered = "\n".join(immersive_audiovisual_composer_prompt_lines(report)).lower()
        self.assertIn("unsupported composer claim risk", rendered)
        self.assertIn("hitl composer question", rendered)
        self.assertNotIn("external_dcc_integration_implemented", rendered)

    def test_roadmap_reality_check_classifies_v8_6_scope(self) -> None:
        assessment = immersive_audiovisual_composer_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertEqual(len(assessment), 36)
        self.assertEqual(
            by_item["Scene Graph Composer"].classification,
            ImmersiveCompositionRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Spatial Audio Planner"].classification,
            ImmersiveCompositionRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Creative Style Profiles"].classification,
            ImmersiveCompositionRoadmapClassification.REUSED_EXISTING_RUNTIME,
        )
        self.assertEqual(
            by_item["Internal Three.js Preview"].classification,
            ImmersiveCompositionRoadmapClassification.REUSED_EXISTING_RUNTIME,
        )
        self.assertEqual(
            by_item["Artifact Preview Loop"].classification,
            ImmersiveCompositionRoadmapClassification.PARTIAL_REUSABLE,
        )
        self.assertEqual(
            by_item["Internal Export Preview"].classification,
            ImmersiveCompositionRoadmapClassification.PARTIAL_REUSABLE,
        )
        self.assertFalse(
            [
                item.item
                for item in assessment
                if item.classification is ImmersiveCompositionRoadmapClassification.MISSING
            ]
        )


if __name__ == "__main__":
    unittest.main()
