import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    SacredArchitectureOperationKind,
    SacredArchitectureRoadmapClassification,
    SacredArchitectureSemanticRole,
    SacredArchitectureValidationSeverity,
    build_v8_4_sacred_architecture_engine,
    detect_sacred_architecture_terms,
    sacred_architecture_prompt_lines,
    sacred_architecture_roadmap_assessment,
)


class SacredArchitectureEngineTests(unittest.TestCase):
    def test_builds_architecture_contracts_with_reused_v8_evidence(self) -> None:
        report = build_v8_4_sacred_architecture_engine(
            "Reverse engineer a sacred temple floor-plan as text: golden ratio "
            "axis, threshold procession, central mandala courtyard, gothic light, "
            "and gallery installation sequence for a Three.js sketch.",
            domains=(CreativeCodingDomain.THREE_JS, CreativeCodingDomain.P5_JS),
        )

        self.assertEqual(report.capability_id, "v8_4_sacred_architecture_engine")
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn("v8_1_creative_knowledge_distillation", report.reused_surface_ids)
        self.assertIn("v8_2_symbolic_translation_engine", report.reused_surface_ids)
        self.assertIn("v8_3_sacred_geometry_engine", report.reused_surface_ids)

        pattern_ids = {pattern.pattern_id for pattern in report.pattern_guidance}
        self.assertIn("axis_threshold_procession", pattern_ids)
        self.assertIn("central_periphery_mandala_plan", pattern_ids)
        self.assertIn("installation_gallery_sequence", pattern_ids)
        self.assertIn("gothic_vertical_light_axis", pattern_ids)

        operation_kinds = {operation.kind for operation in report.operational_guidance}
        self.assertIn(SacredArchitectureOperationKind.PLANIMETRY_LAYOUT, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.AXIS_SYMMETRY, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.THRESHOLD_PROCESSION, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.GEOMETRY_TO_ARCHITECTURE_MAPPING, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.SYMBOLIC_TO_SPATIAL_MAPPING, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.INSTALLATION_PLANNING, operation_kinds)
        self.assertIn(SacredArchitectureOperationKind.REVERSE_ENGINEERING_GUIDANCE, operation_kinds)

        semantic_roles = {node.role for node in report.semantic_nodes}
        self.assertIn(SacredArchitectureSemanticRole.ENTRY, semantic_roles)
        self.assertIn(SacredArchitectureSemanticRole.THRESHOLD, semantic_roles)
        self.assertIn(SacredArchitectureSemanticRole.AXIS, semantic_roles)
        self.assertIn(SacredArchitectureSemanticRole.CENTER, semantic_roles)
        self.assertTrue(report.semantic_edges)

        self.assertTrue(report.architectural_proportion_guidance_implemented)
        self.assertTrue(report.floor_plan_reasoning_implemented)
        self.assertTrue(report.geometry_to_architecture_mapping_implemented)
        self.assertTrue(report.symbolic_to_spatial_mapping_implemented)
        self.assertFalse(report.image_based_reconstruction_implemented)
        self.assertFalse(report.lidar_interpretation_implemented)
        self.assertFalse(report.actual_architectural_analysis_implemented)
        self.assertFalse(report.interactive_architecture_preview_implemented)
        self.assertFalse(report.v8_5_narrative_engine_started)
        self.assertFalse(report.v8_6_immersive_composer_started)

    def test_safe_boundaries_for_image_lidar_and_real_reconstruction_claims(self) -> None:
        report = build_v8_4_sacred_architecture_engine(
            "Use this photo and LIDAR scan to reconstruct the exact CAD blueprint "
            "of an ancient sacred temple and prove its divine geometry.",
            domains=(CreativeCodingDomain.THREE_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertEqual(report.confidence.band.value, "guarded")
        severities = {finding.severity for finding in report.validation_findings}
        self.assertIn(SacredArchitectureValidationSeverity.HITL_REQUIRED, severities)
        self.assertTrue(report.hitl_questions)
        self.assertFalse(report.image_based_reconstruction_implemented)
        self.assertFalse(report.lidar_interpretation_implemented)
        self.assertFalse(report.photogrammetry_implemented)
        self.assertFalse(report.cad_dcc_integration_implemented)

        rendered = "\n".join(sacred_architecture_prompt_lines(report)).lower()
        self.assertIn("image-based reconstruction", rendered)
        self.assertIn("lidar", rendered)
        self.assertIn("unsupported architecture claim risk", rendered)
        self.assertIn("hitl architecture question", rendered)
        self.assertNotIn("image_based_reconstruction_implemented", rendered)

    def test_roadmap_reality_check_classifies_v8_4_scope(self) -> None:
        assessment = sacred_architecture_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertEqual(len(assessment), 26)
        self.assertEqual(
            by_item["Architectural Reverse Engineering"].classification,
            SacredArchitectureRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Planimetry Analysis"].classification,
            SacredArchitectureRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Image-based Reconstruction"].classification,
            SacredArchitectureRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED,
        )
        self.assertEqual(
            by_item["LIDAR Interpretation Layer"].classification,
            SacredArchitectureRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED,
        )
        self.assertEqual(
            by_item["Interactive Architecture Preview"].classification,
            SacredArchitectureRoadmapClassification.LATER_V8_BOUNDARY,
        )
        self.assertFalse(
            [
                item.item
                for item in assessment
                if item.classification is SacredArchitectureRoadmapClassification.MISSING
            ]
        )

    def test_detects_architecture_terms_without_starting_later_capabilities(self) -> None:
        self.assertEqual(
            detect_sacred_architecture_terms("temple axis threshold gallery installation"),
            (
                "axis_threshold_procession",
                "central_periphery_mandala_plan",
                "compressed_threshold_chamber_sequence",
                "installation_gallery_sequence",
                "pavilion_field_environment",
            ),
        )


if __name__ == "__main__":
    unittest.main()
