import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    SacredGeometryOperationKind,
    SacredGeometryRoadmapClassification,
    SacredGeometryValidationSeverity,
    build_v8_3_sacred_geometry_engine,
    detect_sacred_geometry_terms,
    sacred_geometry_prompt_lines,
    sacred_geometry_roadmap_assessment,
)


class SacredGeometryEngineTests(unittest.TestCase):
    def test_builds_geometry_contracts_with_provenance_and_mappings(self) -> None:
        report = build_v8_3_sacred_geometry_engine(
            "Create a golden ratio mandala tessellation with an L-system tree, "
            "flow field particles, reaction diffusion, cellular automata, "
            "geometry to light, and Tone.js harmonic music.",
            domains=(CreativeCodingDomain.P5_JS, CreativeCodingDomain.TONE_JS),
        )

        self.assertEqual(report.capability_id, "v8_3_sacred_geometry_engine")
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn("v8_1_creative_knowledge_distillation", report.reused_surface_ids)
        self.assertIn("v8_2_symbolic_translation_engine", report.reused_surface_ids)

        pattern_ids = {pattern.pattern_id for pattern in report.pattern_guidance}
        self.assertIn("golden_ratio", pattern_ids)
        self.assertIn("mandala_generator", pattern_ids)
        self.assertIn("sacred_tessellation", pattern_ids)
        self.assertIn("l_system_growth", pattern_ids)
        self.assertIn("flow_field", pattern_ids)
        self.assertIn("reaction_diffusion", pattern_ids)
        self.assertIn("cellular_automata", pattern_ids)
        self.assertIn("particle_geometry", pattern_ids)

        operation_kinds = {operation.kind for operation in report.operational_guidance}
        self.assertIn(SacredGeometryOperationKind.MOTION_MAPPING, operation_kinds)
        self.assertIn(SacredGeometryOperationKind.COLOR_LIGHT_MAPPING, operation_kinds)
        self.assertIn(SacredGeometryOperationKind.AUDIO_HARMONIC_MAPPING, operation_kinds)
        self.assertIn(SacredGeometryOperationKind.ARCHITECTURAL_LAYOUT_MAPPING, operation_kinds)
        self.assertIn(SacredGeometryOperationKind.RITUAL_PACING_MAPPING, operation_kinds)

        severities = {finding.severity for finding in report.validation_findings}
        self.assertIn(SacredGeometryValidationSeverity.WARNING, severities)
        self.assertFalse(report.preview_runtime_mutation_implemented)
        self.assertFalse(report.demo_asset_generation_implemented)
        self.assertFalse(report.v8_4_architecture_engine_started)
        self.assertFalse(report.metaphysical_proof_implemented)

    def test_prompt_lines_preserve_safe_sacred_geometry_boundaries(self) -> None:
        report = build_v8_3_sacred_geometry_engine(
            "Make a sacred ancient divine cosmic truth mandala that proves "
            "metaphysical harmony.",
            domains=(CreativeCodingDomain.P5_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertEqual(report.confidence.band.value, "guarded")
        self.assertTrue(report.hitl_questions)

        rendered = "\n".join(sacred_geometry_prompt_lines(report)).lower()
        self.assertIn("metaphysical proof", rendered)
        self.assertIn("unsupported geometry claim risk", rendered)
        self.assertIn("hitl geometry question", rendered)
        self.assertNotIn("metaphysical_proof_implemented", rendered)

    def test_roadmap_reality_check_classifies_v8_3_scope(self) -> None:
        assessment = sacred_geometry_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertGreaterEqual(len(assessment), 31)
        self.assertEqual(
            by_item["Golden Ratio Engine"].classification,
            SacredGeometryRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Reaction Diffusion"].classification,
            SacredGeometryRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Geometry to Architecture Mapping"].classification,
            SacredGeometryRoadmapClassification.PARTIAL_REUSABLE,
        )
        self.assertEqual(
            by_item["Interactive Geometry Preview"].classification,
            SacredGeometryRoadmapClassification.PRODUCT_HITL_REQUIRED,
        )
        self.assertEqual(
            by_item["Geometry Demo Assets"].classification,
            SacredGeometryRoadmapClassification.PRODUCT_HITL_REQUIRED,
        )
        self.assertFalse(
            [
                item.item
                for item in assessment
                if item.classification is SacredGeometryRoadmapClassification.MISSING
            ]
        )

    def test_detects_geometry_terms_without_starting_later_capabilities(self) -> None:
        self.assertEqual(
            detect_sacred_geometry_terms("L-system fractal flow field golden ratio"),
            (
                "l_system_growth",
                "fractal_structure",
                "flow_field",
                "golden_ratio",
                "harmonic_proportion",
            ),
        )


if __name__ == "__main__":
    unittest.main()
