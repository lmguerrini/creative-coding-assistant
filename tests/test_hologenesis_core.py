import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    HoloGenesisGraphKind,
    HoloGenesisRoadmapClassification,
    HoloGenesisValidationSeverity,
    build_v8_7_hologenesis_core,
    hologenesis_core_prompt_lines,
    hologenesis_core_roadmap_assessment,
)


class HoloGenesisCoreTests(unittest.TestCase):
    def test_builds_bounded_unification_report_from_v8_1_through_v8_6(self) -> None:
        report = build_v8_7_hologenesis_core(
            "Compose a phoenix mandala museum installation with golden ratio "
            "geometry, sacred architecture, mythic journey, spatial audio, "
            "GLSL glow, Three.js scene graph, audience path, README, portfolio, "
            "capstone outputs, and Unity export planning notes.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.TONE_JS,
            ),
        )

        self.assertEqual(report.capability_id, "v8_7_hologenesis_core")
        self.assertIn("v8_1_creative_knowledge_distillation", report.reused_surface_ids)
        self.assertIn("v8_2_symbolic_translation_engine", report.reused_surface_ids)
        self.assertIn("v8_3_sacred_geometry_engine", report.reused_surface_ids)
        self.assertIn("v8_4_sacred_architecture_engine", report.reused_surface_ids)
        self.assertIn("v8_5_mythopoetic_engine", report.reused_surface_ids)
        self.assertIn("v8_6_immersive_composer", report.reused_surface_ids)

        self.assertEqual({graph.kind for graph in report.unified_graphs}, set(HoloGenesisGraphKind))
        self.assertEqual(len(report.blackboard_entries), 5)
        self.assertEqual(len(report.symbolic_schedule), 4)
        self.assertGreaterEqual(len(report.creative_plan), 4)
        self.assertEqual(len(report.route_recommendations), 4)
        self.assertGreaterEqual(len(report.artistic_decisions), 4)
        self.assertEqual(len(report.curatorial_assessments), 7)
        self.assertEqual(len(report.readiness_scores), 4)
        self.assertEqual(len(report.external_integration_audit), 6)
        self.assertTrue(report.project_bundle.readme_outline)
        self.assertTrue(report.project_bundle.capstone_outputs)
        self.assertTrue(report.project_bundle.reference_discovery_queries)

        self.assertTrue(report.unified_graphs_implemented)
        self.assertTrue(report.creative_blackboard_implemented)
        self.assertTrue(report.symbolic_scheduler_implemented)
        self.assertTrue(report.creative_planner_implemented)
        self.assertTrue(report.creative_router_implemented)
        self.assertTrue(report.curatorial_engines_implemented)
        self.assertTrue(report.installation_simulation_report_implemented)
        self.assertTrue(report.project_bundle_generator_implemented)
        self.assertTrue(report.external_integration_export_planning_implemented)
        self.assertFalse(report.external_dcc_execution_implemented)
        self.assertFalse(report.mcp_tool_execution_implemented)
        self.assertFalse(report.provider_model_routing_implemented)
        self.assertFalse(report.workflow_control_implemented)
        self.assertFalse(report.storage_write_implemented)
        self.assertFalse(report.frontend_ui_implemented)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)
        self.assertFalse(report.v8_8_demo_showcase_started)

        audit_by_label = {item.label: item for item in report.external_integration_audit}
        self.assertEqual(
            audit_by_label["Unity"].classification,
            HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY,
        )
        self.assertFalse(any(item.live_execution_supported for item in report.external_integration_audit))

        rendered = "\n".join(hologenesis_core_prompt_lines(report)).lower()
        self.assertIn("hologenesis boundary", rendered)
        self.assertIn("unified graph", rendered)
        self.assertIn("creative blackboard", rendered)
        self.assertIn("readiness score", rendered)
        self.assertIn("external integration audit", rendered)
        self.assertIn("creative project bundle", rendered)

    def test_boundary_blocks_holomind_holoiverse_live_dcc_and_mcp_claims(self) -> None:
        report = build_v8_7_hologenesis_core(
            "Build HoloMind and HOLOiVERSE, run live Unity, Unreal, Blender, "
            "Houdini, TouchDesigner, and MCP tool execution, and certify ritual efficacy.",
            domains=(CreativeCodingDomain.THREE_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertEqual(report.confidence.band.value, "guarded")
        self.assertTrue(report.hitl_questions)
        self.assertIn(
            HoloGenesisValidationSeverity.HITL_REQUIRED,
            {finding.severity for finding in report.validation_findings},
        )
        self.assertFalse(report.external_dcc_execution_implemented)
        self.assertFalse(report.mcp_tool_execution_implemented)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)
        self.assertFalse(report.v8_8_demo_showcase_started)
        self.assertFalse(report.provider_model_routing_implemented)
        self.assertFalse(report.workflow_control_implemented)
        self.assertFalse(report.storage_write_implemented)

        rendered = "\n".join(hologenesis_core_prompt_lines(report)).lower()
        self.assertIn("unsupported hologenesis claim risk", rendered)
        self.assertIn("hitl hologenesis question", rendered)
        self.assertNotIn("live_execution_supported: true", rendered)

    def test_roadmap_reality_check_classifies_v8_7_scope(self) -> None:
        assessment = hologenesis_core_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertEqual(len(assessment), 43)
        self.assertEqual(
            by_item["Unified Symbolic Graph"].classification,
            HoloGenesisRoadmapClassification.IMPLEMENTED_REPORT_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Creative Project Bundle Generator"].classification,
            HoloGenesisRoadmapClassification.IMPLEMENTED_REPORT_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Unity Integration Layer"].classification,
            HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY,
        )
        self.assertEqual(
            by_item["MCP Creative Tool Layer"].classification,
            HoloGenesisRoadmapClassification.EXPORT_PLANNING_ONLY,
        )
        self.assertEqual(
            by_item["Future HoloMind Hooks"].classification,
            HoloGenesisRoadmapClassification.FUTURE_HOOK_ONLY,
        )
        self.assertFalse(
            [item.item for item in assessment if item.classification is HoloGenesisRoadmapClassification.MISSING]
        )


if __name__ == "__main__":
    unittest.main()
