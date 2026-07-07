import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    MythopoeticNarrativeOperationKind,
    MythopoeticNarrativeRoadmapClassification,
    MythopoeticNarrativeValidationSeverity,
    build_v8_5_mythopoetic_narrative_engine,
    detect_mythopoetic_narrative_terms,
    mythopoetic_narrative_prompt_lines,
    mythopoetic_narrative_roadmap_assessment,
)


class MythopoeticNarrativeEngineTests(unittest.TestCase):
    def test_builds_narrative_contracts_with_v8_reuse(self) -> None:
        report = build_v8_5_mythopoetic_narrative_engine(
            "Create a mythopoetic phoenix threshold installation journey: "
            "ritual procession through a labyrinth, mandala center, golden "
            "ratio geometry, gallery audience path, Tone.js pulse, demo story, "
            "presentation narrative, and creative brief.",
            domains=(
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.TONE_JS,
            ),
        )

        self.assertEqual(report.capability_id, "v8_5_mythopoetic_engine")
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn("v8_1_creative_knowledge_distillation", report.reused_surface_ids)
        self.assertIn("v8_2_symbolic_translation_engine", report.reused_surface_ids)
        self.assertIn("v8_3_sacred_geometry_engine", report.reused_surface_ids)
        self.assertIn("v8_4_sacred_architecture_engine", report.reused_surface_ids)

        pattern_ids = {pattern.pattern_id for pattern in report.pattern_guidance}
        self.assertIn("death_rebirth_cycle", pattern_ids)
        self.assertIn("threshold_initiation_arc", pattern_ids)
        self.assertIn("ritual_procession", pattern_ids)
        self.assertIn("labyrinth_journey", pattern_ids)
        self.assertIn("installation_audience_journey", pattern_ids)
        self.assertIn("demo_project_story", pattern_ids)

        operation_kinds = {operation.kind for operation in report.operational_guidance}
        self.assertIn(MythopoeticNarrativeOperationKind.SCENE_SEQUENCE, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.SYMBOLIC_DIALOGUE, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.EMOTIONAL_ARC, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.VISUAL_MAPPING, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.MOTION_MAPPING, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.AUDIO_MAPPING, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.SPATIAL_INSTALLATION_MAPPING, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.CREATIVE_BRIEF, operation_kinds)
        self.assertIn(MythopoeticNarrativeOperationKind.AUDIENCE_COMMUNICATION, operation_kinds)

        self.assertEqual(len(report.scene_sequence), 6)
        self.assertTrue(report.symbol_nodes)
        self.assertTrue(report.symbol_edges)
        self.assertTrue(report.creative_brief)
        self.assertTrue(report.concept_explanation)
        self.assertTrue(report.presentation_narrative)
        self.assertTrue(report.demo_story)
        self.assertTrue(report.audience_communication)
        self.assertTrue(report.narrative_symbol_graph_implemented)
        self.assertTrue(report.provenance_confidence_integration_implemented)
        self.assertTrue(report.v8_2_symbolic_translation_reuse_implemented)
        self.assertTrue(report.v8_3_v8_4_geometry_architecture_reuse_implemented)
        self.assertFalse(report.immersive_audiovisual_composer_implemented)
        self.assertFalse(report.preview_runtime_mutation_implemented)
        self.assertFalse(report.external_dcc_integration_implemented)
        self.assertFalse(report.v8_6_immersive_composer_started)
        self.assertFalse(report.v8_7_hologenesis_os_started)
        self.assertFalse(report.v8_8_demo_showcase_started)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)

    def test_safe_boundaries_for_authority_therapy_and_later_v8_claims(self) -> None:
        report = build_v8_5_mythopoetic_narrative_engine(
            "Prove a divine hidden sacred ritual heals trauma through real "
            "spiritual initiation, then build an immersive composer live "
            "preview in Unity.",
            domains=(CreativeCodingDomain.THREE_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertEqual(report.confidence.band.value, "guarded")
        self.assertTrue(report.hitl_questions)
        severities = {finding.severity for finding in report.validation_findings}
        self.assertIn(MythopoeticNarrativeValidationSeverity.HITL_REQUIRED, severities)
        self.assertIn(MythopoeticNarrativeValidationSeverity.WARNING, severities)
        self.assertFalse(report.authoritative_religious_interpretation_implemented)
        self.assertFalse(report.authoritative_esoteric_interpretation_implemented)
        self.assertFalse(report.psychotherapy_or_diagnosis_implemented)
        self.assertFalse(report.ritual_efficacy_claims_implemented)
        self.assertFalse(report.provider_model_routing_implemented)
        self.assertFalse(report.workflow_control_implemented)
        self.assertFalse(report.storage_write_implemented)

        rendered = "\n".join(mythopoetic_narrative_prompt_lines(report)).lower()
        self.assertIn("unsupported narrative claim risk", rendered)
        self.assertIn("hitl narrative question", rendered)
        self.assertIn("v8.6", rendered)
        self.assertNotIn("immersive_audiovisual_composer_implemented", rendered)

    def test_roadmap_reality_check_classifies_v8_5_scope(self) -> None:
        assessment = mythopoetic_narrative_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertEqual(len(assessment), 26)
        self.assertEqual(
            by_item["Narrative Symbol Graph"].classification,
            MythopoeticNarrativeRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Creative Brief Generator"].classification,
            MythopoeticNarrativeRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Mystical Journey Planner"].classification,
            MythopoeticNarrativeRoadmapClassification.PARTIAL_REUSABLE,
        )
        self.assertEqual(
            by_item["Esoteric Narrative Grammar"].classification,
            MythopoeticNarrativeRoadmapClassification.PARTIAL_REUSABLE,
        )
        self.assertFalse(
            [
                item.item
                for item in assessment
                if item.classification is MythopoeticNarrativeRoadmapClassification.MISSING
            ]
        )

    def test_detects_narrative_terms_without_starting_later_capabilities(self) -> None:
        self.assertEqual(
            detect_mythopoetic_narrative_terms("phoenix threshold ritual gallery demo"),
            (
                "death_rebirth_cycle",
                "threshold_initiation_arc",
                "ritual_procession",
                "installation_audience_journey",
                "demo_project_story",
            ),
        )


if __name__ == "__main__":
    unittest.main()
