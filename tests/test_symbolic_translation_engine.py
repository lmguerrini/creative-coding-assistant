import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    SymbolicRoadmapClassification,
    build_v8_1_creative_knowledge_distillation,
    build_v8_2_symbolic_translation_engine,
    detect_symbolic_motif_terms,
    symbolic_translation_prompt_lines,
    symbolic_translation_roadmap_assessment,
)
from creative_coding_assistant.orchestration.creative_translation import (
    derive_creative_translation,
)


class SymbolicTranslationEngineTests(unittest.TestCase):
    def test_translates_symbolic_motifs_into_operational_guidance(self) -> None:
        report = build_v8_2_symbolic_translation_engine(
            "Create an audiovisual phoenix mandala ritual in p5.js with ember "
            "particles, slow pulse, and visible rebirth.",
            domains=(CreativeCodingDomain.P5_JS, CreativeCodingDomain.TONE_JS),
        )

        self.assertEqual(report.capability_id, "v8_2_symbolic_translation_engine")
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn(
            "v8_1_creative_knowledge_distillation",
            report.reused_surface_ids,
        )
        motif_ids = {mapping.motif_id for mapping in report.motif_mappings}
        self.assertIn("phoenix", motif_ids)
        self.assertIn("mandala", motif_ids)
        self.assertIn("pulse", motif_ids)

        operations = {item.kind.value: item for item in report.operational_guidance}
        self.assertIn("visual_structure", operations)
        self.assertIn("audio_mapping", operations)
        self.assertTrue(operations["visual_structure"].parameter_names)
        self.assertIn("p5.js", operations["visual_structure"].runtime_families)
        self.assertIn(
            "symbol_to_art_operational_translation",
            " ".join(report.confidence.v8_1_record_ids),
        )
        self.assertFalse(report.preview_runtime_mutation_implemented)
        self.assertFalse(report.holomind_implemented)
        self.assertFalse(report.holoiverse_implemented)
        self.assertFalse(report.v8_3_sacred_geometry_engine_started)

    def test_prompt_lines_preserve_safe_interpretation_boundaries(self) -> None:
        report = build_v8_2_symbolic_translation_engine(
            "Create a sacred chakra eye mandala that proves ancient cosmic truth.",
            domains=(CreativeCodingDomain.P5_JS,),
        )

        self.assertTrue(report.unsupported_claim_risks)
        self.assertTrue(report.hitl_questions)
        self.assertEqual(report.confidence.band.value, "guarded")
        lines = symbolic_translation_prompt_lines(report)
        rendered = "\n".join(lines).lower()

        self.assertIn("not authoritative", rendered)
        self.assertIn("unsupported symbolic claim risk", rendered)
        self.assertIn("hitl symbolic question", rendered)
        self.assertNotIn("holomind implemented", rendered)
        self.assertNotIn("holoiverse implemented", rendered)

    def test_reuses_existing_translation_without_inventing_universal_dictionary(
        self,
    ) -> None:
        translation = derive_creative_translation(
            "Build a monochrome labyrinth with drifting particles.",
            domains=(CreativeCodingDomain.P5_JS,),
        )
        distillation = build_v8_1_creative_knowledge_distillation(
            indexed_chunk_counts_by_source={"p5_reference": 2},
        )

        report = build_v8_2_symbolic_translation_engine(
            translation.creative_intent,
            domains=(CreativeCodingDomain.P5_JS,),
            creative_translation=translation,
            v8_1_distillation=distillation,
        )

        motif_ids = tuple(mapping.motif_id for mapping in report.motif_mappings)
        self.assertIn("labyrinth", motif_ids)
        self.assertIn("threshold", motif_ids)
        self.assertIn("v3_creative_translation", report.reused_surface_ids)
        self.assertIn(
            "Universal Symbol Dictionary",
            report.risky_hitl_required_items,
        )
        self.assertFalse(report.authoritative_esoteric_interpretation_implemented)
        self.assertFalse(report.comparative_tradition_engine_implemented)

    def test_roadmap_reality_check_classifies_v8_2_scope(self) -> None:
        assessment = symbolic_translation_roadmap_assessment()
        by_item = {item.item: item for item in assessment}

        self.assertGreaterEqual(len(assessment), 28)
        self.assertEqual(
            by_item["Symbol Confidence Engine"].classification,
            SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Creative Pattern Translation"].classification,
            SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        )
        self.assertEqual(
            by_item["Mystical Correspondence Engine"].classification,
            SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        )
        self.assertEqual(
            by_item["Symbol Evolution Engine"].classification,
            SymbolicRoadmapClassification.MISSING,
        )
        self.assertTrue(by_item["Cross-Tradition Symbol Alignment"].hitl_required)

    def test_detects_supported_symbolic_motif_terms(self) -> None:
        self.assertEqual(
            detect_symbolic_motif_terms("spiral phoenix threshold"),
            (
                "spiral",
                "phoenix",
                "fragmentation",
                "reintegration",
                "flame",
                "threshold",
                "gate",
            ),
        )


if __name__ == "__main__":
    unittest.main()
