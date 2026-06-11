import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    creative_translation_prompt_lines,
    derive_creative_translation,
)


class CreativeTranslationTests(unittest.TestCase):
    def test_detects_audiovisual_symbolic_and_geometric_intent(self) -> None:
        translation = derive_creative_translation(
            "Create an audio-reactive mandala using sacred geometry and a "
            "golden ratio spiral. Keep it meditative, with slow pulsing motion."
        )

        self.assertEqual(
            translation.output_modality,
            CreativeOutputModality.AUDIOVISUAL,
        )
        self.assertEqual(
            translation.symbolic_references,
            ("mandala", "sacred geometry"),
        )
        self.assertEqual(
            translation.geometric_references,
            ("golden ratio", "sacred geometry", "spiral"),
        )
        self.assertIn("meditative", translation.mood_atmosphere)
        self.assertIn("pulse", translation.movement_language)
        self.assertEqual(
            translation.runtime_recommendations,
            ("p5.js", "Tone.js"),
        )
        self.assertIsNotNone(translation.sacred_geometry)
        assert translation.sacred_geometry is not None
        self.assertEqual(
            translation.sacred_geometry.concepts,
            ("golden ratio", "mandala", "spiral"),
        )
        self.assertIn(
            "Tone.js",
            translation.sacred_geometry.runtime_recommendations,
        )
        self.assertIsNotNone(translation.visual_style)
        assert translation.visual_style is not None
        self.assertIn("sacred geometry", translation.visual_style.styles)

    def test_uses_domain_and_music_metadata_without_inventing_symbols(self) -> None:
        translation = derive_creative_translation(
            "Build a calm arpeggio with a slow tempo and no autoplay.",
            domains=(CreativeCodingDomain.TONE_JS,),
        )

        self.assertEqual(
            translation.output_modality,
            CreativeOutputModality.AUDIO,
        )
        self.assertEqual(translation.symbolic_references, ())
        self.assertEqual(
            translation.musical_references,
            ("arpeggio", "tempo"),
        )
        self.assertEqual(translation.runtime_recommendations, ("Tone.js",))
        self.assertIn("no autoplay", translation.generation_constraints)
        self.assertIn(
            "Require explicit user interaction before audio playback",
            translation.generation_constraints,
        )
        self.assertIsNone(translation.visual_style)

    def test_prompt_lines_remain_compact_and_evidence_bound(self) -> None:
        translation = derive_creative_translation(
            "Create a monochrome labyrinth with drifting particles.",
            domains=(CreativeCodingDomain.P5_JS,),
            has_image_references=True,
        )

        lines = creative_translation_prompt_lines(translation)

        self.assertIn("Intended modality: visual", lines)
        self.assertIn("Symbolic references: labyrinth", lines)
        self.assertIn("Movement language: drift", lines)
        self.assertIn("Recommended runtime families: p5.js", lines)
        self.assertTrue(
            any("do not invent unsupported symbolic meaning" in line for line in lines)
        )
        self.assertTrue(
            any("supplied image references" in line for line in lines)
        )
        self.assertFalse(
            any(line.startswith("Sacred geometry concepts:") for line in lines)
        )
        self.assertIsNone(translation.shader_presets)
        self.assertIsNotNone(translation.visual_style)
        assert translation.visual_style is not None
        self.assertEqual(
            tuple(style.value for style in translation.visual_style.styles),
            ("monochrome",),
        )
        self.assertIn("Visual style identities: monochrome", lines)

    def test_derives_shader_presets_from_translation_metadata(self) -> None:
        translation = derive_creative_translation(
            "Create an ethereal neon glass sculpture with slow glowing motion.",
            domains=(CreativeCodingDomain.THREE_JS,),
        )

        self.assertIsNotNone(translation.shader_presets)
        assert translation.shader_presets is not None
        self.assertEqual(
            tuple(preset.value for preset in translation.shader_presets.presets),
            ("glass / crystal", "glow", "aura"),
        )
        self.assertEqual(
            translation.shader_presets.runtime_suitability,
            ("Use the selected compatible runtime: Three.js.",),
        )
        lines = creative_translation_prompt_lines(translation)
        self.assertIn(
            "Shader/style presets: glass / crystal, glow, aura",
            lines,
        )
        self.assertIsNotNone(translation.visual_style)
        assert translation.visual_style is not None
        self.assertEqual(
            tuple(style.value for style in translation.visual_style.styles),
            ("ethereal",),
        )
        self.assertIn("Visual style identities: ethereal", lines)

    def test_refinement_preserves_existing_translation_and_adds_new_cues(
        self,
    ) -> None:
        base = derive_creative_translation(
            "Create an audio-reactive cyan mandala with slow drifting motion.",
            domains=(
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.TONE_JS,
            ),
        )

        refined = derive_creative_translation(
            "Make the movement more energetic and pulsing.",
            domains=(CreativeCodingDomain.P5_JS,),
            selected_runtime="p5",
            base_translation=base,
        )

        self.assertEqual(refined.creative_intent, base.creative_intent)
        self.assertEqual(
            refined.output_modality,
            CreativeOutputModality.AUDIOVISUAL,
        )
        self.assertEqual(refined.symbolic_references, ("mandala",))
        self.assertEqual(refined.movement_language, ("drift", "pulse"))
        self.assertEqual(
            refined.runtime_recommendations,
            ("p5.js", "Tone.js"),
        )
        self.assertEqual(refined.sacred_geometry, base.sacred_geometry)
        self.assertEqual(refined.shader_presets, base.shader_presets)
        self.assertEqual(refined.visual_style, base.visual_style)
        self.assertIn("Current refinement:", refined.refinement_targets[-1])


if __name__ == "__main__":
    unittest.main()
