import unittest

from creative_coding_assistant.orchestration.sacred_geometry import (
    derive_sacred_geometry_guidance,
)
from creative_coding_assistant.orchestration.shader_presets import (
    derive_shader_preset_guidance,
)
from creative_coding_assistant.orchestration.visual_styles import (
    VisualStyleId,
    derive_visual_style_guidance,
    detect_visual_styles,
    visual_style_prompt_lines,
)


class VisualStyleTests(unittest.TestCase):
    def test_detects_the_bounded_style_vocabulary(self) -> None:
        cases = {
            "minimalist": VisualStyleId.MINIMAL,
            "cyberpunk": VisualStyleId.CYBERPUNK,
            "organic": VisualStyleId.ORGANIC,
            "ritual": VisualStyleId.RITUAL,
            "sacred geometry": VisualStyleId.SACRED_GEOMETRY,
            "generative modernism": VisualStyleId.GENERATIVE_MODERNISM,
            "retro computational": VisualStyleId.RETRO_COMPUTATIONAL,
            "ethereal": VisualStyleId.ETHEREAL,
            "psychedelic": VisualStyleId.PSYCHEDELIC,
            "architectural": VisualStyleId.ARCHITECTURAL,
            "monochrome": VisualStyleId.MONOCHROME,
            "maximalist": VisualStyleId.MAXIMALIST,
        }

        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertIn(
                    expected,
                    detect_visual_styles(f"Create a {phrase} visual."),
                )

    def test_derives_styles_from_translation_and_sacred_metadata(self) -> None:
        sacred_geometry = derive_sacred_geometry_guidance(
            "Use cathedral geometry.",
            output_modality="visual",
        )
        shader_presets = derive_shader_preset_guidance(
            "Add kaleidoscopic plasma.",
            output_modality="visual",
            selected_runtime="glsl",
            sacred_geometry=sacred_geometry,
        )

        styles = detect_visual_styles(
            "Keep the scene coherent.",
            mood_atmosphere=("minimal", "ethereal"),
            color_material_direction=("monochrome",),
            sacred_geometry=sacred_geometry,
            shader_presets=shader_presets,
        )

        self.assertEqual(
            styles,
            (
                VisualStyleId.MINIMAL,
                VisualStyleId.ETHEREAL,
                VisualStyleId.MONOCHROME,
                VisualStyleId.SACRED_GEOMETRY,
                VisualStyleId.ARCHITECTURAL,
                VisualStyleId.PSYCHEDELIC,
            ),
        )

    def test_builds_runtime_specific_practical_guidance(self) -> None:
        guidance = derive_visual_style_guidance(
            "Create an architectural monochrome scene.",
            output_modality="visual",
            selected_runtime="three",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(
            guidance.styles,
            (VisualStyleId.ARCHITECTURAL, VisualStyleId.MONOCHROME),
        )
        self.assertEqual(
            guidance.runtime_suitability,
            (
                "Use the selected compatible runtime: Three.js.",
                (
                    "Express the style through scene composition, materials, "
                    "lighting, and camera."
                ),
            ),
        )
        self.assertTrue(guidance.palette_behavior)
        self.assertTrue(guidance.spatial_organization)

    def test_uses_bounded_approximation_for_incompatible_runtime(self) -> None:
        guidance = derive_visual_style_guidance(
            "Create an architectural installation.",
            output_modality="visual",
            selected_runtime="hydra",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertIn(
            "bounded stylized approximation in Hydra",
            guidance.runtime_suitability[0],
        )

    def test_returns_none_for_audio_only_or_missing_style_cues(self) -> None:
        self.assertIsNone(
            derive_visual_style_guidance(
                "Compose a minimal synth sequence.",
                output_modality="audio",
            )
        )
        self.assertIsNone(
            derive_visual_style_guidance(
                "Create a visual field.",
                output_modality="visual",
            )
        )

    def test_prompt_lines_are_specific_and_implementation_oriented(self) -> None:
        guidance = derive_visual_style_guidance(
            "Create a generative modernist organic composition.",
            output_modality="visual",
            runtime_recommendations=("p5.js",),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        lines = visual_style_prompt_lines(guidance)

        self.assertIn(
            "Visual style identities: generative modernism, organic",
            lines,
        )
        self.assertTrue(any("grids, modular systems" in line for line in lines))
        self.assertTrue(any("bounded 2D marks" in line for line in lines))

    def test_refinement_preserves_prior_styles_and_adds_new_cues(self) -> None:
        base = derive_visual_style_guidance(
            "Create a minimal composition.",
            output_modality="visual",
            selected_runtime="p5",
        )
        self.assertIsNotNone(base)

        unchanged = derive_visual_style_guidance(
            "Make the motion slower.",
            output_modality="visual",
            selected_runtime="p5",
            base_guidance=base,
        )
        extended = derive_visual_style_guidance(
            "Make it monochrome.",
            output_modality="visual",
            selected_runtime="p5",
            base_guidance=base,
        )

        self.assertEqual(unchanged, base)
        self.assertIsNotNone(extended)
        assert extended is not None
        self.assertEqual(
            extended.styles,
            (VisualStyleId.MINIMAL, VisualStyleId.MONOCHROME),
        )


if __name__ == "__main__":
    unittest.main()
