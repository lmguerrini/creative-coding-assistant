import unittest

from creative_coding_assistant.orchestration.sacred_geometry import (
    derive_sacred_geometry_guidance,
)
from creative_coding_assistant.orchestration.shader_presets import (
    ShaderPresetId,
    derive_shader_preset_guidance,
    detect_shader_presets,
    shader_preset_prompt_lines,
)


class ShaderPresetTests(unittest.TestCase):
    def test_detects_the_bounded_preset_vocabulary(self) -> None:
        cases = {
            "glowing": ShaderPresetId.GLOW,
            "aura": ShaderPresetId.AURA,
            "plasma": ShaderPresetId.PLASMA,
            "bloom-like": ShaderPresetId.BLOOM_EMISSION,
            "refraction": ShaderPresetId.REFRACTION,
            "crystal": ShaderPresetId.GLASS_CRYSTAL,
            "volumetric fog": ShaderPresetId.VOLUMETRIC_ATMOSPHERE,
            "fractal field": ShaderPresetId.FRACTAL_FIELD,
            "kaleidoscopic": ShaderPresetId.KALEIDOSCOPIC_SYMMETRY,
            "sacred light": ShaderPresetId.SACRED_LIGHT,
        }

        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertIn(
                    expected,
                    detect_shader_presets(f"Create a {phrase} visual."),
                )

    def test_derives_presets_from_translation_and_sacred_geometry_metadata(
        self,
    ) -> None:
        sacred_geometry = derive_sacred_geometry_guidance(
            "Create a mandala with fractal symmetry.",
            output_modality="visual",
        )

        presets = detect_shader_presets(
            "Keep the field restrained.",
            mood_atmosphere=("ethereal", "ritual"),
            color_material_direction=("glass", "neon"),
            sacred_geometry=sacred_geometry,
        )

        self.assertEqual(
            presets,
            (
                ShaderPresetId.GLASS_CRYSTAL,
                ShaderPresetId.GLOW,
                ShaderPresetId.AURA,
                ShaderPresetId.SACRED_LIGHT,
                ShaderPresetId.KALEIDOSCOPIC_SYMMETRY,
                ShaderPresetId.FRACTAL_FIELD,
            ),
        )

    def test_builds_practical_guidance_for_a_compatible_runtime(self) -> None:
        guidance = derive_shader_preset_guidance(
            "Create a glass refraction shader with restrained glow.",
            output_modality="visual",
            selected_runtime="glsl",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(
            guidance.presets,
            (
                ShaderPresetId.GLASS_CRYSTAL,
                ShaderPresetId.REFRACTION,
                ShaderPresetId.GLOW,
            ),
        )
        self.assertEqual(
            guidance.runtime_suitability,
            ("Use the selected compatible runtime: GLSL.",),
        )
        self.assertTrue(guidance.shader_structure)
        self.assertTrue(guidance.performance_constraints)

    def test_uses_a_bounded_approximation_for_an_incompatible_runtime(
        self,
    ) -> None:
        guidance = derive_shader_preset_guidance(
            "Create crystal refraction.",
            output_modality="visual",
            selected_runtime="hydra",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertIn(
            "bounded stylized approximation",
            guidance.runtime_suitability[0],
        )

    def test_returns_none_for_audio_only_or_missing_visual_cues(self) -> None:
        self.assertIsNone(
            derive_shader_preset_guidance(
                "Compose a glowing synth drone.",
                output_modality="audio",
            )
        )
        self.assertIsNone(
            derive_shader_preset_guidance(
                "Create a minimal visual field.",
                output_modality="visual",
            )
        )

    def test_prompt_lines_preserve_accuracy_and_performance_boundaries(self) -> None:
        guidance = derive_shader_preset_guidance(
            "Render volumetric fog with bloom.",
            output_modality="visual",
            runtime_recommendations=("Three.js",),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        lines = shader_preset_prompt_lines(guidance)

        self.assertIn(
            "Shader/style presets: bloom-like emission, volumetric atmosphere",
            lines,
        )
        self.assertTrue(any("not proof of physical accuracy" in line for line in lines))
        self.assertTrue(any("Cap ray-march steps" in line for line in lines))

    def test_refinement_preserves_prior_presets_and_adds_new_cues(self) -> None:
        base = derive_shader_preset_guidance(
            "Create a plasma field.",
            output_modality="visual",
            selected_runtime="glsl",
        )
        self.assertIsNotNone(base)

        unchanged = derive_shader_preset_guidance(
            "Make the movement slower.",
            output_modality="visual",
            selected_runtime="glsl",
            base_guidance=base,
        )
        extended = derive_shader_preset_guidance(
            "Add a subtle glow.",
            output_modality="visual",
            selected_runtime="glsl",
            base_guidance=base,
        )

        self.assertEqual(unchanged, base)
        self.assertIsNotNone(extended)
        assert extended is not None
        self.assertEqual(
            extended.presets,
            (ShaderPresetId.PLASMA, ShaderPresetId.GLOW),
        )


if __name__ == "__main__":
    unittest.main()
