import unittest

from creative_coding_assistant.orchestration.sacred_geometry import (
    derive_sacred_geometry_guidance,
    detect_sacred_geometry_concepts,
    sacred_geometry_prompt_lines,
)


class SacredGeometryTests(unittest.TestCase):
    def test_supports_the_bounded_vocabulary(self) -> None:
        cases = {
            "mandala": "mandala",
            "yantra": "yantra",
            "Sri Yantra": "Sri Yantra",
            "Flower of Life": "Flower of Life",
            "Metatron’s Cube": "Metatron's Cube",
            "Merkaba": "Merkaba",
            "torus": "torus",
            "spiral": "spiral",
            "golden ratio": "golden ratio",
            "Fibonacci": "Fibonacci",
            "vesica piscis": "vesica piscis",
            "fractal symmetry": "fractal symmetry",
            "radial symmetry": "radial symmetry",
            "temple geometry": "temple geometry",
            "cathedral geometry": "cathedral geometry",
        }

        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertIn(
                    expected,
                    detect_sacred_geometry_concepts(
                        f"Create a study using {phrase}."
                    ),
                )

    def test_detects_bounded_concepts_without_generic_yantra_duplication(self) -> None:
        concepts = detect_sacred_geometry_concepts(
            "Build a Sri Yantra over a Flower of Life with Fibonacci spacing, "
            "radial symmetry, and cathedral geometry."
        )

        self.assertEqual(
            concepts,
            (
                "Sri Yantra",
                "Flower of Life",
                "radial symmetry",
                "cathedral geometry",
                "Fibonacci",
            ),
        )
        self.assertNotIn("yantra", concepts)

    def test_returns_none_when_no_supported_concept_is_explicit(self) -> None:
        self.assertIsNone(
            derive_sacred_geometry_guidance(
                "Create an abstract field with balanced shapes."
            )
        )

    def test_maps_visual_concepts_to_practical_bounded_guidance(self) -> None:
        guidance = derive_sacred_geometry_guidance(
            "Create a translucent Metatron's Cube inside a rotating torus.",
            output_modality="visual",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(guidance.concepts, ("Metatron's Cube", "torus"))
        self.assertIn("Three.js", guidance.runtime_recommendations)
        self.assertTrue(
            any("connect" in item.lower() for item in guidance.geometric_structure)
        )
        self.assertTrue(
            any("rotational" in item.lower() for item in guidance.symmetry_type)
        )
        self.assertEqual(guidance.audio_implications, ())
        self.assertIn(
            (
                "Treat sacred-geometry terms as practical design motifs, not "
                "authoritative spiritual claims."
            ),
            guidance.generation_constraints,
        )

    def test_adds_audio_implications_only_for_relevant_modalities(self) -> None:
        guidance = derive_sacred_geometry_guidance(
            "Make an audiovisual Fibonacci spiral.",
            output_modality="audiovisual",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertIn("Tone.js", guidance.runtime_recommendations)
        self.assertTrue(guidance.audio_implications)
        self.assertTrue(
            any(
                "explicit user interaction" in item
                for item in guidance.audio_implications
            )
        )

    def test_audio_only_guidance_recommends_an_audio_runtime(self) -> None:
        guidance = derive_sacred_geometry_guidance(
            "Compose a Fibonacci rhythm.",
            output_modality="audio",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(guidance.runtime_recommendations, ("Tone.js",))
        self.assertTrue(guidance.audio_implications)

    def test_prompt_lines_include_practical_safety_boundaries(self) -> None:
        guidance = derive_sacred_geometry_guidance(
            "Render a meditative mandala.",
            output_modality="visual",
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        lines = sacred_geometry_prompt_lines(guidance)

        self.assertIn("Sacred geometry concepts: mandala", lines)
        self.assertTrue(any(line.startswith("Geometric structure:") for line in lines))
        self.assertTrue(
            any("not authoritative spiritual claims" in line for line in lines)
        )
        self.assertTrue(any("not present in the request" in line for line in lines))

    def test_refinement_preserves_prior_guidance_and_adds_new_concepts(self) -> None:
        base = derive_sacred_geometry_guidance(
            "Create a mandala.",
            output_modality="visual",
        )
        self.assertIsNotNone(base)

        unchanged = derive_sacred_geometry_guidance(
            "Make the motion slower.",
            output_modality="visual",
            base_guidance=base,
        )
        extended = derive_sacred_geometry_guidance(
            "Add a vesica piscis layer.",
            output_modality="visual",
            base_guidance=base,
        )

        self.assertEqual(unchanged, base)
        self.assertIsNotNone(extended)
        assert extended is not None
        self.assertEqual(extended.concepts, ("mandala", "vesica piscis"))


if __name__ == "__main__":
    unittest.main()
