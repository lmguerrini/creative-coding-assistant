import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration.reference_fusion import (
    derive_reference_fusion_guidance,
    reference_fusion_prompt_lines,
)


class ReferenceFusionTests(unittest.TestCase):
    def test_returns_none_without_references(self) -> None:
        self.assertIsNone(derive_reference_fusion_guidance(()))

    def test_derives_single_reference_guidance_from_metadata(self) -> None:
        guidance = derive_reference_fusion_guidance(
            (
                _Image(
                    "warm-neon-grid-glass-drift.png",
                    "image/png",
                    128,
                ),
            )
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(guidance.source_count, 1)
        self.assertIn("warm palette bias", guidance.palette_direction)
        self.assertIn("neon accent contrast", guidance.palette_direction)
        self.assertIn("grid-based spatial layout", guidance.composition)
        self.assertIn("glasslike refraction cues", guidance.texture_material_cues)
        self.assertIn("slow drifting motion", guidance.motion_implications)
        self.assertTrue(
            any(
                "Do not identify people" in item
                for item in guidance.safety_constraints
            )
        )

    def test_merges_multiple_references_without_copy_claims(self) -> None:
        guidance = derive_reference_fusion_guidance(
            (
                _Image("cool-spiral-shadow-reference.webp", "image/webp", 256),
                _Image("amber-lattice-glow-pulse-board.jpg", "image/jpeg", 512),
            )
        )

        assert guidance is not None
        self.assertEqual(guidance.source_count, 2)
        self.assertIn("cool blue palette", guidance.palette_direction)
        self.assertIn("amber highlights", guidance.palette_direction)
        self.assertIn("spiral structure", guidance.geometric_structure)
        self.assertIn("lattice structure", guidance.geometric_structure)
        self.assertIn("pulsing temporal behavior", guidance.motion_implications)
        self.assertTrue(
            any("exact copying" in item for item in guidance.safety_constraints)
        )

        lines = reference_fusion_prompt_lines(guidance)

        self.assertIn("Reference fusion sources: 2", lines)
        self.assertTrue(any(line.startswith("Reference palette") for line in lines))
        self.assertTrue(any(line.startswith("Reference safety") for line in lines))

    def test_person_like_reference_uses_only_non_identifying_layout_cues(self) -> None:
        guidance = derive_reference_fusion_guidance(
            (_Image("portrait-face-central-soft.png", "image/png", 128),)
        )

        assert guidance is not None
        self.assertIn(
            "central subject framing without identity assumptions",
            guidance.composition,
        )
        self.assertTrue(
            any(
                "non-identifying layout cues" in item
                for item in guidance.safety_constraints
            )
        )


@dataclass(frozen=True)
class _Image:
    name: str
    mime_type: str
    size_bytes: int
    id: str = "image-reference"


if __name__ == "__main__":
    unittest.main()
