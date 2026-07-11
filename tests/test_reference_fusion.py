import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration.reference_fusion import (
    derive_reference_fusion_guidance,
    reference_fusion_prompt_lines,
)


class ReferenceFusionTests(unittest.TestCase):
    def test_returns_none_without_references(self) -> None:
        self.assertIsNone(derive_reference_fusion_guidance(()))

    def test_preserves_single_reference_metadata_without_visual_inference(self) -> None:
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
        self.assertEqual(guidance.palette_direction, ())
        self.assertEqual(guidance.composition, ())
        self.assertEqual(guidance.texture_material_cues, ())
        self.assertEqual(guidance.motion_implications, ())
        self.assertTrue(
            any(
                "metadata-only" in item for item in guidance.safety_constraints
            )
        )

    def test_merges_multiple_references_without_filename_based_visual_claims(self) -> None:
        guidance = derive_reference_fusion_guidance(
            (
                _Image("cool-spiral-shadow-reference.webp", "image/webp", 256),
                _Image("amber-lattice-glow-pulse-board.jpg", "image/jpeg", 512),
            )
        )

        assert guidance is not None
        self.assertEqual(guidance.source_count, 2)
        self.assertEqual(guidance.palette_direction, ())
        self.assertEqual(guidance.geometric_structure, ())
        self.assertEqual(guidance.motion_implications, ())
        self.assertTrue(
            any("exact copying" in item for item in guidance.safety_constraints)
        )

        lines = reference_fusion_prompt_lines(guidance)

        self.assertIn("Reference fusion sources: 2", lines)
        self.assertFalse(any(line.startswith("Reference palette") for line in lines))
        self.assertTrue(any(line.startswith("Reference safety") for line in lines))

    def test_person_like_filename_does_not_trigger_visual_or_identity_inference(self) -> None:
        guidance = derive_reference_fusion_guidance(
            (_Image("portrait-face-central-soft.png", "image/png", 128),)
        )

        assert guidance is not None
        self.assertEqual(guidance.composition, ())
        self.assertTrue(
            any(
                "Do not identify people" in item
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
