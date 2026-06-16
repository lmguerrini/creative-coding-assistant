import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifact
from creative_coding_assistant.orchestration.creative_translation import (
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.sacred_consistency import (
    evaluate_artifact_sacred_consistency,
)


class SacredConsistencyTests(unittest.TestCase):
    def test_aligned_sacred_geometry_output_produces_bounded_observations(self) -> None:
        artifact = _mandala_artifact()
        original = artifact.model_dump(mode="json")

        evaluation = evaluate_artifact_sacred_consistency(artifact)

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertGreaterEqual(evaluation.overall_score, 0.72)
        self.assertEqual(evaluation.alignment.level, "aligned")
        self.assertEqual(evaluation.claim_safety.level, "aligned")
        self.assertIn("mandala", " ".join(evaluation.alignment.evidence))
        self.assertIn("Bounded motif-analysis score", evaluation.summary)
        self.assertNotIn("spiritual authority", evaluation.summary.lower())
        self.assertEqual(artifact.model_dump(mode="json"), original)

    def test_absent_symbolic_or_geometric_metadata_returns_none(self) -> None:
        artifact = _mandala_artifact().model_copy(
            update={
                "id": "ordinary-field",
                "title": "Ordinary Field",
                "content": "function setup() { createCanvas(640, 360); }",
                "summary": "A simple color field.",
                "creative_translation": derive_creative_translation(
                    "Create a simple p5.js color field.",
                    domains=(CreativeCodingDomain.P5_JS,),
                ),
            }
        )

        evaluation = evaluate_artifact_sacred_consistency(artifact)

        self.assertIsNone(evaluation)

    def test_overclaimed_symbolic_language_is_flagged_without_content_mutation(
        self,
    ) -> None:
        artifact = _mandala_artifact().model_copy(
            update={
                "id": "overclaimed-mandala",
                "title": "Overclaimed Mandala",
                "summary": (
                    "A mandala sketch that activates chakra energy fields and "
                    "guarantees spiritual insight."
                ),
                "content": """
function setup() {
  createCanvas(640, 640);
}
function draw() {
  background(8);
  // This sacred geometry activates chakra energy fields.
  for (let segment = 0; segment < 12; segment += 1) {
    const angle = segment * TWO_PI / 12;
    circle(width / 2 + cos(angle) * 140, height / 2 + sin(angle) * 140, 44);
  }
}
""".strip(),
            }
        )
        original = artifact.model_dump(mode="json")

        evaluation = evaluate_artifact_sacred_consistency(artifact)

        self.assertIsNotNone(evaluation)
        assert evaluation is not None
        self.assertEqual(evaluation.claim_safety.level, "unsupported")
        self.assertLess(evaluation.claim_safety.score, 0.5)
        self.assertIn("overclaim:", " ".join(evaluation.claim_safety.evidence))
        self.assertTrue(
            any(
                "Replace symbolic authority claims" in opportunity
                for opportunity in evaluation.refinement_opportunities
            )
        )
        self.assertEqual(artifact.model_dump(mode="json"), original)


def _mandala_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="mandala-field",
        title="Radial Mandala Field",
        name="mandala-field.p5.js",
        language="javascript",
        source_language="javascript",
        content="""
const rings = [60, 120, 180];
function setup() {
  createCanvas(640, 640);
}
function draw() {
  background(8, 12, 18);
  translate(width / 2, height / 2);
  for (let ring of rings) {
    for (let segment = 0; segment < 16; segment += 1) {
      const angle = segment * TWO_PI / 16 + frameCount * 0.006;
      const radius = ring + sin(frameCount * 0.01 + segment) * 8;
      stroke(220, 180, 90, 160);
      fill(12, 180, 170, 80);
      circle(cos(angle) * radius, sin(angle) * radius, 14);
    }
  }
}
""".strip(),
        summary=(
            "A p5.js radial mandala study using concentric rings, segment counts, "
            "and controlled pulse motion."
        ),
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=True,
        preview_eligible=True,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target="browser_sandbox",
        content_hash="mandala-field-hash",
        creative_translation=derive_creative_translation(
            "Create a p5.js mandala with radial symmetry and concentric motion.",
            domains=(CreativeCodingDomain.P5_JS,),
        ),
    )


if __name__ == "__main__":
    unittest.main()
