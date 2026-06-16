import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.orchestration.artifacts import (
    ArtifactCritiqueDimension,
    WorkflowArtifact,
    WorkflowArtifactCritique,
)
from creative_coding_assistant.orchestration.creative_quality import (
    evaluate_artifact_creative_quality,
)


class CreativeQualityTests(unittest.TestCase):
    def test_high_quality_output_produces_strong_bounded_observations(self) -> None:
        artifact = _strong_artifact()
        original = artifact.model_dump(mode="json")

        evaluation = evaluate_artifact_creative_quality(artifact)

        self.assertGreaterEqual(evaluation.overall_score, 0.78)
        self.assertEqual(evaluation.composition.level, "strong")
        self.assertEqual(evaluation.coherence.level, "strong")
        self.assertEqual(evaluation.expressiveness.level, "strong")
        self.assertGreaterEqual(len(evaluation.strengths), 3)
        self.assertEqual(evaluation.refinement_opportunities, ())
        self.assertEqual(artifact.model_dump(mode="json"), original)

    def test_weak_output_produces_actionable_refinement_opportunities(self) -> None:
        evaluation = evaluate_artifact_creative_quality(_weak_artifact())

        self.assertLess(evaluation.overall_score, 0.5)
        self.assertEqual(evaluation.composition.level, "weak")
        self.assertEqual(evaluation.originality.level, "weak")
        self.assertEqual(evaluation.coherence.level, "weak")
        self.assertIn(
            "Clarify focal hierarchy",
            evaluation.refinement_opportunities[0],
        )
        self.assertLessEqual(len(evaluation.refinement_opportunities), 5)

    def test_legacy_critique_remains_valid_without_creative_evaluation(self) -> None:
        dimension = ArtifactCritiqueDimension(
            score=0.8,
            rationale="Legacy bounded score.",
        )
        critique = WorkflowArtifactCritique(
            artifact_id="legacy",
            artifact_title="legacy.js",
            source_order=1,
            overall_score=0.8,
            rank=1,
            passed=True,
            prompt_alignment=dimension,
            creative_quality=dimension,
            runtime_suitability=dimension,
            code_quality=dimension,
            preview_readiness=dimension,
            domain_appropriateness=dimension,
            rationale="Legacy critique payload.",
        )

        self.assertIsNone(critique.creative_evaluation)
        self.assertIsNone(critique.sacred_consistency)


def _strong_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="orbital-field",
        title="Orbital Color Field",
        name="orbital-field.p5.js",
        language="javascript",
        source_language="javascript",
        content="""
const particles = [];
function setup() {
  createCanvas(720, 480);
  colorMode(HSL, 360, 100, 100, 1);
}
function draw() {
  background(210, 45, 8, 0.12);
  for (let i = 0; i < 96; i += 1) {
    const angle = frameCount * 0.014 + i * 0.21;
    const radius = 70 + sin(frameCount * 0.01 + i) * 44;
    stroke((i * 7 + frameCount) % 360, 82, 66, 0.72);
    fill((i * 5) % 360, 88, 58, 0.35);
    circle(width / 2 + cos(angle) * radius, height / 2 + sin(angle) * radius, 5);
  }
}
""".strip(),
        summary="An expressive orbital field with layered color trails.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=True,
        preview_eligible=True,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target="browser_sandbox",
        content_hash="orbital-field-hash",
    )


def _weak_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="unfinished-note",
        title="Unfinished Note",
        name="unfinished.js",
        language="javascript",
        source_language="javascript",
        content="TODO: implement later",
        summary="Placeholder output.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        content_hash="unfinished-note-hash",
    )


if __name__ == "__main__":
    unittest.main()
