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
from creative_coding_assistant.orchestration.creative_translation import (
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.quality_calibration import (
    calibrate_artifact_quality,
)
from creative_coding_assistant.orchestration.sacred_consistency import (
    evaluate_artifact_sacred_consistency,
)


class QualityCalibrationTests(unittest.TestCase):
    def test_strong_artifact_calibrates_as_bounded_decision_support(self) -> None:
        artifact = _strong_artifact()
        evaluation = calibrate_artifact_quality(
            artifact,
            dimensions=_dimensions(
                prompt=0.9,
                creative=0.88,
                runtime=1.0,
                code=0.9,
                preview=1.0,
                domain=0.9,
            ),
            legacy_score=0.91,
            creative_evaluation=evaluate_artifact_creative_quality(artifact),
            sacred_consistency=None,
            reasons=(),
        )

        self.assertGreaterEqual(evaluation.score, 0.82)
        self.assertEqual(evaluation.decision_band, "strong_candidate")
        self.assertEqual(evaluation.legacy_score, 0.91)
        self.assertIn("not an objective measure", evaluation.summary)
        self.assertTrue(
            any(signal.key == "runtime_preview" for signal in evaluation.signals)
        )

    def test_weak_artifact_is_capped_by_refinement_and_code_quality(self) -> None:
        artifact = _weak_artifact()
        evaluation = calibrate_artifact_quality(
            artifact,
            dimensions=_dimensions(
                prompt=0.42,
                creative=0.34,
                runtime=0.45,
                code=0.38,
                preview=0.32,
                domain=0.38,
            ),
            legacy_score=0.39,
            creative_evaluation=evaluate_artifact_creative_quality(artifact),
            sacred_consistency=None,
            reasons=("overall_quality_below_threshold", "code_quality"),
        )

        self.assertLess(evaluation.score, 0.5)
        self.assertEqual(evaluation.decision_band, "high_risk")
        self.assertTrue(
            any(
                "legacy critique" in adjustment for adjustment in evaluation.adjustments
            )
        )
        self.assertTrue(
            any("code quality" in adjustment for adjustment in evaluation.adjustments)
        )

    def test_unsafe_symbolic_artifact_gets_conservative_claim_cap(self) -> None:
        artifact = _unsafe_symbolic_artifact()
        creative_evaluation = evaluate_artifact_creative_quality(artifact)
        sacred_consistency = evaluate_artifact_sacred_consistency(artifact)

        evaluation = calibrate_artifact_quality(
            artifact,
            dimensions=_dimensions(
                prompt=0.88,
                creative=creative_evaluation.overall_score,
                runtime=1.0,
                code=0.88,
                preview=1.0,
                domain=0.9,
            ),
            legacy_score=0.88,
            creative_evaluation=creative_evaluation,
            sacred_consistency=sacred_consistency,
            reasons=("sacred_claim_safety",),
        )

        self.assertIsNotNone(sacred_consistency)
        self.assertLessEqual(evaluation.score, 0.58)
        self.assertEqual(evaluation.decision_band, "needs_refinement")
        self.assertTrue(
            any(
                "unsupported symbolic claims" in item for item in evaluation.adjustments
            )
        )

    def test_non_previewable_artifact_surfaces_runtime_preview_risk(self) -> None:
        artifact = _weak_artifact()
        evaluation = calibrate_artifact_quality(
            artifact,
            dimensions=_dimensions(
                prompt=0.72,
                creative=0.62,
                runtime=0.45,
                code=0.72,
                preview=0.32,
                domain=0.72,
            ),
            legacy_score=0.68,
            creative_evaluation=evaluate_artifact_creative_quality(artifact),
            sacred_consistency=None,
            reasons=(),
        )

        runtime_signal = next(
            signal for signal in evaluation.signals if signal.key == "runtime_preview"
        )
        self.assertLess(runtime_signal.score, 0.5)
        self.assertTrue(
            any("runtime and preview" in item for item in evaluation.adjustments)
        )

    def test_legacy_critique_remains_valid_without_calibration(self) -> None:
        dimension = ArtifactCritiqueDimension(score=0.8, rationale="Legacy score.")

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

        self.assertIsNone(critique.calibrated_quality)
        self.assertIsNone(critique.legacy_rank)


def _dimensions(
    *,
    prompt: float,
    creative: float,
    runtime: float,
    code: float,
    preview: float,
    domain: float,
) -> dict[str, ArtifactCritiqueDimension]:
    return {
        "prompt_alignment": ArtifactCritiqueDimension(
            score=prompt,
            rationale="Prompt fit.",
        ),
        "creative_quality": ArtifactCritiqueDimension(
            score=creative,
            rationale="Creative score.",
        ),
        "runtime_suitability": ArtifactCritiqueDimension(
            score=runtime,
            rationale="Runtime fit.",
        ),
        "code_quality": ArtifactCritiqueDimension(
            score=code,
            rationale="Code score.",
        ),
        "preview_readiness": ArtifactCritiqueDimension(
            score=preview,
            rationale="Preview readiness.",
        ),
        "domain_appropriateness": ArtifactCritiqueDimension(
            score=domain,
            rationale="Domain fit.",
        ),
    }


def _strong_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="calibrated-field",
        title="Calibrated Field",
        name="calibrated-field.p5.js",
        language="javascript",
        source_language="javascript",
        content="""
function setup() {
  createCanvas(640, 420);
}
function draw() {
  background(8, 16, 24, 20);
  for (let i = 0; i < 80; i += 1) {
    const angle = frameCount * 0.012 + i * 0.21;
    const radius = 72 + sin(frameCount * 0.01 + i) * 36;
    stroke(120, 220, 210, 160);
    fill(30, 180, 170, 90);
    circle(width / 2 + cos(angle) * radius, height / 2 + sin(angle) * radius, 5);
  }
}
""".strip(),
        summary="A layered p5.js orbital field with color and motion variation.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=True,
        preview_eligible=True,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target="browser_sandbox",
        content_hash="calibrated-field-hash",
    )


def _weak_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="weak-note",
        title="Weak Note",
        name="weak-note.js",
        language="javascript",
        source_language="javascript",
        content="TODO: implement later",
        summary="Placeholder output.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        content_hash="weak-note-hash",
    )


def _unsafe_symbolic_artifact() -> WorkflowArtifact:
    return _strong_artifact().model_copy(
        update={
            "id": "unsafe-mandala",
            "title": "Unsafe Mandala",
            "summary": (
                "A mandala sketch that activates chakra energy fields through "
                "radial geometry."
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
            "creative_translation": derive_creative_translation(
                "Create a p5.js mandala with radial symmetry.",
                domains=(CreativeCodingDomain.P5_JS,),
            ),
        }
    )


if __name__ == "__main__":
    unittest.main()
