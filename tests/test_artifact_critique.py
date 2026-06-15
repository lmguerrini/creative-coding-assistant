import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.artifact_critique import (
    ARTIFACT_CRITIQUE_PASS_THRESHOLD,
    critique_workflow_artifacts,
)
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifact
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class ArtifactCritiqueTests(unittest.TestCase):
    def test_ranks_multi_artifact_candidates_and_marks_recommended_default(self) -> None:
        strong, weak = _strong_p5_artifact(), _weak_python_artifact()

        artifacts, summary = critique_workflow_artifacts(
            (weak, strong),
            request=AssistantRequest(
                query="Create a p5.js sketch with orbiting color trails.",
                domain=CreativeCodingDomain.P5_JS,
            ),
            route_decision=_route_decision(CreativeCodingDomain.P5_JS),
        )

        recommended = next(artifact for artifact in artifacts if artifact.is_recommended)
        rejected = next(artifact for artifact in artifacts if artifact.id == weak.id)

        self.assertEqual(summary.artifact_count, 2)
        self.assertEqual(summary.recommended_artifact_id, strong.id)
        self.assertEqual(summary.refinement_required, False)
        self.assertEqual(recommended.id, strong.id)
        self.assertTrue(recommended.is_default)
        self.assertEqual(recommended.quality_rank, 1)
        self.assertIsNotNone(recommended.critique)
        self.assertGreaterEqual(
            recommended.quality_score or 0,
            ARTIFACT_CRITIQUE_PASS_THRESHOLD,
        )
        self.assertFalse(rejected.is_recommended)
        self.assertGreater(rejected.quality_rank or 0, 1)

    def test_single_failing_artifact_carries_refinement_guidance(self) -> None:
        bad_artifact = _weak_python_artifact()

        artifacts, summary = critique_workflow_artifacts(
            (bad_artifact,),
            request=AssistantRequest(
                query="Create a complete p5.js sketch with animated particles.",
                domain=CreativeCodingDomain.P5_JS,
            ),
            route_decision=_route_decision(CreativeCodingDomain.P5_JS),
        )

        artifact = artifacts[0]

        self.assertEqual(summary.recommended_artifact_id, bad_artifact.id)
        self.assertTrue(summary.refinement_required)
        self.assertIn("overall_quality_below_threshold", summary.refinement_reasons)
        self.assertIsNotNone(summary.refinement_guidance)
        self.assertTrue(artifact.is_default)
        self.assertTrue(artifact.is_recommended)
        self.assertIsNotNone(artifact.critique)
        self.assertFalse(artifact.critique.passed)
        self.assertIsNotNone(artifact.critique.creative_evaluation)
        self.assertIn(
            "Creative focus:",
            artifact.critique.refinement_guidance or "",
        )
        self.assertIn(
            "Clarify focal hierarchy",
            artifact.critique.refinement_guidance or "",
        )
        self.assertEqual(artifact.refinement_reason, summary.refinement_guidance)

    def test_code_only_domain_fit_scores_unsupported_domain_without_preview(self) -> None:
        artifact = _weak_python_artifact().model_copy(
            update={
                "id": "hydra-patch",
                "title": "hydra-patch.js",
                "name": "hydra-patch.js",
                "language": "javascript",
                "source_language": "javascript",
                "content": "osc(10, 0.1, 1.2).modulate(shape(4)).out();",
                "summary": "Hydra patch code.",
                "domain": CreativeCodingDomain.HYDRA.value,
                "preview_eligible": False,
                "runtime": None,
                "renderer_id": None,
                "preview_target": None,
            }
        )

        artifacts, summary = critique_workflow_artifacts(
            (artifact,),
            request=AssistantRequest(
                query="Create a Hydra video synth patch.",
                domains=(CreativeCodingDomain.HYDRA,),
            ),
            route_decision=_route_decision(CreativeCodingDomain.HYDRA),
        )

        critique = artifacts[0].critique

        self.assertEqual(summary.recommended_artifact_id, "hydra-patch")
        self.assertIsNotNone(critique)
        self.assertEqual(critique.domain_appropriateness.score, 0.88)
        self.assertIn("correctly code-only", critique.domain_appropriateness.rationale)


def _route_decision(domain: CreativeCodingDomain) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=domain,
    )


def _strong_p5_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="p5-orbit-sketch",
        title="Orbiting Color Trails",
        name="orbiting-color-trails.js",
        language="javascript",
        source_language="javascript",
        content="""
const particles = [];
function setup() {
  createCanvas(640, 480);
  colorMode(HSL, 360, 100, 100, 1);
}
function draw() {
  background(12, 28, 8, 0.14);
  for (let i = 0; i < 96; i += 1) {
    const angle = frameCount * 0.018 + i * 0.22;
    const radius = 80 + sin(frameCount * 0.01 + i) * 42;
    stroke((i * 5 + frameCount) % 360, 80, 62, 0.7);
    fill((i * 7) % 360, 90, 60, 0.4);
    circle(width / 2 + cos(angle) * radius, height / 2 + sin(angle) * radius, 4);
  }
}
""".strip(),
        summary="A p5.js sketch with animated orbiting color trails.",
        source_order=2,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=True,
        is_default=False,
        preview_eligible=True,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target="browser_sandbox",
        content_hash="p5-orbit-hash",
    )


def _weak_python_artifact() -> WorkflowArtifact:
    return WorkflowArtifact(
        id="python-note",
        title="Python Note",
        name="note.py",
        language="python",
        source_language="python",
        content="TODO: describe a sketch later",
        summary="An incomplete implementation note.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=False,
        is_default=True,
        preview_eligible=False,
        runtime=None,
        renderer_id=None,
        preview_target=None,
        content_hash="python-note-hash",
    )
