import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.artifact_critique import (
    critique_workflow_artifacts,
)
from creative_coding_assistant.orchestration.artifacts import (
    RefinementPassRecord,
    WorkflowArtifact,
)
from creative_coding_assistant.orchestration.creative_translation import (
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.refinement_passes import (
    DEFAULT_REFINEMENT_PASS_LIMIT,
    attach_refinement_history,
    complete_latest_refinement_pass,
    plan_next_refinement_pass,
    start_refinement_pass_record,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName


class RefinementPassTests(unittest.TestCase):
    def test_plans_one_pass_from_existing_quality_signals(self) -> None:
        source = _critiqued_sacred_artifact()

        decision = plan_next_refinement_pass(source_artifact=source)

        self.assertTrue(decision.should_continue)
        self.assertEqual(decision.next_pass_number, 1)
        self.assertEqual(decision.max_passes, DEFAULT_REFINEMENT_PASS_LIMIT)
        self.assertIsNotNone(decision.quality_before)
        self.assertIn("Sacred focus", decision.refinement_objective or "")

        record = start_refinement_pass_record(
            source_artifact=source,
            decision=decision,
        )

        self.assertEqual(record.pass_number, 1)
        self.assertEqual(record.source_artifact_id, source.id)
        self.assertEqual(record.quality_before, decision.quality_before)
        self.assertEqual(record.stop_reason, "continue_available")

    def test_completes_pass_when_quality_improves(self) -> None:
        source = _critiqued_sacred_artifact()
        record = start_refinement_pass_record(
            source_artifact=source,
            decision=plan_next_refinement_pass(source_artifact=source),
        )
        improved = _artifact("improved-field").model_copy(
            update={"quality_score": (record.quality_before or 0.4) + 0.08}
        )

        completed = complete_latest_refinement_pass(
            pass_history=(record,),
            result_artifact=improved,
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].result_artifact_id, improved.id)
        self.assertEqual(completed[0].stop_reason, "quality_improved")
        self.assertIsNotNone(completed[0].quality_after)

    def test_stops_at_max_pass_count_without_unbounded_loop(self) -> None:
        source = _critiqued_sacred_artifact()
        first = RefinementPassRecord(
            pass_number=1,
            source_artifact_id=source.id,
            refinement_objective="Improve composition.",
            stop_reason="continue_available",
            summary="Pass 1 stayed below threshold.",
        )
        second = RefinementPassRecord(
            pass_number=2,
            source_artifact_id=source.id,
            refinement_objective="Improve runtime safety.",
            stop_reason="continue_available",
            summary="Pass 2 stayed below threshold.",
        )

        decision = plan_next_refinement_pass(
            source_artifact=source,
            pass_history=(first, second),
        )

        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, "max_passes_reached")

    def test_stops_when_no_useful_refinement_opportunities_exist(self) -> None:
        source = _artifact("plain-field")

        decision = plan_next_refinement_pass(source_artifact=source)

        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, "no_useful_opportunities")

    def test_attaches_history_to_recommended_artifact_without_mutating_source(
        self,
    ) -> None:
        source = _critiqued_sacred_artifact().model_copy(
            update={"is_recommended": False, "is_default": False}
        )
        refined = _artifact("refined-field").model_copy(
            update={"is_recommended": True, "is_default": True}
        )
        pass_record = RefinementPassRecord(
            pass_number=1,
            source_artifact_id=source.id,
            result_artifact_id=refined.id,
            refinement_objective="Refine focal hierarchy.",
            stop_reason="quality_improved",
            summary="Pass 1 improved quality.",
        )

        artifacts = attach_refinement_history((source, refined), (pass_record,))

        self.assertEqual(artifacts[0].refinement_passes, ())
        self.assertEqual(artifacts[1].refinement_passes, (pass_record,))

    def test_legacy_artifacts_remain_valid_without_pass_history(self) -> None:
        artifact = _artifact("legacy-field")

        self.assertEqual(artifact.refinement_passes, ())


def _critiqued_sacred_artifact() -> WorkflowArtifact:
    artifact = _artifact("sacred-field").model_copy(
        update={
            "content": """
const rings = [];
function setup() { createCanvas(640, 640); }
function draw() {
  background(8);
  // This sacred geometry activates chakra energy fields.
  circle(width / 2, height / 2, 120);
}
""".strip(),
            "summary": (
                "A p5.js mandala that claims sacred geometry activates "
                "chakra energy fields."
            ),
        }
    )
    artifacts, _summary = critique_workflow_artifacts(
        (artifact,),
        request=AssistantRequest(
            query="Create a p5.js mandala with radial symmetry.",
            domain=CreativeCodingDomain.P5_JS,
        ),
        route_decision=RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        ),
    )
    return artifacts[0]


def _artifact(artifact_id: str) -> WorkflowArtifact:
    return WorkflowArtifact(
        id=artifact_id,
        title=f"{artifact_id}.p5.js",
        name=f"{artifact_id}.p5.js",
        language="javascript",
        source_language="javascript",
        content="""
function setup() {
  createCanvas(640, 480);
}
function draw() {
  background(8);
  circle(width / 2, height / 2, 120);
}
""".strip(),
        summary="A p5.js sketch.",
        source_order=1,
        domain=CreativeCodingDomain.P5_JS.value,
        is_creative=True,
        is_default=True,
        preview_eligible=True,
        runtime="p5",
        renderer_id="surface.p5",
        preview_target="browser_sandbox",
        content_hash=f"{artifact_id}-hash",
        creative_translation=derive_creative_translation(
            "Create a p5.js mandala with radial symmetry.",
            domains=(CreativeCodingDomain.P5_JS,),
        ),
    )


if __name__ == "__main__":
    unittest.main()
