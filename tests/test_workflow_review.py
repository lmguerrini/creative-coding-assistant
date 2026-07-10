import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    WorkflowReviewOutcome,
    review_assistant_answer,
)
from creative_coding_assistant.orchestration.routing import (
    RouteDecision,
    RouteName,
    route_request,
)
from creative_coding_assistant.orchestration.runtime.artifacts import (
    extract_workflow_artifacts,
)


class WorkflowReviewTests(unittest.TestCase):
    def test_review_passes_practical_answer(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(query="Generate a calm Three.js scene."),
            answer="Use a slow camera move with subtle color shifts.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.PASS)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.score, 1.0)
        self.assertEqual(
            result.rationale,
            "Deterministic review passed without quality gate findings.",
        )

    def test_review_flags_missing_answer(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(query="Generate a scene."),
            answer="",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_answer",))
        self.assertEqual(result.score, 0.75)
        self.assertEqual(
            result.rationale,
            "Deterministic review requested refinement: missing_answer.",
        )

    def test_review_flags_explicit_code_request_without_code_block(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(query="Write code for a p5.js sketch."),
            answer="Create an ellipse in the center of the canvas.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_code_block",))
        self.assertEqual(result.score, 0.75)
        self.assertEqual(
            result.rationale,
            "Deterministic review requested refinement: missing_code_block.",
        )

    def test_review_flags_explain_mode_without_explanation_signals(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(
                query="Explain this shader.",
                mode=AssistantMode.EXPLAIN,
            ),
            answer="Soft gradients and mouse uniforms.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_explanation",))

    def test_explain_mode_sketch_direction_does_not_require_a_code_artifact(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(
                query="Continue the sketch direction.",
                mode=AssistantMode.EXPLAIN,
            ),
            answer="Use the previous palette and keep the motion subtle.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.PASS)
        self.assertEqual(result.reasons, ())

    def test_explain_route_overrides_default_generate_mode_for_deliverables(self) -> None:
        request = AssistantRequest(query="Continue the sketch direction.")
        result = review_assistant_answer(
            request=request,
            answer="Use the previous palette and keep the motion subtle.",
            refinement_count=0,
            route_decision=RouteDecision(
                route=RouteName.EXPLAIN,
                mode=request.mode,
            ),
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.PASS)
        self.assertEqual(result.reasons, ())

    def test_review_flags_debug_mode_without_debug_guidance(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(
                query="Debug this sketch.",
                mode=AssistantMode.DEBUG,
            ),
            answer="Use smoother colors and slower animation.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_debug_guidance",))

    def test_review_flags_unterminated_previewable_artifact_output(self) -> None:
        request = AssistantRequest(
            query=(
                "Design a Three.js kinetic sculpture with camera motion, soft studio "
                "lighting, and a browser-ready animated scene."
            ),
            mode=AssistantMode.GENERATE,
        )
        result = review_assistant_answer(
            request=request,
            answer="```html\n<!doctype html>\n<html><body>",
            refinement_count=0,
            artifacts=(),
            route_decision=route_request(request),
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(
            result.reasons,
            ("unterminated_code_block",),
        )

    def test_review_flags_browser_safe_p5_source_without_a_runnable_preview(self) -> None:
        request = AssistantRequest(
            query=(
                "Create a browser-safe p5.js morphogenesis sketch with setup(), "
                "draw(), and interaction controls."
            ),
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        route_decision = route_request(request)
        answer = "\n".join(
            [
                "```js generated-sketch-1.p5.js",
                "const trail = new Float32Array(64);",
                "function setup() { createCanvas(640, 360); }",
                "function draw() { background(8); circle(20, 20, 10); }",
                "```",
            ]
        )
        artifacts = extract_workflow_artifacts(
            answer,
            request=request,
            route_decision=route_decision,
        )

        result = review_assistant_answer(
            request=request,
            answer=answer,
            refinement_count=0,
            artifacts=artifacts,
            route_decision=route_decision,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_runnable_artifact",))


if __name__ == "__main__":
    unittest.main()
