import unittest

from creative_coding_assistant.contracts import AssistantMode, AssistantRequest
from creative_coding_assistant.orchestration import (
    WorkflowReviewOutcome,
    review_assistant_answer,
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

    def test_review_flags_missing_answer(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(query="Generate a scene."),
            answer="",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_answer",))

    def test_review_flags_explicit_code_request_without_code_block(self) -> None:
        result = review_assistant_answer(
            request=AssistantRequest(query="Write code for a p5.js sketch."),
            answer="Create an ellipse in the center of the canvas.",
            refinement_count=0,
        )

        self.assertEqual(result.outcome, WorkflowReviewOutcome.NEEDS_REFINEMENT)
        self.assertEqual(result.reasons, ("missing_code_block",))

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


if __name__ == "__main__":
    unittest.main()
