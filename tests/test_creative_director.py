import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ArtifactCritiqueSummary,
    ClarificationQuestion,
    ClarificationReason,
    ClarificationRequest,
    JinjaPromptRenderer,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    WorkflowReviewOutcome,
    WorkflowReviewResult,
    build_prompt_input_request,
    build_rendered_prompt_request,
    creative_assistant_director_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_execution_plan,
)


class CreativeAssistantDirectorTests(unittest.TestCase):
    def test_director_brief_coordinates_plan_and_retrieval_signals(self) -> None:
        request = _request()
        route = _route()
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            retrieval_chunk_count=2,
        )

        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            retrieval_chunk_count=2,
        )

        self.assertEqual(director.role, "creative_assistant_director")
        self.assertEqual(director.ambiguity_level, "low")
        self.assertEqual(director.retrieval_posture, "available")
        self.assertEqual(director.modality_direction, "visual")
        self.assertEqual(director.runtime_direction, "p5")
        self.assertFalse(director.hitl_required)
        self.assertIn(
            "Render the prompt and continue through the deterministic workflow.",
            director.next_actions,
        )

    def test_director_brief_keeps_human_in_loop_for_clarification(self) -> None:
        request = AssistantRequest(
            query="Make something evocative about rain.",
            mode=AssistantMode.GENERATE,
        )
        clarification = ClarificationRequest(
            reason=ClarificationReason.AMBIGUOUS_MODALITY,
            confidence=0.44,
            summary="The output modality is unclear.",
            original_query=request.query,
            questions=(
                ClarificationQuestion(
                    id="output_modality",
                    prompt="What should be generated first?",
                    suggested_options=(
                        "Visual sketch",
                        "Audio piece",
                        "Audiovisual piece",
                    ),
                    default_recommendation="Visual sketch",
                ),
            ),
            suggested_options=("Visual sketch", "Audio piece", "Audiovisual piece"),
            default_recommendation="Visual sketch",
        )

        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=RouteDecision(
                route=RouteName.GENERATE,
                mode=AssistantMode.GENERATE,
                capabilities=(RouteCapability.OFFICIAL_DOCS,),
            ),
            creative_translation=None,
            creative_plan=None,
            clarification=clarification,
        )

        self.assertTrue(director.hitl_required)
        self.assertEqual(director.ambiguity_level, "high")
        self.assertEqual(director.hitl_reason, clarification.summary)
        self.assertEqual(
            director.next_actions,
            ("Ask the listed HITL clarification before generation.",),
        )

    def test_director_brief_includes_review_and_refinement_signals(self) -> None:
        review = WorkflowReviewResult(
            outcome=WorkflowReviewOutcome.NEEDS_REFINEMENT,
            reasons=("missing_code_block",),
            refinement_count=1,
            score=0.55,
            rationale="Deterministic review requested refinement.",
        )
        critique = ArtifactCritiqueSummary(
            artifact_count=1,
            recommended_artifact_id="artifact-1",
            recommended_artifact_title="field.p5.js",
            average_score=0.62,
            failed_artifact_count=1,
            refinement_required=True,
            refinement_reasons=("overall_quality_below_threshold",),
            refinement_guidance="Reduce visual density.",
        )

        director = derive_creative_assistant_director_brief(
            request=_request(),
            route_decision=_route(),
            creative_translation=None,
            creative_plan=None,
            artifact_critique_summary=critique,
            review_result=review,
            refinement_count=1,
        )

        self.assertIn(
            "Workflow review outcome: needs_refinement.",
            director.critique_focus,
        )
        self.assertIn("missing_code_block", director.refinement_focus)
        self.assertIn("Reduce visual density.", director.refinement_focus)
        self.assertEqual(
            director.next_actions,
            ("Prepare bounded refinement guidance for the next generation pass.",),
        )

    def test_prompt_renderer_includes_director_guidance(self) -> None:
        request = _request()
        route = _route()
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
        )
        prompt_input = prompt_input.model_copy(
            update={"creative_plan": plan, "creative_director": director}
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )

        system = rendered.sections[0].content
        self.assertIn("Creative Assistant Director:", system)
        self.assertIn("Authority boundary:", system)
        self.assertIn("Runtime direction: p5.", system)
        self.assertIn(
            (
                "Next action: Render the prompt and continue through the "
                "deterministic workflow."
            ),
            system,
        )
        self.assertTrue(creative_assistant_director_prompt_lines(director))


def _request() -> AssistantRequest:
    return AssistantRequest(
        query="Generate a luminous p5.js particle field.",
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
    )


def _route() -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
        domains=(CreativeCodingDomain.P5_JS,),
        capabilities=(RouteCapability.OFFICIAL_DOCS, RouteCapability.TOOL_USE),
    )


def _prompt_input(request: AssistantRequest, route: RouteDecision):
    return StructuredPromptInputBuilder().build(
        build_prompt_input_request(
            assistant_request=request,
            route_decision=route,
            assembled_context=None,
        )
    )


if __name__ == "__main__":
    unittest.main()
