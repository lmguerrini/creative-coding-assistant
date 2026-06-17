import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ClarificationReason,
    PromptInputResponse,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_prompt_input_request,
    derive_creative_translation,
    derive_hitl_clarification,
)


class HitlClarificationTests(unittest.TestCase):
    def test_high_confidence_generation_does_not_ask_questions(self) -> None:
        query = "Generate a p5.js cyan particle field at 60 fps."
        route_decision = _route_decision(domains=(CreativeCodingDomain.P5_JS,))
        translation = derive_creative_translation(
            query,
            domains=route_decision.domains,
        )

        clarification = derive_hitl_clarification(
            query=query,
            route_decision=route_decision,
            creative_translation=translation,
        )

        self.assertIsNone(clarification)

    def test_ambiguous_modality_request_gets_bounded_question(self) -> None:
        query = "Make something evocative about rain."
        clarification = _clarification_for(query)

        self.assertIsNotNone(clarification)
        assert clarification is not None
        self.assertEqual(
            clarification.reason,
            ClarificationReason.AMBIGUOUS_MODALITY,
        )
        self.assertLess(clarification.confidence, 0.5)
        self.assertEqual(len(clarification.questions), 1)
        self.assertEqual(
            clarification.questions[0].suggested_options,
            ("Visual sketch", "Audio piece", "Audiovisual piece"),
        )
        self.assertEqual(clarification.default_recommendation, "Visual sketch")
        self.assertIn("modality=unspecified", clarification.signal_summary)

    def test_conflicting_style_runtime_request_gets_runtime_priority_question(
        self,
    ) -> None:
        query = "Generate a Tone.js fragment shader sculpture."
        clarification = _clarification_for(query)

        self.assertIsNotNone(clarification)
        assert clarification is not None
        self.assertEqual(
            clarification.reason,
            ClarificationReason.CONFLICTING_STYLE_RUNTIME,
        )
        self.assertEqual(clarification.questions[0].id, "runtime_priority")
        self.assertIn(
            "Build an audiovisual bridge",
            clarification.questions[0].suggested_options,
        )

    def test_high_cost_multi_candidate_request_gets_direction_questions(
        self,
    ) -> None:
        query = "Generate several candidate directions for an interactive installation."
        clarification = _clarification_for(query, cost_complexity_estimate=0.82)

        self.assertIsNotNone(clarification)
        assert clarification is not None
        self.assertEqual(
            clarification.reason,
            ClarificationReason.HIGH_COST_MULTI_CANDIDATE,
        )
        self.assertLessEqual(len(clarification.questions), 3)
        self.assertEqual(clarification.questions[0].id, "candidate_priority")
        self.assertEqual(clarification.questions[1].id, "selection_axis")

    def test_clarification_answer_allows_generation_to_continue(self) -> None:
        query = "Make something evocative about rain."
        route_decision = _route_decision()
        translation = derive_creative_translation(query, domains=route_decision.domains)

        clarification = derive_hitl_clarification(
            query=query,
            route_decision=route_decision,
            creative_translation=translation,
            clarification_response="Use a visual p5.js sketch.",
        )

        self.assertIsNone(clarification)

    def test_prompt_input_builder_persists_clarification_metadata(self) -> None:
        assistant_request = AssistantRequest(
            query="Make something evocative about rain."
        )
        route_decision = _route_decision()

        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=route_decision,
                assembled_context=None,
            )
        )

        self.assertIsNotNone(prompt_input.clarification)
        assert prompt_input.clarification is not None
        self.assertEqual(
            prompt_input.clarification.reason,
            ClarificationReason.AMBIGUOUS_MODALITY,
        )
        self.assertEqual(
            prompt_input.clarification.original_query,
            assistant_request.query,
        )

    def test_prompt_input_builder_skips_clarification_after_user_answer(
        self,
    ) -> None:
        assistant_request = AssistantRequest(
            query="Make something evocative about rain.",
            clarificationResponse="Use a visual p5.js sketch.",
        )
        route_decision = _route_decision()

        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=assistant_request,
                route_decision=route_decision,
                assembled_context=None,
            )
        )

        self.assertIsNone(prompt_input.clarification)
        self.assertEqual(
            prompt_input.user_input.clarification_response,
            "Use a visual p5.js sketch.",
        )

    def test_legacy_prompt_input_payload_defaults_to_no_clarification(self) -> None:
        prompt_input = PromptInputResponse.model_validate(
            {
                "request": {
                    "route": "generate",
                    "assistant_request": {"query": "Generate a sketch."},
                },
                "user_input": {
                    "query": "Generate a sketch.",
                    "mode": "generate",
                },
            }
        )

        self.assertIsNone(prompt_input.clarification)


def _clarification_for(
    query: str,
    *,
    cost_complexity_estimate: float | None = None,
):
    route_decision = _route_decision()
    translation = derive_creative_translation(query, domains=route_decision.domains)
    return derive_hitl_clarification(
        query=query,
        route_decision=route_decision,
        creative_translation=translation,
        cost_complexity_estimate=cost_complexity_estimate,
    )


def _route_decision(
    *,
    domains: tuple[CreativeCodingDomain, ...] = (),
) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=domains[0] if len(domains) == 1 else None,
        domains=domains,
        capabilities=(RouteCapability.TOOL_USE,),
    )


if __name__ == "__main__":
    unittest.main()
