import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ModelRecommendationPlan,
    evaluate_execution_policies,
    model_recommendation_by_id,
    model_recommendations_for_policy_posture,
    recommend_model_profile,
    route_model_request,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_MODEL_RECOMMENDATION_FIELDS = {
    "recommendation_id",
    "rank",
    "source_model_route_candidate_id",
    "source_execution_policy_id",
    "source_model_profile_id",
    "model_profile_kind",
    "profile_name",
    "route_name",
    "route_fit_score",
    "route_fit_band",
    "execution_policy_posture",
    "gate_status",
    "recommendation_summary",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "model_recommendation_engine_implemented",
    "model_recommendation_implemented",
    "model_selection_implemented",
    "automatic_model_selection_implemented",
    "configured_model_switching_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "execution_policy_application_implemented",
    "runtime_recommendation_application_implemented",
    "hitl_request_emitted",
    "budget_enforcement_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ModelRecommendationEngineTests(unittest.TestCase):
    def test_default_model_recommendations_use_router_and_policies(self) -> None:
        model_plan = route_model_request(route=RouteName.GENERATE)
        policy_plan = evaluate_execution_policies(route=RouteName.GENERATE)
        plan = recommend_model_profile(
            model_routing=model_plan,
            execution_policies=policy_plan,
        )

        self.assertEqual(plan.role, "model_recommendation_engine")
        self.assertEqual(plan.serialization_version, "model_recommendation_plan.v1")
        self.assertEqual(
            plan.source_model_routing_serialization_version,
            model_plan.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_policy_serialization_version,
            policy_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_recommendation_id,
            "model_recommendation::creative_reasoning_model_profile",
        )
        self.assertEqual(
            plan.recommended_model_profile_id,
            "creative_reasoning_model_profile",
        )
        self.assertEqual(
            plan.recommended_execution_policy_posture,
            "guarded_execution_policy",
        )
        self.assertEqual(plan.recommendation_count, 3)
        self.assertEqual(plan.guarded_recommendation_count, 1)
        self.assertEqual(plan.manual_review_recommendation_count, 0)
        self.assertIn("does not select or switch", plan.authority_boundary)
        self.assertTrue(plan.model_recommendation_engine_implemented)
        self.assertTrue(plan.model_recommendation_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.automatic_model_selection_implemented)
        self.assertFalse(plan.configured_model_switching_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.runtime_recommendation_application_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_model_recommendation_decisions_are_advisory_only(self) -> None:
        plan = recommend_model_profile(route=RouteName.REVIEW)

        for recommendation in plan.recommendations:
            dumped = recommendation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_MODEL_RECOMMENDATION_FIELDS)
            self.assertEqual(
                recommendation.serialization_version,
                "model_recommendation_decision.v1",
            )
            self.assertEqual(recommendation.route_name, RouteName.REVIEW)
            self.assertIn(
                "automatic_model_selection",
                recommendation.blocked_runtime_behaviors,
            )
            self.assertTrue(recommendation.model_recommendation_engine_implemented)
            self.assertTrue(recommendation.model_recommendation_implemented)
            self.assertFalse(recommendation.model_selection_implemented)
            self.assertFalse(recommendation.automatic_model_selection_implemented)
            self.assertFalse(recommendation.configured_model_switching_implemented)
            self.assertFalse(recommendation.provider_model_routing_implemented)
            self.assertFalse(recommendation.provider_execution_implemented)
            self.assertFalse(recommendation.execution_policy_application_implemented)
            self.assertFalse(
                recommendation.runtime_recommendation_application_implemented,
            )
            self.assertFalse(recommendation.hitl_request_emitted)
            self.assertFalse(recommendation.budget_enforcement_implemented)
            self.assertFalse(recommendation.workflow_control_implemented)
            self.assertFalse(recommendation.retry_triggering_implemented)
            self.assertFalse(recommendation.prompt_mutation_implemented)
            self.assertFalse(recommendation.generated_output_mutation_implemented)
            self.assertTrue(recommendation.advisory_only)

    def test_lookup_helpers_return_recommendations_without_model_selection(self) -> None:
        plan = recommend_model_profile(route=RouteName.GENERATE)
        recommended = model_recommendation_by_id(
            "model_recommendation::creative_reasoning_model_profile",
            plan,
        )
        guarded = model_recommendations_for_policy_posture(
            "guarded_execution_policy",
            plan,
        )
        missing = model_recommendation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.rank, 1)
        self.assertIn(recommended, guarded)

    def test_plan_rejects_mismatched_recommendations_or_recommendation(self) -> None:
        plan = recommend_model_profile(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["recommendation_ids"] = (
            "missing",
        ) + tuple(payload["recommendation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "recommendation_ids must match"):
            ModelRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_recommendation_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_recommendation_id must match",
        ):
            ModelRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_model_profile_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_model_profile_id must match",
        ):
            ModelRecommendationPlan(**payload)

    def test_model_recommendations_do_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Generate a model recommendation for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = recommend_model_profile(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_model_recommendations_do_not_declare_selection_terms(self) -> None:
        plan = recommend_model_profile(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for recommendation in plan.recommendations
                    for field in (
                        recommendation.recommendation_id,
                        recommendation.source_model_route_candidate_id,
                        recommendation.source_execution_policy_id,
                        recommendation.source_model_profile_id,
                        recommendation.profile_name,
                        recommendation.recommendation_summary,
                        *recommendation.evidence,
                        *recommendation.advisory_actions,
                        *recommendation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "apply_execution_policy(",
            "apply_runtime_recommendation(",
            "emit_hitl_request(",
            "request_hitl(",
            "enforce_budget(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
