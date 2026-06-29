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
    RuntimeRecommendationPlan,
    evaluate_hitl_budget_gate,
    recommend_runtime_execution,
    route_request,
    runtime_recommendation_by_id,
    runtime_recommendations_for_posture,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_RUNTIME_RECOMMENDATION_FIELDS = {
    "recommendation_id",
    "source_hitl_budget_gate_id",
    "source_budget_policy_id",
    "source_model_profile_id",
    "route_name",
    "gate_status",
    "budget_posture",
    "runtime_posture",
    "recommendation_summary",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "runtime_recommendation_engine_implemented",
    "runtime_recommendation_implemented",
    "runtime_recommendation_application_implemented",
    "hitl_request_emitted",
    "execution_policy_application_implemented",
    "execution_blocking_implemented",
    "budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class RuntimeRecommendationEngineTests(unittest.TestCase):
    def test_default_runtime_recommendations_use_hitl_budget_gate(self) -> None:
        gate_plan = evaluate_hitl_budget_gate(route=RouteName.GENERATE)
        plan = recommend_runtime_execution(hitl_budget_gate=gate_plan)

        self.assertEqual(plan.role, "runtime_recommendation_engine")
        self.assertEqual(plan.serialization_version, "runtime_recommendation_plan.v1")
        self.assertEqual(
            plan.source_hitl_budget_gate_serialization_version,
            gate_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_recommendation_id,
            "runtime_recommendation::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_runtime_posture, "guarded_runtime_review")
        self.assertEqual(plan.recommended_gate_status, "review_recommended")
        self.assertEqual(plan.recommendation_count, 3)
        self.assertEqual(plan.guarded_recommendation_count, 1)
        self.assertEqual(plan.operator_review_required_count, 0)
        self.assertIn("does not apply runtime recommendations", plan.authority_boundary)
        self.assertTrue(plan.runtime_recommendation_engine_implemented)
        self.assertTrue(plan.runtime_recommendation_implemented)
        self.assertFalse(plan.runtime_recommendation_application_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.execution_blocking_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_runtime_recommendation_decisions_are_advisory_only(self) -> None:
        plan = recommend_runtime_execution(route=RouteName.REVIEW)

        for recommendation in plan.recommendations:
            dumped = recommendation.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RUNTIME_RECOMMENDATION_FIELDS)
            self.assertEqual(
                recommendation.serialization_version,
                "runtime_recommendation_decision.v1",
            )
            self.assertEqual(recommendation.route_name, RouteName.REVIEW)
            self.assertIn(
                "runtime_recommendation_application",
                recommendation.blocked_runtime_behaviors,
            )
            self.assertTrue(recommendation.runtime_recommendation_engine_implemented)
            self.assertTrue(recommendation.runtime_recommendation_implemented)
            self.assertFalse(
                recommendation.runtime_recommendation_application_implemented,
            )
            self.assertFalse(recommendation.hitl_request_emitted)
            self.assertFalse(recommendation.execution_policy_application_implemented)
            self.assertFalse(recommendation.execution_blocking_implemented)
            self.assertFalse(recommendation.budget_enforcement_implemented)
            self.assertFalse(recommendation.provider_model_routing_implemented)
            self.assertFalse(recommendation.model_selection_implemented)
            self.assertFalse(recommendation.provider_execution_implemented)
            self.assertFalse(recommendation.workflow_control_implemented)
            self.assertFalse(recommendation.retry_triggering_implemented)
            self.assertFalse(recommendation.prompt_mutation_implemented)
            self.assertFalse(recommendation.generated_output_mutation_implemented)
            self.assertTrue(recommendation.advisory_only)

    def test_lookup_helpers_return_recommendations_without_application(self) -> None:
        plan = recommend_runtime_execution(route=RouteName.GENERATE)
        recommended = runtime_recommendation_by_id(
            "runtime_recommendation::creative_reasoning_model_profile",
            plan,
        )
        guarded = runtime_recommendations_for_posture(
            "guarded_runtime_review",
            plan,
        )
        missing = runtime_recommendation_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, guarded)

    def test_plan_rejects_mismatched_recommendations_or_summary(self) -> None:
        plan = recommend_runtime_execution(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["recommendation_ids"] = (
            "missing",
        ) + tuple(payload["recommendation_ids"][1:])

        with self.assertRaisesRegex(ValueError, "recommendation_ids must match"):
            RuntimeRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_recommendation_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_recommendation_id must match",
        ):
            RuntimeRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_runtime_posture"] = "standard_runtime"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_runtime_posture must match",
        ):
            RuntimeRecommendationPlan(**payload)

    def test_runtime_recommendations_do_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review a high-cost shader workflow.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = recommend_runtime_execution(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_runtime_recommendations_do_not_declare_application_terms(self) -> None:
        plan = recommend_runtime_execution(route=RouteName.REVIEW)
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
                        recommendation.source_hitl_budget_gate_id,
                        recommendation.source_budget_policy_id,
                        recommendation.source_model_profile_id,
                        recommendation.recommendation_summary,
                        *recommendation.evidence,
                        *recommendation.advisory_actions,
                        *recommendation.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_runtime_recommendation(",
            "emit_hitl_request(",
            "request_hitl(",
            "apply_execution_policy(",
            "block_execution(",
            "enforce_budget(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
