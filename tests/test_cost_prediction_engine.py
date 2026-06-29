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
    CostPredictionPlan,
    cost_prediction_by_id,
    cost_predictions_for_band,
    predict_cost_for_route,
    route_request,
)
from creative_coding_assistant.orchestration.hybrid_studio import cost_profile_registry
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_COST_PREDICTION_FIELDS = {
    "prediction_id",
    "source_cost_profile_id",
    "source_model_profile_ids",
    "source_provider_selection_profile_ids",
    "route_name",
    "cost_profile_kind",
    "predicted_cost_band",
    "predicted_cost_range",
    "predicted_cost_midpoint",
    "prediction_confidence",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "cost_prediction_engine_implemented",
    "advisory_cost_prediction_implemented",
    "relative_cost_units_only",
    "provider_pricing_lookup_implemented",
    "live_usage_metering_implemented",
    "cost_scoring_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
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


class CostPredictionEngineTests(unittest.TestCase):
    def test_default_cost_prediction_uses_cost_profiles(self) -> None:
        registry = cost_profile_registry()
        plan = predict_cost_for_route(
            route=RouteName.GENERATE,
            cost_profiles=registry,
        )

        self.assertEqual(plan.role, "cost_prediction_engine")
        self.assertEqual(plan.serialization_version, "cost_prediction_plan.v1")
        self.assertEqual(
            plan.source_cost_profile_serialization_version,
            registry.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_prediction_id,
            "cost_prediction::planning_iteration_cost_profile",
        )
        self.assertEqual(plan.recommended_cost_band, "medium")
        self.assertEqual(plan.recommended_cost_midpoint, 3)
        self.assertEqual(plan.prediction_count, 2)
        self.assertEqual(plan.high_or_guarded_prediction_count, 1)
        self.assertEqual(plan.low_cost_prediction_count, 0)
        self.assertIn("does not look up provider pricing", plan.authority_boundary)
        self.assertTrue(plan.cost_prediction_engine_implemented)
        self.assertTrue(plan.advisory_cost_prediction_implemented)
        self.assertTrue(plan.relative_cost_units_only)
        self.assertFalse(plan.provider_pricing_lookup_implemented)
        self.assertFalse(plan.live_usage_metering_implemented)
        self.assertFalse(plan.cost_scoring_implemented)
        self.assertFalse(plan.budget_enforcement_implemented)
        self.assertFalse(plan.cost_based_routing_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.model_selection_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_cost_prediction_decisions_are_advisory_only(self) -> None:
        plan = predict_cost_for_route(route=RouteName.REVIEW)

        for prediction in plan.predictions:
            dumped = prediction.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_COST_PREDICTION_FIELDS)
            self.assertEqual(
                prediction.serialization_version,
                "cost_prediction_decision.v1",
            )
            self.assertEqual(prediction.route_name, RouteName.REVIEW)
            self.assertEqual(
                prediction.predicted_cost_midpoint,
                (prediction.predicted_cost_range[0] + prediction.predicted_cost_range[1])
                // 2,
            )
            self.assertIn(
                "provider_pricing_lookup",
                prediction.blocked_runtime_behaviors,
            )
            self.assertTrue(prediction.cost_prediction_engine_implemented)
            self.assertTrue(prediction.advisory_cost_prediction_implemented)
            self.assertTrue(prediction.relative_cost_units_only)
            self.assertFalse(prediction.provider_pricing_lookup_implemented)
            self.assertFalse(prediction.live_usage_metering_implemented)
            self.assertFalse(prediction.cost_scoring_implemented)
            self.assertFalse(prediction.budget_enforcement_implemented)
            self.assertFalse(prediction.cost_based_routing_implemented)
            self.assertFalse(prediction.provider_model_routing_implemented)
            self.assertFalse(prediction.model_selection_implemented)
            self.assertFalse(prediction.provider_execution_implemented)
            self.assertFalse(prediction.workflow_control_implemented)
            self.assertFalse(prediction.retry_triggering_implemented)
            self.assertFalse(prediction.prompt_mutation_implemented)
            self.assertFalse(prediction.generated_output_mutation_implemented)
            self.assertTrue(prediction.advisory_only)

    def test_lookup_helpers_return_predictions_without_pricing_lookup(self) -> None:
        plan = predict_cost_for_route(route=RouteName.REVIEW)
        recommended = cost_prediction_by_id(
            "cost_prediction::final_review_cost_profile",
            plan,
        )
        low_cost = cost_predictions_for_band("low", plan)
        missing = cost_prediction_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertEqual(recommended.predicted_cost_band, "low")
        self.assertIn(recommended, low_cost)

    def test_plan_rejects_mismatched_predictions_or_recommendation(self) -> None:
        plan = predict_cost_for_route(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["prediction_ids"] = ("missing",) + tuple(payload["prediction_ids"][1:])

        with self.assertRaisesRegex(ValueError, "prediction_ids must match"):
            CostPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_prediction_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "recommended_prediction_id must match",
        ):
            CostPredictionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_cost_midpoint"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_cost_midpoint must match",
        ):
            CostPredictionPlan(**payload)

    def test_cost_prediction_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Predict review cost for a shader design.",
            mode=AssistantMode.REVIEW,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = predict_cost_for_route(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_cost_prediction_does_not_declare_application_terms(self) -> None:
        plan = predict_cost_for_route(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for prediction in plan.predictions
                    for field in (
                        prediction.prediction_id,
                        prediction.source_cost_profile_id,
                        *prediction.source_model_profile_ids,
                        *prediction.source_provider_selection_profile_ids,
                        *prediction.evidence,
                        *prediction.advisory_actions,
                        *prediction.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "lookup_pricing(",
            "meter_usage(",
            "score_cost(",
            "enforce_budget(",
            "route_by_cost(",
            "select_model(",
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
