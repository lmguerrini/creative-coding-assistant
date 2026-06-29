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
    CostEstimationPlan,
    cost_estimate_scenario_by_id,
    cost_estimate_scenarios_for_confidence,
    estimate_routing_cost,
    optimize_quality_cost,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_COST_SCENARIO_FIELDS = {
    "scenario_id",
    "source_quality_cost_candidate_id",
    "source_model_profile_id",
    "route_name",
    "cost_band",
    "source_advisory_cost_range",
    "estimated_min_cost_units",
    "estimated_max_cost_units",
    "estimated_midpoint_cost_units",
    "estimate_confidence",
    "status",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "cost_estimation_implemented",
    "relative_cost_units_only",
    "provider_pricing_lookup_implemented",
    "live_usage_metering_implemented",
    "cost_prediction_implemented",
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


class CostEstimatorTests(unittest.TestCase):
    def test_default_cost_estimator_uses_quality_cost_ranges(self) -> None:
        quality_cost_plan = optimize_quality_cost(route=RouteName.GENERATE)
        plan = estimate_routing_cost(quality_cost_plan=quality_cost_plan)

        self.assertEqual(plan.role, "cost_estimator")
        self.assertEqual(plan.serialization_version, "cost_estimation_plan.v1")
        self.assertEqual(
            plan.source_quality_cost_serialization_version,
            quality_cost_plan.serialization_version,
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(
            plan.recommended_scenario_id,
            "cost_estimate::creative_reasoning_model_profile",
        )
        self.assertEqual(plan.recommended_max_cost_units, 7)
        self.assertEqual(plan.scenario_count, 3)
        self.assertEqual(plan.high_cost_scenario_count, 1)
        self.assertIn("does not look up provider pricing", plan.authority_boundary)
        self.assertTrue(plan.cost_estimation_implemented)
        self.assertTrue(plan.relative_cost_units_only)
        self.assertFalse(plan.provider_pricing_lookup_implemented)
        self.assertFalse(plan.live_usage_metering_implemented)
        self.assertFalse(plan.cost_prediction_implemented)
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

    def test_cost_estimate_scenarios_are_relative_and_advisory(self) -> None:
        plan = estimate_routing_cost(route=RouteName.REVIEW)

        for scenario in plan.scenarios:
            dumped = scenario.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_COST_SCENARIO_FIELDS)
            self.assertEqual(
                scenario.serialization_version,
                "cost_estimate_scenario.v1",
            )
            self.assertEqual(scenario.route_name, RouteName.REVIEW)
            self.assertEqual(
                scenario.estimated_midpoint_cost_units,
                (
                    scenario.estimated_min_cost_units
                    + scenario.estimated_max_cost_units
                )
                // 2,
            )
            self.assertIn(
                "provider_pricing_lookup",
                scenario.blocked_runtime_behaviors,
            )
            self.assertTrue(scenario.cost_estimation_implemented)
            self.assertTrue(scenario.relative_cost_units_only)
            self.assertFalse(scenario.provider_pricing_lookup_implemented)
            self.assertFalse(scenario.live_usage_metering_implemented)
            self.assertFalse(scenario.cost_prediction_implemented)
            self.assertFalse(scenario.budget_enforcement_implemented)
            self.assertFalse(scenario.provider_model_routing_implemented)
            self.assertFalse(scenario.model_selection_implemented)
            self.assertFalse(scenario.provider_execution_implemented)
            self.assertFalse(scenario.workflow_control_implemented)
            self.assertFalse(scenario.retry_triggering_implemented)
            self.assertFalse(scenario.prompt_mutation_implemented)
            self.assertFalse(scenario.generated_output_mutation_implemented)
            self.assertTrue(scenario.advisory_only)

    def test_lookup_helpers_return_scenarios_without_applying_costs(self) -> None:
        plan = estimate_routing_cost(route=RouteName.REVIEW)
        recommended = cost_estimate_scenario_by_id(
            "cost_estimate::evaluation_review_model_profile",
            plan,
        )
        low_confidence = cost_estimate_scenarios_for_confidence("low", plan)
        missing = cost_estimate_scenario_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertIsNotNone(recommended)
        assert recommended is not None
        self.assertEqual(recommended.status, "recommended")
        self.assertIn(recommended, low_confidence)

    def test_plan_rejects_mismatched_scenarios_or_recommendation(self) -> None:
        plan = estimate_routing_cost(route=RouteName.GENERATE)
        payload = plan.model_dump(mode="json")
        payload["scenario_ids"] = ("missing",) + tuple(payload["scenario_ids"][1:])

        with self.assertRaisesRegex(ValueError, "scenario_ids must match"):
            CostEstimationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_scenario_id"] = "missing"

        with self.assertRaisesRegex(ValueError, "recommended_scenario_id must match"):
            CostEstimationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["recommended_max_cost_units"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "recommended_max_cost_units must match",
        ):
            CostEstimationPlan(**payload)

    def test_cost_estimator_does_not_change_request_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Explain a shader architecture.",
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = estimate_routing_cost(route_decision=baseline_decision)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, baseline_decision.route)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_cost_estimator_does_not_declare_runtime_application_terms(self) -> None:
        plan = estimate_routing_cost(route=RouteName.REVIEW)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for scenario in plan.scenarios
                    for field in (
                        scenario.scenario_id,
                        scenario.source_quality_cost_candidate_id,
                        scenario.source_model_profile_id,
                        *scenario.evidence,
                        *scenario.advisory_actions,
                        *scenario.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "lookup_pricing(",
            "meter_usage(",
            "predict_cost(",
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
