import unittest

from creative_coding_assistant.orchestration import (
    ExecutionCostForecast,
    analyze_workflow_cost,
    execution_cost_forecast_scenario_by_id,
    execution_cost_forecast_scenarios_for_kind,
    forecast_execution_cost,
    plan_workflow_pruning,
)

REQUIRED_SCENARIO_FIELDS = {
    "scenario_id",
    "scenario_kind",
    "source_id",
    "lower_bound_tokens",
    "forecast_tokens",
    "upper_bound_tokens",
    "worst_case_token_estimate",
    "token_delta_from_worst_case",
    "relative_cost",
    "confidence",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "execution_cost_forecasting_implemented",
    "provider_pricing_lookup_implemented",
    "live_usage_metering_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "workflow_pruning_application_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "forecasting_only",
}


class ExecutionCostForecastingTests(unittest.TestCase):
    def test_default_forecast_derives_bounded_scenarios(self) -> None:
        costs = analyze_workflow_cost()
        pruning = plan_workflow_pruning(cost_analysis=costs)
        forecast = forecast_execution_cost(
            cost_analysis=costs,
            pruning_plan=pruning,
        )

        self.assertEqual(forecast.role, "execution_cost_forecaster")
        self.assertEqual(forecast.serialization_version, "execution_cost_forecast.v1")
        self.assertEqual(
            forecast.source_cost_serialization_version,
            costs.serialization_version,
        )
        self.assertEqual(
            forecast.source_pruning_serialization_version,
            pruning.serialization_version,
        )
        self.assertEqual(forecast.scenario_count, 4)
        self.assertEqual(
            forecast.scenario_ids,
            (
                "execution_cost_forecast::minimum_success_path",
                "execution_cost_forecast::single_retry_path",
                "execution_cost_forecast::worst_case_bound",
                "execution_cost_forecast::pruning_adjusted_bound",
            ),
        )
        self.assertEqual(
            forecast.minimum_token_forecast,
            costs.critical_path_token_estimate,
        )
        self.assertEqual(
            forecast.single_retry_token_forecast,
            costs.critical_path_token_estimate
            + min(costs.retry_iteration_token_estimate, costs.retry_token_reserve),
        )
        self.assertEqual(
            forecast.worst_case_token_forecast,
            costs.worst_case_token_estimate,
        )
        self.assertEqual(
            forecast.pruning_adjusted_token_forecast,
            costs.worst_case_token_estimate - pruning.estimated_token_savings,
        )
        self.assertEqual(
            forecast.forecast_token_spread,
            costs.worst_case_token_estimate - costs.critical_path_token_estimate,
        )
        self.assertEqual(forecast.forecast_pressure, costs.estimated_cost_pressure)
        self.assertIn("does not look up provider pricing", forecast.authority_boundary)
        self.assertTrue(forecast.execution_cost_forecasting_implemented)
        self.assertFalse(forecast.provider_pricing_lookup_implemented)
        self.assertFalse(forecast.live_usage_metering_implemented)
        self.assertFalse(forecast.budget_enforcement_implemented)
        self.assertFalse(forecast.cost_based_routing_implemented)
        self.assertFalse(forecast.workflow_pruning_application_implemented)
        self.assertFalse(forecast.provider_model_routing_implemented)
        self.assertFalse(forecast.workflow_control_implemented)
        self.assertFalse(forecast.retry_triggering_implemented)
        self.assertFalse(forecast.persistent_storage_write_implemented)
        self.assertFalse(forecast.generated_output_mutation_implemented)
        self.assertTrue(forecast.forecasting_only)

    def test_scenarios_expose_stable_read_only_contracts(self) -> None:
        forecast = forecast_execution_cost()
        worst = execution_cost_forecast_scenario_by_id(
            "execution_cost_forecast::worst_case_bound",
            forecast,
        )
        pruning = execution_cost_forecast_scenario_by_id(
            "execution_cost_forecast::pruning_adjusted_bound",
            forecast,
        )

        self.assertIsNotNone(worst)
        self.assertIsNotNone(pruning)
        assert worst is not None
        assert pruning is not None
        self.assertEqual(worst.token_delta_from_worst_case, 0)
        self.assertLess(pruning.forecast_tokens, pruning.upper_bound_tokens)
        self.assertEqual(pruning.confidence, "medium")

        for scenario in forecast.scenarios:
            self.assertEqual(
                set(scenario.model_dump(mode="json")),
                REQUIRED_SCENARIO_FIELDS,
            )
            self.assertEqual(
                scenario.serialization_version,
                "execution_cost_forecast_scenario.v1",
            )
            self.assertIn("budget_enforcement", scenario.blocked_runtime_behaviors)
            self.assertTrue(scenario.execution_cost_forecasting_implemented)
            self.assertFalse(scenario.provider_pricing_lookup_implemented)
            self.assertFalse(scenario.live_usage_metering_implemented)
            self.assertFalse(scenario.budget_enforcement_implemented)
            self.assertFalse(scenario.cost_based_routing_implemented)
            self.assertFalse(scenario.workflow_pruning_application_implemented)
            self.assertFalse(scenario.provider_model_routing_implemented)
            self.assertFalse(scenario.workflow_control_implemented)
            self.assertFalse(scenario.retry_triggering_implemented)
            self.assertFalse(scenario.persistent_storage_write_implemented)
            self.assertFalse(scenario.generated_output_mutation_implemented)
            self.assertTrue(scenario.forecasting_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        forecast = forecast_execution_cost()
        single_retry = execution_cost_forecast_scenarios_for_kind(
            "single_retry_path",
            forecast,
        )
        missing = execution_cost_forecast_scenario_by_id("missing", forecast)

        self.assertIsNone(missing)
        self.assertEqual(len(single_retry), 1)
        self.assertIs(
            single_retry[0],
            execution_cost_forecast_scenario_by_id(single_retry[0].scenario_id, forecast),
        )

    def test_forecast_rejects_mismatched_scenarios_or_totals(self) -> None:
        forecast = forecast_execution_cost()
        payload = forecast.model_dump(mode="json")
        payload["scenario_ids"] = ("missing",) + tuple(payload["scenario_ids"][1:])

        with self.assertRaisesRegex(ValueError, "scenario_ids must match"):
            ExecutionCostForecast(**payload)

        payload = forecast.model_dump(mode="json")
        payload["forecast_token_spread"] += 1

        with self.assertRaisesRegex(ValueError, "forecast_token_spread must match"):
            ExecutionCostForecast(**payload)

        payload = forecast.model_dump(mode="json")
        payload["forecast_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "forecast_pressure must match"):
            ExecutionCostForecast(**payload)

    def test_forecast_does_not_declare_runtime_cost_control_terms(self) -> None:
        forecast = forecast_execution_cost()
        combined_text = " ".join(
            (
                forecast.authority_boundary,
                *forecast.blocked_runtime_behaviors,
                *forecast.advisory_actions,
                *(
                    field
                    for scenario in forecast.scenarios
                    for field in (
                        scenario.scenario_id,
                        scenario.source_id,
                        *scenario.evidence,
                        *scenario.advisory_actions,
                        *scenario.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "lookup_provider_price(",
            "meter_live_usage(",
            "enforce_budget(",
            "route_by_cost(",
            "apply_pruning(",
            "route_provider(",
            "control_workflow(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
