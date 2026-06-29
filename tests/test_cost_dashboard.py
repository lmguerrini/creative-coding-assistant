import unittest

from creative_coding_assistant.orchestration import (
    CostDashboard,
    analyze_workflow_cost,
    build_cost_dashboard,
    cost_dashboard_panel_by_id,
    cost_dashboard_panels_for_pressure,
    estimate_routing_cost,
    evaluate_budget_policies,
    predict_cost_for_route,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_COST_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "pressure",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "route_name",
    "relative_cost_units_total",
    "max_relative_cost_units",
    "cost_signal_count",
    "reported_usd_cost",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "cost_dashboard_panel_implemented",
    "provider_pricing_lookup_implemented",
    "live_usage_metering_implemented",
    "cost_scoring_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "hitl_request_implemented",
    "execution_blocking_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class CostDashboardTests(unittest.TestCase):
    def test_default_dashboard_summarizes_cost_metadata(self) -> None:
        workflow = analyze_workflow_cost()
        estimation = estimate_routing_cost(route=RouteName.GENERATE)
        prediction = predict_cost_for_route(route=RouteName.GENERATE)
        policies = evaluate_budget_policies(cost_estimation=estimation)
        dashboard = build_cost_dashboard(
            workflow_cost=workflow,
            cost_estimation=estimation,
            cost_prediction=prediction,
            budget_policies=policies,
        )

        self.assertEqual(dashboard.role, "cost_dashboard")
        self.assertEqual(dashboard.serialization_version, "cost_dashboard.v1")
        self.assertEqual(
            dashboard.source_workflow_cost_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            dashboard.source_cost_estimation_serialization_version,
            estimation.serialization_version,
        )
        self.assertEqual(
            dashboard.source_cost_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            dashboard.source_budget_policy_serialization_version,
            policies.serialization_version,
        )
        self.assertEqual(dashboard.panel_count, 5)
        self.assertEqual(
            dashboard.panel_ids,
            (
                "cost_dashboard::workflow_cost",
                "cost_dashboard::routing_cost_estimate",
                "cost_dashboard::route_cost_prediction",
                "cost_dashboard::budget_policy",
                "cost_dashboard::pricing_boundary",
            ),
        )
        self.assertGreater(dashboard.relative_cost_units_total, 0)
        self.assertGreater(dashboard.highest_relative_cost_units, 0)
        self.assertGreater(dashboard.cost_signal_count, 0)
        self.assertIsNone(dashboard.reported_usd_cost)
        self.assertEqual(dashboard.dashboard_pressure, "guarded")
        self.assertIn(
            "does not look up provider pricing",
            dashboard.authority_boundary,
        )
        self.assertTrue(dashboard.cost_dashboard_implemented)
        self.assertFalse(dashboard.provider_pricing_lookup_implemented)
        self.assertFalse(dashboard.live_usage_metering_implemented)
        self.assertFalse(dashboard.cost_scoring_implemented)
        self.assertFalse(dashboard.budget_enforcement_implemented)
        self.assertFalse(dashboard.cost_based_routing_implemented)
        self.assertFalse(dashboard.hitl_request_implemented)
        self.assertFalse(dashboard.execution_blocking_implemented)
        self.assertFalse(dashboard.provider_model_routing_implemented)
        self.assertFalse(dashboard.model_selection_implemented)
        self.assertFalse(dashboard.provider_execution_implemented)
        self.assertFalse(dashboard.workflow_control_implemented)
        self.assertFalse(dashboard.workflow_execution_implemented)
        self.assertFalse(dashboard.retry_triggering_implemented)
        self.assertFalse(dashboard.prompt_mutation_implemented)
        self.assertFalse(dashboard.persistent_storage_write_implemented)
        self.assertFalse(dashboard.generated_output_mutation_implemented)
        self.assertTrue(dashboard.advisory_only)

    def test_dashboard_panels_are_read_only_and_boundary_explicit(self) -> None:
        dashboard = build_cost_dashboard()

        for panel in dashboard.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_COST_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "cost_dashboard_panel.v1")
            self.assertIsNone(panel.reported_usd_cost)
            self.assertIn(
                "provider_pricing_lookup",
                panel.blocked_runtime_behaviors,
            )
            self.assertTrue(panel.cost_dashboard_panel_implemented)
            self.assertFalse(panel.provider_pricing_lookup_implemented)
            self.assertFalse(panel.live_usage_metering_implemented)
            self.assertFalse(panel.cost_scoring_implemented)
            self.assertFalse(panel.budget_enforcement_implemented)
            self.assertFalse(panel.cost_based_routing_implemented)
            self.assertFalse(panel.hitl_request_implemented)
            self.assertFalse(panel.execution_blocking_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.model_selection_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        boundary = cost_dashboard_panel_by_id(
            "cost_dashboard::pricing_boundary",
            dashboard,
        )
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual(boundary.status, "guarded")
        self.assertEqual(boundary.relative_cost_units_total, 0)
        self.assertEqual(boundary.max_relative_cost_units, 0)
        self.assertEqual(boundary.cost_signal_count, 0)
        self.assertEqual(
            boundary.source_serialization_version,
            "provider_pricing_boundary.v1",
        )

    def test_lookup_helpers_are_stable_and_non_applying(self) -> None:
        dashboard = build_cost_dashboard()
        estimate_panel = cost_dashboard_panel_by_id(
            "cost_dashboard::routing_cost_estimate",
            dashboard,
        )
        guarded = cost_dashboard_panels_for_pressure("guarded", dashboard)
        missing = cost_dashboard_panel_by_id("missing", dashboard)

        self.assertIsNone(missing)
        self.assertIsNotNone(estimate_panel)
        assert estimate_panel is not None
        self.assertEqual(estimate_panel.panel_kind, "routing_cost_estimate")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "cost_dashboard::pricing_boundary",
            tuple(panel.panel_id for panel in guarded),
        )

    def test_dashboard_rejects_mismatched_panel_totals(self) -> None:
        dashboard = build_cost_dashboard()
        payload = dashboard.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            CostDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["relative_cost_units_total"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "relative_cost_units_total must match",
        ):
            CostDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["dashboard_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "dashboard_pressure must match",
        ):
            CostDashboard(**payload)

    def test_dashboard_does_not_declare_runtime_cost_application_terms(self) -> None:
        dashboard = build_cost_dashboard()
        combined_text = " ".join(
            (
                dashboard.authority_boundary,
                *dashboard.blocked_runtime_behaviors,
                *dashboard.advisory_actions,
                *(
                    field
                    for panel in dashboard.panels
                    for field in (
                        panel.panel_id,
                        panel.source_id,
                        *(panel.source_item_ids),
                        *(panel.evidence),
                        *(panel.advisory_actions),
                        *(panel.blocked_runtime_behaviors),
                    )
                ),
            )
        )

        for forbidden_term in (
            "lookup_pricing(",
            "meter_live_usage(",
            "score_cost(",
            "enforce_budget(",
            "route_by_cost(",
            "request_hitl(",
            "block_execution(",
            "select_model(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
