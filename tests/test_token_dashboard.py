import unittest

from creative_coding_assistant.orchestration import (
    TokenDashboard,
    build_token_dashboard,
    forecast_execution_cost,
    optimize_reasoning_budget,
    plan_context_budget,
    token_dashboard_panel_by_id,
    token_dashboard_panels_for_pressure,
)

REQUIRED_TOKEN_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "pressure",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "planned_token_total",
    "reserve_token_total",
    "reported_token_total",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "token_dashboard_panel_implemented",
    "live_usage_metering_implemented",
    "provider_token_collection_implemented",
    "token_budget_enforcement_implemented",
    "runtime_token_allocation_implemented",
    "context_trimming_implemented",
    "prompt_compression_implemented",
    "memory_summarization_implemented",
    "hitl_request_emission_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class TokenDashboardTests(unittest.TestCase):
    def test_default_dashboard_summarizes_token_metadata(self) -> None:
        context = plan_context_budget()
        forecast = forecast_execution_cost()
        reasoning = optimize_reasoning_budget(context_budget=context)
        dashboard = build_token_dashboard(
            context_budget=context,
            execution_cost_forecast=forecast,
            reasoning_budget=reasoning,
        )

        self.assertEqual(dashboard.role, "token_dashboard")
        self.assertEqual(dashboard.serialization_version, "token_dashboard.v1")
        self.assertEqual(
            dashboard.source_context_budget_serialization_version,
            context.serialization_version,
        )
        self.assertEqual(
            dashboard.source_execution_cost_forecast_serialization_version,
            forecast.serialization_version,
        )
        self.assertEqual(
            dashboard.source_reasoning_budget_serialization_version,
            reasoning.serialization_version,
        )
        self.assertEqual(dashboard.panel_count, 4)
        self.assertEqual(
            dashboard.panel_ids,
            (
                "token_dashboard::context_budget",
                "token_dashboard::execution_forecast",
                "token_dashboard::reasoning_budget",
                "token_dashboard::runtime_usage_boundary",
            ),
        )
        self.assertGreater(dashboard.planned_token_total, 0)
        self.assertGreater(dashboard.reserve_token_total, 0)
        self.assertIsNone(dashboard.reported_token_total)
        self.assertEqual(dashboard.dashboard_pressure, "guarded")
        self.assertIn("does not meter live usage", dashboard.authority_boundary)
        self.assertTrue(dashboard.token_dashboard_implemented)
        self.assertFalse(dashboard.live_usage_metering_implemented)
        self.assertFalse(dashboard.provider_token_collection_implemented)
        self.assertFalse(dashboard.token_budget_enforcement_implemented)
        self.assertFalse(dashboard.runtime_token_allocation_implemented)
        self.assertFalse(dashboard.context_trimming_implemented)
        self.assertFalse(dashboard.prompt_compression_implemented)
        self.assertFalse(dashboard.memory_summarization_implemented)
        self.assertFalse(dashboard.hitl_request_emission_implemented)
        self.assertFalse(dashboard.provider_model_routing_implemented)
        self.assertFalse(dashboard.model_selection_implemented)
        self.assertFalse(dashboard.workflow_control_implemented)
        self.assertFalse(dashboard.workflow_execution_implemented)
        self.assertFalse(dashboard.agent_invocation_implemented)
        self.assertFalse(dashboard.node_handler_invocation_implemented)
        self.assertFalse(dashboard.retry_triggering_implemented)
        self.assertFalse(dashboard.persistent_storage_write_implemented)
        self.assertFalse(dashboard.generated_output_mutation_implemented)
        self.assertTrue(dashboard.advisory_only)

    def test_dashboard_panels_are_read_only_and_boundary_explicit(self) -> None:
        dashboard = build_token_dashboard()

        for panel in dashboard.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_TOKEN_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "token_dashboard_panel.v1",
            )
            self.assertIsNone(panel.reported_token_total)
            self.assertIn("live_usage_metering", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.token_dashboard_panel_implemented)
            self.assertFalse(panel.live_usage_metering_implemented)
            self.assertFalse(panel.provider_token_collection_implemented)
            self.assertFalse(panel.token_budget_enforcement_implemented)
            self.assertFalse(panel.runtime_token_allocation_implemented)
            self.assertFalse(panel.context_trimming_implemented)
            self.assertFalse(panel.prompt_compression_implemented)
            self.assertFalse(panel.memory_summarization_implemented)
            self.assertFalse(panel.hitl_request_emission_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.model_selection_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.node_handler_invocation_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        boundary = token_dashboard_panel_by_id(
            "token_dashboard::runtime_usage_boundary",
            dashboard,
        )
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual(boundary.status, "guarded")
        self.assertEqual(boundary.planned_token_total, 0)
        self.assertEqual(boundary.reserve_token_total, 0)
        self.assertEqual(
            boundary.source_serialization_version,
            "runtime_token_usage_boundary.v1",
        )

    def test_lookup_helpers_are_stable_and_non_applying(self) -> None:
        dashboard = build_token_dashboard()
        context_panel = token_dashboard_panel_by_id(
            "token_dashboard::context_budget",
            dashboard,
        )
        guarded = token_dashboard_panels_for_pressure("guarded", dashboard)
        missing = token_dashboard_panel_by_id("missing", dashboard)

        self.assertIsNone(missing)
        self.assertIsNotNone(context_panel)
        assert context_panel is not None
        self.assertEqual(context_panel.panel_kind, "context_budget")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "token_dashboard::runtime_usage_boundary",
            tuple(panel.panel_id for panel in guarded),
        )

    def test_dashboard_rejects_mismatched_panel_totals(self) -> None:
        dashboard = build_token_dashboard()
        payload = dashboard.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            TokenDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["planned_token_total"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "planned_token_total must match",
        ):
            TokenDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["dashboard_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "dashboard_pressure must match",
        ):
            TokenDashboard(**payload)

    def test_dashboard_does_not_declare_runtime_token_application_terms(self) -> None:
        dashboard = build_token_dashboard()
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
                        *panel.source_item_ids,
                        *panel.evidence,
                        *panel.advisory_actions,
                        *panel.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "meter_live_usage(",
            "collect_provider_tokens(",
            "enforce_token_budget(",
            "allocate_runtime_tokens(",
            "trim_context(",
            "compress_prompt(",
            "summarize_memory(",
            "emit_hitl_request(",
            "select_model(",
            "route_provider(",
            "control_workflow(",
            "execute_workflow(",
            "invoke_agent(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
