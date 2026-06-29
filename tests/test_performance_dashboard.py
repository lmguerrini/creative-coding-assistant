import unittest

from creative_coding_assistant.orchestration import (
    PerformanceDashboard,
    build_performance_dashboard,
    detect_performance_regressions,
    optimize_resource_utilization,
    performance_dashboard_panel_by_id,
    performance_dashboard_panels_for_pressure,
    plan_performance_benchmarking,
    predict_performance,
)

REQUIRED_PERFORMANCE_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "pressure",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "relative_performance_units_total",
    "recommended_performance_units",
    "performance_signal_count",
    "measured_latency_ms",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "performance_dashboard_panel_implemented",
    "runtime_performance_measurement_implemented",
    "benchmark_execution_implemented",
    "timer_collection_implemented",
    "profiler_hook_installation_implemented",
    "runtime_trace_collection_implemented",
    "runtime_regression_detection_implemented",
    "resource_allocation_implemented",
    "capacity_enforcement_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class PerformanceDashboardTests(unittest.TestCase):
    def test_default_dashboard_summarizes_performance_metadata(self) -> None:
        prediction = predict_performance()
        benchmarking = plan_performance_benchmarking(
            performance_prediction=prediction,
        )
        regression = detect_performance_regressions(
            performance_prediction=prediction,
            performance_benchmarking=benchmarking,
        )
        resources = optimize_resource_utilization(
            performance_benchmarking=benchmarking,
            performance_regression=regression,
        )
        dashboard = build_performance_dashboard(
            performance_prediction=prediction,
            performance_benchmarking=benchmarking,
            performance_regression=regression,
            resource_utilization=resources,
        )

        self.assertEqual(dashboard.role, "performance_dashboard")
        self.assertEqual(
            dashboard.serialization_version,
            "performance_dashboard.v1",
        )
        self.assertEqual(
            dashboard.source_performance_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            dashboard.source_performance_benchmarking_serialization_version,
            benchmarking.serialization_version,
        )
        self.assertEqual(
            dashboard.source_performance_regression_serialization_version,
            regression.serialization_version,
        )
        self.assertEqual(
            dashboard.source_resource_utilization_serialization_version,
            resources.serialization_version,
        )
        self.assertEqual(dashboard.panel_count, 5)
        self.assertEqual(
            dashboard.panel_ids,
            (
                "performance_dashboard::performance_prediction",
                "performance_dashboard::benchmarking_readiness",
                "performance_dashboard::regression_detection",
                "performance_dashboard::resource_utilization",
                "performance_dashboard::measurement_boundary",
            ),
        )
        self.assertGreater(dashboard.relative_performance_units_total, 0)
        self.assertGreater(dashboard.highest_recommended_performance_units, 0)
        self.assertGreater(dashboard.performance_signal_count, 0)
        self.assertIsNone(dashboard.measured_latency_ms)
        self.assertEqual(dashboard.dashboard_pressure, "guarded")
        self.assertIn(
            "does not measure runtime performance",
            dashboard.authority_boundary,
        )
        self.assertTrue(dashboard.performance_dashboard_implemented)
        self.assertFalse(dashboard.runtime_performance_measurement_implemented)
        self.assertFalse(dashboard.benchmark_execution_implemented)
        self.assertFalse(dashboard.timer_collection_implemented)
        self.assertFalse(dashboard.profiler_hook_installation_implemented)
        self.assertFalse(dashboard.runtime_trace_collection_implemented)
        self.assertFalse(dashboard.runtime_regression_detection_implemented)
        self.assertFalse(dashboard.resource_allocation_implemented)
        self.assertFalse(dashboard.capacity_enforcement_implemented)
        self.assertFalse(dashboard.provider_model_routing_implemented)
        self.assertFalse(dashboard.workflow_control_implemented)
        self.assertFalse(dashboard.workflow_timing_change_implemented)
        self.assertFalse(dashboard.workflow_graph_mutation_implemented)
        self.assertFalse(dashboard.workflow_execution_implemented)
        self.assertFalse(dashboard.agent_invocation_implemented)
        self.assertFalse(dashboard.node_handler_invocation_implemented)
        self.assertFalse(dashboard.retry_triggering_implemented)
        self.assertFalse(dashboard.prompt_mutation_implemented)
        self.assertFalse(dashboard.persistent_storage_write_implemented)
        self.assertFalse(dashboard.generated_output_mutation_implemented)
        self.assertTrue(dashboard.advisory_only)

    def test_dashboard_panels_are_read_only_and_boundary_explicit(self) -> None:
        dashboard = build_performance_dashboard()

        for panel in dashboard.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PERFORMANCE_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "performance_dashboard_panel.v1",
            )
            self.assertIsNone(panel.measured_latency_ms)
            self.assertIn(
                "runtime_performance_measurement",
                panel.blocked_runtime_behaviors,
            )
            self.assertTrue(panel.performance_dashboard_panel_implemented)
            self.assertFalse(panel.runtime_performance_measurement_implemented)
            self.assertFalse(panel.benchmark_execution_implemented)
            self.assertFalse(panel.timer_collection_implemented)
            self.assertFalse(panel.profiler_hook_installation_implemented)
            self.assertFalse(panel.runtime_trace_collection_implemented)
            self.assertFalse(panel.runtime_regression_detection_implemented)
            self.assertFalse(panel.resource_allocation_implemented)
            self.assertFalse(panel.capacity_enforcement_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_timing_change_implemented)
            self.assertFalse(panel.workflow_graph_mutation_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.node_handler_invocation_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        boundary = performance_dashboard_panel_by_id(
            "performance_dashboard::measurement_boundary",
            dashboard,
        )
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual(boundary.status, "guarded")
        self.assertEqual(boundary.relative_performance_units_total, 0)
        self.assertEqual(boundary.recommended_performance_units, 0)
        self.assertEqual(boundary.performance_signal_count, 0)
        self.assertEqual(
            boundary.source_serialization_version,
            "runtime_performance_measurement_boundary.v1",
        )

    def test_lookup_helpers_are_stable_and_non_applying(self) -> None:
        dashboard = build_performance_dashboard()
        prediction_panel = performance_dashboard_panel_by_id(
            "performance_dashboard::performance_prediction",
            dashboard,
        )
        guarded = performance_dashboard_panels_for_pressure("guarded", dashboard)
        missing = performance_dashboard_panel_by_id("missing", dashboard)

        self.assertIsNone(missing)
        self.assertIsNotNone(prediction_panel)
        assert prediction_panel is not None
        self.assertEqual(prediction_panel.panel_kind, "performance_prediction")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "performance_dashboard::measurement_boundary",
            tuple(panel.panel_id for panel in guarded),
        )

    def test_dashboard_rejects_mismatched_panel_totals(self) -> None:
        dashboard = build_performance_dashboard()
        payload = dashboard.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            PerformanceDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["relative_performance_units_total"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "relative_performance_units_total must match",
        ):
            PerformanceDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["dashboard_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "dashboard_pressure must match",
        ):
            PerformanceDashboard(**payload)

    def test_dashboard_does_not_declare_runtime_performance_application_terms(
        self,
    ) -> None:
        dashboard = build_performance_dashboard()
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
            "measure_performance(",
            "run_benchmark(",
            "collect_timer(",
            "install_profiler(",
            "collect_trace(",
            "detect_regression(",
            "allocate_resource(",
            "enforce_capacity(",
            "route_provider(",
            "control_workflow(",
            "execute_workflow(",
            "invoke_agent(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
