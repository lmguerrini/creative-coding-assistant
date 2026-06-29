import unittest

from creative_coding_assistant.orchestration import (
    WorkflowHealthMonitoring,
    build_error_intelligence,
    build_failure_analysis,
    build_performance_dashboard,
    build_production_telemetry,
    build_workflow_diagnostics,
    build_workflow_health_monitoring,
    plan_retry_policies,
    workflow_health_panel_by_id,
    workflow_health_panels_for_status,
)

REQUIRED_WORKFLOW_HEALTH_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "health_signal_count",
    "guardrail_signal_count",
    "observed_runtime_event_count",
    "emitted_health_alert_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "workflow_health_panel_implemented",
    "live_workflow_monitoring_implemented",
    "runtime_event_capture_implemented",
    "health_check_execution_implemented",
    "alert_emission_implemented",
    "telemetry_emission_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_state_mutation_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_replay_execution_implemented",
    "execution_replay_execution_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "escalation_triggering_implemented",
    "agent_invocation_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "workflow_diagnostics",
    "production_telemetry",
    "error_intelligence",
    "failure_analysis",
    "performance_dashboard",
    "retry_policy_plan",
)


class WorkflowHealthMonitoringTests(unittest.TestCase):
    def test_default_monitoring_links_workflow_health_sources(self) -> None:
        workflow = build_workflow_diagnostics()
        telemetry = build_production_telemetry()
        error = build_error_intelligence()
        failure = build_failure_analysis()
        performance = build_performance_dashboard()
        retry = plan_retry_policies()
        monitoring = build_workflow_health_monitoring(
            workflow_diagnostics=workflow,
            production_telemetry=telemetry,
            error_intelligence=error,
            failure_analysis=failure,
            performance_dashboard=performance,
            retry_policy=retry,
        )

        self.assertEqual(monitoring.role, "workflow_health_monitoring")
        self.assertEqual(
            monitoring.serialization_version,
            "workflow_health_monitoring.v1",
        )
        self.assertEqual(
            monitoring.source_workflow_diagnostics_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            monitoring.source_production_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(
            monitoring.source_error_intelligence_serialization_version,
            error.serialization_version,
        )
        self.assertEqual(
            monitoring.source_failure_analysis_serialization_version,
            failure.serialization_version,
        )
        self.assertEqual(
            monitoring.source_performance_dashboard_serialization_version,
            performance.serialization_version,
        )
        self.assertEqual(
            monitoring.source_retry_policy_serialization_version,
            retry.serialization_version,
        )
        self.assertEqual(monitoring.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(monitoring.panel_count, 6)
        self.assertEqual(
            monitoring.panel_ids,
            (
                "workflow_health::diagnostic_health",
                "workflow_health::telemetry_health",
                "workflow_health::error_health",
                "workflow_health::failure_health",
                "workflow_health::performance_health",
                "workflow_health::retry_health",
            ),
        )
        self.assertGreater(monitoring.health_signal_count, 0)
        self.assertGreater(monitoring.guardrail_signal_count, 0)
        self.assertIsNone(monitoring.observed_runtime_event_count)
        self.assertIsNone(monitoring.emitted_health_alert_count)
        self.assertEqual(monitoring.workflow_health_status, "guarded")
        self.assertIn("does not monitor live workflows", monitoring.authority_boundary)
        self.assertTrue(monitoring.workflow_health_monitoring_implemented)
        self.assertFalse(monitoring.live_workflow_monitoring_implemented)
        self.assertFalse(monitoring.runtime_event_capture_implemented)
        self.assertFalse(monitoring.health_check_execution_implemented)
        self.assertFalse(monitoring.alert_emission_implemented)
        self.assertFalse(monitoring.telemetry_emission_implemented)
        self.assertFalse(monitoring.workflow_execution_implemented)
        self.assertFalse(monitoring.workflow_control_implemented)
        self.assertFalse(monitoring.workflow_state_mutation_implemented)
        self.assertFalse(monitoring.workflow_graph_mutation_implemented)
        self.assertFalse(monitoring.workflow_replay_execution_implemented)
        self.assertFalse(monitoring.execution_replay_execution_implemented)
        self.assertFalse(monitoring.node_handler_invocation_implemented)
        self.assertFalse(monitoring.provider_model_routing_implemented)
        self.assertFalse(monitoring.retry_triggering_implemented)
        self.assertFalse(monitoring.refinement_triggering_implemented)
        self.assertFalse(monitoring.escalation_triggering_implemented)
        self.assertFalse(monitoring.agent_invocation_implemented)
        self.assertFalse(monitoring.memory_write_implemented)
        self.assertFalse(monitoring.persistent_storage_write_implemented)
        self.assertFalse(monitoring.generated_output_mutation_implemented)
        self.assertFalse(monitoring.runtime_evolution_implemented)
        self.assertTrue(monitoring.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        monitoring = build_workflow_health_monitoring()

        for panel in monitoring.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_WORKFLOW_HEALTH_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "workflow_health_panel.v1")
            self.assertIsNone(panel.observed_runtime_event_count)
            self.assertIsNone(panel.emitted_health_alert_count)
            self.assertIn("live_workflow_monitoring", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.workflow_health_panel_implemented)
            self.assertFalse(panel.live_workflow_monitoring_implemented)
            self.assertFalse(panel.runtime_event_capture_implemented)
            self.assertFalse(panel.health_check_execution_implemented)
            self.assertFalse(panel.alert_emission_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_state_mutation_implemented)
            self.assertFalse(panel.workflow_graph_mutation_implemented)
            self.assertFalse(panel.workflow_replay_execution_implemented)
            self.assertFalse(panel.execution_replay_execution_implemented)
            self.assertFalse(panel.node_handler_invocation_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        performance = workflow_health_panel_by_id(
            "workflow_health::performance_health",
            monitoring,
        )
        self.assertIsNotNone(performance)
        assert performance is not None
        self.assertEqual(performance.status, "guarded")
        self.assertEqual(
            performance.source_serialization_version,
            "performance_dashboard.v1",
        )
        self.assertGreater(performance.health_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_monitoring(self) -> None:
        monitoring = build_workflow_health_monitoring()
        retry = workflow_health_panel_by_id("workflow_health::retry_health", monitoring)
        guarded = workflow_health_panels_for_status("guarded", monitoring)
        ready = workflow_health_panels_for_status("ready", monitoring)
        missing = workflow_health_panel_by_id("missing", monitoring)

        self.assertIsNone(missing)
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry.panel_kind, "retry_health")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), monitoring.panel_count)

    def test_monitoring_rejects_mismatched_panel_totals(self) -> None:
        monitoring = build_workflow_health_monitoring()
        payload = monitoring.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            WorkflowHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["health_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "health_signal_count must match"):
            WorkflowHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["workflow_health_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "workflow_health_status must match",
        ):
            WorkflowHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            WorkflowHealthMonitoring(**payload)

    def test_monitoring_does_not_declare_runtime_health_terms(self) -> None:
        monitoring = build_workflow_health_monitoring()
        combined_text = " ".join(
            (
                monitoring.authority_boundary,
                *monitoring.blocked_runtime_behaviors,
                *monitoring.advisory_actions,
                *(
                    field
                    for panel in monitoring.panels
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
            "monitor_workflow(",
            "capture_runtime_event(",
            "execute_health_check(",
            "emit_alert(",
            "emit_telemetry(",
            "control_workflow(",
            "mutate_workflow_state(",
            "invoke_node(",
            "trigger_retry(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
