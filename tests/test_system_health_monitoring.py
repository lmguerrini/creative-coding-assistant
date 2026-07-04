import unittest

from creative_coding_assistant.orchestration import (
    SystemHealthMonitoring,
    build_agent_diagnostics,
    build_cost_dashboard,
    build_error_intelligence,
    build_performance_dashboard,
    build_production_telemetry,
    build_quality_dashboard,
    build_system_health_monitoring,
    build_token_dashboard,
    build_workflow_health_monitoring,
    system_health_panel_by_id,
    system_health_panels_for_status,
)

REQUIRED_SYSTEM_HEALTH_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "system_signal_count",
    "guardrail_signal_count",
    "observed_system_event_count",
    "emitted_system_alert_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "system_health_panel_implemented",
    "live_system_monitoring_implemented",
    "runtime_metric_collection_implemented",
    "health_check_execution_implemented",
    "alert_emission_implemented",
    "telemetry_emission_implemented",
    "resource_allocation_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "quality_evaluation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
    "escalation_triggering_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "workflow_health_monitoring",
    "production_telemetry",
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
    "error_intelligence",
    "agent_diagnostics",
)


class SystemHealthMonitoringTests(unittest.TestCase):
    def test_default_monitoring_links_system_health_sources(self) -> None:
        workflow = build_workflow_health_monitoring()
        telemetry = build_production_telemetry()
        token = build_token_dashboard()
        cost = build_cost_dashboard()
        quality = build_quality_dashboard()
        performance = build_performance_dashboard()
        error = build_error_intelligence()
        agent = build_agent_diagnostics()
        monitoring = build_system_health_monitoring(
            workflow_health=workflow,
            production_telemetry=telemetry,
            token_dashboard=token,
            cost_dashboard=cost,
            quality_dashboard=quality,
            performance_dashboard=performance,
            error_intelligence=error,
            agent_diagnostics=agent,
        )

        self.assertEqual(monitoring.role, "system_health_monitoring")
        self.assertEqual(
            monitoring.serialization_version,
            "system_health_monitoring.v1",
        )
        self.assertEqual(
            monitoring.source_workflow_health_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            monitoring.source_production_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(
            monitoring.source_token_dashboard_serialization_version,
            token.serialization_version,
        )
        self.assertEqual(
            monitoring.source_cost_dashboard_serialization_version,
            cost.serialization_version,
        )
        self.assertEqual(
            monitoring.source_quality_dashboard_serialization_version,
            quality.serialization_version,
        )
        self.assertEqual(
            monitoring.source_performance_dashboard_serialization_version,
            performance.serialization_version,
        )
        self.assertEqual(
            monitoring.source_error_intelligence_serialization_version,
            error.serialization_version,
        )
        self.assertEqual(
            monitoring.source_agent_diagnostics_serialization_version,
            agent.serialization_version,
        )
        self.assertEqual(monitoring.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(monitoring.panel_count, 8)
        self.assertEqual(
            monitoring.panel_ids,
            (
                "system_health::workflow_system_health",
                "system_health::telemetry_system_health",
                "system_health::token_system_health",
                "system_health::cost_system_health",
                "system_health::quality_system_health",
                "system_health::performance_system_health",
                "system_health::error_system_health",
                "system_health::agent_system_health",
            ),
        )
        self.assertGreater(monitoring.system_signal_count, 0)
        self.assertGreater(monitoring.guardrail_signal_count, 0)
        self.assertIsNone(monitoring.observed_system_event_count)
        self.assertIsNone(monitoring.emitted_system_alert_count)
        self.assertEqual(monitoring.system_health_status, "guarded")
        self.assertIn("does not monitor live systems", monitoring.authority_boundary)
        self.assertTrue(monitoring.system_health_monitoring_implemented)
        self.assertFalse(monitoring.live_system_monitoring_implemented)
        self.assertFalse(monitoring.runtime_metric_collection_implemented)
        self.assertFalse(monitoring.health_check_execution_implemented)
        self.assertFalse(monitoring.alert_emission_implemented)
        self.assertFalse(monitoring.telemetry_emission_implemented)
        self.assertFalse(monitoring.resource_allocation_implemented)
        self.assertFalse(monitoring.capacity_enforcement_implemented)
        self.assertFalse(monitoring.budget_enforcement_implemented)
        self.assertFalse(monitoring.quality_evaluation_implemented)
        self.assertFalse(monitoring.workflow_execution_implemented)
        self.assertFalse(monitoring.workflow_control_implemented)
        self.assertFalse(monitoring.provider_model_routing_implemented)
        self.assertFalse(monitoring.agent_invocation_implemented)
        self.assertFalse(monitoring.escalation_triggering_implemented)
        self.assertFalse(monitoring.retry_triggering_implemented)
        self.assertFalse(monitoring.refinement_triggering_implemented)
        self.assertFalse(monitoring.memory_write_implemented)
        self.assertFalse(monitoring.persistent_storage_write_implemented)
        self.assertFalse(monitoring.generated_output_mutation_implemented)
        self.assertFalse(monitoring.runtime_evolution_implemented)
        self.assertTrue(monitoring.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        monitoring = build_system_health_monitoring()

        for panel in monitoring.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SYSTEM_HEALTH_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "system_health_panel.v1")
            self.assertIsNone(panel.observed_system_event_count)
            self.assertIsNone(panel.emitted_system_alert_count)
            self.assertIn("live_system_monitoring", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.system_health_panel_implemented)
            self.assertFalse(panel.live_system_monitoring_implemented)
            self.assertFalse(panel.runtime_metric_collection_implemented)
            self.assertFalse(panel.health_check_execution_implemented)
            self.assertFalse(panel.alert_emission_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.resource_allocation_implemented)
            self.assertFalse(panel.capacity_enforcement_implemented)
            self.assertFalse(panel.budget_enforcement_implemented)
            self.assertFalse(panel.quality_evaluation_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        token = system_health_panel_by_id(
            "system_health::token_system_health",
            monitoring,
        )
        self.assertIsNotNone(token)
        assert token is not None
        self.assertEqual(token.status, "guarded")
        self.assertEqual(token.source_serialization_version, "token_dashboard.v1")
        self.assertGreater(token.system_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_monitoring(self) -> None:
        monitoring = build_system_health_monitoring()
        agent = system_health_panel_by_id(
            "system_health::agent_system_health", monitoring
        )
        guarded = system_health_panels_for_status("guarded", monitoring)
        ready = system_health_panels_for_status("ready", monitoring)
        missing = system_health_panel_by_id("missing", monitoring)

        self.assertIsNone(missing)
        self.assertIsNotNone(agent)
        assert agent is not None
        self.assertEqual(agent.panel_kind, "agent_system_health")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), monitoring.panel_count)

    def test_monitoring_rejects_mismatched_panel_totals(self) -> None:
        monitoring = build_system_health_monitoring()
        payload = monitoring.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            SystemHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["system_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "system_signal_count must match"):
            SystemHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["system_health_status"] = "ready"

        with self.assertRaisesRegex(ValueError, "system_health_status must match"):
            SystemHealthMonitoring(**payload)

        payload = monitoring.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            SystemHealthMonitoring(**payload)

    def test_monitoring_does_not_declare_runtime_system_health_terms(self) -> None:
        monitoring = build_system_health_monitoring()
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
            "monitor_system(",
            "collect_runtime_metrics(",
            "execute_health_check(",
            "emit_alert(",
            "emit_telemetry(",
            "allocate_resources(",
            "enforce_capacity(",
            "enforce_budget(",
            "invoke_agent(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
