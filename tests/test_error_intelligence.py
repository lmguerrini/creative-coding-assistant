import unittest

from creative_coding_assistant.orchestration import (
    ErrorIntelligence,
    build_error_intelligence,
    build_escalation_diagnostics,
    build_failure_analysis,
    build_production_telemetry,
    build_routing_diagnostics,
    build_workflow_diagnostics,
    error_intelligence_panel_by_id,
    error_intelligence_panels_for_status,
    langgraph_error_path_audit_registry,
)

REQUIRED_ERROR_INTELLIGENCE_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "error_signal_count",
    "guardrail_signal_count",
    "observed_error_count",
    "remediated_error_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "error_intelligence_panel_implemented",
    "runtime_error_capture_implemented",
    "live_error_classification_implemented",
    "exception_interception_implemented",
    "terminal_failure_routing_implemented",
    "automated_remediation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "alert_emission_implemented",
    "telemetry_emission_implemented",
    "human_review_request_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
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
    "failure_analysis",
    "langgraph_error_path_audit_registry",
    "workflow_diagnostics",
    "production_telemetry",
    "routing_diagnostics",
    "escalation_diagnostics",
)


class ErrorIntelligenceTests(unittest.TestCase):
    def test_default_intelligence_links_error_sources(self) -> None:
        failure = build_failure_analysis()
        langgraph = langgraph_error_path_audit_registry()
        workflow = build_workflow_diagnostics()
        telemetry = build_production_telemetry()
        routing = build_routing_diagnostics()
        escalation = build_escalation_diagnostics()
        intelligence = build_error_intelligence(
            failure_analysis=failure,
            langgraph_error_paths=langgraph,
            workflow_diagnostics=workflow,
            production_telemetry=telemetry,
            routing_diagnostics=routing,
            escalation_diagnostics=escalation,
        )

        self.assertEqual(intelligence.role, "error_intelligence")
        self.assertEqual(
            intelligence.serialization_version,
            "error_intelligence.v1",
        )
        self.assertEqual(
            intelligence.source_failure_analysis_serialization_version,
            failure.serialization_version,
        )
        self.assertEqual(
            intelligence.source_langgraph_error_path_serialization_version,
            langgraph.serialization_version,
        )
        self.assertEqual(
            intelligence.source_workflow_diagnostics_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            intelligence.source_production_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(
            intelligence.source_routing_diagnostics_serialization_version,
            routing.serialization_version,
        )
        self.assertEqual(
            intelligence.source_escalation_diagnostics_serialization_version,
            escalation.serialization_version,
        )
        self.assertEqual(intelligence.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(intelligence.panel_count, 6)
        self.assertEqual(
            intelligence.panel_ids,
            (
                "error_intelligence::error_path_taxonomy",
                "error_intelligence::failure_pattern_summary",
                "error_intelligence::workflow_error_context",
                "error_intelligence::telemetry_error_boundary",
                "error_intelligence::routing_error_boundary",
                "error_intelligence::escalation_error_boundary",
            ),
        )
        self.assertGreater(intelligence.error_signal_count, 0)
        self.assertGreater(intelligence.guardrail_signal_count, 0)
        self.assertIsNone(intelligence.observed_error_count)
        self.assertIsNone(intelligence.remediated_error_count)
        self.assertEqual(intelligence.error_intelligence_status, "guarded")
        self.assertIn(
            "does not capture runtime errors",
            intelligence.authority_boundary,
        )
        self.assertTrue(intelligence.error_intelligence_implemented)
        self.assertFalse(intelligence.runtime_error_capture_implemented)
        self.assertFalse(intelligence.live_error_classification_implemented)
        self.assertFalse(intelligence.exception_interception_implemented)
        self.assertFalse(intelligence.terminal_failure_routing_implemented)
        self.assertFalse(intelligence.automated_remediation_implemented)
        self.assertFalse(intelligence.retry_triggering_implemented)
        self.assertFalse(intelligence.refinement_triggering_implemented)
        self.assertFalse(intelligence.alert_emission_implemented)
        self.assertFalse(intelligence.telemetry_emission_implemented)
        self.assertFalse(intelligence.human_review_request_implemented)
        self.assertFalse(intelligence.workflow_execution_implemented)
        self.assertFalse(intelligence.workflow_control_implemented)
        self.assertFalse(intelligence.workflow_graph_mutation_implemented)
        self.assertFalse(intelligence.provider_model_routing_implemented)
        self.assertFalse(intelligence.provider_execution_implemented)
        self.assertFalse(intelligence.escalation_triggering_implemented)
        self.assertFalse(intelligence.agent_invocation_implemented)
        self.assertFalse(intelligence.memory_write_implemented)
        self.assertFalse(intelligence.persistent_storage_write_implemented)
        self.assertFalse(intelligence.generated_output_mutation_implemented)
        self.assertFalse(intelligence.runtime_evolution_implemented)
        self.assertTrue(intelligence.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        intelligence = build_error_intelligence()

        for panel in intelligence.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ERROR_INTELLIGENCE_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "error_intelligence_panel.v1",
            )
            self.assertIsNone(panel.observed_error_count)
            self.assertIsNone(panel.remediated_error_count)
            self.assertIn("runtime_error_capture", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.error_intelligence_panel_implemented)
            self.assertFalse(panel.runtime_error_capture_implemented)
            self.assertFalse(panel.live_error_classification_implemented)
            self.assertFalse(panel.exception_interception_implemented)
            self.assertFalse(panel.terminal_failure_routing_implemented)
            self.assertFalse(panel.automated_remediation_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.alert_emission_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_graph_mutation_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        telemetry = error_intelligence_panel_by_id(
            "error_intelligence::telemetry_error_boundary",
            intelligence,
        )
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.status, "guarded")
        self.assertEqual(
            telemetry.source_serialization_version,
            "production_telemetry.v1",
        )
        self.assertGreater(telemetry.error_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_classifying(self) -> None:
        intelligence = build_error_intelligence()
        escalation = error_intelligence_panel_by_id(
            "error_intelligence::escalation_error_boundary",
            intelligence,
        )
        guarded = error_intelligence_panels_for_status("guarded", intelligence)
        ready = error_intelligence_panels_for_status("ready", intelligence)
        missing = error_intelligence_panel_by_id("missing", intelligence)

        self.assertIsNone(missing)
        self.assertIsNotNone(escalation)
        assert escalation is not None
        self.assertEqual(escalation.panel_kind, "escalation_error_boundary")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), intelligence.panel_count)

    def test_intelligence_rejects_mismatched_panel_totals(self) -> None:
        intelligence = build_error_intelligence()
        payload = intelligence.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            ErrorIntelligence(**payload)

        payload = intelligence.model_dump(mode="json")
        payload["error_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "error_signal_count must match"):
            ErrorIntelligence(**payload)

        payload = intelligence.model_dump(mode="json")
        payload["error_intelligence_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "error_intelligence_status must match",
        ):
            ErrorIntelligence(**payload)

        payload = intelligence.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            ErrorIntelligence(**payload)

    def test_intelligence_does_not_declare_runtime_error_handling_terms(self) -> None:
        intelligence = build_error_intelligence()
        combined_text = " ".join(
            (
                intelligence.authority_boundary,
                *intelligence.blocked_runtime_behaviors,
                *intelligence.advisory_actions,
                *(
                    field
                    for panel in intelligence.panels
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
            "capture_error(",
            "classify_error(",
            "intercept_exception(",
            "route_terminal_failure(",
            "remediate_error(",
            "trigger_retry(",
            "emit_alert(",
            "emit_telemetry(",
            "request_human_review(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
