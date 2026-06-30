import unittest

from creative_coding_assistant.orchestration import (
    WorkflowExplainabilityDashboard,
    agent_explainability_audit_registry,
    build_error_intelligence,
    build_runtime_timeline,
    build_workflow_diagnostics,
    build_workflow_explainability_dashboard,
    decision_provenance_registry,
    explain_routing_decision,
    workflow_explainability_panel_by_id,
    workflow_explainability_panels_for_status,
)

REQUIRED_WORKFLOW_EXPLAINABILITY_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "explainability_signal_count",
    "guardrail_signal_count",
    "generated_explanation_count",
    "recorded_provenance_count",
    "captured_trace_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "workflow_explainability_panel_implemented",
    "explanation_generation_implemented",
    "live_reasoning_generation_implemented",
    "decision_provenance_recording_implemented",
    "decision_logging_implemented",
    "trace_capture_implemented",
    "trace_emission_implemented",
    "runtime_event_capture_implemented",
    "timeline_reconstruction_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_state_mutation_implemented",
    "live_error_classification_implemented",
    "automated_remediation_implemented",
    "human_review_request_implemented",
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
    "workflow_diagnostics",
    "routing_explainability",
    "agent_explainability_audit_registry",
    "decision_provenance_registry",
    "runtime_timeline",
    "error_intelligence",
)


class WorkflowExplainabilityDashboardTests(unittest.TestCase):
    def test_default_dashboard_links_explainability_sources(self) -> None:
        workflow = build_workflow_diagnostics()
        routing = explain_routing_decision()
        agent = agent_explainability_audit_registry()
        provenance = decision_provenance_registry()
        timeline = build_runtime_timeline(workflow_diagnostics=workflow)
        error = build_error_intelligence(workflow_diagnostics=workflow)
        dashboard = build_workflow_explainability_dashboard(
            workflow_diagnostics=workflow,
            routing_explainability=routing,
            agent_explainability=agent,
            decision_provenance=provenance,
            runtime_timeline=timeline,
            error_intelligence=error,
        )

        self.assertEqual(dashboard.role, "workflow_explainability_dashboard")
        self.assertEqual(
            dashboard.serialization_version,
            "workflow_explainability_dashboard.v1",
        )
        self.assertEqual(
            dashboard.source_workflow_diagnostics_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            dashboard.source_routing_explainability_serialization_version,
            routing.serialization_version,
        )
        self.assertEqual(
            dashboard.source_agent_explainability_serialization_version,
            agent.serialization_version,
        )
        self.assertEqual(
            dashboard.source_decision_provenance_serialization_version,
            provenance.serialization_version,
        )
        self.assertEqual(
            dashboard.source_runtime_timeline_serialization_version,
            timeline.serialization_version,
        )
        self.assertEqual(
            dashboard.source_error_intelligence_serialization_version,
            error.serialization_version,
        )
        self.assertEqual(dashboard.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(dashboard.panel_count, 6)
        self.assertEqual(
            dashboard.panel_ids,
            (
                "workflow_explainability::workflow_reasoning_context",
                "workflow_explainability::routing_explanation_context",
                "workflow_explainability::agent_explainability_audit",
                "workflow_explainability::decision_provenance_context",
                "workflow_explainability::runtime_timeline_context",
                "workflow_explainability::error_explainability_context",
            ),
        )
        self.assertGreater(dashboard.explainability_signal_count, 0)
        self.assertGreater(dashboard.guardrail_signal_count, 0)
        self.assertIsNone(dashboard.generated_explanation_count)
        self.assertIsNone(dashboard.recorded_provenance_count)
        self.assertIsNone(dashboard.captured_trace_count)
        self.assertEqual(dashboard.workflow_explainability_status, "guarded")
        self.assertIn("does not generate live explanations", dashboard.authority_boundary)
        self.assertTrue(dashboard.workflow_explainability_dashboard_implemented)
        self.assertFalse(dashboard.explanation_generation_implemented)
        self.assertFalse(dashboard.live_reasoning_generation_implemented)
        self.assertFalse(dashboard.decision_provenance_recording_implemented)
        self.assertFalse(dashboard.decision_logging_implemented)
        self.assertFalse(dashboard.trace_capture_implemented)
        self.assertFalse(dashboard.trace_emission_implemented)
        self.assertFalse(dashboard.runtime_event_capture_implemented)
        self.assertFalse(dashboard.timeline_reconstruction_implemented)
        self.assertFalse(dashboard.routing_application_implemented)
        self.assertFalse(dashboard.provider_model_routing_implemented)
        self.assertFalse(dashboard.provider_execution_implemented)
        self.assertFalse(dashboard.agent_invocation_implemented)
        self.assertFalse(dashboard.workflow_execution_implemented)
        self.assertFalse(dashboard.workflow_control_implemented)
        self.assertFalse(dashboard.workflow_state_mutation_implemented)
        self.assertFalse(dashboard.live_error_classification_implemented)
        self.assertFalse(dashboard.automated_remediation_implemented)
        self.assertFalse(dashboard.human_review_request_implemented)
        self.assertFalse(dashboard.retry_triggering_implemented)
        self.assertFalse(dashboard.refinement_triggering_implemented)
        self.assertFalse(dashboard.memory_write_implemented)
        self.assertFalse(dashboard.persistent_storage_write_implemented)
        self.assertFalse(dashboard.generated_output_mutation_implemented)
        self.assertFalse(dashboard.runtime_evolution_implemented)
        self.assertTrue(dashboard.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        dashboard = build_workflow_explainability_dashboard()

        for panel in dashboard.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(
                set(dumped),
                REQUIRED_WORKFLOW_EXPLAINABILITY_PANEL_FIELDS,
            )
            self.assertEqual(
                panel.serialization_version,
                "workflow_explainability_panel.v1",
            )
            self.assertIsNone(panel.generated_explanation_count)
            self.assertIsNone(panel.recorded_provenance_count)
            self.assertIsNone(panel.captured_trace_count)
            self.assertIn("explanation_generation", panel.blocked_runtime_behaviors)
            self.assertIn("decision_provenance_recording", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.workflow_explainability_panel_implemented)
            self.assertFalse(panel.explanation_generation_implemented)
            self.assertFalse(panel.live_reasoning_generation_implemented)
            self.assertFalse(panel.decision_provenance_recording_implemented)
            self.assertFalse(panel.decision_logging_implemented)
            self.assertFalse(panel.trace_capture_implemented)
            self.assertFalse(panel.trace_emission_implemented)
            self.assertFalse(panel.runtime_event_capture_implemented)
            self.assertFalse(panel.timeline_reconstruction_implemented)
            self.assertFalse(panel.routing_application_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_state_mutation_implemented)
            self.assertFalse(panel.live_error_classification_implemented)
            self.assertFalse(panel.automated_remediation_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        provenance = workflow_explainability_panel_by_id(
            "workflow_explainability::decision_provenance_context",
            dashboard,
        )
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(provenance.status, "guarded")
        self.assertEqual(
            provenance.source_serialization_version,
            "decision_provenance_registry.v1",
        )
        self.assertGreater(provenance.explainability_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_generating(self) -> None:
        dashboard = build_workflow_explainability_dashboard()
        routing = workflow_explainability_panel_by_id(
            "workflow_explainability::routing_explanation_context",
            dashboard,
        )
        guarded = workflow_explainability_panels_for_status("guarded", dashboard)
        ready = workflow_explainability_panels_for_status("ready", dashboard)
        missing = workflow_explainability_panel_by_id("missing", dashboard)

        self.assertIsNone(missing)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.panel_kind, "routing_explanation_context")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), dashboard.panel_count)

    def test_dashboard_rejects_mismatched_panel_totals(self) -> None:
        dashboard = build_workflow_explainability_dashboard()
        payload = dashboard.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            WorkflowExplainabilityDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["explainability_signal_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "explainability_signal_count must match",
        ):
            WorkflowExplainabilityDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["workflow_explainability_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "workflow_explainability_status must match",
        ):
            WorkflowExplainabilityDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            WorkflowExplainabilityDashboard(**payload)

    def test_dashboard_does_not_declare_runtime_explanation_terms(self) -> None:
        dashboard = build_workflow_explainability_dashboard()
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
            "generate_explanation(",
            "record_provenance(",
            "log_decision(",
            "capture_trace(",
            "apply_routing(",
            "invoke_agent(",
            "execute_workflow(",
            "classify_live_error(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
