import unittest

from creative_coding_assistant.orchestration import (
    FailureAnalysis,
    build_failure_analysis,
    build_workflow_diagnostics,
    execution_optimization_failure_audit_registry,
    failure_analysis_panel_by_id,
    failure_analysis_panels_for_status,
    langgraph_error_path_audit_registry,
    model_routing_failure_path_audit_registry,
    performance_failure_path_audit_registry,
    plan_retry_policies,
)

REQUIRED_FAILURE_ANALYSIS_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "failure_signal_count",
    "guardrail_signal_count",
    "observed_failure_count",
    "handled_failure_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "failure_analysis_panel_implemented",
    "runtime_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_replay_execution_implemented",
    "execution_replay_execution_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "telemetry_emission_implemented",
    "alert_emission_implemented",
    "human_review_request_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "langgraph_error_path_audit_registry",
    "execution_optimization_failure_audit_registry",
    "model_routing_failure_path_audit_registry",
    "performance_failure_path_audit_registry",
    "retry_policy_plan",
    "workflow_diagnostics",
)


class FailureAnalysisTests(unittest.TestCase):
    def test_default_analysis_links_failure_sources(self) -> None:
        langgraph = langgraph_error_path_audit_registry()
        execution = execution_optimization_failure_audit_registry()
        routing = model_routing_failure_path_audit_registry()
        performance = performance_failure_path_audit_registry()
        retry = plan_retry_policies()
        workflow = build_workflow_diagnostics()
        analysis = build_failure_analysis(
            langgraph_error_paths=langgraph,
            execution_failure_audit=execution,
            routing_failure_audit=routing,
            performance_failure_audit=performance,
            retry_policy=retry,
            workflow_diagnostics=workflow,
        )

        self.assertEqual(analysis.role, "failure_analysis")
        self.assertEqual(analysis.serialization_version, "failure_analysis.v1")
        self.assertEqual(
            analysis.source_langgraph_error_path_serialization_version,
            langgraph.serialization_version,
        )
        self.assertEqual(
            analysis.source_execution_failure_audit_serialization_version,
            execution.serialization_version,
        )
        self.assertEqual(
            analysis.source_routing_failure_audit_serialization_version,
            routing.serialization_version,
        )
        self.assertEqual(
            analysis.source_performance_failure_audit_serialization_version,
            performance.serialization_version,
        )
        self.assertEqual(
            analysis.source_retry_policy_serialization_version,
            retry.serialization_version,
        )
        self.assertEqual(
            analysis.source_workflow_diagnostics_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(analysis.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(analysis.panel_count, 6)
        self.assertEqual(
            analysis.panel_ids,
            (
                "failure_analysis::langgraph_error_paths",
                "failure_analysis::execution_failure_audit",
                "failure_analysis::routing_failure_audit",
                "failure_analysis::performance_failure_audit",
                "failure_analysis::retry_failure_boundaries",
                "failure_analysis::observability_failure_boundary",
            ),
        )
        self.assertGreater(analysis.failure_signal_count, 0)
        self.assertGreater(analysis.guardrail_signal_count, 0)
        self.assertIsNone(analysis.observed_failure_count)
        self.assertIsNone(analysis.handled_failure_count)
        self.assertEqual(analysis.failure_analysis_status, "guarded")
        self.assertIn("does not observe runtime failures", analysis.authority_boundary)
        self.assertTrue(analysis.failure_analysis_implemented)
        self.assertFalse(analysis.runtime_failure_observation_implemented)
        self.assertFalse(analysis.live_error_classification_implemented)
        self.assertFalse(analysis.terminal_failure_routing_implemented)
        self.assertFalse(analysis.failure_handling_implemented)
        self.assertFalse(analysis.retry_triggering_implemented)
        self.assertFalse(analysis.refinement_triggering_implemented)
        self.assertFalse(analysis.workflow_execution_implemented)
        self.assertFalse(analysis.workflow_control_implemented)
        self.assertFalse(analysis.workflow_graph_mutation_implemented)
        self.assertFalse(analysis.workflow_replay_execution_implemented)
        self.assertFalse(analysis.execution_replay_execution_implemented)
        self.assertFalse(analysis.provider_model_routing_implemented)
        self.assertFalse(analysis.provider_execution_implemented)
        self.assertFalse(analysis.telemetry_emission_implemented)
        self.assertFalse(analysis.alert_emission_implemented)
        self.assertFalse(analysis.human_review_request_implemented)
        self.assertFalse(analysis.memory_write_implemented)
        self.assertFalse(analysis.persistent_storage_write_implemented)
        self.assertFalse(analysis.generated_output_mutation_implemented)
        self.assertFalse(analysis.runtime_evolution_implemented)
        self.assertTrue(analysis.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        analysis = build_failure_analysis()

        for panel in analysis.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_FAILURE_ANALYSIS_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "failure_analysis_panel.v1")
            self.assertIsNone(panel.observed_failure_count)
            self.assertIsNone(panel.handled_failure_count)
            self.assertIn("runtime_failure_observation", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.failure_analysis_panel_implemented)
            self.assertFalse(panel.runtime_failure_observation_implemented)
            self.assertFalse(panel.live_error_classification_implemented)
            self.assertFalse(panel.terminal_failure_routing_implemented)
            self.assertFalse(panel.failure_handling_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_graph_mutation_implemented)
            self.assertFalse(panel.workflow_replay_execution_implemented)
            self.assertFalse(panel.execution_replay_execution_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.alert_emission_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        routing = failure_analysis_panel_by_id(
            "failure_analysis::routing_failure_audit",
            analysis,
        )
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.status, "guarded")
        self.assertEqual(
            routing.source_serialization_version,
            "model_routing_failure_path_audit_registry.v1",
        )
        self.assertGreater(routing.failure_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_handling(self) -> None:
        analysis = build_failure_analysis()
        retry = failure_analysis_panel_by_id(
            "failure_analysis::retry_failure_boundaries",
            analysis,
        )
        guarded = failure_analysis_panels_for_status("guarded", analysis)
        ready = failure_analysis_panels_for_status("ready", analysis)
        missing = failure_analysis_panel_by_id("missing", analysis)

        self.assertIsNone(missing)
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry.panel_kind, "retry_failure_boundaries")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), analysis.panel_count)

    def test_analysis_rejects_mismatched_panel_totals(self) -> None:
        analysis = build_failure_analysis()
        payload = analysis.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            FailureAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["failure_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "failure_signal_count must match"):
            FailureAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["failure_analysis_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "failure_analysis_status must match",
        ):
            FailureAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            FailureAnalysis(**payload)

    def test_analysis_does_not_declare_runtime_failure_handling_terms(self) -> None:
        analysis = build_failure_analysis()
        combined_text = " ".join(
            (
                analysis.authority_boundary,
                *analysis.blocked_runtime_behaviors,
                *analysis.advisory_actions,
                *(
                    field
                    for panel in analysis.panels
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
            "observe_failure(",
            "classify_error(",
            "route_terminal_failure(",
            "handle_failure(",
            "repair_failure(",
            "trigger_retry(",
            "trigger_refinement(",
            "execute_replay(",
            "emit_alert(",
            "request_human_review(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
