import unittest

from creative_coding_assistant.orchestration import (
    WorkflowDiagnostics,
    analyze_assistant_execution_graph,
    build_production_telemetry,
    build_workflow_diagnostics,
    performance_failure_path_audit_registry,
    plan_execution_replay,
    plan_workflow_replay,
    workflow_diagnostic_panel_by_id,
    workflow_diagnostic_panels_for_status,
)

REQUIRED_WORKFLOW_DIAGNOSTIC_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "diagnostic_signal_count",
    "guardrail_signal_count",
    "observed_runtime_event_count",
    "compiled_graph_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "workflow_diagnostic_panel_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_state_mutation_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "node_handler_invocation_implemented",
    "runtime_event_capture_implemented",
    "workflow_replay_execution_implemented",
    "execution_replay_execution_implemented",
    "telemetry_emission_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "execution_graph_analysis",
    "workflow_state_contract",
    "workflow_replay_plan",
    "execution_replay_plan",
    "performance_failure_path_audit_registry",
    "production_telemetry",
)


class WorkflowDiagnosticsTests(unittest.TestCase):
    def test_default_diagnostics_links_workflow_sources(self) -> None:
        graph = analyze_assistant_execution_graph()
        workflow = plan_workflow_replay(execution_graph=graph)
        execution = plan_execution_replay(workflow_replay=workflow)
        audit = performance_failure_path_audit_registry()
        telemetry = build_production_telemetry()
        diagnostics = build_workflow_diagnostics(
            execution_graph=graph,
            workflow_replay=workflow,
            execution_replay=execution,
            failure_audit=audit,
            production_telemetry=telemetry,
        )

        self.assertEqual(diagnostics.role, "workflow_diagnostics")
        self.assertEqual(
            diagnostics.serialization_version,
            "workflow_diagnostics.v1",
        )
        self.assertEqual(
            diagnostics.source_execution_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_workflow_state_serialization_version,
            "workflow_state_contract.v1",
        )
        self.assertEqual(
            diagnostics.source_workflow_replay_serialization_version,
            workflow.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_execution_replay_serialization_version,
            execution.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_failure_audit_serialization_version,
            audit.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_production_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(diagnostics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(diagnostics.panel_count, 6)
        self.assertEqual(
            diagnostics.panel_ids,
            (
                "workflow_diagnostics::graph_topology",
                "workflow_diagnostics::state_transition_contract",
                "workflow_diagnostics::workflow_replay",
                "workflow_diagnostics::execution_replay",
                "workflow_diagnostics::failure_audit",
                "workflow_diagnostics::telemetry_boundary",
            ),
        )
        self.assertGreater(diagnostics.diagnostic_signal_count, 0)
        self.assertGreater(diagnostics.guardrail_signal_count, 0)
        self.assertIsNone(diagnostics.observed_runtime_event_count)
        self.assertIsNone(diagnostics.compiled_graph_count)
        self.assertEqual(diagnostics.workflow_diagnostics_status, "guarded")
        self.assertIn("does not compile graphs", diagnostics.authority_boundary)
        self.assertTrue(diagnostics.workflow_diagnostics_implemented)
        self.assertFalse(diagnostics.graph_compilation_implemented)
        self.assertFalse(diagnostics.workflow_execution_implemented)
        self.assertFalse(diagnostics.workflow_control_implemented)
        self.assertFalse(diagnostics.workflow_state_mutation_implemented)
        self.assertFalse(diagnostics.workflow_graph_mutation_implemented)
        self.assertFalse(diagnostics.workflow_order_mutation_implemented)
        self.assertFalse(diagnostics.node_handler_invocation_implemented)
        self.assertFalse(diagnostics.runtime_event_capture_implemented)
        self.assertFalse(diagnostics.workflow_replay_execution_implemented)
        self.assertFalse(diagnostics.execution_replay_execution_implemented)
        self.assertFalse(diagnostics.telemetry_emission_implemented)
        self.assertFalse(diagnostics.provider_model_routing_implemented)
        self.assertFalse(diagnostics.retry_triggering_implemented)
        self.assertFalse(diagnostics.prompt_mutation_implemented)
        self.assertFalse(diagnostics.persistent_storage_write_implemented)
        self.assertFalse(diagnostics.generated_output_mutation_implemented)
        self.assertTrue(diagnostics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        diagnostics = build_workflow_diagnostics()

        for panel in diagnostics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_WORKFLOW_DIAGNOSTIC_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "workflow_diagnostic_panel.v1",
            )
            self.assertIsNone(panel.observed_runtime_event_count)
            self.assertIsNone(panel.compiled_graph_count)
            self.assertIn("workflow_execution", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.workflow_diagnostic_panel_implemented)
            self.assertFalse(panel.graph_compilation_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_state_mutation_implemented)
            self.assertFalse(panel.workflow_graph_mutation_implemented)
            self.assertFalse(panel.workflow_order_mutation_implemented)
            self.assertFalse(panel.node_handler_invocation_implemented)
            self.assertFalse(panel.runtime_event_capture_implemented)
            self.assertFalse(panel.workflow_replay_execution_implemented)
            self.assertFalse(panel.execution_replay_execution_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        state_panel = workflow_diagnostic_panel_by_id(
            "workflow_diagnostics::state_transition_contract",
            diagnostics,
        )
        self.assertIsNotNone(state_panel)
        assert state_panel is not None
        self.assertEqual(state_panel.status, "ready")
        self.assertEqual(
            state_panel.source_serialization_version,
            "workflow_state_contract.v1",
        )
        self.assertIn("running", state_panel.source_item_ids)
        self.assertIn("finalization", state_panel.source_item_ids)

    def test_lookup_helpers_are_stable_and_non_executing(self) -> None:
        diagnostics = build_workflow_diagnostics()
        replay_panel = workflow_diagnostic_panel_by_id(
            "workflow_diagnostics::workflow_replay",
            diagnostics,
        )
        guarded = workflow_diagnostic_panels_for_status("guarded", diagnostics)
        ready = workflow_diagnostic_panels_for_status("ready", diagnostics)
        missing = workflow_diagnostic_panel_by_id("missing", diagnostics)

        self.assertIsNone(missing)
        self.assertIsNotNone(replay_panel)
        assert replay_panel is not None
        self.assertEqual(replay_panel.panel_kind, "workflow_replay")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertGreaterEqual(len(ready), 1)
        self.assertIn(
            "workflow_diagnostics::telemetry_boundary",
            tuple(panel.panel_id for panel in guarded),
        )

    def test_diagnostics_rejects_mismatched_panel_totals(self) -> None:
        diagnostics = build_workflow_diagnostics()
        payload = diagnostics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            WorkflowDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["diagnostic_signal_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "diagnostic_signal_count must match",
        ):
            WorkflowDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["workflow_diagnostics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "workflow_diagnostics_status must match",
        ):
            WorkflowDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["source_surfaces"] = ("missing",) + tuple(
            payload["source_surfaces"][1:]
        )

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            WorkflowDiagnostics(**payload)

    def test_diagnostics_does_not_declare_runtime_workflow_application_terms(
        self,
    ) -> None:
        diagnostics = build_workflow_diagnostics()
        combined_text = " ".join(
            (
                diagnostics.authority_boundary,
                *diagnostics.blocked_runtime_behaviors,
                *diagnostics.advisory_actions,
                *(
                    field
                    for panel in diagnostics.panels
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
            "compile_graph(",
            "execute_workflow(",
            "control_workflow(",
            "mutate_workflow_state(",
            "invoke_node_handler(",
            "capture_runtime_event(",
            "replay_workflow(",
            "emit_telemetry(",
            "route_model(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
