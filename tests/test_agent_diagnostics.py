import unittest

from creative_coding_assistant.orchestration import (
    AgentDiagnostics,
    agent_capability_alignment_registry,
    agent_determinism_audit_registry,
    agent_diagnostic_panel_by_id,
    agent_diagnostic_panels_for_status,
    agent_explainability_audit_registry,
    agent_lifecycle_registry,
    agent_metadata_registry,
    agent_reliability_audit_registry,
    agent_telemetry_foundation_registry,
    build_agent_diagnostics,
)

REQUIRED_AGENT_DIAGNOSTIC_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "agent_signal_count",
    "guardrail_signal_count",
    "observed_agent_run_count",
    "runtime_failure_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "agent_diagnostic_panel_implemented",
    "active_agent_execution_implemented",
    "agent_capability_execution_implemented",
    "runtime_lifecycle_engine_implemented",
    "state_transition_execution_implemented",
    "runtime_state_synchronization_implemented",
    "explanation_generation_implemented",
    "trace_capture_implemented",
    "telemetry_emission_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "agent_metadata_registry",
    "agent_lifecycle_registry",
    "agent_capability_alignment_registry",
    "agent_telemetry_foundation_registry",
    "agent_reliability_audit_registry",
    "agent_determinism_audit_registry",
    "agent_explainability_audit_registry",
)


class AgentDiagnosticsTests(unittest.TestCase):
    def test_default_diagnostics_links_agent_sources(self) -> None:
        metadata = agent_metadata_registry()
        lifecycle = agent_lifecycle_registry()
        alignment = agent_capability_alignment_registry()
        telemetry = agent_telemetry_foundation_registry()
        reliability = agent_reliability_audit_registry()
        determinism = agent_determinism_audit_registry()
        explainability = agent_explainability_audit_registry()
        diagnostics = build_agent_diagnostics(
            agent_metadata=metadata,
            agent_lifecycle=lifecycle,
            capability_alignment=alignment,
            agent_telemetry=telemetry,
            reliability_audit=reliability,
            determinism_audit=determinism,
            explainability_audit=explainability,
        )

        self.assertEqual(diagnostics.role, "agent_diagnostics")
        self.assertEqual(diagnostics.serialization_version, "agent_diagnostics.v1")
        self.assertEqual(
            diagnostics.source_agent_metadata_serialization_version,
            metadata.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_agent_lifecycle_serialization_version,
            lifecycle.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_capability_alignment_serialization_version,
            alignment.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_agent_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_reliability_audit_serialization_version,
            reliability.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_determinism_audit_serialization_version,
            determinism.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_explainability_audit_serialization_version,
            explainability.serialization_version,
        )
        self.assertEqual(diagnostics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(diagnostics.panel_count, 7)
        self.assertEqual(
            diagnostics.panel_ids,
            (
                "agent_diagnostics::metadata_coverage",
                "agent_diagnostics::lifecycle_coverage",
                "agent_diagnostics::capability_alignment",
                "agent_diagnostics::telemetry_coverage",
                "agent_diagnostics::reliability_audit",
                "agent_diagnostics::determinism_audit",
                "agent_diagnostics::explainability_audit",
            ),
        )
        self.assertGreater(diagnostics.agent_signal_count, 0)
        self.assertGreater(diagnostics.guardrail_signal_count, 0)
        self.assertIsNone(diagnostics.observed_agent_run_count)
        self.assertIsNone(diagnostics.runtime_failure_count)
        self.assertEqual(diagnostics.agent_diagnostics_status, "guarded")
        self.assertIn("does not invoke agents", diagnostics.authority_boundary)
        self.assertTrue(diagnostics.agent_diagnostics_implemented)
        self.assertFalse(diagnostics.active_agent_execution_implemented)
        self.assertFalse(diagnostics.agent_capability_execution_implemented)
        self.assertFalse(diagnostics.runtime_lifecycle_engine_implemented)
        self.assertFalse(diagnostics.state_transition_execution_implemented)
        self.assertFalse(diagnostics.runtime_state_synchronization_implemented)
        self.assertFalse(diagnostics.explanation_generation_implemented)
        self.assertFalse(diagnostics.trace_capture_implemented)
        self.assertFalse(diagnostics.telemetry_emission_implemented)
        self.assertFalse(diagnostics.runtime_selection_implemented)
        self.assertFalse(diagnostics.provider_model_routing_implemented)
        self.assertFalse(diagnostics.retry_triggering_implemented)
        self.assertFalse(diagnostics.prompt_mutation_implemented)
        self.assertFalse(diagnostics.persistent_storage_write_implemented)
        self.assertFalse(diagnostics.generated_output_mutation_implemented)
        self.assertTrue(diagnostics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        diagnostics = build_agent_diagnostics()

        for panel in diagnostics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AGENT_DIAGNOSTIC_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "agent_diagnostic_panel.v1")
            self.assertIsNone(panel.observed_agent_run_count)
            self.assertIsNone(panel.runtime_failure_count)
            self.assertIn("agent_invocation", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.agent_diagnostic_panel_implemented)
            self.assertFalse(panel.active_agent_execution_implemented)
            self.assertFalse(panel.agent_capability_execution_implemented)
            self.assertFalse(panel.runtime_lifecycle_engine_implemented)
            self.assertFalse(panel.state_transition_execution_implemented)
            self.assertFalse(panel.runtime_state_synchronization_implemented)
            self.assertFalse(panel.explanation_generation_implemented)
            self.assertFalse(panel.trace_capture_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.runtime_selection_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        telemetry = agent_diagnostic_panel_by_id(
            "agent_diagnostics::telemetry_coverage",
            diagnostics,
        )
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.status, "guarded")
        self.assertGreater(telemetry.agent_signal_count, 0)
        self.assertGreater(telemetry.guardrail_signal_count, 0)
        self.assertEqual(
            telemetry.source_serialization_version,
            "agent_telemetry_foundation_registry.v1",
        )

    def test_lookup_helpers_are_stable_and_non_executing(self) -> None:
        diagnostics = build_agent_diagnostics()
        reliability = agent_diagnostic_panel_by_id(
            "agent_diagnostics::reliability_audit",
            diagnostics,
        )
        guarded = agent_diagnostic_panels_for_status("guarded", diagnostics)
        ready = agent_diagnostic_panels_for_status("ready", diagnostics)
        missing = agent_diagnostic_panel_by_id("missing", diagnostics)

        self.assertIsNone(missing)
        self.assertIsNotNone(reliability)
        assert reliability is not None
        self.assertEqual(reliability.panel_kind, "reliability_audit")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), diagnostics.panel_count)

    def test_diagnostics_rejects_mismatched_panel_totals(self) -> None:
        diagnostics = build_agent_diagnostics()
        payload = diagnostics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            AgentDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["agent_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "agent_signal_count must match"):
            AgentDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["agent_diagnostics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "agent_diagnostics_status must match",
        ):
            AgentDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["source_surfaces"] = ("missing",) + tuple(
            payload["source_surfaces"][1:]
        )

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            AgentDiagnostics(**payload)

    def test_diagnostics_does_not_declare_runtime_agent_application_terms(
        self,
    ) -> None:
        diagnostics = build_agent_diagnostics()
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
            "invoke_agent(",
            "execute_agent_capability(",
            "run_lifecycle_transition(",
            "sync_runtime_state(",
            "generate_explanation(",
            "capture_trace(",
            "emit_telemetry(",
            "select_runtime(",
            "route_model(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
