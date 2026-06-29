import unittest

from creative_coding_assistant.orchestration import (
    EscalationDiagnostics,
    adaptive_multi_agent_escalation_registry,
    agent_escalation_signal_registry,
    build_escalation_diagnostics,
    escalation_diagnostic_panel_by_id,
    escalation_diagnostic_panels_for_status,
    escalation_gate_registry,
    escalation_policy_audit_registry,
    escalation_policy_registry,
    escalation_trace_registry,
    hitl_escalation_gate_registry,
)

REQUIRED_ESCALATION_DIAGNOSTIC_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "escalation_signal_count",
    "guardrail_signal_count",
    "triggered_escalation_count",
    "hitl_request_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "escalation_diagnostic_panel_implemented",
    "policy_evaluation_implemented",
    "escalation_triggering_implemented",
    "escalation_execution_implemented",
    "gate_evaluation_implemented",
    "escalation_approval_implemented",
    "human_review_request_implemented",
    "hitl_triggering_implemented",
    "trace_capture_implemented",
    "trace_emission_implemented",
    "adaptation_evaluation_implemented",
    "multi_agent_orchestration_implemented",
    "agent_invocation_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "escalation_policy_audit_registry",
    "escalation_gate_registry",
    "escalation_trace_registry",
    "hitl_escalation_gate_registry",
    "adaptive_multi_agent_escalation_registry",
)


class EscalationDiagnosticsTests(unittest.TestCase):
    def test_default_diagnostics_links_escalation_sources(self) -> None:
        policy = escalation_policy_registry()
        signals = agent_escalation_signal_registry()
        policy_audit = escalation_policy_audit_registry()
        gates = escalation_gate_registry()
        traces = escalation_trace_registry()
        hitl_gates = hitl_escalation_gate_registry()
        adaptive = adaptive_multi_agent_escalation_registry()
        diagnostics = build_escalation_diagnostics(
            policy=policy,
            signals=signals,
            policy_audit=policy_audit,
            gates=gates,
            traces=traces,
            hitl_gates=hitl_gates,
            adaptive=adaptive,
        )

        self.assertEqual(diagnostics.role, "escalation_diagnostics")
        self.assertEqual(
            diagnostics.serialization_version,
            "escalation_diagnostics.v1",
        )
        self.assertEqual(
            diagnostics.source_policy_serialization_version,
            policy.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_signal_serialization_version,
            signals.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_policy_audit_serialization_version,
            policy_audit.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_gate_serialization_version,
            gates.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_trace_serialization_version,
            traces.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_hitl_gate_serialization_version,
            hitl_gates.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_adaptive_serialization_version,
            adaptive.serialization_version,
        )
        self.assertEqual(diagnostics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(diagnostics.panel_count, 7)
        self.assertEqual(
            diagnostics.panel_ids,
            (
                "escalation_diagnostics::policy_rules",
                "escalation_diagnostics::signal_thresholds",
                "escalation_diagnostics::policy_audit",
                "escalation_diagnostics::escalation_gates",
                "escalation_diagnostics::trace_contexts",
                "escalation_diagnostics::hitl_gates",
                "escalation_diagnostics::adaptive_boundary",
            ),
        )
        self.assertGreater(diagnostics.escalation_signal_count, 0)
        self.assertGreater(diagnostics.guardrail_signal_count, 0)
        self.assertIsNone(diagnostics.triggered_escalation_count)
        self.assertIsNone(diagnostics.hitl_request_count)
        self.assertEqual(diagnostics.escalation_diagnostics_status, "guarded")
        self.assertIn("does not evaluate policies", diagnostics.authority_boundary)
        self.assertTrue(diagnostics.escalation_diagnostics_implemented)
        self.assertFalse(diagnostics.policy_evaluation_implemented)
        self.assertFalse(diagnostics.escalation_triggering_implemented)
        self.assertFalse(diagnostics.escalation_execution_implemented)
        self.assertFalse(diagnostics.gate_evaluation_implemented)
        self.assertFalse(diagnostics.escalation_approval_implemented)
        self.assertFalse(diagnostics.human_review_request_implemented)
        self.assertFalse(diagnostics.hitl_triggering_implemented)
        self.assertFalse(diagnostics.trace_capture_implemented)
        self.assertFalse(diagnostics.trace_emission_implemented)
        self.assertFalse(diagnostics.adaptation_evaluation_implemented)
        self.assertFalse(diagnostics.multi_agent_orchestration_implemented)
        self.assertFalse(diagnostics.agent_invocation_implemented)
        self.assertFalse(diagnostics.runtime_selection_implemented)
        self.assertFalse(diagnostics.provider_model_routing_implemented)
        self.assertFalse(diagnostics.workflow_control_implemented)
        self.assertFalse(diagnostics.retry_triggering_implemented)
        self.assertFalse(diagnostics.memory_write_implemented)
        self.assertFalse(diagnostics.persistent_storage_write_implemented)
        self.assertFalse(diagnostics.generated_output_mutation_implemented)
        self.assertTrue(diagnostics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        diagnostics = build_escalation_diagnostics()

        for panel in diagnostics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(
                set(dumped),
                REQUIRED_ESCALATION_DIAGNOSTIC_PANEL_FIELDS,
            )
            self.assertEqual(
                panel.serialization_version,
                "escalation_diagnostic_panel.v1",
            )
            self.assertIsNone(panel.triggered_escalation_count)
            self.assertIsNone(panel.hitl_request_count)
            self.assertIn("policy_evaluation", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.escalation_diagnostic_panel_implemented)
            self.assertFalse(panel.policy_evaluation_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.escalation_execution_implemented)
            self.assertFalse(panel.gate_evaluation_implemented)
            self.assertFalse(panel.escalation_approval_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.hitl_triggering_implemented)
            self.assertFalse(panel.trace_capture_implemented)
            self.assertFalse(panel.trace_emission_implemented)
            self.assertFalse(panel.adaptation_evaluation_implemented)
            self.assertFalse(panel.multi_agent_orchestration_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.runtime_selection_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        hitl_panel = escalation_diagnostic_panel_by_id(
            "escalation_diagnostics::hitl_gates",
            diagnostics,
        )
        self.assertIsNotNone(hitl_panel)
        assert hitl_panel is not None
        self.assertEqual(hitl_panel.status, "guarded")
        self.assertEqual(
            hitl_panel.source_serialization_version,
            "hitl_escalation_gate_registry.v1",
        )
        self.assertGreater(hitl_panel.escalation_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_escalating(self) -> None:
        diagnostics = build_escalation_diagnostics()
        adaptive = escalation_diagnostic_panel_by_id(
            "escalation_diagnostics::adaptive_boundary",
            diagnostics,
        )
        guarded = escalation_diagnostic_panels_for_status("guarded", diagnostics)
        ready = escalation_diagnostic_panels_for_status("ready", diagnostics)
        missing = escalation_diagnostic_panel_by_id("missing", diagnostics)

        self.assertIsNone(missing)
        self.assertIsNotNone(adaptive)
        assert adaptive is not None
        self.assertEqual(adaptive.panel_kind, "adaptive_boundary")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), diagnostics.panel_count)

    def test_diagnostics_rejects_mismatched_panel_totals(self) -> None:
        diagnostics = build_escalation_diagnostics()
        payload = diagnostics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            EscalationDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["escalation_signal_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "escalation_signal_count must match",
        ):
            EscalationDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["escalation_diagnostics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "escalation_diagnostics_status must match",
        ):
            EscalationDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            EscalationDiagnostics(**payload)

    def test_diagnostics_does_not_declare_runtime_escalation_terms(self) -> None:
        diagnostics = build_escalation_diagnostics()
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
            "evaluate_policy(",
            "trigger_escalation(",
            "execute_escalation(",
            "approve_gate(",
            "request_human_review(",
            "capture_trace(",
            "emit_trace(",
            "invoke_agent(",
            "route_model(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
