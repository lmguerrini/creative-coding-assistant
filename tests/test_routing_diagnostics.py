import unittest

from creative_coding_assistant.orchestration import (
    RoutingDiagnostics,
    build_routing_diagnostics,
    model_routing_intelligence_registry,
    provider_availability_registry,
    route_hybrid_model_request,
    route_local_vs_cloud,
    route_model_request,
    routing_diagnostic_panel_by_id,
    routing_diagnostic_panels_for_status,
    routing_provider_profile_registry,
    routing_safety_contract_registry,
    task_aware_routing_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_ROUTING_DIAGNOSTIC_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "routing_signal_count",
    "guardrail_signal_count",
    "applied_route_count",
    "provider_call_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "routing_diagnostic_panel_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "provider_switching_implemented",
    "model_switching_implemented",
    "automatic_provider_selection_implemented",
    "automatic_model_selection_implemented",
    "local_model_discovery_implemented",
    "local_model_download_implemented",
    "automatic_api_key_assumption_implemented",
    "hybrid_routing_application_implemented",
    "budget_enforcement_implemented",
    "hitl_request_emission_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "routing_provider_profile_registry",
    "provider_availability_registry",
    "task_aware_routing_registry",
    "model_routing_plan",
    "local_cloud_routing_plan",
    "hybrid_routing_plan",
    "routing_safety_contract_registry",
    "model_routing_intelligence_registry",
)


class RoutingDiagnosticsTests(unittest.TestCase):
    def test_default_diagnostics_links_routing_sources(self) -> None:
        providers = routing_provider_profile_registry()
        availability = provider_availability_registry()
        tasks = task_aware_routing_registry()
        model = route_model_request(route=RouteName.GENERATE)
        local_cloud = route_local_vs_cloud(model_routing=model)
        hybrid = route_hybrid_model_request(local_cloud_routing=local_cloud)
        safety = routing_safety_contract_registry()
        intelligence = model_routing_intelligence_registry()
        diagnostics = build_routing_diagnostics(
            route=RouteName.GENERATE,
            provider_profiles=providers,
            provider_availability=availability,
            task_routing=tasks,
            model_routing=model,
            local_cloud_routing=local_cloud,
            hybrid_routing=hybrid,
            safety_contracts=safety,
            routing_intelligence=intelligence,
        )

        self.assertEqual(diagnostics.role, "routing_diagnostics")
        self.assertEqual(diagnostics.serialization_version, "routing_diagnostics.v1")
        self.assertEqual(diagnostics.route_name, RouteName.GENERATE)
        self.assertEqual(
            diagnostics.source_provider_profile_serialization_version,
            providers.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_provider_availability_serialization_version,
            availability.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_task_routing_serialization_version,
            tasks.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_model_routing_serialization_version,
            model.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_local_cloud_serialization_version,
            local_cloud.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_hybrid_routing_serialization_version,
            hybrid.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_safety_contract_serialization_version,
            safety.serialization_version,
        )
        self.assertEqual(
            diagnostics.source_intelligence_serialization_version,
            intelligence.serialization_version,
        )
        self.assertEqual(diagnostics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(diagnostics.panel_count, 8)
        self.assertEqual(
            diagnostics.panel_ids,
            (
                "routing_diagnostics::provider_profile_coverage",
                "routing_diagnostics::availability_metadata",
                "routing_diagnostics::task_routing_policy",
                "routing_diagnostics::model_route_recommendation",
                "routing_diagnostics::local_cloud_posture",
                "routing_diagnostics::hybrid_route_posture",
                "routing_diagnostics::safety_contracts",
                "routing_diagnostics::intelligence_summary",
            ),
        )
        self.assertGreater(diagnostics.routing_signal_count, 0)
        self.assertGreater(diagnostics.guardrail_signal_count, 0)
        self.assertIsNone(diagnostics.applied_route_count)
        self.assertIsNone(diagnostics.provider_call_count)
        self.assertEqual(diagnostics.routing_diagnostics_status, "guarded")
        self.assertIn("does not apply routes", diagnostics.authority_boundary)
        self.assertTrue(diagnostics.routing_diagnostics_implemented)
        self.assertFalse(diagnostics.routing_application_implemented)
        self.assertFalse(diagnostics.provider_model_routing_implemented)
        self.assertFalse(diagnostics.provider_execution_implemented)
        self.assertFalse(diagnostics.provider_switching_implemented)
        self.assertFalse(diagnostics.model_switching_implemented)
        self.assertFalse(diagnostics.automatic_provider_selection_implemented)
        self.assertFalse(diagnostics.automatic_model_selection_implemented)
        self.assertFalse(diagnostics.local_model_discovery_implemented)
        self.assertFalse(diagnostics.local_model_download_implemented)
        self.assertFalse(diagnostics.automatic_api_key_assumption_implemented)
        self.assertFalse(diagnostics.hybrid_routing_application_implemented)
        self.assertFalse(diagnostics.budget_enforcement_implemented)
        self.assertFalse(diagnostics.hitl_request_emission_implemented)
        self.assertFalse(diagnostics.workflow_control_implemented)
        self.assertFalse(diagnostics.retry_triggering_implemented)
        self.assertFalse(diagnostics.prompt_mutation_implemented)
        self.assertFalse(diagnostics.persistent_storage_write_implemented)
        self.assertFalse(diagnostics.generated_output_mutation_implemented)
        self.assertTrue(diagnostics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        diagnostics = build_routing_diagnostics()

        for panel in diagnostics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ROUTING_DIAGNOSTIC_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "routing_diagnostic_panel.v1",
            )
            self.assertIsNone(panel.applied_route_count)
            self.assertIsNone(panel.provider_call_count)
            self.assertIn("routing_application", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.routing_diagnostic_panel_implemented)
            self.assertFalse(panel.routing_application_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.provider_switching_implemented)
            self.assertFalse(panel.model_switching_implemented)
            self.assertFalse(panel.automatic_provider_selection_implemented)
            self.assertFalse(panel.automatic_model_selection_implemented)
            self.assertFalse(panel.local_model_discovery_implemented)
            self.assertFalse(panel.local_model_download_implemented)
            self.assertFalse(panel.automatic_api_key_assumption_implemented)
            self.assertFalse(panel.hybrid_routing_application_implemented)
            self.assertFalse(panel.budget_enforcement_implemented)
            self.assertFalse(panel.hitl_request_emission_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        model_panel = routing_diagnostic_panel_by_id(
            "routing_diagnostics::model_route_recommendation",
            diagnostics,
        )
        self.assertIsNotNone(model_panel)
        assert model_panel is not None
        self.assertEqual(model_panel.status, "guarded")
        self.assertEqual(
            model_panel.source_serialization_version,
            "model_routing_plan.v1",
        )
        self.assertGreater(model_panel.routing_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_applying(self) -> None:
        diagnostics = build_routing_diagnostics()
        hybrid = routing_diagnostic_panel_by_id(
            "routing_diagnostics::hybrid_route_posture",
            diagnostics,
        )
        guarded = routing_diagnostic_panels_for_status("guarded", diagnostics)
        ready = routing_diagnostic_panels_for_status("ready", diagnostics)
        missing = routing_diagnostic_panel_by_id("missing", diagnostics)

        self.assertIsNone(missing)
        self.assertIsNotNone(hybrid)
        assert hybrid is not None
        self.assertEqual(hybrid.panel_kind, "hybrid_route_posture")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), diagnostics.panel_count)

    def test_diagnostics_rejects_mismatched_panel_totals(self) -> None:
        diagnostics = build_routing_diagnostics()
        payload = diagnostics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            RoutingDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["routing_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "routing_signal_count must match"):
            RoutingDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["routing_diagnostics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "routing_diagnostics_status must match",
        ):
            RoutingDiagnostics(**payload)

        payload = diagnostics.model_dump(mode="json")
        payload["source_surfaces"] = ("missing",) + tuple(payload["source_surfaces"][1:])

        with self.assertRaisesRegex(
            ValueError,
            "source_surfaces must match",
        ):
            RoutingDiagnostics(**payload)

    def test_diagnostics_does_not_declare_runtime_routing_application_terms(
        self,
    ) -> None:
        diagnostics = build_routing_diagnostics()
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
            "apply_route(",
            "switch_provider(",
            "switch_model(",
            "select_provider(",
            "select_model(",
            "execute_provider(",
            "download_model(",
            "assume_api_key(",
            "emit_hitl_request(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
