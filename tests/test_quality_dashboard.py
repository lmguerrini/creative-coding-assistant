import unittest

from creative_coding_assistant.orchestration import (
    QualityDashboard,
    build_quality_dashboard,
    optimize_quality_cost,
    predict_quality_for_route,
    quality_dashboard_panel_by_id,
    quality_dashboard_panels_for_pressure,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    quality_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_QUALITY_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "pressure",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "route_name",
    "recommended_quality_level",
    "relative_quality_units_total",
    "recommended_quality_units",
    "quality_signal_count",
    "evaluated_output_score",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "quality_dashboard_panel_implemented",
    "generated_output_quality_evaluation_implemented",
    "quality_scoring_implemented",
    "quality_escalation_implemented",
    "refinement_triggering_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "provider_execution_implemented",
    "human_input_request_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class QualityDashboardTests(unittest.TestCase):
    def test_default_dashboard_summarizes_quality_metadata(self) -> None:
        profiles = quality_profile_registry()
        prediction = predict_quality_for_route(route=RouteName.GENERATE)
        tradeoff = optimize_quality_cost(route=RouteName.GENERATE)
        dashboard = build_quality_dashboard(
            quality_profiles=profiles,
            quality_prediction=prediction,
            quality_cost=tradeoff,
        )

        self.assertEqual(dashboard.role, "quality_dashboard")
        self.assertEqual(dashboard.serialization_version, "quality_dashboard.v1")
        self.assertEqual(
            dashboard.source_quality_profile_serialization_version,
            profiles.serialization_version,
        )
        self.assertEqual(
            dashboard.source_quality_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            dashboard.source_quality_cost_serialization_version,
            tradeoff.serialization_version,
        )
        self.assertEqual(dashboard.panel_count, 4)
        self.assertEqual(
            dashboard.panel_ids,
            (
                "quality_dashboard::quality_profile_coverage",
                "quality_dashboard::route_quality_prediction",
                "quality_dashboard::quality_cost_tradeoff",
                "quality_dashboard::evaluation_boundary",
            ),
        )
        self.assertGreater(dashboard.relative_quality_units_total, 0)
        self.assertGreater(dashboard.highest_relative_quality_units, 0)
        self.assertGreater(dashboard.quality_signal_count, 0)
        self.assertIsNone(dashboard.evaluated_output_score)
        self.assertEqual(dashboard.dashboard_pressure, "guarded")
        self.assertIn(
            "does not evaluate generated output",
            dashboard.authority_boundary,
        )
        self.assertTrue(dashboard.quality_dashboard_implemented)
        self.assertFalse(dashboard.generated_output_quality_evaluation_implemented)
        self.assertFalse(dashboard.quality_scoring_implemented)
        self.assertFalse(dashboard.quality_escalation_implemented)
        self.assertFalse(dashboard.refinement_triggering_implemented)
        self.assertFalse(dashboard.provider_model_routing_implemented)
        self.assertFalse(dashboard.model_selection_implemented)
        self.assertFalse(dashboard.provider_execution_implemented)
        self.assertFalse(dashboard.human_input_request_implemented)
        self.assertFalse(dashboard.workflow_control_implemented)
        self.assertFalse(dashboard.workflow_execution_implemented)
        self.assertFalse(dashboard.retry_triggering_implemented)
        self.assertFalse(dashboard.prompt_mutation_implemented)
        self.assertFalse(dashboard.persistent_storage_write_implemented)
        self.assertFalse(dashboard.generated_output_mutation_implemented)
        self.assertTrue(dashboard.advisory_only)

    def test_dashboard_panels_are_read_only_and_boundary_explicit(self) -> None:
        dashboard = build_quality_dashboard()

        for panel in dashboard.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_QUALITY_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version,
                "quality_dashboard_panel.v1",
            )
            self.assertIsNone(panel.evaluated_output_score)
            self.assertIn(
                "generated_output_quality_evaluation",
                panel.blocked_runtime_behaviors,
            )
            self.assertTrue(panel.quality_dashboard_panel_implemented)
            self.assertFalse(panel.generated_output_quality_evaluation_implemented)
            self.assertFalse(panel.quality_scoring_implemented)
            self.assertFalse(panel.quality_escalation_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.model_selection_implemented)
            self.assertFalse(panel.provider_execution_implemented)
            self.assertFalse(panel.human_input_request_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertTrue(panel.advisory_only)

        boundary = quality_dashboard_panel_by_id(
            "quality_dashboard::evaluation_boundary",
            dashboard,
        )
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual(boundary.status, "guarded")
        self.assertEqual(boundary.relative_quality_units_total, 0)
        self.assertEqual(boundary.recommended_quality_units, 0)
        self.assertEqual(boundary.quality_signal_count, 0)
        self.assertEqual(
            boundary.source_serialization_version,
            "generated_output_quality_boundary.v1",
        )

    def test_lookup_helpers_are_stable_and_non_applying(self) -> None:
        dashboard = build_quality_dashboard()
        prediction_panel = quality_dashboard_panel_by_id(
            "quality_dashboard::route_quality_prediction",
            dashboard,
        )
        guarded = quality_dashboard_panels_for_pressure("guarded", dashboard)
        missing = quality_dashboard_panel_by_id("missing", dashboard)

        self.assertIsNone(missing)
        self.assertIsNotNone(prediction_panel)
        assert prediction_panel is not None
        self.assertEqual(prediction_panel.panel_kind, "route_quality_prediction")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "quality_dashboard::evaluation_boundary",
            tuple(panel.panel_id for panel in guarded),
        )

    def test_dashboard_rejects_mismatched_panel_totals(self) -> None:
        dashboard = build_quality_dashboard()
        payload = dashboard.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            QualityDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["relative_quality_units_total"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "relative_quality_units_total must match",
        ):
            QualityDashboard(**payload)

        payload = dashboard.model_dump(mode="json")
        payload["dashboard_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "dashboard_pressure must match",
        ):
            QualityDashboard(**payload)

    def test_dashboard_does_not_declare_runtime_quality_application_terms(self) -> None:
        dashboard = build_quality_dashboard()
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
            "evaluate_quality(",
            "score_quality(",
            "execute_quality_escalation(",
            "trigger_refinement(",
            "select_model(",
            "route_provider(",
            "execute_provider(",
            "request_hitl(",
            "control_workflow(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
