import unittest

from creative_coding_assistant.orchestration import (
    CreativeDiversityAnalytics,
    build_confidence_analytics,
    build_creative_analytics,
    build_creative_diversity_analytics,
    build_system_health_monitoring,
    creative_diversity_analytics_panel_by_id,
    creative_diversity_analytics_panels_for_status,
    creative_diversity_audit_registry,
    creative_exploration_budget_registry,
    predict_creative_diversity,
)

REQUIRED_CREATIVE_DIVERSITY_ANALYTICS_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "diversity_signal_count",
    "guardrail_signal_count",
    "observed_diversity_event_count",
    "generated_variant_count",
    "enforced_budget_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "creative_diversity_analytics_panel_implemented",
    "diversity_metric_collection_implemented",
    "budget_enforcement_implemented",
    "variant_generation_implemented",
    "variant_selection_implemented",
    "artifact_selection_implemented",
    "creative_metric_collection_implemented",
    "generated_output_evaluation_implemented",
    "refinement_triggering_implemented",
    "retry_triggering_implemented",
    "cost_routing_implemented",
    "human_review_request_implemented",
    "escalation_triggering_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "prompt_mutation_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "creative_diversity_prediction_plan",
    "creative_diversity_audit_registry",
    "creative_exploration_budget_registry",
    "creative_analytics",
    "confidence_analytics",
    "system_health_monitoring",
)


class CreativeDiversityAnalyticsTests(unittest.TestCase):
    def test_default_analytics_links_diversity_sources(self) -> None:
        audit = creative_diversity_audit_registry()
        prediction = predict_creative_diversity(diversity_audit=audit)
        budget = creative_exploration_budget_registry()
        creative = build_creative_analytics(diversity_prediction=prediction)
        confidence = build_confidence_analytics(creative_analytics=creative)
        system = build_system_health_monitoring()
        analytics = build_creative_diversity_analytics(
            diversity_prediction=prediction,
            diversity_audit=audit,
            exploration_budget=budget,
            creative_analytics=creative,
            confidence_analytics=confidence,
            system_health=system,
        )

        self.assertEqual(analytics.role, "creative_diversity_analytics")
        self.assertEqual(
            analytics.serialization_version,
            "creative_diversity_analytics.v1",
        )
        self.assertEqual(
            analytics.source_diversity_prediction_serialization_version,
            prediction.serialization_version,
        )
        self.assertEqual(
            analytics.source_diversity_audit_serialization_version,
            audit.serialization_version,
        )
        self.assertEqual(
            analytics.source_exploration_budget_serialization_version,
            budget.serialization_version,
        )
        self.assertEqual(
            analytics.source_creative_analytics_serialization_version,
            creative.serialization_version,
        )
        self.assertEqual(
            analytics.source_confidence_analytics_serialization_version,
            confidence.serialization_version,
        )
        self.assertEqual(
            analytics.source_system_health_serialization_version,
            system.serialization_version,
        )
        self.assertEqual(analytics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(analytics.panel_count, 6)
        self.assertEqual(
            analytics.panel_ids,
            (
                "creative_diversity_analytics::diversity_prediction",
                "creative_diversity_analytics::diversity_audit_coverage",
                "creative_diversity_analytics::exploration_budget_posture",
                "creative_diversity_analytics::creative_diversity_context",
                "creative_diversity_analytics::confidence_diversity_context",
                "creative_diversity_analytics::system_diversity_context",
            ),
        )
        self.assertGreater(analytics.diversity_signal_count, 0)
        self.assertGreater(analytics.guardrail_signal_count, 0)
        self.assertIsNone(analytics.observed_diversity_event_count)
        self.assertIsNone(analytics.generated_variant_count)
        self.assertIsNone(analytics.enforced_budget_count)
        self.assertEqual(analytics.creative_diversity_analytics_status, "guarded")
        self.assertIn("does not collect diversity metrics", analytics.authority_boundary)
        self.assertTrue(analytics.creative_diversity_analytics_implemented)
        self.assertFalse(analytics.diversity_metric_collection_implemented)
        self.assertFalse(analytics.budget_enforcement_implemented)
        self.assertFalse(analytics.variant_generation_implemented)
        self.assertFalse(analytics.variant_selection_implemented)
        self.assertFalse(analytics.artifact_selection_implemented)
        self.assertFalse(analytics.creative_metric_collection_implemented)
        self.assertFalse(analytics.generated_output_evaluation_implemented)
        self.assertFalse(analytics.refinement_triggering_implemented)
        self.assertFalse(analytics.retry_triggering_implemented)
        self.assertFalse(analytics.cost_routing_implemented)
        self.assertFalse(analytics.human_review_request_implemented)
        self.assertFalse(analytics.escalation_triggering_implemented)
        self.assertFalse(analytics.agent_invocation_implemented)
        self.assertFalse(analytics.provider_model_routing_implemented)
        self.assertFalse(analytics.workflow_control_implemented)
        self.assertFalse(analytics.prompt_mutation_implemented)
        self.assertFalse(analytics.memory_write_implemented)
        self.assertFalse(analytics.persistent_storage_write_implemented)
        self.assertFalse(analytics.generated_output_mutation_implemented)
        self.assertFalse(analytics.runtime_evolution_implemented)
        self.assertTrue(analytics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        analytics = build_creative_diversity_analytics()

        for panel in analytics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(
                set(dumped),
                REQUIRED_CREATIVE_DIVERSITY_ANALYTICS_PANEL_FIELDS,
            )
            self.assertEqual(
                panel.serialization_version,
                "creative_diversity_analytics_panel.v1",
            )
            self.assertIsNone(panel.observed_diversity_event_count)
            self.assertIsNone(panel.generated_variant_count)
            self.assertIsNone(panel.enforced_budget_count)
            self.assertIn("diversity_metric_collection", panel.blocked_runtime_behaviors)
            self.assertIn("variant_generation", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.creative_diversity_analytics_panel_implemented)
            self.assertFalse(panel.diversity_metric_collection_implemented)
            self.assertFalse(panel.budget_enforcement_implemented)
            self.assertFalse(panel.variant_generation_implemented)
            self.assertFalse(panel.variant_selection_implemented)
            self.assertFalse(panel.artifact_selection_implemented)
            self.assertFalse(panel.creative_metric_collection_implemented)
            self.assertFalse(panel.generated_output_evaluation_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.cost_routing_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        budget = creative_diversity_analytics_panel_by_id(
            "creative_diversity_analytics::exploration_budget_posture",
            analytics,
        )
        self.assertIsNotNone(budget)
        assert budget is not None
        self.assertEqual(budget.status, "guarded")
        self.assertEqual(
            budget.source_serialization_version,
            "creative_exploration_budget_registry.v1",
        )
        self.assertGreater(budget.diversity_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_generating(self) -> None:
        analytics = build_creative_diversity_analytics()
        prediction = creative_diversity_analytics_panel_by_id(
            "creative_diversity_analytics::diversity_prediction",
            analytics,
        )
        guarded = creative_diversity_analytics_panels_for_status(
            "guarded",
            analytics,
        )
        ready = creative_diversity_analytics_panels_for_status("ready", analytics)
        missing = creative_diversity_analytics_panel_by_id("missing", analytics)

        self.assertIsNone(missing)
        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.panel_kind, "diversity_prediction")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), analytics.panel_count)

    def test_analytics_rejects_mismatched_panel_totals(self) -> None:
        analytics = build_creative_diversity_analytics()
        payload = analytics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            CreativeDiversityAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["diversity_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "diversity_signal_count must match"):
            CreativeDiversityAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["creative_diversity_analytics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "creative_diversity_analytics_status must match",
        ):
            CreativeDiversityAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            CreativeDiversityAnalytics(**payload)

    def test_analytics_does_not_declare_runtime_diversity_terms(self) -> None:
        analytics = build_creative_diversity_analytics()
        combined_text = " ".join(
            (
                analytics.authority_boundary,
                *analytics.blocked_runtime_behaviors,
                *analytics.advisory_actions,
                *(
                    field
                    for panel in analytics.panels
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
            "collect_diversity_metrics(",
            "enforce_budget(",
            "generate_variant(",
            "select_variant(",
            "select_artifact(",
            "trigger_refinement(",
            "route_by_cost(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
