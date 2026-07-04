import unittest

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration import (
    CreativeAnalytics,
    analyze_creative_complexity,
    build_creative_analytics,
    build_quality_dashboard,
    build_system_health_monitoring,
    creative_analytics_panel_by_id,
    creative_analytics_panels_for_status,
    predict_creative_diversity,
)
from creative_coding_assistant.orchestration.creative_analytics import (
    _quality_prediction,
)
from creative_coding_assistant.orchestration.creative_consistency_predictor import (
    predict_creative_consistency,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    derive_creative_score_profile,
)

REQUIRED_CREATIVE_ANALYTICS_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "creative_signal_count",
    "guardrail_signal_count",
    "observed_creative_event_count",
    "evaluated_output_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "creative_analytics_panel_implemented",
    "creative_metric_collection_implemented",
    "generated_output_evaluation_implemented",
    "creative_scoring_execution_implemented",
    "variant_generation_implemented",
    "consistency_validation_execution_implemented",
    "artifact_selection_implemented",
    "prompt_mutation_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
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
    "quality_dashboard",
    "creative_diversity_prediction_plan",
    "creative_consistency_prediction_plan",
    "creative_complexity_analysis",
    "creative_score_profile",
    "system_health_monitoring",
)


class CreativeAnalyticsTests(unittest.TestCase):
    def test_default_analytics_links_creative_sources(self) -> None:
        quality = build_quality_dashboard()
        diversity = predict_creative_diversity()
        quality_prediction = _quality_prediction()
        consistency = predict_creative_consistency(
            creative_quality_prediction=quality_prediction,
        )
        complexity = analyze_creative_complexity()
        request = AssistantRequest(query="Design an audio reactive generative garden.")
        score = derive_creative_score_profile(
            request=request,
            route_decision=None,
            creative_critic=None,
            self_evaluation=None,
            creative_improvement_planner=None,
            reflection_loop=None,
            creative_confidence=None,
            planning_metadata=(),
        )
        system = build_system_health_monitoring()
        analytics = build_creative_analytics(
            quality_dashboard=quality,
            diversity_prediction=diversity,
            consistency_prediction=consistency,
            complexity_analysis=complexity,
            score_profile=score,
            system_health=system,
        )

        self.assertEqual(analytics.role, "creative_analytics")
        self.assertEqual(analytics.serialization_version, "creative_analytics.v1")
        self.assertEqual(
            analytics.source_quality_dashboard_serialization_version,
            quality.serialization_version,
        )
        self.assertEqual(
            analytics.source_diversity_prediction_serialization_version,
            diversity.serialization_version,
        )
        self.assertEqual(
            analytics.source_consistency_prediction_serialization_version,
            consistency.serialization_version,
        )
        self.assertEqual(
            analytics.source_complexity_analysis_serialization_version,
            complexity.serialization_version,
        )
        self.assertEqual(
            analytics.source_score_profile_serialization_version,
            score.serialization_version,
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
                "creative_analytics::quality_readiness",
                "creative_analytics::diversity_readiness",
                "creative_analytics::consistency_readiness",
                "creative_analytics::complexity_profile",
                "creative_analytics::score_profile",
                "creative_analytics::system_context",
            ),
        )
        self.assertGreater(analytics.creative_signal_count, 0)
        self.assertGreater(analytics.guardrail_signal_count, 0)
        self.assertIsNone(analytics.observed_creative_event_count)
        self.assertIsNone(analytics.evaluated_output_count)
        self.assertEqual(analytics.creative_analytics_status, "guarded")
        self.assertIn(
            "does not collect live creative metrics", analytics.authority_boundary
        )
        self.assertTrue(analytics.creative_analytics_implemented)
        self.assertFalse(analytics.creative_metric_collection_implemented)
        self.assertFalse(analytics.generated_output_evaluation_implemented)
        self.assertFalse(analytics.creative_scoring_execution_implemented)
        self.assertFalse(analytics.variant_generation_implemented)
        self.assertFalse(analytics.consistency_validation_execution_implemented)
        self.assertFalse(analytics.artifact_selection_implemented)
        self.assertFalse(analytics.prompt_mutation_implemented)
        self.assertFalse(analytics.workflow_control_implemented)
        self.assertFalse(analytics.provider_model_routing_implemented)
        self.assertFalse(analytics.agent_invocation_implemented)
        self.assertFalse(analytics.retry_triggering_implemented)
        self.assertFalse(analytics.refinement_triggering_implemented)
        self.assertFalse(analytics.memory_write_implemented)
        self.assertFalse(analytics.persistent_storage_write_implemented)
        self.assertFalse(analytics.generated_output_mutation_implemented)
        self.assertFalse(analytics.runtime_evolution_implemented)
        self.assertTrue(analytics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        analytics = build_creative_analytics()

        for panel in analytics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CREATIVE_ANALYTICS_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "creative_analytics_panel.v1")
            self.assertIsNone(panel.observed_creative_event_count)
            self.assertIsNone(panel.evaluated_output_count)
            self.assertIn("creative_metric_collection", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.creative_analytics_panel_implemented)
            self.assertFalse(panel.creative_metric_collection_implemented)
            self.assertFalse(panel.generated_output_evaluation_implemented)
            self.assertFalse(panel.creative_scoring_execution_implemented)
            self.assertFalse(panel.variant_generation_implemented)
            self.assertFalse(panel.consistency_validation_execution_implemented)
            self.assertFalse(panel.artifact_selection_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        score = creative_analytics_panel_by_id(
            "creative_analytics::score_profile",
            analytics,
        )
        self.assertIsNotNone(score)
        assert score is not None
        self.assertEqual(score.status, "guarded")
        self.assertEqual(score.source_serialization_version, "v1")
        self.assertGreater(score.creative_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_evaluating(self) -> None:
        analytics = build_creative_analytics()
        diversity = creative_analytics_panel_by_id(
            "creative_analytics::diversity_readiness",
            analytics,
        )
        guarded = creative_analytics_panels_for_status("guarded", analytics)
        ready = creative_analytics_panels_for_status("ready", analytics)
        missing = creative_analytics_panel_by_id("missing", analytics)

        self.assertIsNone(missing)
        self.assertIsNotNone(diversity)
        assert diversity is not None
        self.assertEqual(diversity.panel_kind, "diversity_readiness")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), analytics.panel_count)

    def test_analytics_rejects_mismatched_panel_totals(self) -> None:
        analytics = build_creative_analytics()
        payload = analytics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            CreativeAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["creative_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "creative_signal_count must match"):
            CreativeAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["creative_analytics_status"] = "ready"

        with self.assertRaisesRegex(ValueError, "creative_analytics_status must match"):
            CreativeAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            CreativeAnalytics(**payload)

    def test_analytics_does_not_declare_runtime_creative_terms(self) -> None:
        analytics = build_creative_analytics()
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
            "collect_creative_metrics(",
            "evaluate_generated_output(",
            "execute_creative_score(",
            "generate_variant(",
            "validate_consistency(",
            "select_artifact(",
            "mutate_prompt(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
