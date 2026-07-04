import unittest

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration import (
    ConfidenceAnalytics,
    agent_confidence_fusion_registry,
    build_confidence_analytics,
    build_creative_analytics,
    build_escalation_diagnostics,
    build_quality_dashboard,
    confidence_analytics_panel_by_id,
    confidence_analytics_panels_for_status,
    confidence_threshold_routing_registry,
    derive_creative_confidence_profile,
)

REQUIRED_CONFIDENCE_ANALYTICS_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "confidence_signal_count",
    "guardrail_signal_count",
    "calculated_confidence_score",
    "evaluated_threshold_count",
    "routed_confidence_decision_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "confidence_analytics_panel_implemented",
    "confidence_score_calculation_implemented",
    "confidence_threshold_evaluation_implemented",
    "confidence_based_routing_implemented",
    "agent_confidence_fusion_execution_implemented",
    "generated_output_evaluation_implemented",
    "quality_scoring_implemented",
    "creative_metric_collection_implemented",
    "human_review_request_implemented",
    "escalation_triggering_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
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
    "creative_confidence_profile",
    "agent_confidence_fusion_registry",
    "confidence_threshold_routing_registry",
    "quality_dashboard",
    "creative_analytics",
    "escalation_diagnostics",
)


class ConfidenceAnalyticsTests(unittest.TestCase):
    def test_default_analytics_links_confidence_sources(self) -> None:
        request = AssistantRequest(query="Design an audio reactive generative garden.")
        confidence = derive_creative_confidence_profile(
            request=request,
            route_decision=None,
            creative_critic=None,
            self_evaluation=None,
            creative_improvement_planner=None,
            reflection_loop=None,
            planning_metadata=(),
        )
        fusion = agent_confidence_fusion_registry()
        threshold = confidence_threshold_routing_registry()
        quality = build_quality_dashboard()
        creative = build_creative_analytics()
        escalation = build_escalation_diagnostics()
        analytics = build_confidence_analytics(
            confidence_profile=confidence,
            agent_confidence_fusion=fusion,
            confidence_threshold_routing=threshold,
            quality_dashboard=quality,
            creative_analytics=creative,
            escalation_diagnostics=escalation,
        )

        self.assertEqual(analytics.role, "confidence_analytics")
        self.assertEqual(analytics.serialization_version, "confidence_analytics.v1")
        self.assertEqual(
            analytics.source_confidence_profile_serialization_version,
            confidence.serialization_version,
        )
        self.assertEqual(
            analytics.source_agent_confidence_fusion_serialization_version,
            fusion.serialization_version,
        )
        self.assertEqual(
            analytics.source_confidence_threshold_serialization_version,
            threshold.serialization_version,
        )
        self.assertEqual(
            analytics.source_quality_dashboard_serialization_version,
            quality.serialization_version,
        )
        self.assertEqual(
            analytics.source_creative_analytics_serialization_version,
            creative.serialization_version,
        )
        self.assertEqual(
            analytics.source_escalation_diagnostics_serialization_version,
            escalation.serialization_version,
        )
        self.assertEqual(analytics.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(analytics.panel_count, 6)
        self.assertEqual(
            analytics.panel_ids,
            (
                "confidence_analytics::confidence_profile",
                "confidence_analytics::agent_confidence_fusion",
                "confidence_analytics::confidence_thresholds",
                "confidence_analytics::quality_confidence_context",
                "confidence_analytics::creative_confidence_context",
                "confidence_analytics::escalation_confidence_context",
            ),
        )
        self.assertGreater(analytics.confidence_signal_count, 0)
        self.assertGreater(analytics.guardrail_signal_count, 0)
        self.assertIsNone(analytics.calculated_confidence_score)
        self.assertIsNone(analytics.evaluated_threshold_count)
        self.assertIsNone(analytics.routed_confidence_decision_count)
        self.assertEqual(analytics.confidence_analytics_status, "guarded")
        self.assertIn(
            "does not calculate confidence scores", analytics.authority_boundary
        )
        self.assertTrue(analytics.confidence_analytics_implemented)
        self.assertFalse(analytics.confidence_score_calculation_implemented)
        self.assertFalse(analytics.confidence_threshold_evaluation_implemented)
        self.assertFalse(analytics.confidence_based_routing_implemented)
        self.assertFalse(analytics.agent_confidence_fusion_execution_implemented)
        self.assertFalse(analytics.generated_output_evaluation_implemented)
        self.assertFalse(analytics.quality_scoring_implemented)
        self.assertFalse(analytics.creative_metric_collection_implemented)
        self.assertFalse(analytics.human_review_request_implemented)
        self.assertFalse(analytics.escalation_triggering_implemented)
        self.assertFalse(analytics.agent_invocation_implemented)
        self.assertFalse(analytics.provider_model_routing_implemented)
        self.assertFalse(analytics.workflow_control_implemented)
        self.assertFalse(analytics.retry_triggering_implemented)
        self.assertFalse(analytics.refinement_triggering_implemented)
        self.assertFalse(analytics.memory_write_implemented)
        self.assertFalse(analytics.persistent_storage_write_implemented)
        self.assertFalse(analytics.generated_output_mutation_implemented)
        self.assertFalse(analytics.runtime_evolution_implemented)
        self.assertTrue(analytics.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        analytics = build_confidence_analytics()

        for panel in analytics.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONFIDENCE_ANALYTICS_PANEL_FIELDS)
            self.assertEqual(
                panel.serialization_version, "confidence_analytics_panel.v1"
            )
            self.assertIsNone(panel.calculated_confidence_score)
            self.assertIsNone(panel.evaluated_threshold_count)
            self.assertIsNone(panel.routed_confidence_decision_count)
            self.assertIn(
                "confidence_score_calculation", panel.blocked_runtime_behaviors
            )
            self.assertIn(
                "confidence_threshold_evaluation",
                panel.blocked_runtime_behaviors,
            )
            self.assertTrue(panel.confidence_analytics_panel_implemented)
            self.assertFalse(panel.confidence_score_calculation_implemented)
            self.assertFalse(panel.confidence_threshold_evaluation_implemented)
            self.assertFalse(panel.confidence_based_routing_implemented)
            self.assertFalse(panel.agent_confidence_fusion_execution_implemented)
            self.assertFalse(panel.generated_output_evaluation_implemented)
            self.assertFalse(panel.quality_scoring_implemented)
            self.assertFalse(panel.creative_metric_collection_implemented)
            self.assertFalse(panel.human_review_request_implemented)
            self.assertFalse(panel.escalation_triggering_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.memory_write_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        threshold = confidence_analytics_panel_by_id(
            "confidence_analytics::confidence_thresholds",
            analytics,
        )
        self.assertIsNotNone(threshold)
        assert threshold is not None
        self.assertEqual(threshold.status, "guarded")
        self.assertEqual(
            threshold.source_serialization_version,
            "confidence_threshold_routing_registry.v1",
        )
        self.assertGreater(threshold.confidence_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_evaluating(self) -> None:
        analytics = build_confidence_analytics()
        fusion = confidence_analytics_panel_by_id(
            "confidence_analytics::agent_confidence_fusion",
            analytics,
        )
        guarded = confidence_analytics_panels_for_status("guarded", analytics)
        ready = confidence_analytics_panels_for_status("ready", analytics)
        missing = confidence_analytics_panel_by_id("missing", analytics)

        self.assertIsNone(missing)
        self.assertIsNotNone(fusion)
        assert fusion is not None
        self.assertEqual(fusion.panel_kind, "agent_confidence_fusion")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), analytics.panel_count)

    def test_analytics_rejects_mismatched_panel_totals(self) -> None:
        analytics = build_confidence_analytics()
        payload = analytics.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            ConfidenceAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["confidence_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "confidence_signal_count must match"):
            ConfidenceAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["confidence_analytics_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError, "confidence_analytics_status must match"
        ):
            ConfidenceAnalytics(**payload)

        payload = analytics.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            ConfidenceAnalytics(**payload)

    def test_analytics_does_not_declare_runtime_confidence_terms(self) -> None:
        analytics = build_confidence_analytics()
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
            "calculate_confidence_score(",
            "evaluate_confidence_threshold(",
            "route_by_confidence(",
            "execute_confidence_fusion(",
            "evaluate_generated_output(",
            "trigger_human_review(",
            "route_provider(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
