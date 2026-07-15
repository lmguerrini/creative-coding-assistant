import unittest

from creative_coding_assistant.orchestration import (
    ProductionCreativeReadinessReview,
    build_production_creative_readiness_review,
    production_creative_readiness_record_by_area,
    production_creative_readiness_records_for_status,
)

REQUIRED_AREAS = (
    "creative_prompt_readiness",
    "visual_preview_readiness",
    "retrieval_context_readiness",
    "creative_quality_readiness",
    "creative_diversity_consistency_readiness",
    "creative_workflow_explainability_readiness",
)
REQUIRED_SOURCE_SURFACES = (
    "production_demo_assets",
    "creative_analytics",
    "production_readiness_review",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "area",
    "status",
    "source_surface_ids",
    "source_serialization_versions",
    "evidence",
    "ready_signals",
    "guarded_findings",
    "blocking_findings",
    "operator_actions",
    "blocked_runtime_behaviors",
    "creative_readiness_record_implemented",
    "generated_output_evaluation_implemented",
    "creative_metric_collection_implemented",
    "creative_scoring_execution_implemented",
    "variant_generation_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "prompt_mutation_implemented",
    "artifact_selection_implemented",
    "artifact_mutation_implemented",
    "generated_output_mutation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "preview_rendering_execution_implemented",
    "human_input_request_implemented",
    "memory_write_implemented",
    "persistent_storage_write_implemented",
    "deployment_execution_implemented",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionCreativeReadinessReviewTests(unittest.TestCase):
    def test_review_aggregates_creative_demo_and_analytics_surfaces(self) -> None:
        review = build_production_creative_readiness_review()

        self.assertEqual(review.role, "production_creative_readiness_review")
        self.assertEqual(
            review.serialization_version,
            "production_creative_readiness_review.v1",
        )
        self.assertEqual(review.source_surfaces, REQUIRED_SOURCE_SURFACES)
        self.assertEqual(review.areas, REQUIRED_AREAS)
        self.assertEqual(review.record_count, 6)
        self.assertEqual(review.creative_readiness_status, "guarded")
        self.assertEqual(review.blocked_record_ids, ())
        self.assertEqual(review.blocking_finding_count, 0)
        self.assertGreater(review.ready_signal_count, 0)
        self.assertGreaterEqual(review.guarded_finding_count, 2)
        self.assertIn(
            "Creative demo materials are available",
            review.capstone_creative_readiness_statement,
        )
        self.assertNotIn(
            "capstone",
            review.capstone_creative_readiness_statement.casefold(),
        )
        self.assertIn("does not evaluate generated output", review.authority_boundary)
        self.assertTrue(review.creative_readiness_review_implemented)
        self.assertTrue(review.prompt_readiness_review_implemented)
        self.assertTrue(review.preview_readiness_review_implemented)
        self.assertTrue(review.retrieval_context_review_implemented)
        self.assertTrue(review.creative_quality_review_implemented)
        self.assertTrue(review.creative_diversity_consistency_review_implemented)
        self.assertTrue(review.workflow_explainability_review_implemented)
        self.assertFalse(review.generated_output_evaluation_implemented)
        self.assertFalse(review.creative_metric_collection_implemented)
        self.assertFalse(review.creative_scoring_execution_implemented)
        self.assertFalse(review.variant_generation_implemented)
        self.assertFalse(review.asset_generation_implemented)
        self.assertFalse(review.retrieval_execution_implemented)
        self.assertFalse(review.prompt_mutation_implemented)
        self.assertFalse(review.artifact_selection_implemented)
        self.assertFalse(review.artifact_mutation_implemented)
        self.assertFalse(review.generated_output_mutation_implemented)
        self.assertFalse(review.provider_model_routing_implemented)
        self.assertFalse(review.provider_execution_implemented)
        self.assertFalse(review.workflow_execution_implemented)
        self.assertFalse(review.workflow_control_implemented)
        self.assertFalse(review.preview_rendering_execution_implemented)
        self.assertFalse(review.human_input_request_implemented)
        self.assertFalse(review.memory_write_implemented)
        self.assertFalse(review.persistent_storage_write_implemented)
        self.assertFalse(review.deployment_execution_implemented)
        self.assertFalse(review.merge_push_tag_implemented)
        self.assertFalse(review.runtime_evolution_implemented)
        self.assertTrue(review.metadata_only)

    def test_records_capture_guarded_analytics_without_output_evaluation(self) -> None:
        review = build_production_creative_readiness_review()
        prompt = production_creative_readiness_record_by_area(
            "creative_prompt_readiness",
            review,
        )
        quality = production_creative_readiness_record_by_area(
            "creative_quality_readiness",
            review,
        )
        diversity_consistency = production_creative_readiness_record_by_area(
            "creative_diversity_consistency_readiness",
            review,
        )
        workflow = production_creative_readiness_record_by_area(
            "creative_workflow_explainability_readiness",
            review,
        )
        guarded = production_creative_readiness_records_for_status("guarded", review)

        self.assertIsNotNone(prompt)
        self.assertIsNotNone(quality)
        self.assertIsNotNone(diversity_consistency)
        self.assertIsNotNone(workflow)
        assert prompt is not None
        assert quality is not None
        assert diversity_consistency is not None
        assert workflow is not None
        self.assertGreaterEqual(len(guarded), 2)
        self.assertEqual(prompt.status, "ready")
        self.assertIn("creative_coding_prompt", prompt.ready_signals)
        self.assertEqual(quality.status, "guarded")
        self.assertIn(
            "creative_analytics::quality_readiness",
            quality.guarded_findings,
        )
        self.assertIn(
            "creative_analytics::diversity_readiness",
            diversity_consistency.guarded_findings,
        )
        self.assertIn(
            "creative_analytics::consistency_readiness",
            diversity_consistency.guarded_findings,
        )
        self.assertEqual(workflow.status, "ready")
        self.assertIn("selected provider", workflow.ready_signals)

        for record in review.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_creative_readiness_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_creative_readiness::{record.area}",
            )
            self.assertEqual(
                len(record.source_surface_ids),
                len(record.source_serialization_versions),
            )
            self.assertFalse(record.blocking_findings)
            self.assertTrue(record.creative_readiness_record_implemented)
            self.assertFalse(record.generated_output_evaluation_implemented)
            self.assertFalse(record.creative_metric_collection_implemented)
            self.assertFalse(record.creative_scoring_execution_implemented)
            self.assertFalse(record.variant_generation_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.artifact_selection_implemented)
            self.assertFalse(record.artifact_mutation_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.preview_rendering_execution_implemented)
            self.assertFalse(record.human_input_request_implemented)
            self.assertFalse(record.memory_write_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_review_rejects_mismatched_records_or_counts(self) -> None:
        review = build_production_creative_readiness_review()
        payload = review.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionCreativeReadinessReview(**payload)

        payload = review.model_dump(mode="json")
        payload["ready_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "ready_signal_count must match"):
            ProductionCreativeReadinessReview(**payload)

        payload = review.model_dump(mode="json")
        payload["creative_readiness_status"] = "ready"

        with self.assertRaisesRegex(ValueError, "creative_readiness_status must match"):
            ProductionCreativeReadinessReview(**payload)


if __name__ == "__main__":
    unittest.main()
