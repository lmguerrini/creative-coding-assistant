import unittest

from creative_coding_assistant.orchestration import (
    ProductionReadinessReview,
    build_production_readiness_review,
    production_readiness_record_by_area,
    production_readiness_records_for_status,
)

REQUIRED_AREAS = (
    "configuration_readiness",
    "safety_readiness",
    "ux_explainability_readiness",
    "deployment_readiness",
    "failure_determinism_readiness",
    "mvp_demo_readiness",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "area",
    "status",
    "source_surface_ids",
    "evidence",
    "guarded_findings",
    "blocking_findings",
    "recommended_operator_actions",
    "blocked_runtime_behaviors",
    "readiness_record_implemented",
    "configuration_mutation_implemented",
    "provider_provisioning_implemented",
    "runtime_installation_implemented",
    "provider_execution_implemented",
    "deployment_execution_implemented",
    "hitl_request_emitted",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "asset_generation_implemented",
    "persistent_storage_write_implemented",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReadinessReviewTests(unittest.TestCase):
    def test_review_aggregates_production_readiness_surfaces(self) -> None:
        review = build_production_readiness_review()

        self.assertEqual(review.role, "production_readiness_review")
        self.assertEqual(review.serialization_version, "production_readiness_review.v1")
        self.assertEqual(review.areas, REQUIRED_AREAS)
        self.assertEqual(review.record_count, 6)
        self.assertEqual(review.production_readiness_status, "guarded")
        self.assertEqual(review.blocked_record_ids, ())
        self.assertEqual(review.blocking_finding_count, 0)
        self.assertGreaterEqual(review.guarded_finding_count, 1)
        self.assertIn(
            "Ready for capstone/portfolio demo", review.mvp_readiness_statement
        )
        self.assertIn("does not change configuration", review.authority_boundary)
        self.assertTrue(review.production_readiness_review_implemented)
        self.assertTrue(review.configuration_review_implemented)
        self.assertTrue(review.safety_review_implemented)
        self.assertTrue(review.ux_explainability_review_implemented)
        self.assertTrue(review.deployment_review_implemented)
        self.assertTrue(review.failure_review_implemented)
        self.assertTrue(review.mvp_demo_review_implemented)
        self.assertFalse(review.configuration_mutation_implemented)
        self.assertFalse(review.provider_provisioning_implemented)
        self.assertFalse(review.runtime_installation_implemented)
        self.assertFalse(review.provider_execution_implemented)
        self.assertFalse(review.deployment_execution_implemented)
        self.assertFalse(review.hitl_request_emitted)
        self.assertFalse(review.workflow_execution_implemented)
        self.assertFalse(review.workflow_control_implemented)
        self.assertFalse(review.asset_generation_implemented)
        self.assertFalse(review.persistent_storage_write_implemented)
        self.assertFalse(review.merge_push_tag_implemented)
        self.assertFalse(review.runtime_evolution_implemented)
        self.assertTrue(review.metadata_only)

    def test_records_surface_guarded_configuration_and_deployment_assumptions(
        self,
    ) -> None:
        review = build_production_readiness_review()
        configuration = production_readiness_record_by_area(
            "configuration_readiness",
            review,
        )
        deployment = production_readiness_record_by_area("deployment_readiness", review)
        ux = production_readiness_record_by_area("ux_explainability_readiness", review)
        guarded = production_readiness_records_for_status("guarded", review)

        self.assertIsNotNone(configuration)
        self.assertIsNotNone(deployment)
        self.assertIsNotNone(ux)
        assert configuration is not None
        assert deployment is not None
        assert ux is not None
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn("missing_api_key", configuration.guarded_findings)
        self.assertIn(
            "production_deployment::external_deployment_manifest",
            deployment.guarded_findings,
        )
        self.assertEqual(ux.status, "ready")

        for record in review.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_readiness_record.v1",
            )
            self.assertEqual(record.record_id, f"production_readiness::{record.area}")
            self.assertFalse(record.blocking_findings)
            self.assertTrue(record.readiness_record_implemented)
            self.assertFalse(record.configuration_mutation_implemented)
            self.assertFalse(record.provider_provisioning_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_review_rejects_mismatched_records_or_counts(self) -> None:
        review = build_production_readiness_review()
        payload = review.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionReadinessReview(**payload)

        payload = review.model_dump(mode="json")
        payload["guarded_finding_count"] += 1

        with self.assertRaisesRegex(ValueError, "guarded_finding_count must match"):
            ProductionReadinessReview(**payload)


if __name__ == "__main__":
    unittest.main()
