import unittest

from creative_coding_assistant.orchestration import (
    ProductionReleaseAudit,
    build_production_release_audit,
    production_release_audit_record_by_area,
    production_release_audit_records_for_status,
)

REQUIRED_SOURCE_SURFACES = (
    "production_release_candidate",
    "production_release_packaging",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
    "production_architecture_freeze",
)
REQUIRED_AREAS = (
    "validation_gate_audit",
    "release_candidate_audit",
    "production_readiness_audit",
    "creative_readiness_audit",
    "deployment_packaging_audit",
    "architecture_freeze_audit",
    "release_control_audit",
)
PENDING_RELEASE_GATES = (
    "codex_engineering_audit_pending",
    "runtime_failure_path_audit_pending",
    "final_validation_pending",
    "cumulative_local_app_smoke_test_pending",
    "capability_acceptance_test_pending",
    "runtime_evolution_review_pending",
    "merge_push_tag_gate_pending",
)
REQUIRED_RECORD_FIELDS = {
    "audit_id",
    "audit_area",
    "audit_status",
    "source_surface_ids",
    "source_serialization_versions",
    "evidence",
    "pass_findings",
    "guarded_findings",
    "blocking_findings",
    "required_followups",
    "release_blocker",
    "blocked_runtime_behaviors",
    "release_audit_record_implemented",
    "release_artifact_creation_implemented",
    "package_build_executed",
    "dependency_installation_implemented",
    "deployment_execution_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "generated_output_mutation_implemented",
    "persistent_storage_write_implemented",
    "hitl_request_emitted",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReleaseAuditTests(unittest.TestCase):
    def test_release_audit_aggregates_guarded_release_posture(self) -> None:
        audit = build_production_release_audit()

        self.assertEqual(audit.role, "production_release_audit")
        self.assertEqual(audit.serialization_version, "production_release_audit.v1")
        self.assertEqual(audit.version, "V5.6.0")
        self.assertEqual(audit.target_branch, "feature/production-release")
        self.assertEqual(audit.target_tag, "v5.6.0")
        self.assertEqual(audit.source_surface_ids, REQUIRED_SOURCE_SURFACES)
        self.assertEqual(audit.pending_release_gate_ids, PENDING_RELEASE_GATES)
        self.assertEqual(audit.audit_areas, REQUIRED_AREAS)
        self.assertEqual(audit.record_count, 7)
        self.assertEqual(audit.release_audit_status, "guarded")
        self.assertEqual(audit.release_audit_outcome, "pass_with_guarded_assumptions")
        self.assertEqual(audit.blocked_audit_ids, ())
        self.assertEqual(audit.blocking_finding_count, 0)
        self.assertEqual(audit.release_blocker_count, 0)
        self.assertGreaterEqual(audit.guarded_finding_count, 1)
        self.assertTrue(audit.release_audit_can_proceed_to_hardening)
        self.assertIn("does not create release artifacts", audit.authority_boundary)
        self.assertTrue(audit.release_audit_implemented)
        self.assertTrue(audit.validation_gate_audit_implemented)
        self.assertTrue(audit.release_candidate_audit_implemented)
        self.assertTrue(audit.production_readiness_audit_implemented)
        self.assertTrue(audit.creative_readiness_audit_implemented)
        self.assertTrue(audit.deployment_packaging_audit_implemented)
        self.assertTrue(audit.architecture_freeze_audit_implemented)
        self.assertTrue(audit.release_control_audit_implemented)
        self.assertFalse(audit.release_artifact_creation_implemented)
        self.assertFalse(audit.package_build_executed)
        self.assertFalse(audit.dependency_installation_implemented)
        self.assertFalse(audit.deployment_execution_implemented)
        self.assertFalse(audit.provider_model_routing_implemented)
        self.assertFalse(audit.provider_execution_implemented)
        self.assertFalse(audit.workflow_execution_implemented)
        self.assertFalse(audit.workflow_control_implemented)
        self.assertFalse(audit.asset_generation_implemented)
        self.assertFalse(audit.retrieval_execution_implemented)
        self.assertFalse(audit.generated_output_mutation_implemented)
        self.assertFalse(audit.persistent_storage_write_implemented)
        self.assertFalse(audit.hitl_request_emitted)
        self.assertFalse(audit.merge_push_tag_implemented)
        self.assertFalse(audit.runtime_evolution_implemented)
        self.assertTrue(audit.metadata_only)

    def test_records_preserve_release_boundaries(self) -> None:
        audit = build_production_release_audit()
        validation = production_release_audit_record_by_area(
            "validation_gate_audit",
            audit,
        )
        deployment = production_release_audit_record_by_area(
            "deployment_packaging_audit",
            audit,
        )
        architecture = production_release_audit_record_by_area(
            "architecture_freeze_audit",
            audit,
        )
        controls = production_release_audit_record_by_area(
            "release_control_audit",
            audit,
        )
        guarded = production_release_audit_records_for_status("guarded", audit)

        self.assertIsNotNone(validation)
        self.assertIsNotNone(deployment)
        self.assertIsNotNone(architecture)
        self.assertIsNotNone(controls)
        assert validation is not None
        assert deployment is not None
        assert architecture is not None
        assert controls is not None
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn("final_validation_pending", validation.guarded_findings)
        self.assertIn("merge_push_tag_gate_pending", controls.guarded_findings)
        self.assertIn(
            "production_deployment::external_deployment_manifest",
            deployment.guarded_findings,
        )
        self.assertIn("no_architecture_expansion_required", architecture.pass_findings)

        for record in audit.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_release_audit_record.v1",
            )
            self.assertEqual(
                record.audit_id,
                f"production_release_audit::{record.audit_area}",
            )
            self.assertEqual(
                len(record.source_surface_ids),
                len(record.source_serialization_versions),
            )
            self.assertFalse(record.blocking_findings)
            self.assertFalse(record.release_blocker)
            self.assertTrue(record.release_audit_record_implemented)
            self.assertFalse(record.release_artifact_creation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_audit_rejects_mismatched_records_or_counts(self) -> None:
        audit = build_production_release_audit()
        payload = audit.model_dump(mode="json")
        payload["audit_ids"] = ("missing",) + tuple(payload["audit_ids"][1:])

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            ProductionReleaseAudit(**payload)

        payload = audit.model_dump(mode="json")
        payload["guarded_finding_count"] += 1

        with self.assertRaisesRegex(ValueError, "guarded_finding_count must match"):
            ProductionReleaseAudit(**payload)

        payload = audit.model_dump(mode="json")
        payload["release_audit_outcome"] = "blocked"

        with self.assertRaisesRegex(ValueError, "release_audit_outcome must match"):
            ProductionReleaseAudit(**payload)


if __name__ == "__main__":
    unittest.main()
