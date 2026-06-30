import unittest

from creative_coding_assistant.orchestration import (
    ProductionFinalHardening,
    build_production_final_hardening,
    production_hardening_record_by_area,
    production_hardening_records_for_status,
)

REQUIRED_AREAS = (
    "configuration_hardening",
    "deployment_hardening",
    "release_gate_hardening",
    "creative_demo_hardening",
    "architecture_boundary_hardening",
    "failure_path_hardening",
)
REQUIRED_RECORD_FIELDS = {
    "hardening_id",
    "hardening_area",
    "hardening_status",
    "source_surface_ids",
    "source_serialization_versions",
    "evidence",
    "hardening_actions",
    "guarded_findings",
    "blocking_findings",
    "release_blocker",
    "blocked_runtime_behaviors",
    "hardening_record_implemented",
    "hardening_action_execution_implemented",
    "configuration_mutation_implemented",
    "provider_provisioning_implemented",
    "dependency_installation_implemented",
    "runtime_installation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "release_artifact_creation_implemented",
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


class ProductionFinalHardeningTests(unittest.TestCase):
    def test_final_hardening_records_guarded_release_actions(self) -> None:
        hardening = build_production_final_hardening()

        self.assertEqual(hardening.role, "production_final_hardening")
        self.assertEqual(
            hardening.serialization_version,
            "production_final_hardening.v1",
        )
        self.assertEqual(hardening.hardening_areas, REQUIRED_AREAS)
        self.assertEqual(hardening.record_count, 6)
        self.assertEqual(hardening.final_hardening_status, "guarded")
        self.assertEqual(
            hardening.final_hardening_outcome,
            "guarded_ready_for_consistency_pass",
        )
        self.assertEqual(hardening.blocked_hardening_ids, ())
        self.assertEqual(hardening.blocking_finding_count, 0)
        self.assertEqual(hardening.release_blocker_count, 0)
        self.assertGreater(hardening.hardening_action_count, 0)
        self.assertGreaterEqual(hardening.guarded_finding_count, 1)
        self.assertTrue(hardening.can_proceed_to_architecture_consistency_pass)
        self.assertIn("does not mutate configuration", hardening.authority_boundary)
        self.assertTrue(hardening.final_hardening_implemented)
        self.assertTrue(hardening.configuration_hardening_implemented)
        self.assertTrue(hardening.deployment_hardening_implemented)
        self.assertTrue(hardening.release_gate_hardening_implemented)
        self.assertTrue(hardening.creative_demo_hardening_implemented)
        self.assertTrue(hardening.architecture_boundary_hardening_implemented)
        self.assertTrue(hardening.failure_path_hardening_implemented)
        self.assertFalse(hardening.hardening_action_execution_implemented)
        self.assertFalse(hardening.configuration_mutation_implemented)
        self.assertFalse(hardening.provider_provisioning_implemented)
        self.assertFalse(hardening.dependency_installation_implemented)
        self.assertFalse(hardening.runtime_installation_implemented)
        self.assertFalse(hardening.package_build_executed)
        self.assertFalse(hardening.deployment_execution_implemented)
        self.assertFalse(hardening.release_artifact_creation_implemented)
        self.assertFalse(hardening.provider_model_routing_implemented)
        self.assertFalse(hardening.provider_execution_implemented)
        self.assertFalse(hardening.workflow_execution_implemented)
        self.assertFalse(hardening.workflow_control_implemented)
        self.assertFalse(hardening.asset_generation_implemented)
        self.assertFalse(hardening.retrieval_execution_implemented)
        self.assertFalse(hardening.generated_output_mutation_implemented)
        self.assertFalse(hardening.persistent_storage_write_implemented)
        self.assertFalse(hardening.hitl_request_emitted)
        self.assertFalse(hardening.merge_push_tag_implemented)
        self.assertFalse(hardening.runtime_evolution_implemented)
        self.assertTrue(hardening.metadata_only)

    def test_records_carry_guarded_findings_without_applying_actions(self) -> None:
        hardening = build_production_final_hardening()
        configuration = production_hardening_record_by_area(
            "configuration_hardening",
            hardening,
        )
        deployment = production_hardening_record_by_area(
            "deployment_hardening",
            hardening,
        )
        gates = production_hardening_record_by_area(
            "release_gate_hardening",
            hardening,
        )
        architecture = production_hardening_record_by_area(
            "architecture_boundary_hardening",
            hardening,
        )
        guarded = production_hardening_records_for_status("guarded", hardening)

        self.assertIsNotNone(configuration)
        self.assertIsNotNone(deployment)
        self.assertIsNotNone(gates)
        self.assertIsNotNone(architecture)
        assert configuration is not None
        assert deployment is not None
        assert gates is not None
        assert architecture is not None
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "production_readiness::configuration_readiness",
            configuration.guarded_findings,
        )
        self.assertIn(
            "production_deployment::external_deployment_manifest",
            deployment.guarded_findings,
        )
        self.assertIn("final_validation_pending", gates.guarded_findings)
        self.assertIn("runtime_evolution_review_pending", gates.guarded_findings)
        self.assertIn("missing_api_key", architecture.guarded_findings)

        for record in hardening.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_hardening_record.v1",
            )
            self.assertEqual(
                record.hardening_id,
                f"production_hardening::{record.hardening_area}",
            )
            self.assertEqual(
                len(record.source_surface_ids),
                len(record.source_serialization_versions),
            )
            self.assertFalse(record.blocking_findings)
            self.assertFalse(record.release_blocker)
            self.assertTrue(record.hardening_record_implemented)
            self.assertFalse(record.hardening_action_execution_implemented)
            self.assertFalse(record.configuration_mutation_implemented)
            self.assertFalse(record.provider_provisioning_implemented)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.release_artifact_creation_implemented)
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

    def test_hardening_rejects_mismatched_records_or_counts(self) -> None:
        hardening = build_production_final_hardening()
        payload = hardening.model_dump(mode="json")
        payload["hardening_ids"] = ("missing",) + tuple(payload["hardening_ids"][1:])

        with self.assertRaisesRegex(ValueError, "hardening_ids must match"):
            ProductionFinalHardening(**payload)

        payload = hardening.model_dump(mode="json")
        payload["hardening_action_count"] += 1

        with self.assertRaisesRegex(ValueError, "hardening_action_count must match"):
            ProductionFinalHardening(**payload)

        payload = hardening.model_dump(mode="json")
        payload["final_hardening_outcome"] = "blocked"

        with self.assertRaisesRegex(ValueError, "final_hardening_outcome must match"):
            ProductionFinalHardening(**payload)


if __name__ == "__main__":
    unittest.main()
