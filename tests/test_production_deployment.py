import unittest

from creative_coding_assistant.orchestration import (
    ProductionDeploymentPlan,
    build_production_deployment_plan,
    production_deployment_record_by_surface,
    production_deployment_records_for_status,
)

REQUIRED_SURFACES = (
    "backend_runtime_entrypoint",
    "frontend_runtime_entrypoint",
    "environment_configuration",
    "runtime_data_paths",
    "external_deployment_manifest",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "surface_id",
    "status",
    "source_refs",
    "required_items",
    "present_items",
    "missing_items",
    "deployment_notes",
    "blocked_runtime_behaviors",
    "deployment_record_implemented",
    "deployment_execution_implemented",
    "server_start_implemented",
    "dependency_installation_implemented",
    "package_build_executed",
    "container_image_build_implemented",
    "provider_provisioning_implemented",
    "environment_variable_mutation_implemented",
    "runtime_data_write_implemented",
    "workflow_execution_implemented",
    "provider_execution_implemented",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionDeploymentTests(unittest.TestCase):
    def test_deployment_plan_documents_local_runtime_and_external_gap(self) -> None:
        plan = build_production_deployment_plan()

        self.assertEqual(plan.role, "production_deployment")
        self.assertEqual(plan.serialization_version, "production_deployment_plan.v1")
        self.assertEqual(plan.backend_host, "127.0.0.1")
        self.assertEqual(plan.backend_port, 8000)
        self.assertEqual(
            plan.backend_paths, ("/api/assistant/stream", "/api/workspace/session")
        )
        self.assertEqual(plan.frontend_scripts, ("build", "start"))
        self.assertEqual(plan.surface_ids, REQUIRED_SURFACES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.deployment_status, "guarded")
        self.assertIn(
            "production_deployment::external_deployment_manifest",
            plan.guarded_record_ids,
        )
        self.assertIn("does not deploy services", plan.authority_boundary)
        self.assertTrue(plan.deployment_metadata_implemented)
        self.assertTrue(plan.local_backend_entrypoint_documented)
        self.assertTrue(plan.local_frontend_entrypoint_documented)
        self.assertTrue(plan.environment_configuration_documented)
        self.assertTrue(plan.runtime_data_paths_documented)
        self.assertTrue(plan.external_deployment_assumptions_documented)
        self.assertFalse(plan.deployment_execution_implemented)
        self.assertFalse(plan.server_start_implemented)
        self.assertFalse(plan.dependency_installation_implemented)
        self.assertFalse(plan.package_build_executed)
        self.assertFalse(plan.container_image_build_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.environment_variable_mutation_implemented)
        self.assertFalse(plan.runtime_data_write_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.merge_push_tag_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.metadata_only)

    def test_records_are_metadata_only_and_guard_external_manifest_absence(
        self,
    ) -> None:
        plan = build_production_deployment_plan()
        manifest = production_deployment_record_by_surface(
            "external_deployment_manifest",
            plan,
        )
        backend = production_deployment_record_by_surface(
            "backend_runtime_entrypoint",
            plan,
        )
        guarded = production_deployment_records_for_status("guarded", plan)

        self.assertIsNotNone(manifest)
        self.assertIsNotNone(backend)
        assert manifest is not None
        assert backend is not None
        self.assertEqual(len(guarded), 1)
        self.assertEqual(manifest.status, "guarded")
        self.assertIn("Dockerfile", manifest.present_items)
        self.assertIn("docker-compose.yml", manifest.present_items)
        self.assertIn("vercel.json", manifest.missing_items)
        self.assertEqual(backend.status, "ready")

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_deployment_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_deployment::{record.surface_id}",
            )
            self.assertTrue(record.deployment_record_implemented)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.server_start_implemented)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.container_image_build_implemented)
            self.assertFalse(record.provider_provisioning_implemented)
            self.assertFalse(record.environment_variable_mutation_implemented)
            self.assertFalse(record.runtime_data_write_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_plan_rejects_mismatched_records_or_status(self) -> None:
        plan = build_production_deployment_plan()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionDeploymentPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["deployment_status"] = "ready"

        with self.assertRaisesRegex(ValueError, "deployment_status must match"):
            ProductionDeploymentPlan(**payload)


if __name__ == "__main__":
    unittest.main()
