import unittest

from creative_coding_assistant.orchestration import (
    ProductionPackagingPlan,
    build_production_packaging_plan,
    production_packaging_record_by_surface,
    production_packaging_records_for_status,
)

REQUIRED_SURFACES = (
    "python_package_metadata",
    "frontend_package_metadata",
    "environment_template",
    "runtime_data_placeholders",
    "release_scripts",
)
REQUIRED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "CCA_OPENAI_API_KEY",
    "CCA_OPENAI_MODEL",
    "CCA_OPENAI_EMBEDDING_MODEL",
    "CCA_DEFAULT_GENERATION_PROVIDER",
    "CCA_DEFAULT_DOMAIN",
    "CCA_DEFAULT_MODE",
    "CCA_CHROMA_PERSIST_DIR",
    "CCA_ARTIFACT_DIR",
    "CCA_EVAL_DATA_PATH",
    "CCA_EVAL_RAGAS_RESULTS_PATH",
    "CCA_LOG_LEVEL",
}
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "surface_id",
    "status",
    "source_paths",
    "required_items",
    "present_items",
    "missing_items",
    "evidence",
    "packaging_actions",
    "blocked_runtime_behaviors",
    "packaging_record_implemented",
    "dependency_installation_implemented",
    "package_build_executed",
    "archive_creation_implemented",
    "container_image_build_implemented",
    "provider_provisioning_implemented",
    "environment_variable_mutation_implemented",
    "runtime_data_write_implemented",
    "workflow_execution_implemented",
    "provider_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "merge_push_tag_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReleasePackagingTests(unittest.TestCase):
    def test_packaging_plan_reads_repository_metadata(self) -> None:
        plan = build_production_packaging_plan()

        self.assertEqual(plan.role, "production_release_packaging")
        self.assertEqual(
            plan.serialization_version,
            "production_release_packaging_plan.v1",
        )
        self.assertEqual(plan.python_package_name, "creative-coding-assistant")
        self.assertEqual(plan.python_package_version, "0.1.0")
        self.assertEqual(plan.frontend_package_name, "@creative-coding-assistant/nextjs-client")
        self.assertEqual(plan.frontend_package_version, "0.1.0")
        self.assertTrue(plan.frontend_private_package)
        self.assertEqual(plan.packaging_commands, ("python -m build", "npm --prefix clients/nextjs run build"))
        self.assertEqual(plan.surface_ids, REQUIRED_SURFACES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.guarded_record_ids, ())
        self.assertEqual(plan.packaging_status, "ready")
        self.assertTrue(REQUIRED_ENV_KEYS.issubset(set(plan.environment_variable_keys)))
        self.assertEqual(
            plan.runtime_placeholder_paths,
            (
                "data/chroma/.gitkeep",
                "data/artifacts/.gitkeep",
                "data/eval/.gitkeep",
            ),
        )
        self.assertIn("does not install dependencies", plan.authority_boundary)
        self.assertTrue(plan.packaging_metadata_implemented)
        self.assertTrue(plan.python_package_review_implemented)
        self.assertTrue(plan.frontend_package_review_implemented)
        self.assertTrue(plan.environment_template_review_implemented)
        self.assertTrue(plan.runtime_placeholder_review_implemented)
        self.assertTrue(plan.release_script_review_implemented)
        self.assertFalse(plan.dependency_installation_implemented)
        self.assertFalse(plan.package_build_executed)
        self.assertFalse(plan.archive_creation_implemented)
        self.assertFalse(plan.container_image_build_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.environment_variable_mutation_implemented)
        self.assertFalse(plan.runtime_data_write_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertFalse(plan.merge_push_tag_implemented)
        self.assertTrue(plan.metadata_only)

    def test_records_are_read_only_and_complete(self) -> None:
        plan = build_production_packaging_plan()
        frontend = production_packaging_record_by_surface(
            "frontend_package_metadata",
            plan,
        )
        env = production_packaging_record_by_surface("environment_template", plan)
        ready_records = production_packaging_records_for_status("ready", plan)

        self.assertIsNotNone(frontend)
        self.assertIsNotNone(env)
        assert frontend is not None
        assert env is not None
        self.assertEqual(len(ready_records), 5)
        self.assertIn("build", frontend.present_items)
        self.assertIn("start", frontend.present_items)
        self.assertIn("OPENAI_API_KEY", env.present_items)
        self.assertFalse(env.missing_items)

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_release_packaging_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_packaging::{record.surface_id}",
            )
            self.assertEqual(record.status, "ready")
            self.assertFalse(record.missing_items)
            self.assertTrue(record.packaging_record_implemented)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.archive_creation_implemented)
            self.assertFalse(record.container_image_build_implemented)
            self.assertFalse(record.provider_provisioning_implemented)
            self.assertFalse(record.environment_variable_mutation_implemented)
            self.assertFalse(record.runtime_data_write_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertTrue(record.metadata_only)

    def test_plan_rejects_mismatched_records_or_commands(self) -> None:
        plan = build_production_packaging_plan()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionPackagingPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["packaging_commands"] = ("python -m build", "npm install")

        with self.assertRaisesRegex(ValueError, "packaging_commands must remain"):
            ProductionPackagingPlan(**payload)


if __name__ == "__main__":
    unittest.main()
