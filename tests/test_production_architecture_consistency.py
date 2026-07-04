import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ProductionArchitectureConsistencyRegistry,
    production_architecture_consistency_by_surface,
    production_architecture_consistency_records_for_layer,
    production_architecture_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "production_release_final_optimization",
    "production_release_packaging",
    "production_release_candidate",
    "production_demo_assets",
    "production_deployment",
    "production_readiness_review",
    "production_creative_readiness_review",
    "production_architecture_freeze",
    "production_release_audit",
    "production_final_hardening",
)
EXPECTED_ARCHITECTURE_LAYERS = (
    "release_readiness_boundary",
    "packaging_deployment_boundary",
    "demo_creative_boundary",
    "readiness_audit_boundary",
    "architecture_release_control_boundary",
    "hardening_boundary",
)
REQUIRED_VERSION_RULES = (
    "v5_6_surface_role_declared",
    "serialization_version_declared",
    "metadata_only_declared",
    "v5_architecture_boundary_preserved",
    "v4_boundary_compatibility_confirmed",
    "provider_model_routing_not_applied",
    "provider_execution_not_applied",
    "workflow_execution_not_applied",
    "generated_output_mutation_blocked",
    "release_operations_human_controlled",
    "hitl_not_emitted",
    "runtime_evolution_not_applied",
)
REQUIRED_RECORD_FIELDS = {
    "surface_id",
    "architecture_layer",
    "architecture_stage",
    "source_role",
    "source_serialization_version",
    "source_count_field",
    "source_count",
    "validated_version_rules",
    "passive_boundary_flags",
    "source_blocked_runtime_behaviors",
    "source_active_runtime_flags",
    "missing_coverage_items",
    "source_metadata_only_declared",
    "v5_architecture_consistency_confirmed",
    "v4_boundary_compatibility_confirmed",
    "version_runtime_rules_confirmed",
    "architecture_consistency_status",
    "architecture_expansion_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "dependency_installation_implemented",
    "runtime_installation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "release_artifact_creation_implemented",
    "hitl_request_emitted",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionArchitectureConsistencyTests(unittest.TestCase):
    def test_registry_covers_v5_6_production_surfaces(self) -> None:
        registry = production_architecture_consistency_registry()

        self.assertEqual(
            registry.role,
            "production_architecture_consistency_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "production_architecture_consistency_registry.v1",
        )
        self.assertEqual(
            registry.architecture_stage,
            "v5_6_architecture_consistency_pass",
        )
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.architecture_layers, EXPECTED_ARCHITECTURE_LAYERS)
        self.assertEqual(registry.record_count, 10)
        self.assertTrue(registry.all_surfaces_covered)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.v5_architecture_consistency_confirmed)
        self.assertTrue(registry.v4_boundaries_preserved)
        self.assertTrue(registry.version_runtime_rules_confirmed)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("runtime_evolution_not_applied", registry.validated_version_rules)
        self.assertIn(
            "provider_or_model_routing_mutation",
            registry.blocked_runtime_behaviors,
        )
        self.assertIn(
            "V5.6 production release architecture consistency",
            registry.authority_boundary,
        )
        self.assertFalse(registry.architecture_expansion_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.dependency_installation_implemented)
        self.assertFalse(registry.runtime_installation_implemented)
        self.assertFalse(registry.package_build_executed)
        self.assertFalse(registry.deployment_execution_implemented)
        self.assertFalse(registry.asset_generation_implemented)
        self.assertFalse(registry.retrieval_execution_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.release_artifact_creation_implemented)
        self.assertFalse(registry.hitl_request_emitted)
        self.assertFalse(registry.merge_push_tag_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.metadata_only)

    def test_records_are_metadata_only_and_source_aligned(self) -> None:
        registry = production_architecture_consistency_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_architecture_consistency_record.v1",
            )
            self.assertEqual(record.architecture_stage, registry.architecture_stage)
            self.assertTrue(record.source_serialization_version.endswith(".v1"))
            self.assertIn(
                record.source_count_field,
                {"record_count", "asset_count", "guarded_finding_count"},
            )
            self.assertGreaterEqual(record.source_count, 0)
            self.assertEqual(record.validated_version_rules, REQUIRED_VERSION_RULES)
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertTrue(record.source_blocked_runtime_behaviors)
            self.assertFalse(record.source_active_runtime_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.source_metadata_only_declared)
            self.assertTrue(record.v5_architecture_consistency_confirmed)
            self.assertTrue(record.v4_boundary_compatibility_confirmed)
            self.assertTrue(record.version_runtime_rules_confirmed)
            self.assertEqual(record.architecture_consistency_status, "pass")
            self.assertFalse(record.architecture_expansion_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.release_artifact_creation_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_lookup_and_layer_filters_are_stable(self) -> None:
        registry = production_architecture_consistency_registry()
        deployment = production_architecture_consistency_by_surface(
            "production_deployment",
            registry,
        )
        missing = production_architecture_consistency_by_surface("missing", registry)
        packaging_records = production_architecture_consistency_records_for_layer(
            "packaging_deployment_boundary",
            registry,
        )
        creative_records = production_architecture_consistency_records_for_layer(
            "demo_creative_boundary",
            registry,
        )

        self.assertIsNone(missing)
        self.assertIsNotNone(deployment)
        assert deployment is not None
        self.assertEqual(deployment.source_role, "production_deployment")
        self.assertEqual(
            tuple(record.surface_id for record in packaging_records),
            ("production_release_packaging", "production_deployment"),
        )
        self.assertEqual(
            tuple(record.surface_id for record in creative_records),
            ("production_demo_assets", "production_creative_readiness_review"),
        )

    def test_registry_rejects_mismatched_records_or_rules(self) -> None:
        registry = production_architecture_consistency_registry()
        payload = registry.model_dump(mode="json")
        payload["surface_ids"] = ("missing",) + tuple(payload["surface_ids"][1:])

        with self.assertRaisesRegex(ValueError, "surface_ids must match"):
            ProductionArchitectureConsistencyRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["validated_version_rules"] = tuple(
            payload["validated_version_rules"][1:]
        ) + (payload["validated_version_rules"][0],)

        with self.assertRaisesRegex(ValueError, "validated_version_rules"):
            ProductionArchitectureConsistencyRegistry(**payload)

    def test_consistency_pass_does_not_change_routing(self) -> None:
        request = AssistantRequest(
            query="Create a production architecture consistency test scene.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )

        before = route_request(request)
        registry = production_architecture_consistency_registry()
        after = route_request(request)

        self.assertEqual(before.route, RouteName.GENERATE)
        self.assertEqual(after.route, before.route)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)


if __name__ == "__main__":
    unittest.main()
