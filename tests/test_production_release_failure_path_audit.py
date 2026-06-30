import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ProductionReleaseFailurePathAuditRegistry,
    production_release_failure_path_audit_by_id,
    production_release_failure_path_audit_registry,
    production_release_failure_path_audits_for_check,
    production_release_failure_path_audits_for_surface,
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
EXPECTED_REQUIRED_CHECKS = (
    "node_level_failure_paths",
    "terminal_failure_routing",
    "provider_failures",
    "model_routing_failures",
    "stream_failures",
    "scheduling_failures",
    "retry_failures",
    "planning_helper_failures",
    "prompt_rendering_failures",
    "serialization_failures",
    "preview_workstation_frontend_backend_failures",
    "registry_import_loading_failures",
    "telemetry_observability_failures",
    "cache_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
EXPECTED_APPLICABLE_CHECKS = tuple(
    check for check in EXPECTED_REQUIRED_CHECKS if check != "cache_failures"
)
EXPECTED_NOT_APPLICABLE_CHECKS = ("cache_failures",)
REQUIRED_RECORD_FIELDS = {
    "audit_id",
    "check_kind",
    "checklist_source",
    "source_surface_ids",
    "evidence",
    "invariant_assertions",
    "failure_response_boundary",
    "audit_status",
    "blocked_runtime_behaviors",
    "checklist_item_applicable",
    "runtime_failure_path_audit_implemented",
    "runtime_failure_handler_creation_implemented",
    "terminal_failure_routing_mutation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "dependency_installation_implemented",
    "runtime_installation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "preview_rendering_execution_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReleaseFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = production_release_failure_path_audit_registry()

        self.assertEqual(
            registry.role,
            "production_release_failure_path_audit_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "production_release_failure_path_audit_registry.v1",
        )
        self.assertEqual(registry.checklist_source, "runtime/RUNTIME_FAILURE_PATH_AUDIT.md")
        self.assertEqual(
            registry.architecture_registry_serialization_version,
            "production_architecture_consistency_registry.v1",
        )
        self.assertEqual(registry.architecture_registry_record_count, 10)
        self.assertEqual(registry.source_surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.required_checks, EXPECTED_REQUIRED_CHECKS)
        self.assertEqual(registry.applicable_required_checks, EXPECTED_APPLICABLE_CHECKS)
        self.assertEqual(
            registry.not_applicable_required_checks,
            EXPECTED_NOT_APPLICABLE_CHECKS,
        )
        self.assertEqual(registry.check_kinds, EXPECTED_APPLICABLE_CHECKS)
        self.assertEqual(registry.record_count, 18)
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.all_applicable_checks_covered)
        self.assertTrue(registry.deployment_failure_boundary_preserved)
        self.assertTrue(registry.release_operation_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("runtime/RUNTIME_FAILURE_PATH_AUDIT.md", registry.authority_boundary)
        self.assertFalse(registry.runtime_failure_handler_creation_implemented)
        self.assertFalse(registry.terminal_failure_routing_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.dependency_installation_implemented)
        self.assertFalse(registry.runtime_installation_implemented)
        self.assertFalse(registry.package_build_executed)
        self.assertFalse(registry.deployment_execution_implemented)
        self.assertFalse(registry.asset_generation_implemented)
        self.assertFalse(registry.retrieval_execution_implemented)
        self.assertFalse(registry.preview_rendering_execution_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.hitl_request_emission_implemented)
        self.assertFalse(registry.human_review_request_implemented)
        self.assertFalse(registry.merge_push_tag_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = production_release_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("production_release_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "production_release_failure_path_audit_record.v1",
            )
            self.assertTrue(record.source_surface_ids)
            self.assertTrue(record.evidence)
            self.assertTrue(record.invariant_assertions)
            self.assertTrue(record.failure_response_boundary)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            self.assertFalse(record.runtime_failure_handler_creation_implemented)
            self.assertFalse(record.terminal_failure_routing_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.dependency_installation_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.preview_rendering_execution_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.refinement_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.hitl_request_emission_implemented)
            self.assertFalse(record.human_review_request_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_lookup_helpers_filter_by_id_check_and_surface(self) -> None:
        registry = production_release_failure_path_audit_registry()
        provider = production_release_failure_path_audit_by_id(
            "production_release_failure::provider_failures",
            registry,
        )
        missing = production_release_failure_path_audit_by_id("missing", registry)
        output_records = production_release_failure_path_audits_for_check(
            "generated_output_mutation_boundaries",
            registry,
        )
        deployment_records = production_release_failure_path_audits_for_surface(
            "production_deployment",
            registry,
        )

        self.assertIsNone(missing)
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.check_kind, "provider_failures")
        self.assertEqual(len(output_records), 1)
        self.assertGreaterEqual(len(deployment_records), 1)
        self.assertTrue(
            all("production_deployment" in record.source_surface_ids for record in deployment_records)
        )

    def test_registry_rejects_mismatched_records_or_checks(self) -> None:
        registry = production_release_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ("missing",) + tuple(payload["audit_ids"][1:])

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            ProductionReleaseFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["required_checks"] = tuple(payload["required_checks"][1:]) + (
            payload["required_checks"][0],
        )

        with self.assertRaisesRegex(ValueError, "required_checks"):
            ProductionReleaseFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_routing(self) -> None:
        request = AssistantRequest(
            query="Create a failure audit routing preservation scene.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )

        before = route_request(request)
        registry = production_release_failure_path_audit_registry()
        after = route_request(request)

        self.assertEqual(before.route, RouteName.GENERATE)
        self.assertEqual(after.route, before.route)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)


if __name__ == "__main__":
    unittest.main()
