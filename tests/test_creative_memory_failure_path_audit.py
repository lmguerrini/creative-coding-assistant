import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    CreativeMemoryFailurePathAuditRegistry,
    creative_memory_failure_path_audit_by_id,
    creative_memory_failure_path_audit_registry,
    creative_memory_failure_path_audits_for_check,
    creative_memory_failure_path_audits_for_surface,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "long_term_creative_memory",
    "user_preferences",
    "style_profiles",
    "project_memory",
    "creative_dna",
    "personalization_engine",
    "session_memory_evolution",
    "artifact_history",
    "creative_lineage",
    "creative_ontology",
    "creative_memory_core_surface",
    "creative_memory_secondary_surface",
    "creative_memory_governance_safety",
)
EXPECTED_SOURCE_SERIALIZATION_VERSIONS = (
    "long_term_creative_memory_plan.v1",
    "user_preferences_plan.v1",
    "style_profile_plan.v1",
    "project_memory_plan.v1",
    "creative_dna_plan.v1",
    "personalization_engine_plan.v1",
    "session_memory_evolution_plan.v1",
    "artifact_history_plan.v1",
    "creative_lineage_plan.v1",
    "creative_ontology_plan.v1",
    "creative_memory_core_surface_plan.v1",
    "creative_memory_secondary_surface_plan.v1",
    "creative_memory_governance_plan.v1",
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
EXPECTED_APPLICABLE_CHECKS = (
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
    "registry_import_loading_failures",
    "telemetry_observability_failures",
    "budget_cost_prediction_failures",
    "workflow_state_integrity_after_failure",
    "provider_model_routing_preservation",
    "generated_output_mutation_boundaries",
    "passive_registry_activation_boundaries",
)
EXPECTED_NOT_APPLICABLE_CHECKS = (
    "preview_workstation_frontend_backend_failures",
    "cache_failures",
)
FALSE_FLAG_FIELDS = (
    "memory_storage_write_implemented",
    "memory_retrieval_execution_implemented",
    "memory_consolidation_execution_implemented",
    "preference_learning_execution_implemented",
    "user_model_application_implemented",
    "personalization_application_implemented",
    "creative_dna_application_implemented",
    "artifact_history_persistence_implemented",
    "creative_lineage_inference_implemented",
    "ontology_relationship_inference_implemented",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "automation_activation_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "runtime_probe_implemented",
    "local_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "telemetry_collection_implemented",
    "live_memory_observation_implemented",
    "live_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "failure_repair_implemented",
    "generated_output_evaluation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "replay_execution_implemented",
    "model_training_implemented",
    "runtime_mutation_implemented",
    "automatic_preference_mutation_implemented",
    "automatic_remediation_implemented",
    "artifact_execution_implemented",
    "dependency_installation_implemented",
    "graph_compilation_implemented",
    "prompt_rendering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
)
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
    "serialization_version",
    "metadata_only",
    *FALSE_FLAG_FIELDS,
}


class CreativeMemoryFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = creative_memory_failure_path_audit_registry()

        self.assertEqual(
            registry.role,
            "creative_memory_failure_path_audit_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "creative_memory_failure_path_audit_registry.v1",
        )
        self.assertEqual(
            registry.checklist_source,
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
        )
        self.assertEqual(registry.source_surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(
            registry.source_serialization_versions,
            EXPECTED_SOURCE_SERIALIZATION_VERSIONS,
        )
        self.assertEqual(registry.required_checks, EXPECTED_REQUIRED_CHECKS)
        self.assertEqual(
            registry.applicable_required_checks,
            EXPECTED_APPLICABLE_CHECKS,
        )
        self.assertEqual(
            registry.not_applicable_required_checks,
            EXPECTED_NOT_APPLICABLE_CHECKS,
        )
        self.assertEqual(registry.check_kinds, EXPECTED_APPLICABLE_CHECKS)
        self.assertEqual(registry.record_count, 17)
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.all_applicable_checks_covered)
        self.assertTrue(registry.memory_storage_boundary_preserved)
        self.assertTrue(registry.preference_learning_boundary_preserved)
        self.assertTrue(registry.governance_safety_boundary_preserved)
        self.assertTrue(registry.runtime_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("V6.2 creative memory", registry.authority_boundary)
        self.assertIn(
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
            registry.authority_boundary,
        )
        for field_name in FALSE_FLAG_FIELDS:
            self.assertFalse(getattr(registry, field_name))
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = creative_memory_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("creative_memory_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "creative_memory_failure_path_audit_record.v1",
            )
            self.assertTrue(record.source_surface_ids)
            self.assertTrue(record.evidence)
            self.assertTrue(record.invariant_assertions)
            self.assertTrue(record.failure_response_boundary)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            for field_name in FALSE_FLAG_FIELDS:
                self.assertFalse(getattr(record, field_name))
            self.assertTrue(record.metadata_only)
            for source_surface_id in record.source_surface_ids:
                self.assertIn(source_surface_id, EXPECTED_SURFACE_IDS)

    def test_lookup_helpers_filter_by_id_check_and_surface(self) -> None:
        registry = creative_memory_failure_path_audit_registry()
        telemetry_records = creative_memory_failure_path_audits_for_check(
            "telemetry_observability_failures",
            registry,
        )
        telemetry_record = creative_memory_failure_path_audit_by_id(
            "creative_memory_failure::telemetry_observability_failures",
            registry,
        )
        governance_records = creative_memory_failure_path_audits_for_surface(
            "creative_memory_governance_safety",
            registry,
        )
        missing_record = creative_memory_failure_path_audit_by_id(
            "missing",
            registry,
        )
        missing_surface_records = creative_memory_failure_path_audits_for_surface(
            "missing",
            registry,
        )

        self.assertEqual(len(telemetry_records), 1)
        self.assertIs(telemetry_records[0], telemetry_record)
        self.assertIsNone(missing_record)
        self.assertEqual(missing_surface_records, ())
        self.assertIn(
            "terminal_failure_routing",
            tuple(record.check_kind for record in governance_records),
        )
        self.assertIn(
            "provider_model_routing_preservation",
            tuple(record.check_kind for record in governance_records),
        )

    def test_registry_rejects_mismatched_coverage(self) -> None:
        registry = creative_memory_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ["missing", *payload["audit_ids"][1:]]

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            CreativeMemoryFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_serialization_versions"][0] = "missing.v1"

        with self.assertRaisesRegex(ValueError, "source_serialization_versions"):
            CreativeMemoryFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = [
            "serialization_failures",
            *payload["check_kinds"][1:],
        ]

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            CreativeMemoryFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["records"][0]["source_surface_ids"] = ["unknown"]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            CreativeMemoryFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate creative memory failure audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        creative_memory_failure_path_audit_registry()
        creative_memory_failure_path_audit_by_id(
            "creative_memory_failure::telemetry_observability_failures"
        )
        creative_memory_failure_path_audits_for_check("serialization_failures")
        creative_memory_failure_path_audits_for_surface(
            "creative_memory_governance_safety"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_failure_audit_metadata_does_not_declare_active_terms(self) -> None:
        registry = creative_memory_failure_path_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.required_checks,
                *registry.applicable_required_checks,
                *registry.not_applicable_required_checks,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.audit_id,
                        record.check_kind,
                        record.failure_response_boundary,
                        *record.evidence,
                        *record.invariant_assertions,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_memory(",
            "retrieve_memory(",
            "consolidate_memory(",
            "learn_preference(",
            "apply_user_model(",
            "apply_personalization(",
            "apply_creative_dna(",
            "persist_artifact_history(",
            "infer_creative_lineage(",
            "infer_ontology(",
            "enforce_governance(",
            "enforce_safety(",
            "activate_automation(",
            "emit_hitl_request(",
            "request_human_review(",
            "route_provider(",
            "route_model(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_runtime(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "collect_telemetry(",
            "observe_memory(",
            "observe_failure(",
            "classify_live_error(",
            "route_terminal_failure(",
            "handle_failure(",
            "repair_failure(",
            "evaluate_generated_output(",
            "execute_workflow(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "trigger_retry(",
            "trigger_refinement(",
            "execute_replay(",
            "train_model(",
            "mutate_runtime(",
            "mutate_preference(",
            "remediate_failure(",
            "execute_artifact(",
            "install_dependency(",
            "compile_graph(",
            "render_prompt(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
