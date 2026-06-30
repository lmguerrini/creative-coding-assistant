import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AdaptiveLearningFailurePathAuditRegistry,
    adaptive_learning_failure_path_audit_by_id,
    adaptive_learning_failure_path_audit_registry,
    adaptive_learning_failure_path_audits_for_check,
    adaptive_learning_failure_path_audits_for_surface,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "adaptive_learning_engine",
    "workflow_success_tracking",
    "failure_tracking",
    "strategy_learning",
    "technique_learning",
    "runtime_learning",
    "routing_learning",
    "artifact_learning",
    "evaluation_learning",
    "continuous_improvement_signals",
    "success_pattern_discovery",
    "failure_pattern_discovery",
    "learning_governance",
)
EXPECTED_SOURCE_SERIALIZATION_VERSIONS = (
    "adaptive_learning_plan.v1",
    "workflow_success_tracking_plan.v1",
    "failure_tracking_plan.v1",
    "strategy_learning_plan.v1",
    "technique_learning_plan.v1",
    "runtime_learning_plan.v1",
    "routing_learning_plan.v1",
    "artifact_learning_plan.v1",
    "evaluation_learning_plan.v1",
    "continuous_improvement_signal_plan.v1",
    "success_pattern_discovery_plan.v1",
    "failure_pattern_discovery_plan.v1",
    "learning_governance_plan.v1",
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
    "learning_memory_persistence_implemented",
    "learning_feedback_application_implemented",
    "learning_policy_update_implemented",
    "learning_policy_enforcement_implemented",
    "strategy_mutation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "runtime_probe_implemented",
    "local_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "budget_enforcement_implemented",
    "telemetry_collection_implemented",
    "live_success_observation_implemented",
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
    "technique_application_implemented",
    "runtime_selection_implemented",
    "artifact_execution_implemented",
    "dependency_installation_implemented",
    "preview_behavior_change_implemented",
    "graph_compilation_implemented",
    "prompt_rendering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "success_metric_persistence_implemented",
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


class AdaptiveLearningFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = adaptive_learning_failure_path_audit_registry()

        self.assertEqual(
            registry.role,
            "adaptive_learning_failure_path_audit_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "adaptive_learning_failure_path_audit_registry.v1",
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
        self.assertTrue(registry.learning_memory_boundary_preserved)
        self.assertTrue(registry.feedback_policy_boundary_preserved)
        self.assertTrue(registry.runtime_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("V6.1 adaptive learning", registry.authority_boundary)
        self.assertIn(
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
            registry.authority_boundary,
        )
        for field_name in FALSE_FLAG_FIELDS:
            self.assertFalse(getattr(registry, field_name))
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = adaptive_learning_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("adaptive_learning_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "adaptive_learning_failure_path_audit_record.v1",
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
        registry = adaptive_learning_failure_path_audit_registry()
        telemetry_records = adaptive_learning_failure_path_audits_for_check(
            "telemetry_observability_failures",
            registry,
        )
        telemetry_record = adaptive_learning_failure_path_audit_by_id(
            "adaptive_learning_failure::telemetry_observability_failures",
            registry,
        )
        governance_records = adaptive_learning_failure_path_audits_for_surface(
            "learning_governance",
            registry,
        )
        missing_record = adaptive_learning_failure_path_audit_by_id(
            "missing",
            registry,
        )
        missing_surface_records = adaptive_learning_failure_path_audits_for_surface(
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
        registry = adaptive_learning_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ["missing", *payload["audit_ids"][1:]]

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            AdaptiveLearningFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_serialization_versions"][0] = "missing.v1"

        with self.assertRaisesRegex(ValueError, "source_serialization_versions"):
            AdaptiveLearningFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = [
            "serialization_failures",
            *payload["check_kinds"][1:],
        ]

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            AdaptiveLearningFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["records"][0]["source_surface_ids"] = ["unknown"]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            AdaptiveLearningFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate adaptive learning failure audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        adaptive_learning_failure_path_audit_registry()
        adaptive_learning_failure_path_audit_by_id(
            "adaptive_learning_failure::telemetry_observability_failures"
        )
        adaptive_learning_failure_path_audits_for_check("serialization_failures")
        adaptive_learning_failure_path_audits_for_surface("learning_governance")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_failure_audit_metadata_does_not_declare_active_terms(self) -> None:
        registry = adaptive_learning_failure_path_audit_registry()
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
            "persist_learning_memory(",
            "apply_learning_feedback(",
            "update_learning_policy(",
            "enforce_learning_policy(",
            "mutate_strategy(",
            "route_provider(",
            "route_model(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "probe_runtime(",
            "download_model(",
            "invoke_agent(",
            "allocate_resource(",
            "emit_hitl_request(",
            "request_human_review(",
            "enforce_budget(",
            "collect_telemetry(",
            "observe_success(",
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
            "apply_technique(",
            "select_runtime(",
            "execute_artifact(",
            "install_dependency(",
            "change_preview_behavior(",
            "compile_graph(",
            "render_prompt(",
            "mutate_prompt(",
            "write_storage(",
            "persist_success_metric(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
