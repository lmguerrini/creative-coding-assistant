import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AdaptiveExecutionFailurePathAuditRegistry,
    adaptive_execution_failure_path_audit_by_id,
    adaptive_execution_failure_path_audit_registry,
    adaptive_execution_failure_path_audits_for_check,
    adaptive_execution_failure_path_audits_for_surface,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "adaptive_hybrid_workflow_optimizer",
    "adaptive_escalation_optimizer",
    "agent_activation_optimizer",
    "adaptive_cost_quality_optimizer",
    "adaptive_latency_optimizer",
    "adaptive_execution_strategy_selection",
    "adaptive_execution_policy_engine",
    "dynamic_agent_allocation",
    "dynamic_resource_allocation",
    "workflow_self_tuning_policies",
    "execution_confidence_engine",
    "workflow_risk_engine",
    "creative_exploration_optimizer",
    "emergence_optimizer",
    "agent_diversity_optimizer",
    "reflection_budget_optimizer",
    "adaptive_policy_explainability",
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
    "policy_application_implemented",
    "execution_policy_application_implemented",
    "strategy_application_implemented",
    "routing_application_implemented",
    "risk_decision_application_implemented",
    "confidence_application_implemented",
    "self_tuning_application_implemented",
    "emergence_behavior_application_implemented",
    "agent_diversity_behavior_application_implemented",
    "reflection_budget_enforcement_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "agent_invocation_implemented",
    "agent_activation_implemented",
    "agent_instantiation_implemented",
    "runtime_agent_allocation_implemented",
    "resource_allocation_implemented",
    "runtime_resource_measurement_implemented",
    "budget_enforcement_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "graph_compilation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class AdaptiveExecutionFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = adaptive_execution_failure_path_audit_registry()

        self.assertEqual(
            registry.role,
            "adaptive_execution_failure_path_audit_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "adaptive_execution_failure_path_audit_registry.v1",
        )
        self.assertEqual(
            registry.checklist_source,
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
        )
        self.assertEqual(
            registry.architecture_registry_serialization_version,
            "adaptive_execution_architecture_consistency_registry.v1",
        )
        self.assertEqual(registry.architecture_registry_record_count, 17)
        self.assertEqual(registry.source_surface_ids, EXPECTED_SURFACE_IDS)
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
        self.assertTrue(registry.adaptive_policy_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("controlled adaptive policy", registry.authority_boundary)
        self.assertFalse(registry.policy_application_implemented)
        self.assertFalse(registry.execution_policy_application_implemented)
        self.assertFalse(registry.strategy_application_implemented)
        self.assertFalse(registry.routing_application_implemented)
        self.assertFalse(registry.risk_decision_application_implemented)
        self.assertFalse(registry.confidence_application_implemented)
        self.assertFalse(registry.self_tuning_application_implemented)
        self.assertFalse(registry.emergence_behavior_application_implemented)
        self.assertFalse(registry.agent_diversity_behavior_application_implemented)
        self.assertFalse(registry.reflection_budget_enforcement_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.automatic_provider_switching_implemented)
        self.assertFalse(registry.automatic_model_switching_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.agent_activation_implemented)
        self.assertFalse(registry.agent_instantiation_implemented)
        self.assertFalse(registry.runtime_agent_allocation_implemented)
        self.assertFalse(registry.resource_allocation_implemented)
        self.assertFalse(registry.runtime_resource_measurement_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.hitl_request_emission_implemented)
        self.assertFalse(registry.human_review_request_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.graph_compilation_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.refinement_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = adaptive_execution_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("adaptive_execution_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "adaptive_execution_failure_path_audit_record.v1",
            )
            self.assertTrue(record.source_surface_ids)
            self.assertTrue(record.evidence)
            self.assertTrue(record.invariant_assertions)
            self.assertTrue(record.failure_response_boundary)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            self.assertFalse(record.policy_application_implemented)
            self.assertFalse(record.execution_policy_application_implemented)
            self.assertFalse(record.strategy_application_implemented)
            self.assertFalse(record.routing_application_implemented)
            self.assertFalse(record.risk_decision_application_implemented)
            self.assertFalse(record.confidence_application_implemented)
            self.assertFalse(record.self_tuning_application_implemented)
            self.assertFalse(record.emergence_behavior_application_implemented)
            self.assertFalse(record.agent_diversity_behavior_application_implemented)
            self.assertFalse(record.reflection_budget_enforcement_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.automatic_provider_switching_implemented)
            self.assertFalse(record.automatic_model_switching_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.agent_activation_implemented)
            self.assertFalse(record.agent_instantiation_implemented)
            self.assertFalse(record.runtime_agent_allocation_implemented)
            self.assertFalse(record.resource_allocation_implemented)
            self.assertFalse(record.runtime_resource_measurement_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.hitl_request_emission_implemented)
            self.assertFalse(record.human_review_request_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.graph_compilation_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.refinement_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)
            for source_surface_id in record.source_surface_ids:
                self.assertIn(source_surface_id, EXPECTED_SURFACE_IDS)

    def test_lookup_helpers_are_stable(self) -> None:
        registry = adaptive_execution_failure_path_audit_registry()
        telemetry_records = adaptive_execution_failure_path_audits_for_check(
            "telemetry_observability_failures",
            registry,
        )
        telemetry_record = adaptive_execution_failure_path_audit_by_id(
            "adaptive_execution_failure::telemetry_observability_failures",
            registry,
        )
        policy_records = adaptive_execution_failure_path_audits_for_surface(
            "adaptive_policy_explainability",
            registry,
        )
        missing_record = adaptive_execution_failure_path_audit_by_id(
            "missing",
            registry,
        )
        missing_surface_records = adaptive_execution_failure_path_audits_for_surface(
            "missing",
            registry,
        )

        self.assertEqual(len(telemetry_records), 1)
        self.assertIs(telemetry_records[0], telemetry_record)
        self.assertIsNone(missing_record)
        self.assertEqual(missing_surface_records, ())
        self.assertIn(
            "terminal_failure_routing",
            tuple(record.check_kind for record in policy_records),
        )
        self.assertIn(
            "telemetry_observability_failures",
            tuple(record.check_kind for record in policy_records),
        )
        self.assertIn(
            "provider_model_routing_preservation",
            tuple(record.check_kind for record in policy_records),
        )

    def test_registry_rejects_mismatched_coverage(self) -> None:
        registry = adaptive_execution_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ["missing", *payload["audit_ids"][1:]]

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            AdaptiveExecutionFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_surface_ids"] = payload["source_surface_ids"][1:]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            AdaptiveExecutionFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = [
            "serialization_failures",
            *payload["check_kinds"][1:],
        ]

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            AdaptiveExecutionFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["records"][0]["source_surface_ids"] = ["unknown"]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            AdaptiveExecutionFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate adaptive execution failure path audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        adaptive_execution_failure_path_audit_registry()
        adaptive_execution_failure_path_audit_by_id(
            "adaptive_execution_failure::telemetry_observability_failures"
        )
        adaptive_execution_failure_path_audits_for_check(
            "serialization_failures"
        )
        adaptive_execution_failure_path_audits_for_surface(
            "adaptive_policy_explainability"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_failure_audit_metadata_does_not_declare_active_terms(self) -> None:
        registry = adaptive_execution_failure_path_audit_registry()
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
            "apply_policy(",
            "apply_execution_policy(",
            "apply_strategy(",
            "apply_routing(",
            "apply_risk_decision(",
            "apply_confidence(",
            "apply_self_tuning(",
            "apply_emergence(",
            "apply_agent_diversity(",
            "enforce_reflection_budget(",
            "route_provider(",
            "route_model(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "activate_agent(",
            "allocate_agent(",
            "allocate_resource(",
            "measure_runtime_resource(",
            "enforce_budget(",
            "emit_hitl_request(",
            "request_human_review(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "compile_graph(",
            "trigger_retry(",
            "trigger_refinement(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
