import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ResearchFailurePathAuditRegistry,
    research_failure_path_audit_by_id,
    research_failure_path_audit_registry,
    research_failure_path_audits_for_check,
    research_failure_path_audits_for_surface,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "research_planner",
    "research_decomposer",
    "paper_research",
    "web_research",
    "cross_source_comparison",
    "knowledge_distillation",
    "automatic_kb_enrichment",
    "research_reports",
    "research_memory",
    "source_validation_engine",
    "source_credibility_engine",
    "contradiction_detection",
    "research_confidence_engine",
    "research_gap_discovery",
    "research_recommendation_engine",
    "research_execution_policy",
    "research_hitl_policies",
    "creative_research_engine",
    "cross_domain_inspiration_discovery",
    "research_core_surface",
    "research_secondary_surface",
    "research_governance_safety",
)
EXPECTED_SOURCE_SERIALIZATION_VERSIONS = (
    "research_planner_plan.v1",
    "research_decomposer_plan.v1",
    "paper_research_plan.v1",
    "web_research_plan.v1",
    "cross_source_comparison_plan.v1",
    "knowledge_distillation_plan.v1",
    "automatic_kb_enrichment_plan.v1",
    "research_report_plan.v1",
    "research_memory_plan.v1",
    "source_validation_plan.v1",
    "source_credibility_plan.v1",
    "contradiction_detection_plan.v1",
    "research_confidence_plan.v1",
    "research_gap_plan.v1",
    "research_recommendation_plan.v1",
    "research_execution_policy_plan.v1",
    "research_hitl_policy_plan.v1",
    "creative_research_plan.v1",
    "cross_domain_inspiration_plan.v1",
    "research_core_surface_plan.v1",
    "research_secondary_surface_plan.v1",
    "research_governance_plan.v1",
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
    "research_execution_implemented",
    "research_plan_mutation_implemented",
    "research_task_creation_implemented",
    "paper_research_execution_implemented",
    "web_research_execution_implemented",
    "external_source_fetch_implemented",
    "web_browsing_implemented",
    "paper_download_implemented",
    "cross_source_comparison_execution_implemented",
    "knowledge_distillation_execution_implemented",
    "kb_enrichment_execution_implemented",
    "kb_storage_write_implemented",
    "research_report_generation_implemented",
    "research_memory_write_implemented",
    "source_validation_execution_implemented",
    "source_credibility_scoring_execution_implemented",
    "contradiction_detection_execution_implemented",
    "research_confidence_scoring_execution_implemented",
    "research_gap_discovery_execution_implemented",
    "research_recommendation_generation_implemented",
    "research_execution_policy_application_implemented",
    "hitl_request_emission_implemented",
    "hitl_decision_application_implemented",
    "creative_output_generation_implemented",
    "inspiration_discovery_execution_implemented",
    "live_cross_domain_search_implemented",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "automation_activation_implemented",
    "human_review_request_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "runtime_probe_implemented",
    "local_model_download_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "telemetry_collection_implemented",
    "live_research_observation_implemented",
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
    "automatic_remediation_implemented",
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


class ResearchFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = research_failure_path_audit_registry()

        self.assertEqual(registry.role, "research_failure_path_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "research_failure_path_audit_registry.v1",
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
        self.assertTrue(registry.all_roadmap_surfaces_traceable)
        self.assertTrue(registry.external_source_boundary_preserved)
        self.assertTrue(registry.research_execution_boundary_preserved)
        self.assertTrue(registry.knowledge_storage_boundary_preserved)
        self.assertTrue(registry.governance_safety_boundary_preserved)
        self.assertTrue(registry.runtime_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("V6.4 research", registry.authority_boundary)
        self.assertIn(
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
            registry.authority_boundary,
        )
        for field_name in FALSE_FLAG_FIELDS:
            self.assertFalse(getattr(registry, field_name))
        self.assertTrue(registry.metadata_only)

    def test_records_are_passive_and_source_scoped(self) -> None:
        registry = research_failure_path_audit_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertTrue(record.audit_id.startswith("research_failure::"))
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertIn(record.check_kind, EXPECTED_APPLICABLE_CHECKS)
            self.assertEqual(
                record.serialization_version,
                "research_failure_path_audit_record.v1",
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
        registry = research_failure_path_audit_registry()
        telemetry_records = research_failure_path_audits_for_check(
            "telemetry_observability_failures",
            registry,
        )
        telemetry_record = research_failure_path_audit_by_id(
            "research_failure::telemetry_observability_failures",
            registry,
        )
        governance_records = research_failure_path_audits_for_surface(
            "research_governance_safety",
            registry,
        )
        missing_record = research_failure_path_audit_by_id("missing", registry)
        missing_surface_records = research_failure_path_audits_for_surface(
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
        registry = research_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["audit_ids"] = ["missing", *payload["audit_ids"][1:]]

        with self.assertRaisesRegex(ValueError, "audit_ids must match"):
            ResearchFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["source_serialization_versions"][0] = "missing.v1"

        with self.assertRaisesRegex(ValueError, "source_serialization_versions"):
            ResearchFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["check_kinds"] = [
            "serialization_failures",
            *payload["check_kinds"][1:],
        ]

        with self.assertRaisesRegex(ValueError, "check_kinds"):
            ResearchFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["records"][0]["source_surface_ids"] = ["unknown"]

        with self.assertRaisesRegex(ValueError, "source_surface_ids"):
            ResearchFailurePathAuditRegistry(**payload)

    def test_failure_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate research failure audit metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        research_failure_path_audit_registry()
        research_failure_path_audit_by_id(
            "research_failure::telemetry_observability_failures"
        )
        research_failure_path_audits_for_check("serialization_failures")
        research_failure_path_audits_for_surface("research_governance_safety")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_failure_audit_metadata_does_not_declare_active_terms(self) -> None:
        registry = research_failure_path_audit_registry()
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
            "execute_research(",
            "mutate_research_plan(",
            "create_research_task(",
            "execute_paper_research(",
            "execute_web_research(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "execute_cross_source_comparison(",
            "execute_knowledge_distillation(",
            "execute_kb_enrichment(",
            "write_kb_storage(",
            "generate_research_report(",
            "write_research_memory(",
            "execute_source_validation(",
            "score_source_credibility(",
            "execute_contradiction_detection(",
            "score_research_confidence(",
            "discover_research_gap(",
            "generate_research_recommendation(",
            "apply_research_execution_policy(",
            "emit_hitl_request(",
            "apply_hitl_decision(",
            "generate_creative_output(",
            "execute_inspiration_discovery(",
            "perform_live_cross_domain_search(",
            "enforce_governance(",
            "enforce_safety(",
            "activate_automation(",
            "request_human_review(",
            "apply_routing(",
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
            "observe_research(",
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
            "remediate_failure(",
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
