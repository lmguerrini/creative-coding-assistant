import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    SelfEvolutionFailurePathAuditRecord,
    SelfEvolutionFailurePathAuditRegistry,
    build_self_evolution_governance,
    route_request,
    self_evolution_failure_path_audit_by_id,
    self_evolution_failure_path_audit_registry,
    self_evolution_failure_path_audits_for_check,
    self_evolution_failure_path_audits_for_surface,
)
from creative_coding_assistant.orchestration.self_evolution_common import (
    CROSS_CUTTING_CONTRACTS,
    UPSTREAM_CAPABILITIES,
)
from creative_coding_assistant.orchestration.self_evolution_core_surface import (
    CORE_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.self_evolution_failure_path_audit import (
    APPLICABLE_FAILURE_PATH_CHECKS,
    FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
    NOT_APPLICABLE_FAILURE_PATH_CHECKS,
    REQUIRED_FAILURE_PATH_CHECKS,
)

EXPECTED_SOURCE_SURFACE_IDS = (
    "prompt_evolution",
    "workflow_evolution",
    "benchmark_engine",
    "quality_trends",
    "cost_trends",
    "autonomous_optimization_suggestions",
    "architecture_evolution_engine",
    "workflow_mutation_engine",
    "strategy_evolution_engine",
    "agent_evolution_policies",
    "routing_evolution_policies",
    "memory_evolution_policies",
    "retrieval_evolution_policies",
    "self_improvement_proposals",
    "creative_evolution_policies",
    "taste_evolution_engine",
    "reasoning_evolution_engine",
    "improvement_ranking_engine",
    "cost_benefit_analysis",
    "risk_analysis",
    "expected_impact_estimator",
    "rollback_strategy_generator",
    "self_evolution_core_surface",
    "self_evolution_secondary_surface",
    "self_evolution_governance_safety",
)
EXPECTED_SOURCE_SERIALIZATION_VERSIONS = (
    "prompt_evolution_plan.v1",
    "workflow_evolution_plan.v1",
    "benchmark_engine_plan.v1",
    "quality_trends_plan.v1",
    "cost_trends_plan.v1",
    "autonomous_optimization_suggestions_plan.v1",
    "architecture_evolution_engine_plan.v1",
    "workflow_mutation_engine_plan.v1",
    "strategy_evolution_engine_plan.v1",
    "agent_evolution_policies_plan.v1",
    "routing_evolution_policies_plan.v1",
    "memory_evolution_policies_plan.v1",
    "retrieval_evolution_policies_plan.v1",
    "self_improvement_proposals_plan.v1",
    "creative_evolution_policies_plan.v1",
    "taste_evolution_engine_plan.v1",
    "reasoning_evolution_engine_plan.v1",
    "improvement_ranking_engine_plan.v1",
    "cost_benefit_analysis_plan.v1",
    "risk_analysis_plan.v1",
    "expected_impact_estimator_plan.v1",
    "rollback_strategy_generator_plan.v1",
    "self_evolution_core_surface.v1",
    "self_evolution_secondary_surface.v1",
    "self_evolution_governance_plan.v1",
)
FALSE_FLAG_FIELDS = (
    "audit_enforcement_implemented",
    "live_failure_observation_implemented",
    "live_error_classification_implemented",
    "terminal_failure_routing_implemented",
    "failure_handling_implemented",
    "failure_repair_implemented",
    "automatic_remediation_implemented",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "hitl_request_emission_implemented",
    "human_review_request_implemented",
    "hitl_decision_application_implemented",
    "automation_activation_implemented",
    "proposal_application_implemented",
    "rollback_execution_implemented",
    "report_artifact_generation_implemented",
    "storage_write_implemented",
    "prompt_rendering_implemented",
    "prompt_mutation_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "routing_mutation_implemented",
    "provider_model_routing_implemented",
    "memory_mutation_implemented",
    "retrieval_mutation_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "telemetry_collection_implemented",
    "runtime_probe_implemented",
    "dependency_installation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "generated_output_evaluation_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
)


class SelfEvolutionFailurePathAuditTests(unittest.TestCase):
    def test_registry_covers_runtime_failure_path_checklist(self) -> None:
        registry = self_evolution_failure_path_audit_registry()

        self.assertEqual(
            registry.role,
            "self_evolution_failure_path_audit_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "self_evolution_failure_path_audit_registry.v1",
        )
        self.assertEqual(
            registry.checklist_source,
            "runtime/RUNTIME_FAILURE_PATH_AUDIT.md",
        )
        self.assertEqual(registry.source_surface_ids, EXPECTED_SOURCE_SURFACE_IDS)
        self.assertEqual(
            registry.source_serialization_versions,
            EXPECTED_SOURCE_SERIALIZATION_VERSIONS,
        )
        self.assertEqual(registry.required_checks, REQUIRED_FAILURE_PATH_CHECKS)
        self.assertEqual(
            registry.applicable_required_checks,
            APPLICABLE_FAILURE_PATH_CHECKS,
        )
        self.assertEqual(
            registry.not_applicable_required_checks,
            NOT_APPLICABLE_FAILURE_PATH_CHECKS,
        )
        self.assertEqual(registry.check_kinds, APPLICABLE_FAILURE_PATH_CHECKS)
        self.assertEqual(registry.record_count, 17)
        self.assertEqual(registry.covered_roadmap_items, CORE_ROADMAP_ITEMS)
        self.assertEqual(registry.covered_roadmap_item_count, 22)
        self.assertEqual(registry.proposal_count, 110)
        self.assertEqual(registry.governance_boundary_count, 22)
        self.assertEqual(registry.upstream_capabilities, UPSTREAM_CAPABILITIES)
        self.assertEqual(registry.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
        self.assertEqual(
            registry.blocked_runtime_behaviors,
            FAILURE_PATH_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(registry.metadata_only_rule_satisfied)
        self.assertTrue(registry.active_behavior_rule_satisfied)
        self.assertTrue(registry.all_applicable_checks_covered)
        self.assertTrue(registry.all_roadmap_items_traceable)
        self.assertTrue(registry.all_proposals_traceable)
        self.assertTrue(registry.upstream_signal_sources_traceable)
        self.assertTrue(registry.governance_safety_boundary_preserved)
        self.assertTrue(registry.runtime_failure_boundary_preserved)
        self.assertTrue(registry.workflow_state_integrity_boundary_preserved)
        self.assertTrue(registry.provider_model_routing_preserved)
        self.assertTrue(registry.generated_output_mutation_boundary_preserved)
        self.assertTrue(registry.passive_registry_activation_boundary_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("V6.5 Self Evolution", registry.authority_boundary)
        self.assertIn("metadata only", registry.authority_boundary)
        for field_name in FALSE_FLAG_FIELDS:
            self.assertFalse(getattr(registry, field_name))
        self.assertFalse(registry.applied_audit_fix_ids)
        self.assertFalse(registry.handled_failure_ids)
        self.assertFalse(registry.routed_terminal_failure_ids)
        self.assertFalse(registry.applied_evolution_proposal_ids)
        self.assertFalse(registry.emitted_hitl_request_ids)
        self.assertFalse(registry.generated_report_artifact_ids)
        self.assertFalse(registry.written_storage_record_ids)
        self.assertFalse(registry.provider_execution_ids)
        self.assertFalse(registry.mutated_output_ids)
        self.assertTrue(registry.metadata_only)

    def test_records_are_metadata_only_and_trace_all_surfaces(self) -> None:
        registry = self_evolution_failure_path_audit_registry()

        for record in registry.records:
            self.assertEqual(
                record.audit_id,
                f"self_evolution_failure_path_audit::{record.check_kind}",
            )
            self.assertEqual(record.checklist_source, registry.checklist_source)
            self.assertEqual(record.source_surface_ids, registry.source_surface_ids)
            self.assertEqual(record.covered_roadmap_items, CORE_ROADMAP_ITEMS)
            self.assertEqual(record.proposal_count, 110)
            self.assertEqual(record.upstream_capabilities, UPSTREAM_CAPABILITIES)
            self.assertEqual(record.cross_cutting_contracts, CROSS_CUTTING_CONTRACTS)
            self.assertEqual(record.audit_status, "pass")
            self.assertTrue(record.checklist_item_applicable)
            self.assertTrue(record.runtime_failure_path_audit_implemented)
            self.assertTrue(record.self_evolution_orchestration_layer_verified)
            self.assertTrue(record.v6_signal_sources_integrated)
            self.assertTrue(record.proposal_traceability_verified)
            self.assertTrue(record.governance_boundary_verified)
            self.assertTrue(record.metadata_only_rule_satisfied)
            self.assertIn("metadata coverage only", record.failure_response_boundary)
            for field_name in FALSE_FLAG_FIELDS:
                self.assertFalse(getattr(record, field_name))
            self.assertFalse(record.applied_audit_fix_ids)
            self.assertFalse(record.handled_failure_ids)
            self.assertFalse(record.routed_terminal_failure_ids)
            self.assertFalse(record.applied_evolution_proposal_ids)
            self.assertFalse(record.emitted_hitl_request_ids)
            self.assertFalse(record.generated_report_artifact_ids)
            self.assertFalse(record.written_storage_record_ids)
            self.assertFalse(record.provider_execution_ids)
            self.assertFalse(record.mutated_output_ids)
            self.assertTrue(record.metadata_only)

        record = self_evolution_failure_path_audit_by_id(
            "self_evolution_failure_path_audit::provider_failures",
            registry,
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.check_kind, "provider_failures")
        self.assertEqual(
            len(
                self_evolution_failure_path_audits_for_check(
                    "provider_failures",
                    registry,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                self_evolution_failure_path_audits_for_surface(
                    "prompt_evolution",
                    registry,
                )
            ),
            17,
        )

    def test_registry_rejects_mismatched_or_mutating_payloads(self) -> None:
        registry = self_evolution_failure_path_audit_registry()
        payload = registry.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            SelfEvolutionFailurePathAuditRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["applied_audit_fix_ids"] = ("audit_fix",)

        with self.assertRaisesRegex(
            ValueError,
            "failure path registry mutation ids must be empty",
        ):
            SelfEvolutionFailurePathAuditRegistry(**payload)

        record_payload = registry.records[0].model_dump(mode="json")
        record_payload["handled_failure_ids"] = ("failure",)

        with self.assertRaisesRegex(
            ValueError,
            "failure path audit mutation ids must be empty",
        ):
            SelfEvolutionFailurePathAuditRecord(**record_payload)

    def test_failure_path_audit_does_not_change_routing_or_provider(self) -> None:
        request = AssistantRequest(
            query="Review the self evolution failure path audit.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        governance_plan = build_self_evolution_governance()
        registry = self_evolution_failure_path_audit_registry(governance_plan)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(registry.proposal_count, 110)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")


if __name__ == "__main__":
    unittest.main()
