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
    KnowledgeEvolutionGovernancePlan,
    build_knowledge_evolution_governance,
    knowledge_evolution_governance_boundaries_for_status,
    knowledge_evolution_governance_boundary_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "knowledge_evolution_secondary_surface",
    "learning_governance",
    "hitl_budget_gate",
    "routing_explainability",
)
EXPECTED_ROADMAP_ITEMS = (
    "Automatic KB Updates",
    "Documentation Intelligence",
    "Embedding Refresh",
    "Retrieval Evolution",
    "Ranking Optimization",
    "Knowledge Health Monitoring",
    "Knowledge Quality Scoring",
    "Knowledge Gap Detection",
    "Knowledge Conflict Resolver",
    "Knowledge Drift Detection",
    "Source Reliability Engine",
    "Knowledge Consolidation",
    "Knowledge Lifecycle Management",
    "Knowledge Provenance Evolution",
    "Knowledge Versioning",
    "Knowledge Snapshot Engine",
    "Knowledge Rollback",
    "Knowledge Freshness Tracking",
    "Knowledge Trust Score",
)
REQUIRED_BOUNDARY_FIELDS = {
    "boundary_id",
    "boundary_kind",
    "status",
    "priority",
    "route_name",
    "task_type",
    "execution_mode_id",
    "governed_area",
    "roadmap_items",
    "roadmap_item_count",
    "source_plan_roles",
    "source_serialization_versions",
    "source_item_ids",
    "source_item_count",
    "hitl_requirement_count",
    "explainability_signal_count",
    "no_automation_weight",
    "safety_weight",
    "governance_score",
    "hitl_required_before_governance_application",
    "governed_surface_summary",
    "review_requirement",
    "explainability_requirement",
    "no_automation_boundary",
    "safety_boundary",
    "governance_tags",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_evolution_governance_implemented",
    "governance_boundary_metadata_implemented",
    "hitl_boundary_metadata_implemented",
    "explainability_boundary_metadata_implemented",
    "no_automation_boundary_metadata_implemented",
    "all_roadmap_items_traceable",
    "v5_v6_governance_sources_used",
    "governance_policy_enforcement_implemented",
    "safety_policy_enforcement_implemented",
    "hitl_request_emitted",
    "human_input_request_implemented",
    "automation_activation_implemented",
    "secondary_surface_activation_implemented",
    "adaptive_learning_application_implemented",
    "execution_policy_application_implemented",
    "automatic_kb_update_execution_implemented",
    "documentation_fetch_execution_implemented",
    "embedding_refresh_execution_implemented",
    "retrieval_execution_implemented",
    "ranking_mutation_implemented",
    "knowledge_health_monitoring_execution_implemented",
    "quality_score_computation_implemented",
    "knowledge_gap_detection_execution_implemented",
    "knowledge_conflict_resolution_execution_implemented",
    "knowledge_drift_detection_execution_implemented",
    "source_reliability_scoring_execution_implemented",
    "knowledge_consolidation_execution_implemented",
    "knowledge_lifecycle_management_execution_implemented",
    "provenance_graph_mutation_implemented",
    "version_graph_mutation_implemented",
    "knowledge_snapshot_engine_execution_implemented",
    "knowledge_rollback_execution_implemented",
    "freshness_scan_execution_implemented",
    "trust_score_computation_implemented",
    "kb_storage_write_implemented",
    "source_record_update_implemented",
    "routing_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class KnowledgeEvolutionGovernanceTests(unittest.TestCase):
    def test_plan_builds_governance_and_safety_metadata(self) -> None:
        plan = build_knowledge_evolution_governance(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_evolution_governance_safety")
        self.assertEqual(
            plan.serialization_version,
            "knowledge_evolution_governance_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "knowledge_evolution_secondary_surface_plan.v1",
                "learning_governance_plan.v1",
                "hitl_budget_gate_plan.v1",
                "routing_explainability_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 19)
        self.assertEqual(len(plan.source_item_ids), 19)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.boundary_count, 5)
        self.assertEqual(plan.guarded_boundary_count, 5)
        self.assertEqual(plan.hitl_required_boundary_count, 5)
        self.assertFalse(plan.applied_governance_boundary_ids)
        self.assertFalse(plan.enforced_safety_policy_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.requested_human_input_ids)
        self.assertFalse(plan.activated_automation_ids)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.applied_learning_ids)
        self.assertFalse(plan.applied_execution_policy_ids)
        self.assertFalse(plan.executed_knowledge_operation_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_governance_posture, "guarded")
        self.assertIn("HITL", plan.authority_boundary)
        self.assertIn("no-automation boundaries", plan.authority_boundary)
        self.assertTrue(plan.knowledge_evolution_governance_implemented)
        self.assertTrue(plan.governance_boundary_metadata_implemented)
        self.assertTrue(plan.hitl_boundary_metadata_implemented)
        self.assertTrue(plan.explainability_boundary_metadata_implemented)
        self.assertTrue(plan.no_automation_boundary_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.v5_v6_governance_sources_used)
        self.assertFalse(plan.governance_policy_enforcement_implemented)
        self.assertFalse(plan.safety_policy_enforcement_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.human_input_request_implemented)
        self.assertFalse(plan.automation_activation_implemented)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.adaptive_learning_application_implemented)
        self.assertFalse(plan.execution_policy_application_implemented)
        self.assertFalse(plan.automatic_kb_update_execution_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.knowledge_health_monitoring_execution_implemented)
        self.assertFalse(plan.quality_score_computation_implemented)
        self.assertFalse(plan.knowledge_gap_detection_execution_implemented)
        self.assertFalse(plan.knowledge_conflict_resolution_execution_implemented)
        self.assertFalse(plan.knowledge_drift_detection_execution_implemented)
        self.assertFalse(plan.source_reliability_scoring_execution_implemented)
        self.assertFalse(plan.knowledge_consolidation_execution_implemented)
        self.assertFalse(plan.knowledge_lifecycle_management_execution_implemented)
        self.assertFalse(plan.provenance_graph_mutation_implemented)
        self.assertFalse(plan.version_graph_mutation_implemented)
        self.assertFalse(plan.knowledge_snapshot_engine_execution_implemented)
        self.assertFalse(plan.knowledge_rollback_execution_implemented)
        self.assertFalse(plan.freshness_scan_execution_implemented)
        self.assertFalse(plan.trust_score_computation_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.routing_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_boundaries_score_governance_without_enforcement(self) -> None:
        plan = build_knowledge_evolution_governance(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for boundary in plan.boundaries:
            dumped = boundary.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BOUNDARY_FIELDS)
            self.assertEqual(
                boundary.serialization_version,
                "knowledge_evolution_governance_boundary.v1",
            )
            self.assertEqual(boundary.route_name, RouteName.GENERATE)
            self.assertEqual(
                boundary.boundary_id,
                f"knowledge_evolution_governance::{boundary.boundary_kind}",
            )
            self.assertEqual(boundary.roadmap_item_count, len(boundary.roadmap_items))
            self.assertEqual(boundary.source_item_count, len(boundary.source_item_ids))
            self.assertTrue(set(boundary.source_item_ids).issubset(source_items))
            self.assertTrue(set(boundary.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                boundary.governance_score,
                min(
                    1000,
                    max(
                        0,
                        boundary.source_item_count * 18
                        + boundary.hitl_requirement_count * 65
                        + boundary.explainability_signal_count * 25
                        + boundary.no_automation_weight
                        + boundary.safety_weight,
                    ),
                ),
            )
            self.assertEqual(boundary.status, "guarded")
            self.assertEqual(boundary.priority, "guarded")
            self.assertIn("governance_safety", boundary.governance_tags)
            self.assertIn("automation_activation", boundary.blocked_runtime_behaviors)
            self.assertIn("hitl_request_emission", boundary.blocked_runtime_behaviors)
            self.assertTrue(boundary.hitl_required_before_governance_application)
            self.assertTrue(boundary.knowledge_evolution_governance_implemented)
            self.assertTrue(boundary.governance_boundary_metadata_implemented)
            self.assertTrue(boundary.hitl_boundary_metadata_implemented)
            self.assertTrue(boundary.explainability_boundary_metadata_implemented)
            self.assertTrue(boundary.no_automation_boundary_metadata_implemented)
            self.assertTrue(boundary.all_roadmap_items_traceable)
            self.assertFalse(boundary.governance_policy_enforcement_implemented)
            self.assertFalse(boundary.safety_policy_enforcement_implemented)
            self.assertFalse(boundary.hitl_request_emitted)
            self.assertFalse(boundary.human_input_request_implemented)
            self.assertFalse(boundary.automation_activation_implemented)
            self.assertFalse(boundary.secondary_surface_activation_implemented)
            self.assertFalse(boundary.adaptive_learning_application_implemented)
            self.assertFalse(boundary.execution_policy_application_implemented)
            self.assertFalse(boundary.automatic_kb_update_execution_implemented)
            self.assertFalse(boundary.documentation_fetch_execution_implemented)
            self.assertFalse(boundary.embedding_refresh_execution_implemented)
            self.assertFalse(boundary.retrieval_execution_implemented)
            self.assertFalse(boundary.ranking_mutation_implemented)
            self.assertFalse(boundary.knowledge_health_monitoring_execution_implemented)
            self.assertFalse(boundary.quality_score_computation_implemented)
            self.assertFalse(boundary.knowledge_gap_detection_execution_implemented)
            self.assertFalse(
                boundary.knowledge_conflict_resolution_execution_implemented
            )
            self.assertFalse(boundary.knowledge_drift_detection_execution_implemented)
            self.assertFalse(boundary.source_reliability_scoring_execution_implemented)
            self.assertFalse(boundary.knowledge_consolidation_execution_implemented)
            self.assertFalse(
                boundary.knowledge_lifecycle_management_execution_implemented
            )
            self.assertFalse(boundary.provenance_graph_mutation_implemented)
            self.assertFalse(boundary.version_graph_mutation_implemented)
            self.assertFalse(boundary.knowledge_snapshot_engine_execution_implemented)
            self.assertFalse(boundary.knowledge_rollback_execution_implemented)
            self.assertFalse(boundary.freshness_scan_execution_implemented)
            self.assertFalse(boundary.trust_score_computation_implemented)
            self.assertFalse(boundary.kb_storage_write_implemented)
            self.assertFalse(boundary.source_record_update_implemented)
            self.assertFalse(boundary.routing_application_implemented)
            self.assertFalse(boundary.provider_model_routing_implemented)
            self.assertFalse(boundary.provider_execution_implemented)
            self.assertFalse(boundary.agent_invocation_implemented)
            self.assertFalse(boundary.workflow_control_implemented)
            self.assertFalse(boundary.workflow_graph_mutation_implemented)
            self.assertFalse(boundary.workflow_execution_implemented)
            self.assertFalse(boundary.generated_output_mutation_implemented)
            self.assertFalse(boundary.runtime_evolution_implemented)
            self.assertTrue(boundary.advisory_only)

        no_automation = knowledge_evolution_governance_boundary_by_id(
            "knowledge_evolution_governance::recovery_trust_no_automation_governance",
            plan,
        )
        self.assertIsNotNone(no_automation)
        assert no_automation is not None
        self.assertEqual(no_automation.status, "guarded")
        self.assertEqual(no_automation.explainability_signal_count, 6)
        self.assertEqual(no_automation.roadmap_item_count, 4)
        self.assertEqual(
            len(knowledge_evolution_governance_boundaries_for_status("guarded", plan)),
            5,
        )

    def test_plan_rejects_mismatched_governance_metadata(self) -> None:
        plan = build_knowledge_evolution_governance()
        payload = plan.model_dump(mode="json")
        payload["boundary_ids"] = ("missing",) + tuple(payload["boundary_ids"][1:])

        with self.assertRaisesRegex(ValueError, "boundary_ids must match"):
            KnowledgeEvolutionGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_governance_score must match",
        ):
            KnowledgeEvolutionGovernancePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["emitted_hitl_request_ids"] = (plan.boundary_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "emitted_hitl_request_ids must remain empty",
        ):
            KnowledgeEvolutionGovernancePlan(**payload)

    def test_governance_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge evolution governance.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_evolution_governance(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_governance_does_not_declare_enforcement_terms(self) -> None:
        plan = build_knowledge_evolution_governance(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for boundary in plan.boundaries
                    for field in (
                        boundary.boundary_id,
                        boundary.boundary_kind,
                        boundary.status,
                        boundary.priority,
                        boundary.governed_area,
                        *boundary.roadmap_items,
                        *boundary.source_plan_roles,
                        *boundary.source_serialization_versions,
                        *boundary.source_item_ids,
                        boundary.governed_surface_summary,
                        boundary.review_requirement,
                        boundary.explainability_requirement,
                        boundary.no_automation_boundary,
                        boundary.safety_boundary,
                        *boundary.governance_tags,
                        *boundary.advisory_actions,
                        *boundary.evidence,
                        *boundary.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_governance(",
            "enforce_safety(",
            "emit_hitl_request(",
            "request_human_input(",
            "activate_automation(",
            "activate_secondary_surface(",
            "apply_adaptive_learning(",
            "apply_execution_policy(",
            "execute_knowledge_operation(",
            "execute_automatic_kb_update(",
            "fetch_documentation(",
            "refresh_embedding(",
            "execute_retrieval(",
            "mutate_ranking(",
            "compute_quality_score(",
            "execute_gap_detection(",
            "resolve_conflict(",
            "detect_drift(",
            "score_source_reliability(",
            "consolidate_knowledge(",
            "manage_lifecycle(",
            "mutate_provenance_graph(",
            "mutate_version_graph(",
            "execute_snapshot(",
            "execute_rollback(",
            "execute_freshness_scan(",
            "compute_trust_score(",
            "write_kb_storage(",
            "update_source_record(",
            "apply_routing(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
