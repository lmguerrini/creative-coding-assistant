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
    KnowledgeEvolutionSecondarySurfacePlan,
    build_knowledge_evolution_secondary_surface,
    knowledge_evolution_secondary_surface_entries_for_confidence,
    knowledge_evolution_secondary_surface_entries_for_status,
    knowledge_evolution_secondary_surface_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "knowledge_evolution_core_surface",
    "adaptive_learning_engine",
    "adaptive_execution_policy_engine",
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
REQUIRED_ENTRY_FIELDS = {
    "secondary_surface_id",
    "surface_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "surface_axis",
    "roadmap_items",
    "roadmap_item_count",
    "source_plan_roles",
    "source_serialization_versions",
    "source_item_ids",
    "source_item_count",
    "surface_summary",
    "roadmap_traceability_score",
    "source_composition_score",
    "governance_alignment_score",
    "v5_v6_foundation_score",
    "activation_risk_score",
    "governance_weight",
    "secondary_surface_score",
    "hitl_required_before_secondary_surface_activation",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "secondary_surface_implemented",
    "secondary_surface_metadata_implemented",
    "all_roadmap_items_traceable",
    "v5_policy_foundation_used",
    "v6_learning_foundation_used",
    "v6_knowledge_core_foundation_used",
    "all_sources_metadata_only",
    "secondary_surface_activation_implemented",
    "adaptive_learning_application_implemented",
    "adaptive_execution_policy_application_implemented",
    "knowledge_operation_execution_implemented",
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


class KnowledgeEvolutionSecondarySurfaceTests(unittest.TestCase):
    def test_plan_builds_secondary_surface_metadata(self) -> None:
        plan = build_knowledge_evolution_secondary_surface(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_evolution_secondary_surface")
        self.assertEqual(
            plan.serialization_version,
            "knowledge_evolution_secondary_surface_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "knowledge_evolution_core_surface_plan.v1",
                "adaptive_learning_plan.v1",
                "adaptive_execution_policy_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 15)
        self.assertEqual(len(plan.source_item_ids), 15)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 19)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 2)
        self.assertEqual(plan.guarded_entry_count, 2)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.applied_learning_ids)
        self.assertFalse(plan.applied_policy_ids)
        self.assertFalse(plan.executed_knowledge_operation_ids)
        self.assertFalse(plan.mutated_retrieval_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_secondary_surface_posture, "guarded")
        self.assertIn("V5 controlled execution policy", plan.authority_boundary)
        self.assertIn("does not activate secondary surfaces", plan.authority_boundary)
        self.assertTrue(plan.secondary_surface_implemented)
        self.assertTrue(plan.secondary_surface_metadata_implemented)
        self.assertTrue(plan.all_roadmap_items_traceable)
        self.assertTrue(plan.v5_policy_foundation_used)
        self.assertTrue(plan.v6_learning_foundation_used)
        self.assertTrue(plan.v6_knowledge_core_foundation_used)
        self.assertTrue(plan.all_sources_metadata_only)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.adaptive_learning_application_implemented)
        self.assertFalse(plan.adaptive_execution_policy_application_implemented)
        self.assertFalse(plan.knowledge_operation_execution_implemented)
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
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_secondary_surface_without_activation(self) -> None:
        plan = build_knowledge_evolution_secondary_surface(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for entry in plan.entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "knowledge_evolution_secondary_surface_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.secondary_surface_id,
                f"knowledge_evolution_secondary::{entry.surface_kind}",
            )
            self.assertEqual(entry.roadmap_item_count, len(entry.roadmap_items))
            self.assertEqual(entry.source_item_count, len(entry.source_item_ids))
            self.assertTrue(set(entry.source_item_ids).issubset(source_items))
            self.assertTrue(set(entry.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                entry.secondary_surface_score,
                min(
                    1000,
                    max(
                        0,
                        entry.roadmap_traceability_score * 2
                        + entry.source_composition_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.v5_v6_foundation_score * 2
                        + entry.activation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_evolution_secondary_surface", entry.context_tags)
            self.assertIn(
                "secondary_surface_activation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("kb_storage_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.explainability_notes)
            self.assertTrue(entry.advisory_actions)
            self.assertTrue(entry.evidence)
            self.assertTrue(entry.hitl_required_before_secondary_surface_activation)
            self.assertTrue(entry.secondary_surface_implemented)
            self.assertTrue(entry.secondary_surface_metadata_implemented)
            self.assertTrue(entry.all_roadmap_items_traceable)
            self.assertTrue(entry.v5_policy_foundation_used)
            self.assertTrue(entry.v6_learning_foundation_used)
            self.assertTrue(entry.v6_knowledge_core_foundation_used)
            self.assertTrue(entry.all_sources_metadata_only)
            self.assertFalse(entry.secondary_surface_activation_implemented)
            self.assertFalse(entry.adaptive_learning_application_implemented)
            self.assertFalse(entry.adaptive_execution_policy_application_implemented)
            self.assertFalse(entry.knowledge_operation_execution_implemented)
            self.assertFalse(entry.automatic_kb_update_execution_implemented)
            self.assertFalse(entry.documentation_fetch_execution_implemented)
            self.assertFalse(entry.embedding_refresh_execution_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.ranking_mutation_implemented)
            self.assertFalse(entry.knowledge_health_monitoring_execution_implemented)
            self.assertFalse(entry.quality_score_computation_implemented)
            self.assertFalse(entry.knowledge_gap_detection_execution_implemented)
            self.assertFalse(entry.knowledge_conflict_resolution_execution_implemented)
            self.assertFalse(entry.knowledge_drift_detection_execution_implemented)
            self.assertFalse(entry.source_reliability_scoring_execution_implemented)
            self.assertFalse(entry.knowledge_consolidation_execution_implemented)
            self.assertFalse(entry.knowledge_lifecycle_management_execution_implemented)
            self.assertFalse(entry.provenance_graph_mutation_implemented)
            self.assertFalse(entry.version_graph_mutation_implemented)
            self.assertFalse(entry.knowledge_snapshot_engine_execution_implemented)
            self.assertFalse(entry.knowledge_rollback_execution_implemented)
            self.assertFalse(entry.freshness_scan_execution_implemented)
            self.assertFalse(entry.trust_score_computation_implemented)
            self.assertFalse(entry.kb_storage_write_implemented)
            self.assertFalse(entry.source_record_update_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        operational = knowledge_evolution_secondary_surface_entry_by_id(
            "knowledge_evolution_secondary::knowledge_operational_support_surface",
            plan,
        )
        self.assertIsNotNone(operational)
        assert operational is not None
        self.assertEqual(operational.status, "guarded")
        self.assertEqual(operational.confidence, "guarded")
        self.assertEqual(
            len(
                knowledge_evolution_secondary_surface_entries_for_status(
                    "review_required",
                    plan,
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                knowledge_evolution_secondary_surface_entries_for_confidence(
                    "high",
                    plan,
                )
            ),
            1,
        )

    def test_plan_rejects_mismatched_secondary_surface_metadata(self) -> None:
        plan = build_knowledge_evolution_secondary_surface()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            KnowledgeEvolutionSecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_secondary_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_secondary_surface_score must match",
        ):
            KnowledgeEvolutionSecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activated_secondary_surface_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "activated_secondary_surface_ids must remain empty",
        ):
            KnowledgeEvolutionSecondarySurfacePlan(**payload)

    def test_secondary_surface_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review the knowledge evolution secondary surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_evolution_secondary_surface(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_secondary_surface_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_knowledge_evolution_secondary_surface(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for entry in plan.entries
                    for field in (
                        entry.secondary_surface_id,
                        entry.surface_kind,
                        entry.status,
                        entry.confidence,
                        entry.surface_axis,
                        *entry.roadmap_items,
                        *entry.source_plan_roles,
                        *entry.source_serialization_versions,
                        *entry.source_item_ids,
                        entry.surface_summary,
                        *entry.context_tags,
                        *entry.explainability_notes,
                        *entry.advisory_actions,
                        *entry.evidence,
                        *entry.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "activate_secondary_surface(",
            "apply_adaptive_learning(",
            "apply_execution_policy(",
            "execute_knowledge_operation(",
            "execute_automatic_kb_update(",
            "fetch_documentation(",
            "refresh_embedding(",
            "execute_retrieval(",
            "mutate_ranking(",
            "run_health_monitor(",
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
